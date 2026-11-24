"""Submit applications to the six-question model and record its decision.

The decision is the model's own: DECLINED when probability_declined >= the threshold
packaged with the model (locked by Person 1), otherwise APPROVED. No language model
participates, and the explanation can never change the decision.
"""

import logging
import os
from math import isfinite
from re import fullmatch
from time import perf_counter
from typing import Any, get_args
from uuid import UUID

from abc_credit.db.decision_repository import (
    DecisionRecord,
    IncompleteAnswersError,
    SQLAlchemyDecisionRepository,
)
from abc_credit.db.repository import RepositoryNotFoundError, RepositoryStateConflictError
from abc_credit.integrations.model import (
    LoanModel,
    ModelPayload,
    ModelProvider,
    ModelUnavailableError,
)
from abc_credit.modeling.reference_tables import build_minimal_payload
from abc_credit.schemas.decisions import DecisionOutcome
from abc_credit.schemas.questionnaire import AnswerValue, QuestionId
from abc_credit.services.questionnaire import QUESTION_IDS, model_answers
from abc_credit.services.reference_data import ReferenceData

EXPECTED_MODEL_VERSION_ENV = "ABC_EXPECTED_MODEL_VERSION"
HIDDEN_FACTORS_ENV = "ABC_HIDDEN_EXPLANATION_FACTORS"
# Location can act as a proxy for protected characteristics, so it is hidden by default.
DEFAULT_HIDDEN_FACTORS = "Location"
MODEL_VERSION_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"

logger = logging.getLogger(__name__)


class DecisionNotFoundError(LookupError):
    """Raised when an application identifier is unknown."""


class DecisionStateError(ValueError):
    """Raised when the application cannot be submitted or has no decision yet."""

    def __init__(self, application_status: str) -> None:
        self.application_status = application_status
        super().__init__(application_status)


class MissingAnswersError(ValueError):
    """Raised when required answers are absent at submission time."""

    def __init__(self, missing_questions: list[str]) -> None:
        self.missing_questions = missing_questions
        super().__init__(", ".join(missing_questions))


class InvalidSubmissionError(ValueError):
    """Raised when stored answers cannot form a valid model payload."""


def validate_model(model: LoanModel) -> None:
    """Refuse a model whose identity cannot be audited or was not approved."""

    version = getattr(model, "version", None)
    if not isinstance(version, str) or fullmatch(MODEL_VERSION_PATTERN, version) is None:
        raise ValueError("model version must be a short identifier such as 'catboost-minimal-v2'")
    if getattr(model, "provider", None) not in get_args(ModelProvider):
        raise ValueError("model provider must be 'mock' or 'trained'")
    threshold = getattr(model, "threshold", None)
    if not isinstance(threshold, int | float) or not 0 < threshold < 1:
        raise ValueError("model threshold must be between 0 and 1")
    expected = os.getenv(EXPECTED_MODEL_VERSION_ENV)
    if expected and version != expected:
        raise ValueError(
            f"loaded model version {version!r} does not match "
            f"{EXPECTED_MODEL_VERSION_ENV}={expected!r}"
        )


def build_model_payload(
    answers: dict[QuestionId, AnswerValue],
    reference: ReferenceData,
) -> ModelPayload:
    """Turn stored answers into the model's ten contract fields (Person 1's helper)."""

    return build_minimal_payload(model_answers(answers), reference.tables)


def masked_payload(payload: ModelPayload) -> dict[str, Any]:
    """The ten model fields for logs, with the pincode reduced to its first three digits."""

    masked = dict(payload)
    pincode = payload.get("Pincode")
    masked["Pincode"] = f"{str(pincode)[:3]}XXX" if pincode else None
    return masked


def hidden_factors() -> frozenset[str]:
    configured = os.getenv(HIDDEN_FACTORS_ENV, DEFAULT_HIDDEN_FACTORS)
    return frozenset(name.strip() for name in configured.split(",") if name.strip())


def public_explanation(explanation: dict[str, Any]) -> dict[str, Any]:
    """Keep factor names and strengths only; drop numbers and hidden (sensitive) factors."""

    hidden = hidden_factors()

    def clean(items: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {"factor": str(item["factor"]), "strength": str(item["strength"])}
            for item in items
            if item.get("factor") not in hidden
        ]

    return {
        "factors_raising_decline_score": clean(explanation.get("factors_raising_decline_score", [])),
        "factors_lowering_decline_score": clean(
            explanation.get("factors_lowering_decline_score", [])
        ),
        "caveat": str(explanation.get("caveat", "")),
    }


class DecisionService:
    """Coordinate one-time submission, model inference, and decision recording."""

    def __init__(
        self,
        repository: SQLAlchemyDecisionRepository,
        model: LoanModel | None,
        reference_data: ReferenceData | None,
    ) -> None:
        if model is not None:
            validate_model(model)
        self._repository = repository
        self._model = model
        self._reference_data = reference_data

    @property
    def model(self) -> LoanModel | None:
        return self._model

    @property
    def reference_data(self) -> ReferenceData | None:
        return self._reference_data

    @property
    def ready(self) -> bool:
        return self._model is not None and self._reference_data is not None

    def submit(self, application_id: UUID) -> DecisionRecord:
        model, reference = self._model, self._reference_data
        if model is None or reference is None:
            # Checked before claiming, so the application stays READY_FOR_REVIEW for a retry.
            # The unknown-application case still returns 404 through get().
            self._ensure_exists(application_id)
            raise ModelUnavailableError("model or reference tables are not loaded")

        try:
            answers = self._repository.claim_for_submission(
                application_id,
                required_questions=QUESTION_IDS,
            )
        except RepositoryNotFoundError as error:
            raise DecisionNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise DecisionStateError(error.application_status) from error
        except IncompleteAnswersError as error:
            raise MissingAnswersError(error.missing_questions) from error

        try:
            payload = build_model_payload(answers, reference)
        except ValueError as error:
            self._repository.release_submission(application_id, reason="InvalidPayload")
            raise InvalidSubmissionError(str(error)) from error

        # The applicant waits for both inference and the explanation. Measure
        # that complete model-processing path against the brief's Y target.
        started = perf_counter()
        try:
            prediction = model.predict(payload)
            if (
                not isfinite(prediction.probability_declined)
                or not 0 <= prediction.probability_declined <= 1
            ):
                raise ModelUnavailableError("model returned a probability outside [0, 1]")
        except Exception as error:
            # Any model fault leaves the application retryable instead of guessing an outcome.
            self._repository.release_submission(application_id, reason=type(error).__name__)
            logger.warning(
                "model_inference_failed model_version=%s error_type=%s",
                model.version,
                type(error).__name__,
            )
            raise ModelUnavailableError("model inference failed") from error
        explanation = self._explain(model, payload)
        latency_ms = round((perf_counter() - started) * 1000, 3)
        decision = DecisionOutcome(prediction.decision)
        masked = masked_payload(payload)
        record = self._repository.record_decision(
            application_id,
            decision=decision,
            probability_declined=prediction.probability_declined,
            threshold=prediction.threshold,
            model_version=prediction.model_version,
            model_provider=model.provider,
            latency_ms=latency_ms,
            explanation=explanation,
            audit_details={"model_inputs": masked},
        )
        logger.info(
            "model_decision_recorded model_version=%s threshold=%.5f probability_declined=%.5f "
            "decision=%s latency_ms=%.3f model_inputs=%s",
            record.model_version,
            record.threshold,
            record.probability_declined,
            record.decision.value,
            record.latency_ms,
            masked,
        )
        return record

    def get(self, application_id: UUID) -> DecisionRecord:
        try:
            return self._repository.get_decision(application_id)
        except RepositoryNotFoundError as error:
            raise DecisionNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise DecisionStateError(error.application_status) from error

    def _ensure_exists(self, application_id: UUID) -> None:
        try:
            self._repository.get_decision(application_id)
        except RepositoryNotFoundError as error:
            raise DecisionNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError:
            return

    @staticmethod
    def _explain(model: LoanModel, payload: ModelPayload) -> dict[str, Any] | None:
        """An explanation failure must never block or change the decision."""

        try:
            return public_explanation(model.explain(payload))
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "model_explanation_failed model_version=%s error_type=%s",
                model.version,
                type(error).__name__,
            )
            return None
