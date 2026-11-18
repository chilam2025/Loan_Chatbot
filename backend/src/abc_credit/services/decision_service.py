"""Submit applications to the predictive model and apply the deterministic decision rule."""

import logging
import os
from math import isfinite
from re import fullmatch
from time import perf_counter
from typing import get_args
from uuid import UUID

from abc_credit.db.decision_repository import (
    DecisionRecord,
    IncompleteAnswersError,
    SQLAlchemyDecisionRepository,
)
from abc_credit.db.repository import RepositoryNotFoundError, RepositoryStateConflictError
from abc_credit.integrations.model import (
    LoanModel,
    ModelFeatures,
    ModelProvider,
    ModelUnavailableError,
)
from abc_credit.schemas.decisions import DecisionOutcome
from abc_credit.schemas.questionnaire import AnswerValue, QuestionId
from abc_credit.services.questionnaire import QUESTION_BY_ID, calculate_ltv

DECISION_THRESHOLD_ENV = "ABC_DECISION_THRESHOLD"
DEFAULT_DECISION_THRESHOLD = 0.5
EXPECTED_MODEL_VERSION_ENV = "ABC_EXPECTED_MODEL_VERSION"
MODEL_VERSION_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"

# Model columns from the training contract that the questionnaire does not collect yet.
# They are sent as None so the trained pipeline's imputers handle them explicitly.
UNCOLLECTED_MODEL_FEATURES = (
    "Product_Code",
    "Final_Tier",
    "Make_Code",
    "Model_Description",
    "Model_Variant",
)

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


def get_decision_threshold() -> float:
    """Read the approval threshold, rejecting values that would make decisions trivial."""

    configured = os.getenv(DECISION_THRESHOLD_ENV)
    if not configured:
        return DEFAULT_DECISION_THRESHOLD
    try:
        threshold = float(configured)
    except ValueError as error:
        raise ValueError(f"{DECISION_THRESHOLD_ENV} must be a number") from error
    if not 0 < threshold < 1:
        raise ValueError(f"{DECISION_THRESHOLD_ENV} must be between 0 and 1")
    return threshold


def validate_model(model: LoanModel) -> None:
    """Refuse to start with a model whose identity cannot be audited or was not approved."""

    version = getattr(model, "version", None)
    if not isinstance(version, str) or fullmatch(MODEL_VERSION_PATTERN, version) is None:
        raise ValueError("model version must be a short identifier such as 'mock-0.1.0'")
    if getattr(model, "provider", None) not in get_args(ModelProvider):
        raise ValueError("model provider must be 'mock' or 'trained'")
    expected = os.getenv(EXPECTED_MODEL_VERSION_ENV)
    if expected and version != expected:
        raise ValueError(
            f"loaded model version {version!r} does not match "
            f"{EXPECTED_MODEL_VERSION_ENV}={expected!r}"
        )


def build_model_features(answers: dict[QuestionId, AnswerValue]) -> ModelFeatures:
    """Map validated questionnaire answers to the model's training column names."""

    loan_amount = answers.get(QuestionId.LOAN_AMOUNT)
    vehicle_price = answers.get(QuestionId.VEHICLE_PRICE)
    ltv = (
        calculate_ltv(loan_amount, vehicle_price)
        if isinstance(loan_amount, int) and isinstance(vehicle_price, int)
        else None
    )
    features: ModelFeatures = {
        "Age": answers.get(QuestionId.AGE),
        "Loan_Amount": loan_amount,
        "LTV": ltv,
        "Net_salary": answers.get(QuestionId.NET_SALARY),
        "Qualifications": answers.get(QuestionId.QUALIFICATIONS),
        "Employment_Type": answers.get(QuestionId.EMPLOYMENT_TYPE),
        "Pincode": answers.get(QuestionId.PINCODE),
        "Gender": answers.get(QuestionId.GENDER),
        "Resident_Type": answers.get(QuestionId.RESIDENT_TYPE),
        "PAST_LOANS_ACTIVE": answers.get(QuestionId.PAST_LOANS_ACTIVE),
    }
    features.update(dict.fromkeys(UNCOLLECTED_MODEL_FEATURES))
    return features


def apply_decision_rule(approval_probability: float, threshold: float) -> DecisionOutcome:
    """The only place an outcome is derived; no language model participates."""

    if approval_probability >= threshold:
        return DecisionOutcome.APPROVED
    return DecisionOutcome.DECLINED


class DecisionService:
    """Coordinate one-time submission, model inference, and decision recording."""

    def __init__(
        self,
        repository: SQLAlchemyDecisionRepository,
        model: LoanModel,
        *,
        threshold: float | None = None,
    ) -> None:
        validate_model(model)
        self._repository = repository
        self._model = model
        if threshold is None:
            # Precedence: explicit ABC_DECISION_THRESHOLD, then the threshold packaged
            # with the model (its validated operating point), then the default.
            model_threshold = getattr(model, "default_threshold", None)
            if os.getenv(DECISION_THRESHOLD_ENV) or model_threshold is None:
                threshold = get_decision_threshold()
            else:
                threshold = float(model_threshold)
        self._threshold = threshold
        if not 0 < self._threshold < 1:
            raise ValueError("decision threshold must be between 0 and 1")

    @property
    def model(self) -> LoanModel:
        return self._model

    def submit(self, application_id: UUID) -> DecisionRecord:
        try:
            answers = self._repository.claim_for_submission(
                application_id,
                required_questions=frozenset(QUESTION_BY_ID),
            )
        except RepositoryNotFoundError as error:
            raise DecisionNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise DecisionStateError(error.application_status) from error
        except IncompleteAnswersError as error:
            raise MissingAnswersError(error.missing_questions) from error

        started = perf_counter()
        try:
            probability = float(
                self._model.predict_approval_probability(build_model_features(answers))
            )
            if not isfinite(probability) or not 0 <= probability <= 1:
                raise ModelUnavailableError("model returned a probability outside [0, 1]")
        except Exception as error:
            # Any model fault leaves the application retryable instead of guessing an outcome.
            self._repository.release_submission(application_id, reason=type(error).__name__)
            logger.warning(
                "model_inference_failed model_version=%s error_type=%s",
                self._model.version,
                type(error).__name__,
            )
            raise ModelUnavailableError("model inference failed") from error
        latency_ms = round((perf_counter() - started) * 1000, 3)

        decision = apply_decision_rule(probability, self._threshold)
        record = self._repository.record_decision(
            application_id,
            decision=decision,
            approval_probability=probability,
            threshold=self._threshold,
            model_version=self._model.version,
            model_provider=self._model.provider,
            latency_ms=latency_ms,
        )
        logger.info(
            "model_decision_recorded model_version=%s probability=%.4f threshold=%.4f "
            "decision=%s latency_ms=%.3f",
            record.model_version,
            record.approval_probability,
            record.threshold,
            record.decision.value,
            record.latency_ms,
        )
        return record

    def get(self, application_id: UUID) -> DecisionRecord:
        try:
            return self._repository.get_decision(application_id)
        except RepositoryNotFoundError as error:
            raise DecisionNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise DecisionStateError(error.application_status) from error
