"""Predictive-model boundary: Person 1's packaged six-question model and a test mock.

Every model scores the probability that the historical decision was DECLINED and
declines at or above its own locked threshold. The language model never participates.
"""

import logging
import os
from dataclasses import dataclass
from math import exp, isfinite
from pathlib import Path
from typing import Any, Literal, Protocol

ModelPayload = dict[str, int | float | str | None]
ModelProvider = Literal["mock", "trained"]
Explanation = dict[str, Any]

MODEL_PROVIDER_ENV = "ABC_MODEL_PROVIDER"
MODEL_PATH_ENV = "ABC_MODEL_PATH"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRAINED_MODEL_PATH = (
    REPOSITORY_ROOT / "models" / "catboost_minimal_v2" / "decision_model.joblib"
)

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when the predictive model cannot produce a trustworthy score (HTTP 503)."""


@dataclass(frozen=True)
class ModelPrediction:
    decision: Literal["APPROVED", "DECLINED"]
    probability_declined: float
    threshold: float
    model_version: str


class LoanModel(Protocol):
    """Contract every predictive model must satisfy to be used for decisions."""

    @property
    def version(self) -> str: ...

    @property
    def provider(self) -> ModelProvider: ...

    @property
    def threshold(self) -> float: ...

    def predict(self, payload: ModelPayload) -> ModelPrediction: ...

    def explain(self, payload: ModelPayload) -> Explanation: ...


def _decision(probability_declined: float, threshold: float) -> Literal["APPROVED", "DECLINED"]:
    return "DECLINED" if probability_declined >= threshold else "APPROVED"


class MockLoanModel:
    """Deterministic stand-in with the same ten-field contract; not a credit model.

    Used by automated tests and for demos when the private artifact is absent
    (ABC_MODEL_PROVIDER=mock). High LTV and an active loan raise the decline score.
    """

    version = "mock-minimal-0.2.0"
    provider: ModelProvider = "mock"
    threshold = 0.5

    def _score(self, payload: ModelPayload) -> dict[str, float]:
        ltv = payload.get("LTV")
        if not isinstance(ltv, int | float) or not isfinite(ltv):
            raise ModelUnavailableError("mock model requires a numeric LTV")
        return {
            "Loan size relative to vehicle price": (ltv - 75) / 10,
            "Active-loan status": (
                1.0 if payload.get("PAST_LOANS_ACTIVE") == "PAST_LOANS_ACTIVE" else -0.3
            ),
            "Location": 0.1,
        }

    def predict(self, payload: ModelPayload) -> ModelPrediction:
        score = sum(self._score(payload).values()) - 1.0
        probability = round(1 / (1 + exp(-score)), 4)
        return ModelPrediction(
            decision=_decision(probability, self.threshold),
            probability_declined=probability,
            threshold=self.threshold,
            model_version=self.version,
        )

    def explain(self, payload: ModelPayload) -> Explanation:
        factors = self._score(payload)
        prediction = self.predict(payload)

        def describe(items: list[tuple[str, float]]) -> list[dict[str, object]]:
            return [
                {
                    "factor": name,
                    "strength": "strong" if abs(v) >= 0.5 else "moderate" if abs(v) >= 0.2 else "slight",
                    "log_odds_contribution": round(v, 4),
                }
                for name, v in items
            ]

        raising = sorted(((k, v) for k, v in factors.items() if v > 0), key=lambda i: -i[1])
        lowering = sorted(((k, v) for k, v in factors.items() if v < 0), key=lambda i: i[1])
        return {
            "decision": prediction.decision,
            "probability_declined": prediction.probability_declined,
            "threshold": self.threshold,
            "model_version": self.version,
            "factors_raising_decline_score": describe(raising[:3]),
            "factors_lowering_decline_score": describe(lowering[:3]),
            "caveat": "Mock explanation for testing only.",
        }


class TrainedLoanModel:
    """Adapter for Person 1's packaged model (``catboost-minimal-v2``)."""

    provider: ModelProvider = "trained"

    def __init__(self, bundle: dict[str, Any]) -> None:
        metadata = bundle["metadata"]
        self._bundle = bundle
        self.version: str = metadata["model_version"]
        self.threshold: float = float(metadata["threshold"])
        self.feature_columns: list[str] = list(metadata["feature_columns"])

    @classmethod
    def from_path(cls, path: str | Path) -> "TrainedLoanModel":
        """Load a trusted artifact once at startup; joblib must never load untrusted files."""

        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(
                f"trained model artifact not found at {artifact}; unzip model_handoff.zip "
                "into the repository root or set ABC_MODEL_PATH"
            )
        # Imported lazily so the mock provider works without loading CatBoost.
        from abc_credit.modeling.inference import load_model

        return cls(load_model(artifact))

    def predict(self, payload: ModelPayload) -> ModelPrediction:
        from abc_credit.modeling.inference import predict_applications

        try:
            result = predict_applications(self._bundle, [payload]).iloc[0]
        except ValueError as error:
            raise ModelUnavailableError(f"trained model rejected the input: {error}") from error
        probability = float(result["probability_declined"])
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ModelUnavailableError("trained model returned an invalid probability")
        return ModelPrediction(
            decision=str(result["decision"]),  # type: ignore[arg-type]
            probability_declined=probability,
            threshold=float(result["threshold"]),
            model_version=str(result["model_version"]),
        )

    def explain(self, payload: ModelPayload) -> Explanation:
        from abc_credit.modeling.explain import explain_application

        return explain_application(self._bundle, payload, top_k=3)


def load_configured_model() -> LoanModel:
    """Choose the model from ABC_MODEL_PROVIDER (`trained` by default, or `mock`)."""

    provider = os.getenv(MODEL_PROVIDER_ENV, "trained").strip().lower()
    if provider == "mock":
        return MockLoanModel()
    if provider == "trained":
        return TrainedLoanModel.from_path(os.getenv(MODEL_PATH_ENV) or DEFAULT_TRAINED_MODEL_PATH)
    raise ValueError(f"{MODEL_PROVIDER_ENV} must be 'mock' or 'trained'")


def try_load_configured_model() -> LoanModel | None:
    """Load the model at startup; on failure log it and return None so requests get 503."""

    try:
        model = load_configured_model()
    except Exception as error:  # noqa: BLE001 - any load failure must leave the API up
        logger.error(
            "model_load_failed provider=%s path=%s error_type=%s error=%s",
            os.getenv(MODEL_PROVIDER_ENV, "trained"),
            os.getenv(MODEL_PATH_ENV) or DEFAULT_TRAINED_MODEL_PATH,
            type(error).__name__,
            error,
        )
        return None
    logger.info(
        "model_loaded provider=%s model_version=%s threshold=%.5f",
        model.provider,
        model.version,
        model.threshold,
    )
    return model
