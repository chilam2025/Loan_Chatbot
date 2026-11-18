"""Predictive-model boundary: the academic mock and an adapter for Person 1's trained model."""

import os
from math import exp, isfinite
from pathlib import Path
from typing import Any, Literal, Protocol

ModelFeatures = dict[str, int | float | str | None]
ModelProvider = Literal["mock", "trained"]

MODEL_PROVIDER_ENV = "ABC_MODEL_PROVIDER"
MODEL_PATH_ENV = "ABC_MODEL_PATH"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRAINED_MODEL_PATH = (
    REPOSITORY_ROOT / "models" / "catboost_pincode_prefix_v1" / "decision_model.joblib"
)


class ModelUnavailableError(RuntimeError):
    """Raised when the predictive model cannot produce a trustworthy probability."""


class LoanModel(Protocol):
    """Contract every predictive model must satisfy to be used for decisions."""

    @property
    def version(self) -> str: ...

    @property
    def provider(self) -> ModelProvider: ...

    def predict_approval_probability(self, features: ModelFeatures) -> float: ...


class MockLoanModel:
    """Deterministic stand-in for Person 1's model; it is not a trained credit model.

    The score rises with annual income relative to the requested loan and falls when
    the applicant already has active loans or borrows above 90% of the vehicle price,
    so demonstrations produce both outcomes.
    """

    version = "mock-0.1.0"
    provider: ModelProvider = "mock"

    def predict_approval_probability(self, features: ModelFeatures) -> float:
        net_salary = features.get("Net_salary")
        loan_amount = features.get("Loan_Amount")
        if not isinstance(net_salary, int | float) or not isinstance(loan_amount, int | float):
            raise ModelUnavailableError("mock model requires Net_salary and Loan_Amount")
        if loan_amount <= 0:
            raise ModelUnavailableError("mock model requires a positive Loan_Amount")

        income_cover = min(net_salary * 12 / loan_amount, 4.0)
        score = -1.0 + 1.5 * income_cover
        if features.get("PAST_LOANS_ACTIVE") == "PAST_LOANS_ACTIVE":
            score -= 0.6
        ltv = features.get("LTV")
        if isinstance(ltv, int | float) and ltv > 90:
            score -= 0.5
        return round(1 / (1 + exp(-score)), 4)


class TrainedLoanModel:
    """Adapter for Person 1's packaged decision model.

    The packaged model scores the probability of a historical DECLINED decision and
    declines at or above its own threshold. This adapter converts that score to an
    approval probability and exposes the equivalent approval threshold, so the
    decision service's single rule (approval >= threshold -> APPROVED) is unchanged.
    """

    provider: ModelProvider = "trained"

    def __init__(self, bundle: dict[str, Any]) -> None:
        metadata = bundle["metadata"]
        self._bundle = bundle
        self.version: str = metadata["model_version"]
        self.decline_threshold = float(metadata["threshold"])
        self.default_threshold = 1 - self.decline_threshold

    @classmethod
    def from_path(cls, path: str | Path) -> "TrainedLoanModel":
        """Load a trusted artifact once at startup; joblib must never load untrusted files."""

        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(
                f"trained model artifact not found at {artifact}; build it with "
                "`uv run --project backend python -m "
                "abc_credit.modeling.package_catboost_pincode_prefix` from the repository root"
            )
        # Imported lazily so the mock provider works without the modeling dependencies.
        from abc_credit.modeling.inference import load_model

        return cls(load_model(artifact))

    def predict_approval_probability(self, features: ModelFeatures) -> float:
        from abc_credit.modeling.inference import predict_applications

        try:
            result = predict_applications(self._bundle, [features])
        except ValueError as error:
            raise ModelUnavailableError(f"trained model rejected the input: {error}") from error
        declined = float(result["probability_declined"].iloc[0])
        if not isfinite(declined) or not 0 <= declined <= 1:
            raise ModelUnavailableError("trained model returned an invalid probability")
        return 1 - declined


def load_configured_model() -> LoanModel:
    """Choose the model from ABC_MODEL_PROVIDER (`mock` by default, or `trained`)."""

    provider = os.getenv(MODEL_PROVIDER_ENV, "mock").strip().lower()
    if provider == "mock":
        return MockLoanModel()
    if provider == "trained":
        return TrainedLoanModel.from_path(
            os.getenv(MODEL_PATH_ENV) or DEFAULT_TRAINED_MODEL_PATH
        )
    raise ValueError(f"{MODEL_PROVIDER_ENV} must be 'mock' or 'trained'")
