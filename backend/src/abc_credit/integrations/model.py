"""Predictive-model boundary and the academic mock used until the trained model ships."""

from math import exp
from typing import Literal, Protocol

ModelFeatures = dict[str, int | float | str | None]
ModelProvider = Literal["mock", "trained"]


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
