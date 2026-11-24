import pytest

from abc_credit.benchmark import benchmark_model, percentile_95
from abc_credit.integrations.model import MockLoanModel


def test_nearest_rank_p95() -> None:
    assert percentile_95([5, 1, 4, 2, 3]) == 5
    with pytest.raises(ValueError, match="at least one"):
        percentile_95([])


def test_benchmark_covers_prediction_and_explanation() -> None:
    model = MockLoanModel()
    payload = {"LTV": 75.0, "PAST_LOANS_ACTIVE": "NO_PAST_LOANS"}

    result = benchmark_model(
        model,
        payload,
        iterations=3,
        warmups=1,
        target_ms=2_000,
    )

    assert result["model_version"] == model.version
    assert result["scope"] == "prediction_plus_explanation"
    assert result["iterations"] == 3
    assert result["passed"] is True
    assert result["p95_ms"] >= 0


@pytest.mark.parametrize(
    ("iterations", "warmups", "target_ms"),
    [(0, 0, 1), (1, -1, 1), (1, 0, 0)],
)
def test_benchmark_rejects_invalid_settings(iterations, warmups, target_ms) -> None:
    with pytest.raises(ValueError, match="iterations and target"):
        benchmark_model(
            MockLoanModel(),
            {"LTV": 75.0},
            iterations=iterations,
            warmups=warmups,
            target_ms=target_ms,
        )
