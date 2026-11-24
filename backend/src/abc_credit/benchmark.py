"""Measure the complete model decision path against the project latency target."""

import argparse
import json
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter

from abc_credit.integrations.model import DEFAULT_TRAINED_MODEL_PATH, LoanModel, TrainedLoanModel
from abc_credit.schemas.questionnaire import QuestionId
from abc_credit.services.decision_service import build_model_payload
from abc_credit.services.reference_data import DEFAULT_REFERENCE_TABLES_PATH, ReferenceData

DEFAULT_ITERATIONS = 30
DEFAULT_WARMUPS = 3
DEFAULT_TARGET_MS = 2_000.0


def percentile_95(values: list[float]) -> float:
    """Return the nearest-rank 95th percentile for a non-empty sample."""

    if not values:
        raise ValueError("at least one latency measurement is required")
    ordered = sorted(values)
    return ordered[ceil(0.95 * len(ordered)) - 1]


def benchmark_model(
    model: LoanModel,
    payload: dict[str, int | float | str | None],
    *,
    iterations: int,
    warmups: int,
    target_ms: float,
) -> dict[str, object]:
    """Measure prediction plus applicant-facing explanation generation."""

    if iterations < 1 or warmups < 0 or target_ms <= 0:
        raise ValueError("iterations and target must be positive; warmups cannot be negative")

    for _ in range(warmups):
        model.predict(payload)
        model.explain(payload)

    timings: list[float] = []
    for _ in range(iterations):
        started = perf_counter()
        prediction = model.predict(payload)
        explanation = model.explain(payload)
        timings.append((perf_counter() - started) * 1_000)
        if explanation.get("decision") != prediction.decision:
            raise RuntimeError("model prediction and explanation decisions disagree")

    p95_ms = percentile_95(timings)
    return {
        "model_version": model.version,
        "iterations": iterations,
        "warmups": warmups,
        "median_ms": round(median(timings), 3),
        "p95_ms": round(p95_ms, 3),
        "max_ms": round(max(timings), 3),
        "target_ms": target_ms,
        "passed": p95_ms <= target_ms,
        "scope": "prediction_plus_explanation",
    }


def representative_payload(reference: ReferenceData) -> dict[str, int | float | str | None]:
    """Build one valid synthetic payload without using an applicant record."""

    pincode = next(
        (
            value
            for value in sorted(reference.tables["pincode_tier"])
            if len(value) == 6 and value.isdigit()
        ),
        None,
    )
    if pincode is None:
        raise ValueError("reference tables contain no valid six-digit pincode")
    answers = {
        QuestionId.AGE: 30,
        QuestionId.PINCODE: pincode,
        QuestionId.VEHICLE: reference.vehicles[0].value,
        QuestionId.VEHICLE_PRICE: 100_000,
        QuestionId.LOAN_AMOUNT: 75_000,
        QuestionId.PAST_LOANS_ACTIVE: "NO_PAST_LOANS",
    }
    return build_model_payload(answers, reference)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_TRAINED_MODEL_PATH)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_TABLES_PATH)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--target-ms", type=float, default=DEFAULT_TARGET_MS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    model = TrainedLoanModel.from_path(args.model)
    reference = ReferenceData.from_path(args.reference)
    result = benchmark_model(
        model,
        representative_payload(reference),
        iterations=args.iterations,
        warmups=args.warmups,
        target_ms=args.target_ms,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
