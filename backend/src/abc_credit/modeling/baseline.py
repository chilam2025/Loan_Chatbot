"""Evaluate an all-approved benchmark on the existing validation split."""

import json
from pathlib import Path

import numpy as np

from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.phase_b import evaluate


def main():
    splits = load_splits()
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    # A zero decline score predicts APPROVED for every application.
    probabilities = np.zeros(len(y_validation))
    metrics = evaluate(y_validation, probabilities, threshold=0.5)

    report = {
        "baseline": "Always predict APPROVED",
        "partition": "validation",
        "positive_class": "DECLINED",
        "records": len(y_validation),
        "metrics": metrics,
        "interpretation": (
            "High accuracy reflects the majority approval class. "
            "This benchmark detects no historical declines."
        ),
    }

    output = Path("reports/generated/baseline")
    output.mkdir(parents=True, exist_ok=True)
    (output / "all_approved_validation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("Test partition was not evaluated.")


if __name__ == "__main__":
    main()