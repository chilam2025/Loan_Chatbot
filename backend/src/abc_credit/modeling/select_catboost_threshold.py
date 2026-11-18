"""Compare CatBoost operating thresholds using validation data only."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

CANDIDATE = Path(
    "reports/generated/catboost_selection/training_candidate.joblib"
)
OUTPUT = Path("reports/generated/catboost_thresholds")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load only the trusted candidate generated locally.
    bundle = joblib.load(CANDIDATE)
    model = bundle["pipeline"]

    splits = load_splits()
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(
        splits.X_validation
    )[:, positive_index]

    table = threshold_table(y_validation, probabilities)

    # Existing illustrative historical-decision disagreement costs.
    table["assumed_cost_5_to_1"] = (
        5 * table["historically_declined_predicted_approved"]
        + table["historically_approved_predicted_declined"]
    )

    operating_points = {}

    cost_choice = table.sort_values(
        ["assumed_cost_5_to_1", "threshold"],
        ascending=[True, False],
    ).iloc[0]

    operating_points["minimum_cost_5_to_1"] = float(
        cost_choice["threshold"]
    )

    for target in [0.20, 0.40, 0.60]:
        selected = (
            table.loc[table["declined_recall"] >= target]
            .sort_values("threshold", ascending=False)
            .iloc[0]
        )
        operating_points[f"recall_at_least_{target:.0%}"] = float(
            selected["threshold"]
        )

    # Fixed default threshold is included as a reference.
    operating_points["default_0.5"] = 0.5

    rows = []
    approved_count = int(y_validation.eq(0).sum())

    for name, threshold in operating_points.items():
        metrics = evaluate(
            y_validation, probabilities, threshold
        )
        matrix = metrics.pop(
            "confusion_matrix_approved_declined"
        )

        rows.append({
            "operating_point": name,
            "threshold": threshold,
            **metrics,
            "correct_approvals": matrix[0][0],
            "false_declines": matrix[0][1],
            "missed_declines": matrix[1][0],
            "detected_declines": matrix[1][1],
            "false_decline_rate": (
                matrix[0][1] / approved_count
            ),
            "predicted_declines": (
                matrix[0][1] + matrix[1][1]
            ),
            "assumed_cost_5_to_1": (
                5 * matrix[1][0] + matrix[0][1]
            ),
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        OUTPUT / "operating_point_comparison.csv",
        index=False,
    )

    table.to_csv(
        OUTPUT / "all_thresholds.csv",
        index=False,
    )

    context = {
        "candidate_path": str(CANDIDATE),
        "candidate_sha256": hashlib.sha256(
            CANDIDATE.read_bytes()
        ).hexdigest(),
        "selected_parameters": bundle["metadata"]["selected_parameters"],
        "calibration": "None selected",
        "partition": "Existing development validation",
        "validation_records": len(y_validation),
        "historical_declines": int(y_validation.sum()),
        "status": "Comparison only; final threshold not locked",
        "cost_assumption": (
            "False approval cost 5; false decline cost 1. "
            "Illustrative historical-label disagreement costs."
        ),
    }

    (OUTPUT / "comparison_context.json").write_text(
        json.dumps(context, indent=2),
        encoding="utf-8",
    )

    print(comparison.round(5).to_string(index=False))
    print(f"\nSaved to: {OUTPUT}")
    print("Candidate artifact was not modified.")
    print("Test partition was not evaluated.")


if __name__ == "__main__":
    main()