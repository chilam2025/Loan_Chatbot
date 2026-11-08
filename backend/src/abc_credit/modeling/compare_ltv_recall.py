"""Compare LTV treatments at validation recall targets."""

from pathlib import Path

import pandas as pd

from abc_credit.modeling.compare_ltv_fix import build_model
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/ltv_recall_comparison")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    results = []

    for name, apply_fix in [
        ("original_ltv", False),
        ("zero_ltv_imputed", True),
    ]:
        model = build_model(apply_fix)
        model.fit(splits.X_train, y_train)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(
            splits.X_validation
        )[:, positive_index]

        table = threshold_table(y_validation, probabilities)

        for target in [0.20, 0.40, 0.60]:
            eligible = table.loc[
                table["declined_recall"] >= target
            ]

            # Highest threshold achieving at least the requested recall.
            # Tied scores remain together, so recall can exceed the target.
            selected = eligible.sort_values(
                "threshold", ascending=False
            ).iloc[0]

            false_declines = int(
                selected["historically_approved_predicted_declined"]
            )
            missed_declines = int(
                selected["historically_declined_predicted_approved"]
            )

            results.append({
                "experiment": name,
                "target_recall": target,
                "actual_recall": float(selected["declined_recall"]),
                "threshold": float(selected["threshold"]),
                "declined_precision": float(
                    selected["declined_precision"]
                ),
                "declined_f1": float(selected["declined_f1"]),
                "detected_declines": int(
                    selected["correctly_detected_declines"]
                ),
                "missed_declines": missed_declines,
                "false_declines": false_declines,
                "false_decline_rate": (
                    false_declines / int(y_validation.eq(0).sum())
                ),
                "assumed_cost_5_to_1": (
                    5 * missed_declines + false_declines
                ),
            })

        print(f"Completed: {name}", flush=True)

    comparison = pd.DataFrame(results).sort_values(
        ["target_recall", "experiment"]
    )

    comparison.to_csv(
        OUTPUT / "matched_recall_comparison.csv",
        index=False,
    )

    print(comparison.round(5).to_string(index=False))
    print(f"\nSaved to: {OUTPUT}")
    print("Test partition and packaged model were not changed.")


if __name__ == "__main__":
    main()