"""Compare two fixed CatBoost pipelines on development validation data."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss

from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/final_catboost_feature_comparison")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    summaries = []
    recall_rows = []

    for variant in ["existing", "vehicle_value"]:
        print(f"Training: {variant}", flush=True)

        # Both use depth=4, L2=10 and zero-LTV imputation.
        model = make_model(variant)
        model.fit(splits.X_train, y_train)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(
            splits.X_validation
        )[:, positive_index]

        table = threshold_table(y_validation, probabilities)
        table["cost"] = (
            5 * table["historically_declined_predicted_approved"]
            + table["historically_approved_predicted_declined"]
        )

        selected = table.sort_values(
            ["cost", "threshold"],
            ascending=[True, False],
        ).iloc[0]

        threshold = float(selected["threshold"])
        metrics = evaluate(
            y_validation, probabilities, threshold
        )
        matrix = metrics.pop(
            "confusion_matrix_approved_declined"
        )

        summaries.append({
            "variant": variant,
            "threshold": threshold,
            **metrics,
            "log_loss": log_loss(
                y_validation, probabilities, labels=[0, 1]
            ),
            "false_declines": matrix[0][1],
            "missed_declines": matrix[1][0],
            "detected_declines": matrix[1][1],
            "assumed_cost_5_to_1": (
                5 * matrix[1][0] + matrix[0][1]
            ),
        })

        for target in [0.20, 0.40, 0.60]:
            row = (
                table.loc[table["declined_recall"] >= target]
                .sort_values("threshold", ascending=False)
                .iloc[0]
            )

            recall_rows.append({
                "variant": variant,
                "target_recall": target,
                "actual_recall": float(row["declined_recall"]),
                "threshold": float(row["threshold"]),
                "declined_precision": float(
                    row["declined_precision"]
                ),
                "false_declines": int(
                    row["historically_approved_predicted_declined"]
                ),
                "detected_declines": int(
                    row["correctly_detected_declines"]
                ),
            })

        # Save progress without exporting applicant-level predictions.
        pd.DataFrame(summaries).to_csv(
            OUTPUT / "validation_comparison.csv",
            index=False,
        )
        pd.DataFrame(recall_rows).to_csv(
            OUTPUT / "matched_recall_comparison.csv",
            index=False,
        )

        print(f"Completed: {variant}", flush=True)

    print("\nVALIDATION COMPARISON")
    print(pd.DataFrame(summaries).round(6).to_string(index=False))

    print("\nMATCHED RECALL")
    print(pd.DataFrame(recall_rows).round(6).to_string(index=False))

    print(f"\nResults saved to: {OUTPUT}")
    print("Test partition was not evaluated.")
    print("Existing artifacts were not modified.")


if __name__ == "__main__":
    main()