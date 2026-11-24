"""Compare Logistic Regression class weights on validation data only."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from abc_credit.modeling.feature_analysis import load_splits, make_pipeline
from abc_credit.modeling.phase_b import choose_threshold, evaluate

OUTPUT = Path("reports/generated/tree_model_comparison")


def result_row(name, y, probabilities, threshold, training_seconds, prediction_seconds):
    metrics = evaluate(y, probabilities, threshold)
    matrix = metrics.pop("confusion_matrix_approved_declined")

    return {
        "model": name,
        "threshold": threshold,
        **metrics,
        "historically_approved_predicted_declined": matrix[0][1],
        "historically_declined_predicted_approved": matrix[1][0],
        "validation_cost_5_to_1": 5 * matrix[1][0] + matrix[0][1],
        "training_seconds": training_seconds,
        "validation_batch_prediction_seconds": prediction_seconds,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    approved_count = int(y_train.eq(0).sum())
    declined_count = int(y_train.eq(1).sum())

    print(f"Training APPROVED: {approved_count:,}")
    print(f"Training DECLINED: {declined_count:,}")

    models = {
        "logistic_standard": make_pipeline(
            "without_gender"
        ).named_steps["classifier"],

        "random_forest_standard": RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=5,
            class_weight=None,
            random_state=42,
            n_jobs=2,
        ),

        "random_forest_balanced": RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=2,
        ),
    }
    selected_rows = []
    fixed_rows = []
    threshold_rows = []

    for name, classifier in models.items():
        pipeline = make_pipeline("without_gender")
        pipeline.set_params(classifier=clone(classifier))

        # Guard against accidentally running a different feature set.
        preprocessor = pipeline.named_steps["preprocessor"]
        used_columns = [
            column
            for _, _, columns in preprocessor.transformers
            for column in columns
        ]
        assert "Gender" not in used_columns
        assert "ID" not in used_columns
        assert "Decision" not in used_columns

        start = perf_counter()
        pipeline.fit(splits.X_train, y_train)
        training_seconds = perf_counter() - start

        start = perf_counter()
        probabilities = pipeline.predict_proba(splits.X_validation)[:, 1]
        prediction_seconds = perf_counter() - start

        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        # First compare both models at a fixed threshold.
        fixed_rows.append(
            result_row(
                name, y_validation, probabilities, 0.5,
                training_seconds, prediction_seconds,
            )
        )

        # Then tune each threshold using the same illustrative cost assumption.
        threshold = choose_threshold(
            y_validation,
            probabilities,
            false_approval_cost=5.0,
            false_decline_cost=1.0,
        )
        selected_rows.append(
            result_row(
                name, y_validation, probabilities, threshold,
                training_seconds, prediction_seconds,
            )
        )

        # Save the full threshold trade-off without applicant-level predictions.
        for value in np.linspace(0, 1, 201):
            row = result_row(
                name, y_validation, probabilities, float(value),
                training_seconds, prediction_seconds,
            )
            threshold_rows.append({
                key: row[key]
                for key in [
                    "model",
                    "threshold",
                    "declined_precision",
                    "declined_recall",
                    "declined_f1",
                    "historically_approved_predicted_declined",
                    "historically_declined_predicted_approved",
                    "validation_cost_5_to_1",
                ]
            })

        print(f"Completed: {name}", flush=True)

    fixed = pd.DataFrame(fixed_rows)
    selected = pd.DataFrame(selected_rows)

    fixed.to_csv(OUTPUT / "fixed_threshold_comparison.csv", index=False)
    selected.to_csv(OUTPUT / "tuned_threshold_comparison.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(
        OUTPUT / "threshold_tradeoffs.csv", index=False
    )

    print("\nFIXED THRESHOLD = 0.5")
    print(fixed.round(4).to_string(index=False))

    print("\nVALIDATION-TUNED THRESHOLDS")
    print(selected.round(4).to_string(index=False))

    print(f"\nResults saved to: {OUTPUT}")
    print("No test-set evaluation performed.")


if __name__ == "__main__":
    main()