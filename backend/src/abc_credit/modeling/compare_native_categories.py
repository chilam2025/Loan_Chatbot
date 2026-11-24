"""Compare categorical representations using existing development splits."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from abc_credit.data_quality import CATEGORICAL_COLUMNS
from abc_credit.modeling.compare_boosting import make_candidate
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/native_category_comparison")

CATEGORICAL_FIELDS = [
    column for column in CATEGORICAL_COLUMNS
    if column != "Gender"
]


def make_native_pipeline():
    pipeline = make_candidate("hist_gradient_boosting", False)
    original = pipeline.named_steps["preprocessor"]

    # Preserve the same numeric preprocessing and categorical imputation.
    transformers = []

    for name, transformer, columns in original.transformers:
        transformer = clone(transformer)

        if name == "categorical":
            transformer.set_params(
                encoder=OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    min_frequency=10,
                    max_categories=100,
                    dtype=np.float64,
                )
            )

        transformers.append((name, transformer, columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    pipeline.set_params(
        preprocessor=preprocessor,
        classifier__categorical_features=CATEGORICAL_FIELDS,
    )

    return pipeline


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    models = {
        "boosting_one_hot": make_candidate(
            "hist_gradient_boosting", False
        ),
        "boosting_native_categories": make_native_pipeline(),
    }

    summaries = []
    recall_results = []

    for name, model in models.items():
        print(f"Training: {name}", flush=True)

        with threadpool_limits(limits=2):
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
            "model": name,
            "threshold": threshold,
            **metrics,
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

            recall_results.append({
                "model": name,
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

        # Check inference with an unseen category.
        sample = splits.X_validation.head(1).copy()
        sample["Make_Code"] = "UNSEEN_EXPERIMENTAL_MAKE"

        with threadpool_limits(limits=2):
            unknown_probability = model.predict_proba(sample)

        assert np.isfinite(unknown_probability).all()

        pd.DataFrame(summaries).to_csv(
            OUTPUT / "validation_comparison.csv",
            index=False,
        )
        pd.DataFrame(recall_results).to_csv(
            OUTPUT / "matched_recall_comparison.csv",
            index=False,
        )

        print(f"Completed: {name}", flush=True)

    print("\nVALIDATION COMPARISON")
    print(pd.DataFrame(summaries).round(5).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT}")
    print("Test partition was not evaluated.")
    print("Packaged model was not modified.")


if __name__ == "__main__":
    main()