"""Compare CatBoost LTV treatments on existing development splits."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from abc_credit.data_quality import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/catboost_comparison")

CATEGORICAL_FIELDS = [
    column for column in CATEGORICAL_COLUMNS
    if column != "Gender"
]
INPUT_FIELDS = NUMERIC_COLUMNS + CATEGORICAL_FIELDS


class PrepareCatBoost(BaseEstimator, TransformerMixin):
    def __init__(self, fix_zero_ltv=False):
        self.fix_zero_ltv = fix_zero_ltv

    def fit(self, X, y=None):
        valid_ltv = X["LTV"].where(
            X["LTV"].between(0, 100, inclusive="right")
        )
        self.ltv_median_ = valid_ltv.median()

        if self.fix_zero_ltv and pd.isna(self.ltv_median_):
            raise ValueError("No valid training LTV available for imputation")

        return self

    def transform(self, X):
        result = X[INPUT_FIELDS].copy()

        for column in CATEGORICAL_FIELDS:
            result[column] = (
                result[column]
                .astype("string")
                .str.strip()
                .fillna("__MISSING__")
                .astype(object)
            )

        for column in NUMERIC_COLUMNS:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            ).astype("float64")

        if self.fix_zero_ltv:
            missing_or_zero = (
                result["LTV"].isna() | result["LTV"].eq(0)
            )
            result["LTV_missing_or_zero"] = (
                missing_or_zero.astype(float)
            )
            result["LTV"] = result["LTV"].mask(
                missing_or_zero, self.ltv_median_
            )

        return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    summaries = []
    recall_rows = []

    for name, apply_fix in [
        ("catboost_original_ltv", False),
        ("catboost_imputed_ltv", True),
    ]:
        model = Pipeline([
            (
                "prepare",
                PrepareCatBoost(fix_zero_ltv=apply_fix),
            ),
            (
                "classifier",
                CatBoostClassifier(
                    iterations=600,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3,
                    loss_function="Logloss",
                    cat_features=CATEGORICAL_FIELDS,
                    random_seed=42,
                    thread_count=2,
                    verbose=100,
                    allow_writing_files=False,
                ),
            ),
        ])

        print(f"\nTraining: {name}", flush=True)
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

        threshold = float(
            table.sort_values(
                ["cost", "threshold"],
                ascending=[True, False],
            ).iloc[0]["threshold"]
        )

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

            recall_rows.append({
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

        # Confirm that an unseen category does not break prediction.
        sample = splits.X_validation.head(1).copy()
        sample["Make_Code"] = "UNSEEN_EXPERIMENTAL_MAKE"
        assert np.isfinite(model.predict_proba(sample)).all()

        pd.DataFrame(summaries).to_csv(
            OUTPUT / "validation_comparison.csv",
            index=False,
        )
        pd.DataFrame(recall_rows).to_csv(
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