"""Compare CatBoost with and without financing-related features."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from abc_credit.modeling.compare_catboost import (
    CATEGORICAL_FIELDS,
    PrepareCatBoost,
)
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/catboost_feature_comparison")


class PrepareFinancingFeatures(PrepareCatBoost):
    """Extend existing preparation with deterministic numeric features."""

    def transform(self, X):
        # Uses the parent's training-fitted LTV median.
        result = super().transform(X)

        income = pd.to_numeric(
            X["Net_salary"], errors="coerce"
        ).astype(float)

        loan = pd.to_numeric(
            X["Loan_Amount"], errors="coerce"
        ).astype(float)

        original_ltv = pd.to_numeric(
            X["LTV"], errors="coerce"
        ).astype(float)

        positive_income = income.where(income > 0)
        valid_original_ltv = original_ltv.where(
            original_ltv.between(0, 100, inclusive="right")
        )

        result["loan_to_monthly_income"] = loan / positive_income
        result["zero_income"] = income.eq(0).astype(float)
        result["income_missing"] = income.isna().astype(float)

        # Calculate before LTV imputation; do not infer a value from
        # the median-filled ratio.
        result["implied_vehicle_value"] = (
            loan / (valid_original_ltv / 100)
        )

        return result


def make_model(engineered):
    transformer = (
        PrepareFinancingFeatures(fix_zero_ltv=True)
        if engineered
        else PrepareCatBoost(fix_zero_ltv=True)
    )

    return Pipeline([
        ("prepare", transformer),
        (
            "classifier",
            CatBoostClassifier(
                iterations=600,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=10,
                loss_function="Logloss",
                cat_features=list(CATEGORICAL_FIELDS),
                random_seed=42,
                thread_count=2,
                verbose=False,
                allow_writing_files=False,
            ),
        ),
    ])


def operating_thresholds(y, probabilities):
    table = threshold_table(y, probabilities)

    recall_choice = (
        table.loc[table["declined_recall"] >= 0.40]
        .sort_values("threshold", ascending=False)
        .iloc[0]
    )

    table["cost"] = (
        5 * table["historically_declined_predicted_approved"]
        + table["historically_approved_predicted_declined"]
    )

    cost_choice = table.sort_values(
        ["cost", "threshold"],
        ascending=[True, False],
    ).iloc[0]

    return {
        "inner_recall_40_percent": float(recall_choice["threshold"]),
        "minimum_inner_cost_5_to_1": float(cost_choice["threshold"]),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    X = splits.X_train
    y = splits.y_train.eq("DECLINED").astype(int)

    outer_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    results = []

    for fold, (development_positions, evaluation_positions) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_development = X.iloc[development_positions]
        y_development = y.iloc[development_positions]

        X_evaluation = X.iloc[evaluation_positions]
        y_evaluation = y.iloc[evaluation_positions]

        X_fit, X_threshold, y_fit, y_threshold = train_test_split(
            X_development,
            y_development,
            test_size=0.20,
            stratify=y_development,
            random_state=42 + fold,
        )

        assert set(X_fit.index).isdisjoint(X_threshold.index)
        assert set(X_fit.index).isdisjoint(X_evaluation.index)
        assert set(X_threshold.index).isdisjoint(X_evaluation.index)

        for name, engineered in [
            ("existing_features", False),
            ("financing_features", True),
        ]:
            print(f"Training fold {fold}: {name}", flush=True)
            model = make_model(engineered)
            model.fit(X_fit, y_fit)

            positive_index = list(model.classes_).index(1)

            threshold_scores = model.predict_proba(
                X_threshold
            )[:, positive_index]

            thresholds = operating_thresholds(
                y_threshold, threshold_scores
            )

            evaluation_scores = model.predict_proba(
                X_evaluation
            )[:, positive_index]

            assert np.isfinite(evaluation_scores).all()

            for point, threshold in thresholds.items():
                metrics = evaluate(
                    y_evaluation,
                    evaluation_scores,
                    threshold,
                )
                matrix = metrics.pop(
                    "confusion_matrix_approved_declined"
                )

                results.append({
                    "fold": fold,
                    "model": name,
                    "operating_point": point,
                    "threshold": threshold,
                    **metrics,
                    "false_declines": matrix[0][1],
                    "missed_declines": matrix[1][0],
                    "detected_declines": matrix[1][1],
                    "false_decline_rate": (
                        matrix[0][1] / int(y_evaluation.eq(0).sum())
                    ),
                    "cost_per_1000": (
                        1000
                        * (5 * matrix[1][0] + matrix[0][1])
                        / len(y_evaluation)
                    ),
                })

            pd.DataFrame(results).to_csv(
                OUTPUT / "fold_results.csv",
                index=False,
            )
            print(f"Completed fold {fold}: {name}", flush=True)

    results = pd.DataFrame(results)

    metrics = [
        "declined_precision",
        "declined_recall",
        "declined_f1",
        "declined_pr_auc_average_precision",
        "brier_score",
        "false_decline_rate",
        "cost_per_1000",
    ]

    summary = results.groupby(
        ["model", "operating_point"]
    )[metrics].agg(["mean", "std"])

    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()
    summary.to_csv(OUTPUT / "feature_summary.csv", index=False)

    paired = results.loc[
        results["operating_point"].eq("inner_recall_40_percent")
    ].pivot(
        index="fold",
        columns="model",
        values=[
            "declined_pr_auc_average_precision",
            "declined_precision",
            "declined_recall",
            "false_decline_rate",
        ],
    )

    paired.columns = [
        f"{metric}__{model}"
        for metric, model in paired.columns
    ]
    paired.to_csv(OUTPUT / "paired_comparison.csv")

    print("\nFEATURE COMPARISON")
    print(summary.round(5).to_string(index=False))
    print(f"\nSaved to: {OUTPUT}")
    print("Original validation and test partitions were not evaluated.")
    print("Saved model artifacts were not changed.")


if __name__ == "__main__":
    main()