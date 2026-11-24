"""Training-only ablation of financing features for CatBoost."""

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

OUTPUT = Path("reports/generated/catboost_feature_ablation")

VARIANTS = [
    "existing",
    "income_features",
    "vehicle_value",
    "full_bundle",
]


class PrepareAblation(PrepareCatBoost):
    def __init__(self, variant="existing"):
        super().__init__(fix_zero_ltv=True)
        self.variant = variant

    def transform(self, X):
        result = super().transform(X)

        loan = pd.to_numeric(
            X["Loan_Amount"], errors="coerce"
        ).astype(float)

        if self.variant in {"income_features", "full_bundle"}:
            income = pd.to_numeric(
                X["Net_salary"], errors="coerce"
            ).astype(float)

            result["loan_to_monthly_income"] = (
                loan / income.where(income > 0)
            )
            result["zero_income"] = income.eq(0).astype(float)
            result["income_missing"] = income.isna().astype(float)

        if self.variant in {"vehicle_value", "full_bundle"}:
            original_ltv = pd.to_numeric(
                X["LTV"], errors="coerce"
            ).astype(float)

            valid_ltv = original_ltv.where(
                original_ltv.between(0, 100, inclusive="right")
            )
            result["implied_vehicle_value"] = loan / (valid_ltv / 100)

        return result


def make_model(variant):
    return Pipeline([
        ("prepare", PrepareAblation(variant=variant)),
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


def select_operating_points(y, probabilities):
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

    rows = []

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

        for variant in VARIANTS:
            print(f"Training fold {fold}: {variant}", flush=True)

            model = make_model(variant)
            model.fit(X_fit, y_fit)

            positive_index = list(model.classes_).index(1)

            threshold_scores = model.predict_proba(
                X_threshold
            )[:, positive_index]

            operating_points = select_operating_points(
                y_threshold, threshold_scores
            )

            evaluation_scores = model.predict_proba(
                X_evaluation
            )[:, positive_index]

            assert np.isfinite(evaluation_scores).all()

            for operating_point, threshold in operating_points.items():
                metrics = evaluate(
                    y_evaluation,
                    evaluation_scores,
                    threshold,
                )
                matrix = metrics.pop(
                    "confusion_matrix_approved_declined"
                )

                rows.append({
                    "fold": fold,
                    "variant": variant,
                    "operating_point": operating_point,
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

            pd.DataFrame(rows).to_csv(
                OUTPUT / "fold_results.csv", index=False
            )
            print(f"Completed fold {fold}: {variant}", flush=True)

    results = pd.DataFrame(rows)

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
        ["variant", "operating_point"]
    )[metrics].agg(["mean", "std"])

    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()

    summary.to_csv(
        OUTPUT / "ablation_summary.csv", index=False
    )

    # Ranking is threshold-independent: one row per fold and variant.
    ranking = results.loc[
        results["operating_point"].eq("inner_recall_40_percent")
    ].pivot(
        index="fold",
        columns="variant",
        values="declined_pr_auc_average_precision",
    )
    ranking.to_csv(OUTPUT / "paired_average_precision.csv")

    print("\nABLATION SUMMARY")
    print(summary.round(5).to_string(index=False))
    print(f"\nSaved to: {OUTPUT}")
    print("Original validation and test partitions were not evaluated.")
    print("Saved model artifacts were not modified.")


if __name__ == "__main__":
    main()