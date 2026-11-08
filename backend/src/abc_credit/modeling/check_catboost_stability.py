"""Compare candidate stability inside the original training partition."""

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from abc_credit.modeling.compare_catboost import (
    CATEGORICAL_FIELDS,
    PrepareCatBoost,
)
from abc_credit.modeling.compare_ltv_fix import build_model
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/catboost_stability")


def make_candidate(name):
    if name == "forest_imputed_ltv":
        return build_model(True)

    return Pipeline([
        (
            "prepare",
            PrepareCatBoost(
                fix_zero_ltv=(name == "catboost_imputed_ltv")
            ),
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
                verbose=False,
                allow_writing_files=False,
            ),
        ),
    ])


def select_threshold(y, probabilities):
    table = threshold_table(y, probabilities)
    table["cost"] = (
        5 * table["historically_declined_predicted_approved"]
        + table["historically_approved_predicted_declined"]
    )

    return float(
        table.sort_values(
            ["cost", "threshold"],
            ascending=[True, False],
        ).iloc[0]["threshold"]
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    # Use only the original training partition.
    X = splits.X_train
    y = splits.y_train.eq("DECLINED").astype(int)

    outer_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    names = [
        "forest_imputed_ltv",
        "catboost_original_ltv",
        "catboost_imputed_ltv",
    ]

    results = []

    for fold, (development_positions, evaluation_positions) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_development = X.iloc[development_positions]
        y_development = y.iloc[development_positions]

        X_evaluation = X.iloc[evaluation_positions]
        y_evaluation = y.iloc[evaluation_positions]

        # Reserve an inner subset for threshold selection.
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

        for name in names:
            print(f"Training fold {fold}: {name}", flush=True)
            model = make_candidate(name)

            # Each candidate learns preprocessing only from X_fit.
            model.fit(X_fit, y_fit)
            positive_index = list(model.classes_).index(1)

            threshold_scores = model.predict_proba(
                X_threshold
            )[:, positive_index]

            threshold = select_threshold(
                y_threshold, threshold_scores
            )

            # Evaluate the same fitted model at its locked inner threshold.
            evaluation_scores = model.predict_proba(
                X_evaluation
            )[:, positive_index]

            metrics = evaluate(
                y_evaluation, evaluation_scores, threshold
            )
            matrix = metrics.pop(
                "confusion_matrix_approved_declined"
            )

            cost = 5 * matrix[1][0] + matrix[0][1]
            all_approved_cost = 5 * int(y_evaluation.sum())

            results.append({
                "fold": fold,
                "model": name,
                "threshold": threshold,
                **metrics,
                "false_declines": matrix[0][1],
                "missed_declines": matrix[1][0],
                "detected_declines": matrix[1][1],
                "cost_per_1000": (
                    1000 * cost / len(y_evaluation)
                ),
                "cost_reduction_vs_all_approved_percent": (
                    100 * (all_approved_cost - cost)
                    / all_approved_cost
                ),
            })

            pd.DataFrame(results).to_csv(
                OUTPUT / "fold_results.csv",
                index=False,
            )
            print(f"Completed fold {fold}: {name}", flush=True)

    results = pd.DataFrame(results)

    columns = [
        "declined_precision",
        "declined_recall",
        "declined_f1",
        "declined_pr_auc_average_precision",
        "brier_score",
        "cost_per_1000",
        "cost_reduction_vs_all_approved_percent",
    ]

    summary = results.groupby("model")[columns].agg(
        ["mean", "std"]
    )
    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()
    summary.to_csv(OUTPUT / "stability_summary.csv", index=False)

    paired = results.pivot(
        index="fold",
        columns="model",
        values="cost_per_1000",
    )

    for name in [
        "catboost_original_ltv",
        "catboost_imputed_ltv",
    ]:
        paired[f"{name}_minus_forest"] = (
            paired[name] - paired["forest_imputed_ltv"]
        )

    paired.to_csv(OUTPUT / "paired_cost_comparison.csv")

    print("\nSTABILITY SUMMARY")
    print(summary.round(5).to_string(index=False))
    print("\nPAIRED COST COMPARISON")
    print(paired.round(5).to_string())
    print(f"\nResults saved to: {OUTPUT}")
    print("Original validation and test partitions were not evaluated.")


if __name__ == "__main__":
    main()