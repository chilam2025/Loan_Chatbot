"""Training-only comparison of moderate CatBoost class weights."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.nested_model_comparison import make_catboost
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/catboost_weight_comparison")

# Positive class is DECLINED=1.
WEIGHTS = {
    "unweighted": [1.0, 1.0],
    "declined_weight_2": [1.0, 2.0],
    "declined_weight_5": [1.0, 5.0],
}


def choose_operating_points(y, probabilities):
    table = threshold_table(y, probabilities)

    # Exploratory target, not an agreed business requirement.
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

        for name, weights in WEIGHTS.items():
            print(f"Training fold {fold}: {name}", flush=True)

            # Uses zero-LTV imputation fitted inside this fold.
            model = make_catboost()
            model.set_params(
                classifier__depth=4,
                classifier__l2_leaf_reg=10,
                classifier__class_weights=weights,
            )
            model.fit(X_fit, y_fit)

            positive_index = list(model.classes_).index(1)

            threshold_scores = model.predict_proba(
                X_threshold
            )[:, positive_index]

            operating_points = choose_operating_points(
                y_threshold,
                threshold_scores,
            )

            evaluation_scores = model.predict_proba(
                X_evaluation
            )[:, positive_index]

            for point, threshold in operating_points.items():
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

            # Save progress after each candidate completes.
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

    summary.to_csv(
        OUTPUT / "weight_summary.csv",
        index=False,
    )

    # Paired fold comparisons at thresholds targeting inner 40% recall.
    recall_results = results.loc[
        results["operating_point"].eq("inner_recall_40_percent")
    ]

    paired = recall_results.pivot(
        index="fold",
        columns="model",
        values=[
            "declined_precision",
            "declined_recall",
            "declined_pr_auc_average_precision",
            "false_decline_rate",
        ],
    )
    paired.columns = [
        f"{metric}__{model}"
        for metric, model in paired.columns
    ]
    paired.to_csv(OUTPUT / "paired_recall_comparison.csv")

    print("\nWEIGHT COMPARISON")
    print(summary.round(5).to_string(index=False))
    print(f"\nSaved to: {OUTPUT}")
    print("Original validation and test partitions were not evaluated.")
    print("Existing model artifacts were not changed.")


if __name__ == "__main__":
    main()