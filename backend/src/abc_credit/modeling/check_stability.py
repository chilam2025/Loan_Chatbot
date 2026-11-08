"""Check model stability using nested splits inside training data only."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from abc_credit.modeling.feature_analysis import load_splits, make_pipeline
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/model_stability")

# Illustrative assumption; retain only if justified/agreed by the team.
FALSE_APPROVAL_COST = 5
FALSE_DECLINE_COST = 1


def select_threshold(y, probabilities):
    table = threshold_table(y, probabilities)

    table["cost"] = (
        FALSE_APPROVAL_COST
        * table["historically_declined_predicted_approved"]
        + FALSE_DECLINE_COST
        * table["historically_approved_predicted_declined"]
    )

    # Break cost ties using the highest threshold.
    best = table.sort_values(
        ["cost", "threshold"],
        ascending=[True, False],
    ).iloc[0]

    return float(best["threshold"])


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

        # Inside the development fold, reserve data for choosing a threshold.
        X_fit, X_threshold, y_fit, y_threshold = train_test_split(
            X_development,
            y_development,
            test_size=0.20,
            stratify=y_development,
            random_state=42 + fold,
        )

        # Check that all three subsets are disjoint.
        assert set(X_fit.index).isdisjoint(X_threshold.index)
        assert set(X_fit.index).isdisjoint(X_evaluation.index)
        assert set(X_threshold.index).isdisjoint(X_evaluation.index)

        for name in ["logistic", "random_forest"]:
            model = make_pipeline("without_gender")

            if name == "random_forest":
                model.set_params(
                    classifier=RandomForestClassifier(
                        n_estimators=150,
                        max_depth=18,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=2,
                    )
                )

            # Preprocessing is fitted only on this fold's fitting subset.
            model.fit(X_fit, y_fit)
            positive_index = list(model.classes_).index(1)

            threshold_probabilities = model.predict_proba(
                X_threshold
            )[:, positive_index]

            threshold = select_threshold(
                y_threshold, threshold_probabilities
            )

            # Evaluate the same fitted model at the locked inner threshold.
            evaluation_probabilities = model.predict_proba(
                X_evaluation
            )[:, positive_index]

            metrics = evaluate(
                y_evaluation,
                evaluation_probabilities,
                threshold,
            )
            matrix = metrics.pop(
                "confusion_matrix_approved_declined"
            )

            cost = (
                FALSE_APPROVAL_COST * matrix[1][0]
                + FALSE_DECLINE_COST * matrix[0][1]
            )
            all_approved_cost = (
                FALSE_APPROVAL_COST * int(y_evaluation.sum())
            )

            results.append({
                "fold": fold,
                "model": name,
                "fit_rows": len(X_fit),
                "threshold_selection_rows": len(X_threshold),
                "evaluation_rows": len(X_evaluation),
                "threshold": threshold,
                **metrics,
                "historically_approved_predicted_declined": matrix[0][1],
                "historically_declined_predicted_approved": matrix[1][0],
                "correctly_detected_declines": matrix[1][1],
                "assumed_cost": cost,
                "cost_per_1000_applications": (
                    1000 * cost / len(X_evaluation)
                ),
                "cost_reduction_vs_all_approved_percent": (
                    100 * (all_approved_cost - cost)
                    / all_approved_cost
                ),
            })

            print(
                f"Completed fold {fold}: {name}",
                flush=True,
            )

    results = pd.DataFrame(results)
    results.to_csv(OUTPUT / "fold_results.csv", index=False)

    summary_columns = [
        "accuracy",
        "declined_precision",
        "declined_recall",
        "declined_f1",
        "declined_pr_auc_average_precision",
        "brier_score",
        "threshold",
        "cost_per_1000_applications",
        "cost_reduction_vs_all_approved_percent",
    ]

    summary = results.groupby("model")[summary_columns].agg(
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
        values="cost_per_1000_applications",
    )
    paired["forest_minus_logistic_cost_per_1000"] = (
        paired["random_forest"] - paired["logistic"]
    )
    paired.to_csv(OUTPUT / "paired_cost_comparison.csv")

    print("\nSTABILITY SUMMARY")
    print(summary.round(4).to_string(index=False))
    print("\nPAIRED COST COMPARISON")
    print(paired.round(4).to_string())
    print(f"\nResults saved to: {OUTPUT}")
    print("Original validation and test partitions were not used.")


if __name__ == "__main__":
    main()