"""Training-only nested comparison of CatBoost and Random Forest."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from abc_credit.modeling.compare_catboost import (
    CATEGORICAL_FIELDS,
    PrepareCatBoost,
)
from abc_credit.modeling.compare_ltv_fix import build_model
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/nested_model_comparison")


def make_catboost():
    """Construct a fresh pipeline, avoiding CatBoost cloning."""
    return Pipeline([
        (
            "prepare",
            PrepareCatBoost(fix_zero_ltv=True),
        ),
        (
            "classifier",
            CatBoostClassifier(
                iterations=600,
                learning_rate=0.05,
                loss_function="Logloss",
                cat_features=list(CATEGORICAL_FIELDS),
                random_seed=42,
                thread_count=2,
                verbose=False,
                allow_writing_files=False,
            ),
        ),
    ])


def manual_search(X, y, parameter_grid, inner_cv):
    """Select parameters using inner-fold average precision."""
    rows = []

    # Reuse identical inner partitions for every configuration.
    fold_positions = list(inner_cv.split(X, y))

    for parameters in ParameterGrid(parameter_grid):
        scores = []

        for fold, (fit_positions, score_positions) in enumerate(
            fold_positions, start=1
        ):
            model = make_catboost()
            model.set_params(**parameters)

            model.fit(
                X.iloc[fit_positions],
                y.iloc[fit_positions],
            )

            positive_index = list(model.classes_).index(1)
            probabilities = model.predict_proba(
                X.iloc[score_positions]
            )[:, positive_index]

            score = average_precision_score(
                y.iloc[score_positions],
                probabilities,
            )
            scores.append(float(score))

            print(
                f"Parameters: {parameters}; "
                f"inner fold {fold}; "
                f"average precision: {score:.5f}",
                flush=True,
            )

        rows.append({
            "params": parameters,
            "mean_test_score": float(np.mean(scores)),
            "std_test_score": float(np.std(scores)),
            **{
                f"split{index}_test_score": score
                for index, score in enumerate(scores)
            },
        })

    # Ties retain the first configuration in ParameterGrid order.
    best = max(rows, key=lambda row: row["mean_test_score"])

    selected_model = make_catboost()
    selected_model.set_params(**best["params"])

    # Refit only on the inner fitting data.
    # Threshold-selection and outer evaluation records remain excluded.
    selected_model.fit(X, y)

    return SimpleNamespace(
        best_params_=best["params"],
        best_score_=best["mean_test_score"],
        best_estimator_=selected_model,
        cv_results_=rows,
    )


def choose_inner_threshold(y, probabilities):
    """Select a threshold under the illustrative 5:1 cost assumption."""
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

    # Original validation and test partitions are not used.
    X = splits.X_train
    y = splits.y_train.eq("DECLINED").astype(int)

    outer_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    # Four predefined configurations.
    parameter_grid = {
        "classifier__depth": [4, 6],
        "classifier__l2_leaf_reg": [3, 10],
    }

    results = []
    selections = []

    for fold, (development_positions, evaluation_positions) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_development = X.iloc[development_positions]
        y_development = y.iloc[development_positions]

        X_evaluation = X.iloc[evaluation_positions]
        y_evaluation = y.iloc[evaluation_positions]

        # Separate threshold selection from hyperparameter tuning.
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

        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=100 + fold,
        )

        print(
            f"\nOUTER FOLD {fold}: tuning CatBoost",
            flush=True,
        )

        search = manual_search(
            X_fit,
            y_fit,
            parameter_grid,
            inner_cv,
        )

        pd.DataFrame(search.cv_results_).to_csv(
            OUTPUT / f"fold_{fold}_inner_search.csv",
            index=False,
        )

        selections.append({
            "fold": fold,
            "selected_depth": (
                search.best_params_["classifier__depth"]
            ),
            "selected_l2_leaf_reg": (
                search.best_params_["classifier__l2_leaf_reg"]
            ),
            "inner_mean_average_precision": float(
                search.best_score_
            ),
        })

        print(
            f"Outer fold {fold} selected: {search.best_params_}",
            flush=True,
        )

        forest = build_model(True)
        forest.fit(X_fit, y_fit)

        models = {
            "catboost_tuned": search.best_estimator_,
            "forest_reference": forest,
        }

        for name, model in models.items():
            positive_index = list(model.classes_).index(1)

            threshold_scores = model.predict_proba(
                X_threshold
            )[:, positive_index]

            threshold = choose_inner_threshold(
                y_threshold,
                threshold_scores,
            )

            # Recall thresholds are also chosen on inner data only.
            inner_table = threshold_table(
                y_threshold,
                threshold_scores,
            )
            operating_points = {
                "cost_5_to_1": threshold,
            }

            for target in [0.20, 0.40, 0.60]:
                selected = (
                    inner_table.loc[
                        inner_table["declined_recall"] >= target
                    ]
                    .sort_values(
                        "threshold",
                        ascending=False,
                    )
                    .iloc[0]
                )

                operating_points[
                    f"inner_recall_{target:.0%}"
                ] = float(selected["threshold"])

            # Evaluate the same fitted model without refitting.
            evaluation_scores = model.predict_proba(
                X_evaluation
            )[:, positive_index]

            for operating_point, locked_threshold in operating_points.items():
                metrics = evaluate(
                    y_evaluation,
                    evaluation_scores,
                    locked_threshold,
                )

                matrix = metrics.pop(
                    "confusion_matrix_approved_declined"
                )
                cost = 5 * matrix[1][0] + matrix[0][1]

                results.append({
                    "fold": fold,
                    "model": name,
                    "operating_point": operating_point,
                    "threshold": locked_threshold,
                    **metrics,
                    "false_declines": matrix[0][1],
                    "missed_declines": matrix[1][0],
                    "detected_declines": matrix[1][1],
                    "cost_per_1000": (
                        1000 * cost / len(y_evaluation)
                    ),
                })

            print(
                f"Evaluated outer fold {fold}: {name}",
                flush=True,
            )

        # Save completed-fold checkpoints.
        pd.DataFrame(results).to_csv(
            OUTPUT / "outer_fold_results.csv",
            index=False,
        )
        pd.DataFrame(selections).to_csv(
            OUTPUT / "selected_parameters.csv",
            index=False,
        )

    results = pd.DataFrame(results)

    metrics_to_summarise = [
        "declined_precision",
        "declined_recall",
        "declined_f1",
        "declined_pr_auc_average_precision",
        "brier_score",
        "cost_per_1000",
    ]

    summary = results.groupby(
        ["model", "operating_point"]
    )[metrics_to_summarise].agg(["mean", "std"])

    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()

    summary.to_csv(
        OUTPUT / "outer_summary.csv",
        index=False,
    )

    print("\nOUTER EVALUATION SUMMARY")
    print(summary.round(5).to_string(index=False))
    print(f"\nSaved to: {OUTPUT}")
    print("Original validation and test partitions were not evaluated.")
    print("Packaged model was not modified.")


if __name__ == "__main__":
    main()