"""Tune boosting using training-only CV, then compare on validation."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from abc_credit.modeling.compare_boosting import make_candidate
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/boosting_tuning")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    # Original LTV treatment, consistent across all tuning candidates.
    pipeline = make_candidate("hist_gradient_boosting", False)

    # Six predefined configurations, each evaluated on the same three folds.
    grid = {
        "classifier__max_leaf_nodes": [7, 15, 31],
        "classifier__min_samples_leaf": [30, 100],
    }

    folds = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="average_precision",
        cv=folds,
        refit=True,
        n_jobs=1,
        verbose=2,
        error_score="raise",
        return_train_score=False,
    )

    print("Starting 18 cross-validation fits plus the selected refit.")
    print("Preprocessing is fitted separately inside each training fold.")

    with threadpool_limits(limits=2):
        search.fit(splits.X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(OUTPUT / "cv_results.csv", index=False)

    settings = {
        "selection_metric": "average_precision",
        "cv_folds": 3,
        "best_parameters": search.best_params_,
        "best_mean_cv_average_precision": float(search.best_score_),
        "fixed_settings": {
            "learning_rate": 0.05,
            "max_iter": 250,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "ltv_treatment": "original",
        },
        "note": (
            "CV score was used for selection and is not an independent "
            "final performance estimate."
        ),
    }

    (OUTPUT / "selected_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    # The initial boosting model is already fitted if the selected
    # configuration matches it; otherwise fit it for comparison.
    initial_boosting = make_candidate("hist_gradient_boosting", False)
    forest = make_candidate("random_forest", False)

    models = {
        "random_forest_reference": forest,
        "boosting_initial": initial_boosting,
        "boosting_cv_selected": search.best_estimator_,
    }

    summaries = []
    matched_recall = []

    for name, model in models.items():
        if name != "boosting_cv_selected":
            with threadpool_limits(limits=2):
                model.fit(splits.X_train, y_train)

        with threadpool_limits(limits=2):
            positive_index = list(model.classes_).index(1)
            probabilities = model.predict_proba(
                splits.X_validation
            )[:, positive_index]

        table = threshold_table(y_validation, probabilities)
        table["cost"] = (
            5 * table["historically_declined_predicted_approved"]
            + table["historically_approved_predicted_declined"]
        )

        best_threshold = float(
            table.sort_values(
                ["cost", "threshold"],
                ascending=[True, False],
            ).iloc[0]["threshold"]
        )

        metrics = evaluate(
            y_validation, probabilities, best_threshold
        )
        matrix = metrics.pop("confusion_matrix_approved_declined")

        summaries.append({
            "model": name,
            "threshold": best_threshold,
            **metrics,
            "false_declines": matrix[0][1],
            "missed_declines": matrix[1][0],
            "detected_declines": matrix[1][1],
            "assumed_cost_5_to_1": 5 * matrix[1][0] + matrix[0][1],
        })

        for target in [0.20, 0.40, 0.60]:
            selected = (
                table.loc[table["declined_recall"] >= target]
                .sort_values("threshold", ascending=False)
                .iloc[0]
            )

            matched_recall.append({
                "model": name,
                "target_recall": target,
                "actual_recall": float(selected["declined_recall"]),
                "threshold": float(selected["threshold"]),
                "declined_precision": float(
                    selected["declined_precision"]
                ),
                "false_declines": int(
                    selected[
                        "historically_approved_predicted_declined"
                    ]
                ),
                "detected_declines": int(
                    selected["correctly_detected_declines"]
                ),
            })

    pd.DataFrame(summaries).to_csv(
        OUTPUT / "validation_comparison.csv", index=False
    )
    pd.DataFrame(matched_recall).to_csv(
        OUTPUT / "matched_recall_comparison.csv", index=False
    )

    print("\nSELECTED SETTINGS")
    print(json.dumps(settings, indent=2))

    print("\nVALIDATION COMPARISON")
    print(pd.DataFrame(summaries).round(5).to_string(index=False))

    print(f"\nResults saved to: {OUTPUT}")
    print("Test partition was not evaluated.")
    print("Packaged model was not modified.")


if __name__ == "__main__":
    main()