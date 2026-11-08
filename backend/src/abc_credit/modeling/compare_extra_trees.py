"""Compare Extra Trees and CatBoost at matched validation recall."""

from sklearn.ensemble import ExtraTreesClassifier

from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.feature_analysis import (
    FEATURE_SETS,
    load_splits,
    make_pipeline,
)
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table


def main():
    splits = load_splits()
    features = FEATURE_SETS["without_gender"]

    X_train = splits.X_train[features]
    X_validation = splits.X_validation[features]
    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    extra_trees = make_pipeline("without_gender")
    extra_trees.set_params(
        classifier=ExtraTreesClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=2,
        )
    )

    models = {
        "CatBoost + implied vehicle value": make_model("vehicle_value"),
        "Extra Trees": extra_trees,
    }

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        model.fit(X_train, y_train)
        positive_index = list(model.classes_).index(1)
        scores = model.predict_proba(X_validation)[:, positive_index]
        thresholds = threshold_table(y_validation, scores)

        for recall_target in (0.20, 0.40):
            eligible = thresholds.loc[
                thresholds["declined_recall"] >= recall_target
            ]
            threshold = float(eligible["threshold"].max())
            metrics = evaluate(y_validation, scores, threshold)
            (tn, fp), (fn, tp) = metrics["confusion_matrix_approved_declined"]

            print(
                f"Target recall={recall_target:.0%} | "
                f"Actual recall={metrics['declined_recall']:.2%} | "
                f"Precision={metrics['declined_precision']:.2%} | "
                f"AP={metrics['declined_pr_auc_average_precision']:.4f} | "
                f"Declines caught={tp}/{tp + fn} | "
                f"False declines={fp}/{tn + fp}"
            )


if __name__ == "__main__":
    main()