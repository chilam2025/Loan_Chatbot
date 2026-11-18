"""Select a CatBoost–Random Forest blend using training-fold predictions."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.feature_analysis import (
    FEATURE_SETS,
    load_splits,
    make_pipeline,
)
from abc_credit.modeling.threshold_analysis import threshold_table


def fit_and_score(X_fit, y_fit, X_score):
    catboost = make_model("vehicle_value")
    forest = make_pipeline("without_gender")
    forest.set_params(
        classifier=RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=2,
        )
    )

    scores = []
    for model in (catboost, forest):
        model.fit(X_fit, y_fit)
        decline_index = list(model.classes_).index(1)
        scores.append(model.predict_proba(X_score)[:, decline_index])

    return scores[0], scores[1]


def report(name, y, scores):
    print(f"\n{name} | AP={average_precision_score(y, scores):.4f}")
    table = threshold_table(y, scores)

    for target in (0.20, 0.40):
        selected = (
            table.loc[table["declined_recall"] >= target]
            .sort_values("threshold", ascending=False)
            .iloc[0]
        )
        caught = int(selected["correctly_detected_declines"])
        false_declines = int(selected["historically_approved_predicted_declined"])

        print(
            f"Target recall={target:.0%} | "
            f"Actual recall={selected['declined_recall']:.2%} | "
            f"Precision={selected['declined_precision']:.2%} | "
            f"Declines caught={caught} | False declines={false_declines}"
        )


def main():
    splits = load_splits()
    features = FEATURE_SETS["without_gender"]
    X_train = splits.X_train[features]
    y_train = splits.y_train.eq("DECLINED").astype(int)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cat_oof = np.empty(len(y_train))
    forest_oof = np.empty(len(y_train))

    for fold, (fit, held_out) in enumerate(cv.split(X_train, y_train), start=1):
        print(f"Training fold {fold}/3...", flush=True)
        cat_oof[held_out], forest_oof[held_out] = fit_and_score(
            X_train.iloc[fit], y_train.iloc[fit], X_train.iloc[held_out]
        )

    # A weight of 1.00 means CatBoost alone. It wins ties.
    weights = (1.00, 0.75, 0.50, 0.25, 0.00)
    ap_by_weight = {
        weight: average_precision_score(
            y_train, weight * cat_oof + (1 - weight) * forest_oof
        )
        for weight in weights
    }

    print("\nTraining-fold results")
    for weight, ap in ap_by_weight.items():
        print(f"CatBoost weight={weight:.2f} | AP={ap:.5f}")

    selected_weight = max(weights, key=ap_by_weight.__getitem__)
    print(f"\nSelected CatBoost weight={selected_weight:.2f}", flush=True)

    X_validation = splits.X_validation[features]
    y_validation = splits.y_validation.eq("DECLINED").astype(int)
    cat_scores, forest_scores = fit_and_score(X_train, y_train, X_validation)
    selected_scores = (
        selected_weight * cat_scores + (1 - selected_weight) * forest_scores
    )

    report("CatBoost alone", y_validation, cat_scores)
    report("Random Forest alone", y_validation, forest_scores)
    report("Selected combination", y_validation, selected_scores)


if __name__ == "__main__":
    main()