"""Compare the prefix model with and without Qualifications."""

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.compare_pincode_signal import represent_pincode
from abc_credit.modeling.feature_analysis import FEATURE_SETS, load_splits
from abc_credit.modeling.threshold_analysis import threshold_table

TARGET_DECLINES_CAUGHT = 1595


def held_out_scores(X, y, folds, name):
    scores = np.empty(len(y))

    for fold, (fit, held_out) in enumerate(folds, start=1):
        print(f"{name}: training fold {fold}/{len(folds)}", flush=True)

        model = make_model("vehicle_value")
        model.set_params(classifier__depth=4)
        model.fit(X.iloc[fit], y.iloc[fit])

        decline_index = list(model.classes_).index(1)
        scores[held_out] = model.predict_proba(X.iloc[held_out])[:, decline_index]

    return scores


def report(name, y, scores):
    table = threshold_table(y, scores)
    eligible = table.loc[
        table["correctly_detected_declines"].ge(TARGET_DECLINES_CAUGHT)
    ]

    if eligible.empty:
        raise ValueError(
            f"Cannot catch {TARGET_DECLINES_CAUGHT} historical declines "
            f"on a split containing {int(y.sum())} declines."
        )

    matched = eligible.sort_values("threshold", ascending=False).iloc[0]
    ap = average_precision_score(y, scores)
    caught = int(matched["correctly_detected_declines"])
    false_flags = int(matched["historically_approved_predicted_declined"])

    print(
        f"{name} | Pooled AP={ap:.5f} | "
        f"Declines caught={caught} | Historical approvals flagged={false_flags}"
    )


def main():
    splits = load_splits()
    features = FEATURE_SETS["without_gender"]
    X_train = represent_pincode(splits.X_train[features], "prefix_3")
    y_train = splits.y_train.eq("DECLINED").astype(int)

    X_reduced = X_train.copy()
    X_reduced["Qualifications"] = "__CONSTANT__"

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(cv.split(X_train, y_train))

    full_scores = held_out_scores(X_train, y_train, folds, "Full model")
    reduced_scores = held_out_scores(
        X_reduced, y_train, folds, "Without qualification"
    )

    print("\nPooled held-out training results")
    report("Full model", y_train, full_scores)
    report("Without qualification", y_train, reduced_scores)


if __name__ == "__main__":
    main()