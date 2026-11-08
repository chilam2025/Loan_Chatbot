"""Compare pincode representations on held-out training folds."""

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.compare_model_blend import report
from abc_credit.modeling.feature_analysis import FEATURE_SETS, load_splits
from abc_credit.modeling.threshold_analysis import threshold_table

VARIANTS = ("exact", "prefix_3", "excluded")


def represent_pincode(frame, variant):
    frame = frame.copy()

    if variant == "prefix_3":
        pincode = frame["Pincode"].astype("string").str.strip()
        valid = pincode.str.fullmatch(r"[0-9]{6}", na=False)
        frame["Pincode"] = pincode.where(valid).str[:3]
    elif variant == "excluded":
        # A constant preserves the pipeline schema and carries no location signal.
        frame["Pincode"] = "__CONSTANT__"

    return frame


def main():
    splits = load_splits()
    X = splits.X_train[FEATURE_SETS["without_gender"]]
    y = splits.y_train.eq("DECLINED").astype(int)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(cv.split(X, y))

    for variant in VARIANTS:
        inputs = represent_pincode(X, variant)
        scores = np.empty(len(y))

        for fold, (fit, held_out) in enumerate(folds, start=1):
            print(f"{variant}: training fold {fold}/3", flush=True)

            model = make_model("vehicle_value")
            model.fit(inputs.iloc[fit], y.iloc[fit])

            decline_index = list(model.classes_).index(1)
            fold_scores = model.predict_proba(
                inputs.iloc[held_out]
            )[:, decline_index]
            scores[held_out] = fold_scores

            ap = average_precision_score(y.iloc[held_out], fold_scores)
            print(f"  Fold AP={ap:.5f}", flush=True)

        report(f"{variant}: pooled held-out training predictions", y, scores)

def compare_on_validation():
    splits = load_splits()
    features = FEATURE_SETS["without_gender"]
    X_train = splits.X_train[features]
    X_validation = splits.X_validation[features]
    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    for variant in ("exact", "prefix_3"):
        print(f"\nTraining {variant} on the full training split...", flush=True)

        model = make_model("vehicle_value")
        model.fit(represent_pincode(X_train, variant), y_train)

        decline_index = list(model.classes_).index(1)
        scores = model.predict_proba(
            represent_pincode(X_validation, variant)
        )[:, decline_index]

        report(f"{variant}: development validation", y_validation, scores)

        table = threshold_table(y_validation, scores)
        table["assumed_cost"] = (
            5 * table["historically_declined_predicted_approved"]
            + table["historically_approved_predicted_declined"]
        )
        best = table.sort_values(
            ["assumed_cost", "threshold"],
            ascending=[True, False],
        ).iloc[0]

        print(
            f"{variant} | Threshold={best['threshold']:.16g} | "
            f"Recall={best['declined_recall']:.2%} | "
            f"Precision={best['declined_precision']:.2%} | "
            f"Declines caught={int(best['correctly_detected_declines'])} | "
            f"False declines={int(best['historically_approved_predicted_declined'])} | "
            f"Cost={int(best['assumed_cost'])}"
        )


if __name__ == "__main__":
    compare_on_validation()
if __name__ == "__main__":
    compare_on_validation()