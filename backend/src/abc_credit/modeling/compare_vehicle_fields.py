"""Compare full vehicle inputs with Model_Description alone."""
import numpy as np
from sklearn.metrics import average_precision_score

from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.compare_pincode_signal import represent_pincode
from abc_credit.modeling.feature_analysis import FEATURE_SETS, load_splits
from abc_credit.modeling.threshold_analysis import threshold_table

REMOVED_FIELDS = ("Make_Code", "Model_Variant", "Product_Code")
TARGET_DECLINES_CAUGHT = 340


def held_out_scores(X, y, folds, name):
    scores = np.empty(len(y))

    for fold, (fit, held_out) in enumerate(folds, start=1):
        print(f"{name}: training fold {fold}/{len(folds)}", flush=True)

        model = make_model("vehicle_value")
        model.set_params(classifier__depth=4)
        model.fit(X.iloc[fit], y.iloc[fit])

        decline_index = list(model.classes_).index(1)
        scores[held_out] = model.predict_proba(
            X.iloc[held_out]
        )[:, decline_index]

    return scores


def report(name, y, scores):
    table = threshold_table(y, scores)
    matched = (
        table.loc[
            table["correctly_detected_declines"].ge(TARGET_DECLINES_CAUGHT)
        ]
        .sort_values("threshold", ascending=False)
        .iloc[0]
    )

    ap = average_precision_score(y, scores)
    caught = int(matched["correctly_detected_declines"])
    false_flags = int(matched["historically_approved_predicted_declined"])

    print(
        f"{name} | Validation AP={ap:.5f} | "
        f"Declines caught={caught} | Historical approvals flagged={false_flags}"
    )


def main():
    splits = load_splits()
    features = FEATURE_SETS["without_gender"]

    X_train = represent_pincode(splits.X_train[features], "prefix_3")
    y_train = splits.y_train.eq("DECLINED").astype(int)
    X_validation = represent_pincode(splits.X_validation[features], "prefix_3")
    y_validation = splits.y_validation.eq("DECLINED").astype(int)
    
    y_train = splits.y_train.eq("DECLINED").astype(int)
   

    for name, reduce_vehicle_fields in (
        ("All vehicle fields", False),
        ("Model_Description only", True),
    ):
        X_fit = X_train.copy()
        X_score = X_validation.copy()

        if reduce_vehicle_fields:
            for column in REMOVED_FIELDS:
                X_fit[column] = "__CONSTANT__"
                X_score[column] = "__CONSTANT__"

        print(f"\n{name}: fitting on the full training split", flush=True)
        model = make_model("vehicle_value")
        model.set_params(classifier__depth=4)
        model.fit(X_fit, y_train)

        decline_index = list(model.classes_).index(1)
        scores = model.predict_proba(X_score)[:, decline_index]
        report(name, y_validation, scores)

if __name__ == "__main__":
    main()