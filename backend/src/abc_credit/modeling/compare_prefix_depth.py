"""Select cutoffs on training folds and compare them on development validation."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.compare_pincode_signal import represent_pincode
from abc_credit.modeling.feature_analysis import FEATURE_SETS, load_splits
from abc_credit.modeling.threshold_analysis import threshold_table

PACKAGED_CUTOFF = 0.2014630886001852


def fit_and_score(X_fit, y_fit, X_score):
    model = make_model("vehicle_value")
    model.set_params(classifier__depth=6)
    model.fit(X_fit, y_fit)
    decline_index = list(model.classes_).index(1)
    return model.predict_proba(X_score)[:, decline_index]


def select_cutoffs(y_train, scores):
    table = threshold_table(y_train, scores).copy()
    table["cost_5_to_1"] = (
        5 * table["historically_declined_predicted_approved"]
        + table["historically_approved_predicted_declined"]
    )

    best_f1 = table.sort_values(
        ["declined_f1", "threshold"], ascending=[False, False]
    ).iloc[0]
    best_cost = table.sort_values(
        ["cost_5_to_1", "threshold"], ascending=[True, False]
    ).iloc[0]
    recall_40 = table.loc[
        table["declined_recall"].ge(0.40), "threshold"
    ].max()

    return {
        "Packaged reference": PACKAGED_CUTOFF,
        "Best training-fold decline F1": float(best_f1["threshold"]),
        "Lowest training-fold 5:1 cost": float(best_cost["threshold"]),
        "40% training-fold recall target": float(recall_40),
    }


def report(name, actual, predicted):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    precision_text = f"{precision:.2%}" if precision is not None else "N/A"

    print(f"\n{name}")
    print(
        f"  Accuracy={accuracy:.2%} | Decline precision={precision_text} | "
        f"Decline recall={recall:.2%} | Decline F1={f1:.2%}"
    )
    print(f"  Declines caught={tp} | Missed={fn} | False flags={fp}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")
    print(f"  Illustrative 5:1 cost={5 * fn + fp}")


def main():
    splits = load_splits()
    features = FEATURE_SETS["without_gender"]

    X_train = represent_pincode(splits.X_train[features], "prefix_3")
    y_train = splits.y_train.eq("DECLINED").astype(int)
    
    scores = np.empty(len(y_train))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for fold, (fit, held_out) in enumerate(cv.split(X_train, y_train), start=1):
        print(f"Training fold {fold}/3", flush=True)
        scores[held_out] = fit_and_score(
            X_train.iloc[fit], y_train.iloc[fit], X_train.iloc[held_out]
        )

    print(f"\nDepth 6 pooled training-fold AP={average_precision_score(y_train, scores):.5f}")

    table = threshold_table(y_train, scores)
    matched = table.loc[
        table["correctly_detected_declines"].ge(1595)
    ].sort_values("threshold", ascending=False).iloc[0]

    print(
        "Depth 6 at matched training-fold recall | "
        f"Declines caught={int(matched['correctly_detected_declines'])} | "
        f"False flags={int(matched['historically_approved_predicted_declined'])}"
    )
    


if __name__ == "__main__":
    main()