"""Check whether training-selected cutoffs transfer across held-out folds."""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from abc_credit.modeling.compare_pincode_signal import represent_pincode
from abc_credit.modeling.compare_prefix_cutoffs import fit_and_score, select_cutoffs
from abc_credit.modeling.feature_analysis import FEATURE_SETS, load_splits

RULES = (
    "Best training-fold decline F1",
    "Lowest training-fold 5:1 cost",
)


def summarize(counts):
    _tn, fp, fn, tp = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    return (
        f"Precision={precision:.2%} | Recall={recall:.2%} | F1={f1:.2%} | "
        f"Caught={tp} | Missed={fn} | False flags={fp} | Cost={5 * fn + fp}"
    )


def main():
    splits = load_splits()
    features = FEATURE_SETS["without_gender"]
    X = represent_pincode(splits.X_train[features], "prefix_3")
    y = splits.y_train.eq("DECLINED").astype(int)

    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    totals = {rule: np.zeros(4, dtype=int) for rule in RULES}

    for fold, (development, held_out) in enumerate(outer_cv.split(X, y), start=1):
        X_dev, y_dev = X.iloc[development], y.iloc[development]
        X_holdout, y_holdout = X.iloc[held_out], y.iloc[held_out]
        inner_scores = np.empty(len(y_dev))
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for inner_fold, (fit, score) in enumerate(
            inner_cv.split(X_dev, y_dev), start=1
        ):
            print(f"Outer fold {fold}/3, inner fit {inner_fold}/3", flush=True)
            inner_scores[score] = fit_and_score(
                X_dev.iloc[fit], y_dev.iloc[fit], X_dev.iloc[score]
            )

        cutoffs = select_cutoffs(y_dev, inner_scores)
        print(f"Outer fold {fold}/3, fitting for held-out scoring", flush=True)
        holdout_scores = fit_and_score(X_dev, y_dev, X_holdout)

        for rule in RULES:
            cutoff = cutoffs[rule]
            counts = confusion_matrix(
                y_holdout, holdout_scores >= cutoff, labels=[0, 1]
            ).ravel()
            totals[rule] += counts
            print(f"  {rule} | Cutoff={cutoff:.6f} | {summarize(counts)}")

    print("\nPooled outer-fold results")
    for rule in RULES:
        print(f"{rule} | {summarize(totals[rule])}")


if __name__ == "__main__":
    main()