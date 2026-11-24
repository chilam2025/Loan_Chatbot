"""Training-only search for a stable six-question CatBoost improvement.

The original validation and internal test partitions are deliberately not used.
Candidates are compared on the same profile-grouped folds and at matched decline
recall. Generated results contain aggregate metrics only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from abc_credit.data_quality import NUMERIC_COLUMNS
from abc_credit.modeling.compare_catboost import CATEGORICAL_FIELDS
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.minimal_model import MINIMAL_FIELDS, build_minimal_v2_pipeline
from abc_credit.modeling.package_catboost_minimal_v2 import profile_ids
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/minimal_v3_search")
TARGET_RECALL = 0.40

CANDIDATES = {
    "v2_baseline": {
        "classifier__depth": 4,
        "classifier__l2_leaf_reg": 10,
        "classifier__max_ctr_complexity": 4,
        "classifier__random_strength": 1.0,
    },
    "higher_categorical_interactions": {
        "classifier__depth": 4,
        "classifier__l2_leaf_reg": 10,
        "classifier__max_ctr_complexity": 5,
        "classifier__random_strength": 1.0,
    },
    "deeper_current_interactions": {
        "classifier__depth": 5,
        "classifier__l2_leaf_reg": 15,
        "classifier__max_ctr_complexity": 4,
        "classifier__random_strength": 1.0,
    },
}


def matched_recall_metrics(labels: pd.Series, scores: np.ndarray) -> dict[str, float | int]:
    table = threshold_table(labels, scores)
    selected = (
        table.loc[table["declined_recall"].ge(TARGET_RECALL)]
        .sort_values("threshold", ascending=False)
        .iloc[0]
    )
    caught = int(selected["correctly_detected_declines"])
    false_flags = int(selected["historically_approved_predicted_declined"])
    return {
        "threshold": float(selected["threshold"]),
        "declined_precision": float(selected["declined_precision"]),
        "declined_recall": float(selected["declined_recall"]),
        "declines_caught": caught,
        "historical_approvals_flagged": false_flags,
        "flagged_share": (caught + false_flags) / len(labels),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    all_fields = [*NUMERIC_COLUMNS, *CATEGORICAL_FIELDS]
    inputs = splits.X_train[MINIMAL_FIELDS].reset_index(drop=True)
    labels = splits.y_train.eq("DECLINED").astype(int).reset_index(drop=True)
    groups = profile_ids(splits.X_train[all_fields]).reset_index(drop=True)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(cv.split(inputs, labels, groups))

    fold_rows: list[dict[str, float | int | str]] = []
    pooled_scores: dict[str, np.ndarray] = {
        name: np.empty(len(labels), dtype=float) for name in CANDIDATES
    }

    for name, parameters in CANDIDATES.items():
        for fold, (fit, held_out) in enumerate(folds, start=1):
            print(f"{name}: fold {fold}/{len(folds)}", flush=True)
            model = build_minimal_v2_pipeline(seed=42 + fold)
            model.set_params(
                classifier__thread_count=8,
                **parameters,
            )
            model.fit(inputs.iloc[fit], labels.iloc[fit])
            positive_index = list(model.classes_).index(1)
            scores = model.predict_proba(inputs.iloc[held_out])[:, positive_index]
            pooled_scores[name][held_out] = scores
            metrics = matched_recall_metrics(labels.iloc[held_out], scores)
            fold_rows.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "average_precision": average_precision_score(
                        labels.iloc[held_out], scores
                    ),
                    **metrics,
                }
            )
            pd.DataFrame(fold_rows).to_csv(OUTPUT / "fold_results.csv", index=False)

    summary_rows = []
    for name, scores in pooled_scores.items():
        metrics = matched_recall_metrics(labels, scores)
        fold_frame = pd.DataFrame(fold_rows)
        candidate_folds = fold_frame.loc[fold_frame["candidate"].eq(name)]
        summary_rows.append(
            {
                "candidate": name,
                "pooled_average_precision": average_precision_score(labels, scores),
                "minimum_fold_average_precision": candidate_folds[
                    "average_precision"
                ].min(),
                "mean_fold_average_precision": candidate_folds[
                    "average_precision"
                ].mean(),
                **metrics,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        "pooled_average_precision", ascending=False
    )
    baseline = pooled_scores["v2_baseline"]
    paired_rows = []
    for name, scores in pooled_scores.items():
        if name == "v2_baseline":
            continue
        for fold, (_, held_out) in enumerate(folds, start=1):
            paired_rows.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "average_precision_delta": average_precision_score(
                        labels.iloc[held_out], scores[held_out]
                    )
                    - average_precision_score(labels.iloc[held_out], baseline[held_out]),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUTPUT / "paired_deltas.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)

    print("\nTRAINING-ONLY GROUPED RESULTS")
    print(summary.round(6).to_string(index=False))
    print("\nPAIRED FOLD AP DELTAS VS V2")
    print(paired.round(6).to_string(index=False))
    print(f"\nSaved aggregate results to {OUTPUT}")
    print("Original validation and internal test partitions were not evaluated.")


if __name__ == "__main__":
    main()
