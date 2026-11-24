"""Train, select on validation, and evaluate one frozen loan-decision pipeline."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from abc_credit.data_quality import load_raw_data, normalize_for_analysis
from abc_credit.preprocessing import FEATURE_COLUMNS, build_preprocessor, stratified_split


def evaluate(y, probabilities, threshold):
    predicted = np.asarray(probabilities) >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "declined_precision": float(precision), "declined_recall": float(recall),
        "declined_f1": float(f1),
        "declined_pr_auc_average_precision": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "confusion_matrix_approved_declined": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def choose_threshold(y, probabilities, false_approval_cost=5.0, false_decline_cost=1.0):
    """Illustrative relative costs, not measured financial losses."""
    candidates = []
    for threshold in np.linspace(0, 1, 201):
        predicted = probabilities >= threshold
        false_approvals = int(((np.asarray(y) == 1) & ~predicted).sum())
        false_declines = int(((np.asarray(y) == 0) & predicted).sum())
        cost = false_approval_cost * false_approvals + false_decline_cost * false_declines
        candidates.append((cost, float(threshold)))
    return min(candidates)[1]


def predict_applications(bundle, raw):
    """Same deterministic normalization and fitted pipeline used at evaluation."""
    missing = set(bundle["feature_columns"]) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required inputs: {sorted(missing)}")
    frame = raw.copy().astype("string")
    frame["ID"] = "inference"
    frame["Decision"] = "APPROVED"  # Placeholder only for shared schema normalization.
    features = normalize_for_analysis(frame).labelled[bundle["feature_columns"]]
    probabilities = bundle["pipeline"].predict_proba(features)[:, 1]
    return pd.DataFrame({
        "decision": np.where(probabilities >= bundle["threshold"], "DECLINED", "APPROVED"),
        "probability_declined": probabilities,
    }, index=raw.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/training_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/phase_b"))
    parser.add_argument("--false-approval-cost", type=float, default=5.0)
    parser.add_argument("--false-decline-cost", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.false_approval_cost, args.false_decline_cost) <= 0:
        parser.error("Costs must be positive")
    labelled = normalize_for_analysis(load_raw_data(args.input)).labelled
    # Collapse identical input records before splitting: they must not leak across partitions.
    conflict = labelled.groupby(FEATURE_COLUMNS, dropna=False)["Decision"].transform("nunique") > 1
    conflicting_rows = int(conflict.sum())
    usable = labelled.loc[~conflict].drop_duplicates(subset=FEATURE_COLUMNS)
    splits = stratified_split(usable)
    y_train = (splits.y_train == "DECLINED").astype(int)
    y_val = (splits.y_validation == "DECLINED").astype(int)
    candidates = {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("scale", StandardScaler(with_mean=False)),
            ("classifier", LogisticRegression(max_iter=1500, solver="liblinear", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(n_estimators=150, max_depth=18,
              min_samples_leaf=5, n_jobs=1, random_state=42)),
        ]),
    }
    results = {}
    for name, pipeline in candidates.items():
        start = time.perf_counter()
        pipeline.fit(splits.X_train, y_train)
        probabilities = pipeline.predict_proba(splits.X_validation)[:, 1]
        threshold = choose_threshold(y_val, probabilities, args.false_approval_cost,
                                     args.false_decline_cost)
        metrics = evaluate(y_val, probabilities, threshold)
        cm = metrics["confusion_matrix_approved_declined"]
        results[name] = {"threshold": threshold, "validation": metrics,
            "validation_cost": cm[1][0]*args.false_approval_cost + cm[0][1]*args.false_decline_cost,
            "training_seconds": time.perf_counter()-start}
        print(name, json.dumps(results[name]), flush=True)
    winner = min(results, key=lambda name: (results[name]["validation_cost"],
                 -results[name]["validation"]["declined_pr_auc_average_precision"]))
    bundle = {"pipeline": candidates[winner], "threshold": results[winner]["threshold"],
              "feature_columns": FEATURE_COLUMNS, "positive_class": "DECLINED",
              "model_name": winner, "label_mapping": {"APPROVED": 0, "DECLINED": 1}}
    # Only the selected model is inspected on the final internal test partition.
    y_test = (splits.y_test == "DECLINED").astype(int)
    start = time.perf_counter()
    predictions = predict_applications(bundle, splits.X_test)
    batch_seconds = time.perf_counter()-start
    test = evaluate(y_test, predictions.probability_declined, bundle["threshold"])
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output / "decision_pipeline.joblib")
    restored = joblib.load(args.output / "decision_pipeline.joblib")
    np.testing.assert_allclose(predict_applications(restored, splits.X_test.head()).probability_declined,
                               predictions.probability_declined.head(), rtol=1e-12, atol=1e-15)
    report = {"selected_model": winner, "positive_class": "DECLINED", "seed": 42,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        "cost_assumption": {"false_approval": args.false_approval_cost,
            "false_decline": args.false_decline_cost,
            "note": "Illustrative costs of disagreement with historical labels, not default losses."},
        "data": {"labelled": len(labelled), "conflicting_rows_quarantined": conflicting_rows,
            "identical_feature_duplicates_removed": len(labelled)-conflicting_rows-len(usable),
            "train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "candidates": results, "selected_test": test,
        "all_approved_test": evaluate(y_test, np.zeros(len(y_test)), 0.5),
        "test_batch_seconds": batch_seconds,
        "mean_batch_seconds_per_row": batch_seconds/len(y_test),
        "limitations": ["Historical approval prediction, not default prediction.",
            "Scores have not been probability-calibrated; Brier score is diagnostic only.",
            "No fairness, feature reduction, or end-to-end chatbot latency claims yet.",
            "Batch timing is not a single-request latency benchmark."]}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    print("Selected:", winner, "Test:", json.dumps(test), flush=True)


if __name__ == "__main__":
    main()
