"""Score a CSV of applications with a packaged model.

The input needs the dataset's original column names (extra columns are
ignored). If it also has a labeled ``Decision`` column, metrics are printed.

    python -m abc_credit.modeling.predict_batch --input applications.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from abc_credit.modeling.inference import load_model, predict_applications

DEFAULT_MODEL = Path("models/catboost_minimal_v2/decision_model.joblib")
DEFAULT_OUTPUT = Path("reports/generated/predictions.csv")


def run_batch(bundle, frame):
    """Return one prediction row per input row, in the same order."""
    features = bundle["metadata"]["feature_columns"]
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    result = predict_applications(bundle, frame[features])

    output = pd.DataFrame(index=frame.index)
    if "ID" in frame.columns:
        output["ID"] = frame["ID"]
    output["Decision"] = result["decision"].to_numpy()
    output["probability_declined"] = result[
        "probability_declined"
    ].to_numpy()
    output["threshold"] = bundle["metadata"]["threshold"]
    output["model_version"] = bundle["metadata"]["model_version"]
    return output


def score_labeled(truth, output):
    """Metrics for both classes when the input carries true decisions."""
    known = truth.isin(["APPROVED", "DECLINED"])
    y = truth[known].eq("DECLINED").astype(int).to_numpy()
    predicted = output.loc[known, "Decision"].eq("DECLINED").to_numpy()
    probability = output.loc[known, "probability_declined"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y, predicted.astype(int), labels=[0, 1]
    ).ravel()

    def ratio(a, b):
        return a / b if b else 0.0

    approved_precision, approved_recall = ratio(tn, tn + fn), ratio(tn, tn + fp)
    declined_precision, declined_recall = ratio(tp, tp + fp), ratio(tp, tp + fn)

    def f1(p, r):
        return ratio(2 * p * r, p + r)

    return {
        "rows_scored": int(known.sum()),
        "accuracy": ratio(tn + tp, len(y)),
        "balanced_accuracy": (approved_recall + declined_recall) / 2,
        "approved_precision": approved_precision,
        "approved_recall": approved_recall,
        "approved_f1": f1(approved_precision, approved_recall),
        "declined_precision": declined_precision,
        "declined_recall": declined_recall,
        "declined_f1": f1(declined_precision, declined_recall),
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix_approved_declined": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input, dtype=str)
    for column in frame.columns:
        frame[column] = frame[column].str.strip()

    bundle = load_model(args.model)
    output = run_batch(bundle, frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    declined = int(output["Decision"].eq("DECLINED").sum())
    print(
        f"Scored {len(output)} applications with "
        f"{bundle['metadata']['model_version']} "
        f"(cutoff {bundle['metadata']['threshold']:.4f}): "
        f"{declined} DECLINED ({declined / len(output):.1%}), "
        f"{len(output) - declined} APPROVED."
    )
    print("Saved predictions to", args.output)

    if "Decision" in frame.columns:
        metrics = score_labeled(frame["Decision"], output)
        print("Labeled-data metrics:")
        for name, value in metrics.items():
            shown = f"{value:.4f}" if isinstance(value, float) else value
            print(f"  {name}: {shown}")

    return output


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()