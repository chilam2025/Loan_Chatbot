"""Score the internal test partition with the packaged model at its locked cutoff.

The packaged model (catboost-minimal-v2) was fitted on the training partition
and its cutoff was chosen on held-out training folds. The internal test
partition (15%) was not used to fit, tune or choose the cutoff of this model,
so this is the closest stand-in for the instructor's hidden test set.

    uv run --project backend python -m abc_credit.modeling.evaluate_internal_test

Writes ``reports/generated/internal_test_metrics.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.inference import load_model, predict_applications
from abc_credit.modeling.predict_batch import DEFAULT_MODEL, score_labeled

DEFAULT_OUTPUT = Path("reports/generated/internal_test_metrics.json")
# Illustrative cost of approving a risky applicant relative to declining a good one.
COST_RATIOS = [3, 5, 8, 10]


def _metrics_at(truth, probability, cutoff):
    output = pd.DataFrame(
        {
            "Decision": np.where(probability >= cutoff, "DECLINED", "APPROVED"),
            "probability_declined": probability,
        },
        index=truth.index,
    )
    metrics = score_labeled(truth, output)
    metrics["cutoff"] = float(cutoff)
    metrics["share_flagged"] = float((probability >= cutoff).mean())
    return metrics


def _score(bundle, X, y):
    features = bundle["metadata"]["feature_columns"]
    probability = predict_applications(bundle, X[features])["probability_declined"].to_numpy()
    return probability


def run(model_path: Path) -> dict:
    bundle = load_model(model_path)
    metadata = bundle["metadata"]
    cutoff = metadata["threshold"]
    splits = load_splits()

    # Guard: the split must be the one the model was packaged with.
    val_probability = _score(bundle, splits.X_validation, splits.y_validation)
    val = _metrics_at(splits.y_validation, val_probability, cutoff)
    expected = metadata["validation_metrics"]["confusion_matrix_approved_declined"]
    if val["confusion_matrix_approved_declined"] != expected:
        raise RuntimeError(
            f"Validation confusion matrix {val['confusion_matrix_approved_declined']} "
            f"does not match the packaged model's {expected}; the split differs."
        )

    y = splits.y_test
    probability = _score(bundle, splits.X_test, y)
    declined = y.eq("DECLINED").to_numpy()

    result = {
        "model_version": metadata["model_version"],
        "partition": "internal test (15%, stratified, seed-fixed; not used to fit or tune this model)",
        "rows": len(y),
        "historical_declines": int(declined.sum()),
        "historical_decline_rate": float(declined.mean()),
        "locked_cutoff": cutoff,
        "at_locked_cutoff": _metrics_at(y, probability, cutoff),
        "always_approve_baseline_accuracy": float(1 - declined.mean()),
        "validation_check": "validation confusion matrix reproduced exactly",
        "operating_points": [
            {"target_recall_on_training_folds": point["target_recall"],
             **_metrics_at(y, probability, point["cutoff"])}
            for point in metadata["operating_points_on_held_out_training_scores"]
        ],
    }

    # Which training-fold cutoff minimises an illustrative cost on test?
    costs = []
    for ratio in COST_RATIOS:
        best = None
        for point in result["operating_points"]:
            (_, fp), (fn, _) = point["confusion_matrix_approved_declined"]
            cost = ratio * fn + fp
            if best is None or cost < best[0]:
                best = (cost, point["cutoff"])
        costs.append({"missed_decline_cost_vs_lost_customer": ratio, "cheapest_cutoff": best[1]})
    result["cost_view"] = costs

    gender = splits.X_test["Gender"].fillna("<MISSING>")
    result["subgroups_gender"] = {
        group: {
            "rows": int(mask.sum()),
            "historical_decline_rate": float(declined[mask].mean()),
            "predicted_decline_rate": float((probability[mask] >= cutoff).mean()),
            "decline_recall": float(((probability >= cutoff) & declined & mask).sum() / max(1, (declined & mask).sum())),
        }
        for group in ["Female", "Male"]
        for mask in [gender.eq(group).to_numpy()]
    }
    return result


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args(argv)
    result = run(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
