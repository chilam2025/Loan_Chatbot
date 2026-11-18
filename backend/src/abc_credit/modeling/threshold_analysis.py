"""Validation-only threshold comparison under illustrative error costs."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from abc_credit.modeling.feature_analysis import load_splits, make_pipeline
from abc_credit.modeling.phase_b import evaluate

OUTPUT = Path("reports/generated/threshold_analysis")


def threshold_table(y, probabilities):
    """Evaluate all possible decisions, keeping tied scores together."""
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    order = np.argsort(-probabilities, kind="stable")
    scores = probabilities[order]
    labels = y[order]

    # At each boundary, include every application sharing the same score.
    ends = np.r_[
        np.flatnonzero(scores[:-1] != scores[1:]),
        len(scores) - 1,
    ]

    cumulative_declined = np.cumsum(labels)
    predicted_declined = ends + 1

    true_declined = cumulative_declined[ends]
    false_declined = predicted_declined - true_declined
    total_declined = labels.sum()
    total_approved = len(labels) - total_declined

    # First row represents approving everyone.
    thresholds = np.r_[
        np.nextafter(scores[0], np.inf),
        scores[ends],
    ]
    true_declined = np.r_[0, true_declined]
    false_declined = np.r_[0, false_declined]
    missed_declined = total_declined - true_declined

    precision = np.divide(
        true_declined,
        true_declined + false_declined,
        out=np.zeros(len(thresholds), dtype=float),
        where=(true_declined + false_declined) > 0,
    )
    recall = true_declined / total_declined
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(thresholds), dtype=float),
        where=(precision + recall) > 0,
    )

    return pd.DataFrame({
        "threshold": thresholds,
        "accuracy": (
            true_declined + total_approved - false_declined
        ) / len(labels),
        "declined_precision": precision,
        "declined_recall": recall,
        "declined_f1": f1,
        "historically_approved_predicted_declined": false_declined,
        "historically_declined_predicted_approved": missed_declined,
        "correctly_detected_declines": true_declined,
    })


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    logistic = make_pipeline("without_gender")

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

    models = {
        "logistic": logistic,
        "random_forest": forest,
    }

    summaries = []

    for name, model in models.items():
        model.fit(splits.X_train, y_train)
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(
            splits.X_validation
        )[:, positive_index]

        table = threshold_table(y_validation, probabilities)

        # Cost of approving a historical decline versus declining an approval.
        for false_approval_cost in [1, 5, 10]:
            costs = (
                false_approval_cost
                * table["historically_declined_predicted_approved"]
                + table["historically_approved_predicted_declined"]
            )

            # Among equally costly choices, select the highest threshold.
            best = table.loc[costs.eq(costs.min())].sort_values(
                "threshold", ascending=False
            ).iloc[0]

            threshold = float(best["threshold"])
            metrics = evaluate(
                y_validation, probabilities, threshold
            )
            matrix = metrics["confusion_matrix_approved_declined"]

            summaries.append({
                "model": name,
                "false_approval_cost": false_approval_cost,
                "false_decline_cost": 1,
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "declined_precision": metrics["declined_precision"],
                "declined_recall": metrics["declined_recall"],
                "declined_f1": metrics["declined_f1"],
                "declined_pr_auc": (
                    metrics["declined_pr_auc_average_precision"]
                ),
                "historically_approved_predicted_declined": matrix[0][1],
                "historically_declined_predicted_approved": matrix[1][0],
                "correctly_detected_declines": matrix[1][1],
                "total_assumed_cost": (
                    false_approval_cost * matrix[1][0]
                    + matrix[0][1]
                ),
                "all_approved_assumed_cost": (
                    false_approval_cost * int(y_validation.sum())
                ),
            })

        table.to_csv(
            OUTPUT / f"{name}_threshold_tradeoffs.csv",
            index=False,
        )
        print(f"Completed: {name}", flush=True)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "cost_scenario_comparison.csv", index=False)

    print("\nCOST SCENARIOS")
    print(summary.round(4).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT}")
    print("No test-set evaluation performed.")


if __name__ == "__main__":
    main()