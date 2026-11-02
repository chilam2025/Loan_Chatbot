"""Compare sigmoid calibration using training folds and validation data."""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from abc_credit.modeling.feature_analysis import load_splits, make_pipeline
from abc_credit.modeling.phase_b import choose_threshold, evaluate

OUTPUT = Path("reports/generated/calibration_comparison")


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

    candidates = {
        "logistic": logistic,
        "random_forest": forest,
    }

    folds = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    results = []
    reliability_tables = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for axis, (name, pipeline) in zip(
        axes, candidates.items(), strict=True
    ):
        versions = {
            "uncalibrated": clone(pipeline),
            "sigmoid_calibrated": CalibratedClassifierCV(
                estimator=clone(pipeline),
                method="sigmoid",
                cv=folds,
                ensemble=False,
                n_jobs=1,
            ),
        }

        axis.plot(
            [0, 1], [0, 1],
            linestyle="--",
            color="grey",
            label="Perfect calibration",
        )

        for version, estimator in versions.items():
            start = perf_counter()
            estimator.fit(splits.X_train, y_train)
            training_seconds = perf_counter() - start

            start = perf_counter()
            positive_index = list(estimator.classes_).index(1)
            probabilities = estimator.predict_proba(
                splits.X_validation
            )[:, positive_index]
            prediction_seconds = perf_counter() - start

            assert np.isfinite(probabilities).all()
            assert ((probabilities >= 0) & (probabilities <= 1)).all()

            threshold = choose_threshold(
                y_validation,
                probabilities,
                false_approval_cost=5.0,
                false_decline_cost=1.0,
            )

            metrics = evaluate(
                y_validation, probabilities, threshold
            )
            matrix = metrics.pop(
                "confusion_matrix_approved_declined"
            )

            label = f"{name}_{version}"

            results.append({
                "model": label,
                "threshold": threshold,
                **metrics,
                "log_loss": log_loss(
                    y_validation,
                    probabilities,
                    labels=[0, 1],
                ),
                "historically_approved_predicted_declined": matrix[0][1],
                "historically_declined_predicted_approved": matrix[1][0],
                "validation_cost_5_to_1": (
                    5 * matrix[1][0] + matrix[0][1]
                ),
                "training_seconds": training_seconds,
                "validation_batch_prediction_seconds": prediction_seconds,
            })

            observed, predicted = calibration_curve(
                y_validation,
                probabilities,
                n_bins=10,
                strategy="quantile",
            )

            axis.plot(
                predicted,
                observed,
                marker="o",
                label=version,
            )

            reliability_tables.append(pd.DataFrame({
                "model": label,
                "bin": np.arange(1, len(predicted) + 1),
                "mean_predicted_decline_probability": predicted,
                "observed_decline_fraction": observed,
            }))

            print(f"Completed: {label}", flush=True)

        axis.set_title(name)
        axis.set_xlabel("Mean predicted historical-decline probability")
        axis.set_ylabel("Observed historical-decline fraction")
        axis.grid(alpha=0.3)
        axis.legend()

        # Zoom around the probability range occupied by the reliability points.
        point_values = np.concatenate([
            np.asarray(line.get_xdata(), dtype=float)
            for line in axis.lines[1:]
        ] + [
            np.asarray(line.get_ydata(), dtype=float)
            for line in axis.lines[1:]
        ])
        limit = min(1.0, max(0.1, float(point_values.max()) * 1.15))
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)

    fig.tight_layout()
    fig.savefig(OUTPUT / "reliability_plots.png", dpi=160)
    plt.close(fig)

    comparison = pd.DataFrame(results)
    comparison.to_csv(
        OUTPUT / "calibration_comparison.csv",
        index=False,
    )

    pd.concat(reliability_tables, ignore_index=True).to_csv(
        OUTPUT / "reliability_bins.csv",
        index=False,
    )

    print("\nVALIDATION RESULTS")
    print(comparison.round(5).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT}")
    print("No test-set evaluation performed.")


if __name__ == "__main__":
    main()