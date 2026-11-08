"""Check calibration for CatBoost with implied vehicle value."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold

from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.feature_analysis import load_splits

OUTPUT = Path("reports/generated/vehicle_value_calibration")


def raw_scores(pipeline, frame):
    prepared = pipeline.named_steps["prepare"].transform(frame)

    return np.asarray(
        pipeline.named_steps["classifier"].predict(
            prepared,
            prediction_type="RawFormulaVal",
        ),
        dtype=float,
    ).reshape(-1)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    X_train = splits.X_train
    y_train = splits.y_train.eq("DECLINED").astype(int)
    X_validation = splits.X_validation
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    folds = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    oof_scores = np.full(len(X_train), np.nan)
    assignments = np.zeros(len(X_train), dtype=int)

    for fold, (fit_positions, score_positions) in enumerate(
        folds.split(X_train, y_train), start=1
    ):
        print(f"Training calibration fold {fold}/3...", flush=True)

        model = make_model("vehicle_value")
        model.fit(
            X_train.iloc[fit_positions],
            y_train.iloc[fit_positions],
        )

        oof_scores[score_positions] = raw_scores(
            model,
            X_train.iloc[score_positions],
        )
        assignments[score_positions] += 1

    assert np.isfinite(oof_scores).all()
    assert (assignments == 1).all()

    calibrator = LogisticRegression(
        C=1_000_000,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    calibrator.fit(oof_scores.reshape(-1, 1), y_train)

    print("Fitting the full-training candidate...", flush=True)
    full_model = make_model("vehicle_value")
    full_model.fit(X_train, y_train)

    positive_index = list(full_model.classes_).index(1)
    original = full_model.predict_proba(
        X_validation
    )[:, positive_index]

    validation_scores = raw_scores(full_model, X_validation)
    calibration_index = list(calibrator.classes_).index(1)

    calibrated = calibrator.predict_proba(
        validation_scores.reshape(-1, 1)
    )[:, calibration_index]

    rows = []
    bin_tables = []
    fig, axis = plt.subplots(figsize=(7, 6))
    upper_limit = 0.10

    for name, probabilities in {
        "uncalibrated": original,
        "sigmoid_calibrated": calibrated,
    }.items():
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        rows.append({
            "version": name,
            "average_precision": average_precision_score(
                y_validation, probabilities
            ),
            "brier_score": brier_score_loss(
                y_validation, probabilities
            ),
            "log_loss": log_loss(
                y_validation, probabilities, labels=[0, 1]
            ),
            "mean_predicted_decline_probability": probabilities.mean(),
            "observed_decline_fraction": y_validation.mean(),
        })

        observed, predicted = calibration_curve(
            y_validation,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )

        axis.plot(predicted, observed, marker="o", label=name)

        upper_limit = max(
            upper_limit,
            float(predicted.max()) * 1.15,
            float(observed.max()) * 1.15,
        )

        bin_tables.append(pd.DataFrame({
            "version": name,
            "mean_predicted_probability": predicted,
            "observed_decline_fraction": observed,
        }))

    upper_limit = min(1.0, upper_limit)
    axis.plot(
        [0, upper_limit],
        [0, upper_limit],
        "--",
        color="grey",
        label="Perfect calibration",
    )
    axis.set_xlim(0, upper_limit)
    axis.set_ylim(0, upper_limit)
    axis.set_xlabel("Mean predicted historical-decline probability")
    axis.set_ylabel("Observed historical-decline fraction")
    axis.set_title("CatBoost with implied vehicle value")
    axis.grid(alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT / "reliability_plot.png", dpi=160)
    plt.close(fig)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        OUTPUT / "calibration_comparison.csv", index=False
    )
    pd.concat(bin_tables, ignore_index=True).to_csv(
        OUTPUT / "reliability_bins.csv", index=False
    )

    print("\nCALIBRATION COMPARISON")
    print(comparison.round(8).to_string(index=False))

    slope = float(calibrator.coef_[0, 0])
    print("\nSigmoid slope:", slope)
    if slope <= 0:
        print("WARNING: Non-positive slope; investigate before adoption.")

    print(f"\nSaved to: {OUTPUT}")
    print("No threshold was selected.")
    print("Test partition and saved model artifacts were not changed.")


if __name__ == "__main__":
    main()