"""Evaluate CatBoost sigmoid calibration without using the test set."""

import json
from pathlib import Path

import joblib
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

from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.nested_model_comparison import make_catboost

CANDIDATE = Path(
    "reports/generated/catboost_selection/training_candidate.joblib"
)
OUTPUT = Path("reports/generated/catboost_calibration")


def raw_scores(pipeline, frame):
    """Get CatBoost margins before conversion to probabilities."""
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

    # Load only your trusted, locally generated artifact.
    candidate = joblib.load(CANDIDATE)
    selected_parameters = candidate["metadata"]["selected_parameters"]

    splits = load_splits()
    X_train = splits.X_train
    y_train = splits.y_train.eq("DECLINED").astype(int)
    X_validation = splits.X_validation
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    if len(X_train) != candidate["metadata"]["training_rows"]:
        raise ValueError("Training row count differs from the saved candidate")

    folds = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    oof_margins = np.full(len(X_train), np.nan)
    assigned = np.zeros(len(X_train), dtype=int)

    for fold, (fit_positions, score_positions) in enumerate(
        folds.split(X_train, y_train), start=1
    ):
        print(f"Training calibration fold {fold}/3...", flush=True)

        pipeline = make_catboost()
        pipeline.set_params(**selected_parameters)

        pipeline.fit(
            X_train.iloc[fit_positions],
            y_train.iloc[fit_positions],
        )

        oof_margins[score_positions] = raw_scores(
            pipeline,
            X_train.iloc[score_positions],
        )
        assigned[score_positions] += 1

    assert np.isfinite(oof_margins).all()
    assert (assigned == 1).all()

    # Logistic regression on raw margins provides a sigmoid mapping.
    # No class weighting: preserve the observed training prevalence.
    calibrator = LogisticRegression(
        C=1_000_000,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    calibrator.fit(
        oof_margins.reshape(-1, 1),
        y_train,
    )

    # Evaluate the saved full-training model, not an outer-fold model.
    full_model = candidate["pipeline"]
    positive_index = list(full_model.classes_).index(1)

    original_probabilities = full_model.predict_proba(
        X_validation
    )[:, positive_index]

    validation_margins = raw_scores(full_model, X_validation)
    calibration_positive_index = list(calibrator.classes_).index(1)

    calibrated_probabilities = calibrator.predict_proba(
        validation_margins.reshape(-1, 1)
    )[:, calibration_positive_index]

    versions = {
        "uncalibrated": original_probabilities,
        "sigmoid_calibrated": calibrated_probabilities,
    }

    rows = []
    bin_tables = []
    fig, axis = plt.subplots(figsize=(7, 6))
    maximum_point = 0.1

    for name, probabilities in versions.items():
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

        axis.plot(
            predicted, observed, marker="o", label=name
        )

        maximum_point = max(
            maximum_point,
            float(predicted.max()),
            float(observed.max()),
        )

        bin_tables.append(pd.DataFrame({
            "version": name,
            "mean_predicted_probability": predicted,
            "observed_decline_fraction": observed,
        }))

    limit = min(1.0, maximum_point * 1.15)
    axis.plot(
        [0, limit],
        [0, limit],
        "--",
        color="grey",
        label="Perfect calibration",
    )
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("Mean predicted historical-decline probability")
    axis.set_ylabel("Observed historical-decline fraction")
    axis.set_title("CatBoost: validation reliability")
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

    calibration_info = {
        "method": "Sigmoid mapping fitted to training-only OOF raw margins",
        "folds": 3,
        "selected_model_parameters": selected_parameters,
        "slope": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        "status": "Experimental; calibration and threshold not locked",
        "limitation": (
            "Model settings were selected using this training partition "
            "before generating calibration folds. This is development "
            "calibration, not a fully nested final performance estimate."
        ),
    }

    (OUTPUT / "calibration_info.json").write_text(
        json.dumps(calibration_info, indent=2),
        encoding="utf-8",
    )

    # Save only the experimental calibration mapping.
    # This does not overwrite the candidate or packaged forest.
    joblib.dump(
        {
            "calibrator": calibrator,
            "metadata": calibration_info,
        },
        OUTPUT / "sigmoid_mapping.joblib",
    )

    print("\nVALIDATION CALIBRATION COMPARISON")
    print(comparison.round(6).to_string(index=False))
    print("\nSigmoid slope:", calibration_info["slope"])

    if calibration_info["slope"] <= 0:
        print(
            "WARNING: Non-positive slope. Investigate before adopting "
            "this calibration mapping."
        )

    print(f"\nSaved to: {OUTPUT}")
    print("No decision threshold was selected.")
    print("Test partition was not evaluated.")


if __name__ == "__main__":
    main()