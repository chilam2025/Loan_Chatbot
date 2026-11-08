"""Package the CatBoost pincode-prefix candidate with a fixed threshold."""

import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.preprocessing import FunctionTransformer

from abc_credit.data_quality import NUMERIC_COLUMNS
from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.compare_catboost import CATEGORICAL_FIELDS
from abc_credit.modeling.compare_pincode_signal import represent_pincode
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.inference import load_model, predict_applications
from abc_credit.modeling.phase_b import evaluate

OUTPUT = Path("models/catboost_pincode_prefix_v1")
MODEL_VERSION = "catboost-pincode-prefix-v1"
THRESHOLD = 0.2014630886001852


def main():
    artifact_path = OUTPUT / "decision_model.joblib"
    metadata_path = OUTPUT / "metadata.json"

    if artifact_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"{MODEL_VERSION} already exists in {OUTPUT}. "
            "Use a new version and output directory to create another package."
        )

    splits = load_splits()
    features = list(NUMERIC_COLUMNS) + list(CATEGORICAL_FIELDS)

    X_train = splits.X_train[features].copy()
    X_validation = splits.X_validation[features].copy()
    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    model = make_model("vehicle_value")

    # Keep the original six-digit pincode in the input contract.
    # The same conversion runs during training and inference.
    model.steps.insert(
        0,
        (
            "pincode_prefix",
            FunctionTransformer(
                represent_pincode,
                kw_args={"variant": "prefix_3"},
            ),
        ),
    )

    print(
        "Training CatBoost with implied vehicle value and pincode prefix...",
        flush=True,
    )
    started = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - started

    positive_index = list(model.classes_).index(1)

    started = perf_counter()
    probabilities = model.predict_proba(X_validation)[:, positive_index]
    batch_seconds = perf_counter() - started

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite validation probabilities")

    metrics = evaluate(y_validation, probabilities, THRESHOLD)
    matrix = metrics["confusion_matrix_approved_declined"]

    metadata = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Predict historical loan approval decisions",
        "status": "Academic prototype with limited decline recall",
        "model_type": "CatBoostClassifier",
        "feature_columns": features,
        "excluded_predictors": ["ID", "Decision", "Gender"],
        "derived_features": [
            "LTV_missing_or_zero",
            "implied_vehicle_value",
        ],
        "label_mapping": {"APPROVED": 0, "DECLINED": 1},
        "positive_class": "DECLINED",
        "positive_class_index": positive_index,
        "threshold": THRESHOLD,
        "decision_rule": (
            "DECLINED if probability_declined >= threshold; "
            "otherwise APPROVED"
        ),
        "calibration": "No separate calibration applied",
        "preprocessing": {
            "pincode": (
                "Valid six-digit pincodes are converted to their first "
                "three digits. Invalid and missing values are treated as missing."
            ),
            "ltv": (
                "Zero or missing LTV is replaced with the training median "
                "and marked with a missing-or-zero indicator."
            ),
            "training_ltv_median": float(
                model.named_steps["prepare"].ltv_median_
            ),
            "implied_vehicle_value": (
                "Loan_Amount / (original LTV / 100) when original LTV "
                "is greater than zero and at most 100. Otherwise missing."
            ),
            "vehicle_value_interpretation": (
                "Derived estimate based on loan amount and original LTV."
            ),
            "categorical": "Native CatBoost categorical handling",
        },
        "hyperparameters": {
            "iterations": 600,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 10,
            "loss_function": "Logloss",
            "random_seed": 42,
            "thread_count": 2,
            "class_weights": None,
        },
        "threshold_selection": {
            "data": "Previously reused development validation partition",
            "method": "Previously selected minimum illustrative 5:1 cost",
            "false_approval_cost": 5,
            "false_decline_cost": 1,
            "reselected_during_packaging": False,
            "assumption": "Illustrative historical-decision disagreement costs",
        },
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "validation_metrics": metrics,
        "validation_assumed_cost_5_to_1": (
            5 * int(matrix[1][0]) + int(matrix[0][1])
        ),
        "timing": {
            "training_seconds": training_seconds,
            "validation_batch_prediction_seconds": batch_seconds,
        },
        "versions": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in [
                    "catboost",
                    "scikit-learn",
                    "pandas",
                    "numpy",
                    "joblib",
                ]
            },
        },
        "limitations": [
            "Labels represent historical decisions rather than repayment outcomes.",
            "Declined-class recall remains limited.",
            "Validation was reused during model development.",
            "The internal test partition was inspected in earlier work.",
            "Hidden evaluation performance is unknown.",
            "Fairness has not been established.",
            "Loading requires compatible dependencies and project source.",
        ],
    }

    bundle = {"pipeline": model, "metadata": metadata}

    # Verify that the public inference function agrees with direct scoring.
    sample = X_validation.head(20)
    expected = predict_applications(bundle, sample)
    np.testing.assert_allclose(
        expected["probability_declined"],
        probabilities[:len(sample)],
        rtol=1e-12,
        atol=1e-15,
    )

    one_application = sample.head(1)
    predict_applications(bundle, one_application)

    durations_ms = []
    for _ in range(50):
        started = perf_counter()
        predict_applications(bundle, one_application)
        durations_ms.append((perf_counter() - started) * 1000)

    metadata["timing"].update({
        "single_application_latency_runs": len(durations_ms),
        "single_application_latency_median_ms": float(
            np.median(durations_ms)
        ),
        "single_application_latency_p95_ms": float(
            np.percentile(durations_ms, 95)
        ),
        "latency_scope": (
            "Local warm inference including input normalization. "
            "Excludes startup, network, and API overhead."
        ),
    })

    metadata_json = json.dumps(metadata, indent=2, allow_nan=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)

    try:
        restored = load_model(artifact_path)
        actual = predict_applications(restored, sample)

        np.testing.assert_allclose(
            expected["probability_declined"],
            actual["probability_declined"],
            rtol=1e-12,
            atol=1e-15,
        )
        np.testing.assert_array_equal(
            expected["decision"],
            actual["decision"],
        )
        metadata_path.write_text(metadata_json + "\n", encoding="utf-8")
    except Exception:
        # Allow a retry if package verification fails.
        artifact_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise

    print("\nModel version:", MODEL_VERSION)
    print("Fixed threshold:", THRESHOLD)
    print("Validation metrics:")
    print(json.dumps(metrics, indent=2))
    print("Illustrative cost:", metadata["validation_assumed_cost_5_to_1"])
    print("\nTiming:")
    print(json.dumps(metadata["timing"], indent=2))
    print("\nSerialization and inference checks passed.")
    print("Saved to:", OUTPUT)
    print("Test partition was not evaluated.")


if __name__ == "__main__":
    main()