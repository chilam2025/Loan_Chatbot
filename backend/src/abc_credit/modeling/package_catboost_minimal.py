"""Package the five-question CatBoost model (ten contract fields)."""

import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from abc_credit.data_quality import NUMERIC_COLUMNS
from abc_credit.modeling.compare_catboost import CATEGORICAL_FIELDS
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.inference import load_model, predict_applications
from abc_credit.modeling.minimal_model import (
    MINIMAL_FIELDS,
    build_minimal_pipeline,
)
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("models/catboost_minimal_v1")
MODEL_VERSION = "catboost-minimal-v1"
PROFILE_KEY = [
    "Pincode",
    "Age",
    "Net_salary",
    "Qualifications",
    "Employment_Type",
    "Resident_Type",
    "Final_Tier",
    "PAST_LOANS_ACTIVE",
]
RECALL_TARGETS = (0.30, 0.40, 0.50, 0.60)


def profile_ids(frame):
    key = frame[PROFILE_KEY].astype("string").fillna("__MISSING__")
    return key.agg("|".join, axis=1)


def curve(y, scores):
    """Precision, recall, F1 and counts at every possible cutoff."""
    table = threshold_table(y, scores)
    caught = table["correctly_detected_declines"].to_numpy(dtype=float)
    false_flags = table[
        "historically_approved_predicted_declined"
    ].to_numpy(dtype=float)
    flagged = caught + false_flags
    total = float(np.asarray(y).sum())

    precision = np.divide(
        caught, flagged, out=np.zeros_like(caught), where=flagged > 0
    )
    recall = caught / total
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(caught),
        where=denominator > 0,
    )
    return table["threshold"].to_numpy(), precision, recall, f1, flagged


def f1_threshold(y, scores):
    thresholds, _, _, f1, _ = curve(y, scores)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def operating_points(y, scores):
    """Cutoffs that reach several recall levels, on held-out scores."""
    thresholds, precision, recall, _, flagged = curve(y, scores)
    points = []
    for target in RECALL_TARGETS:
        row = int(np.argmax(recall >= target))
        points.append({
            "target_recall": target,
            "cutoff": float(thresholds[row]),
            "precision": float(precision[row]),
            "recall": float(recall[row]),
            "share_of_applications_flagged": float(
                flagged[row] / len(y)
            ),
        })
    return points


def held_out_scores(X, y, groups):
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    scores = np.empty(len(y))

    for fold, (fit, held_out) in enumerate(cv.split(X, y, groups), 1):
        print(f"Cutoff folds: fold {fold}/3", flush=True)
        model = build_minimal_pipeline()
        model.fit(X.iloc[fit], y.iloc[fit])
        index = list(model.classes_).index(1)
        scores[held_out] = model.predict_proba(X.iloc[held_out])[:, index]

    return scores


def subgroup_report(frame, column, y, predicted):
    report = {}
    for value, rows in frame.groupby(column, dropna=False).groups.items():
        mask = frame.index.isin(rows)
        declines = int(y[mask].sum())
        report[str(value)] = {
            "rows": int(mask.sum()),
            "historical_decline_rate": float(y[mask].mean()),
            "predicted_decline_rate": float(predicted[mask].mean()),
            "decline_recall": (
                float(predicted[mask & (y == 1)].mean())
                if declines
                else None
            ),
        }
    return report


def main():
    artifact_path = OUTPUT / "decision_model.joblib"
    metadata_path = OUTPUT / "metadata.json"

    if artifact_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"{MODEL_VERSION} already exists in {OUTPUT}. "
            "Use a new version and output directory."
        )

    splits = load_splits()
    all_fields = list(NUMERIC_COLUMNS) + list(CATEGORICAL_FIELDS)

    X_train = splits.X_train[MINIMAL_FIELDS].copy()
    X_validation = splits.X_validation[MINIMAL_FIELDS].copy()
    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)
    train_profiles = profile_ids(splits.X_train[all_fields])
    validation_profiles = profile_ids(splits.X_validation[all_fields])

    held_out = held_out_scores(X_train, y_train, train_profiles)
    threshold, cutoff_f1 = f1_threshold(y_train, held_out)
    points = operating_points(y_train, held_out)
    print(f"Selected cutoff {threshold:.6f} (training F1 {cutoff_f1:.4f})")

    print("Training the final model...", flush=True)
    started = perf_counter()
    model = build_minimal_pipeline().fit(X_train, y_train)
    training_seconds = perf_counter() - started

    positive_index = list(model.classes_).index(1)
    started = perf_counter()
    probabilities = model.predict_proba(X_validation)[:, positive_index]
    batch_seconds = perf_counter() - started

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite validation probabilities")

    metrics = evaluate(y_validation, probabilities, threshold)

    seen = validation_profiles.isin(set(train_profiles))
    unseen = ~seen.to_numpy()
    unseen_metrics = evaluate(
        y_validation[unseen], probabilities[unseen], threshold
    )

    predicted = probabilities >= threshold
    validation_frame = splits.X_validation.copy().reset_index(drop=True)
    fairness = {
        column: subgroup_report(
            validation_frame, column, y_validation.to_numpy(), predicted
        )
        for column in ("Gender", "Resident_Type")
        if column in validation_frame
    }

    metadata = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Predict historical loan approval decisions",
        "status": "Academic prototype with limited decline recall",
        "model_type": "CatBoostClassifier pipeline, five applicant questions",
        "feature_columns": MINIMAL_FIELDS,
        "applicant_questions": [
            "age",
            "pincode",
            "vehicle (catalogue selection)",
            "loan_amount",
            "vehicle_price",
            "past_loans_active",
        ],
        "derived_by_backend": {
            "LTV": "loan_amount / vehicle_price * 100, must be in (0, 100]",
            "Final_Tier": "lookup from pincode (reference_tables.json)",
            "Make_Code, Model_Variant, Product_Code": (
                "lookup from the selected Model_Description"
            ),
        },
        "unused_inputs": [
            "Qualifications",
            "Employment_Type",
            "Resident_Type",
            "Net_salary",
            "Gender",
        ],
        "label_mapping": {"APPROVED": 0, "DECLINED": 1},
        "positive_class": "DECLINED",
        "positive_class_index": positive_index,
        "threshold": threshold,
        "decision_rule": (
            "DECLINED if probability_declined >= threshold; "
            "otherwise APPROVED"
        ),
        "calibration": "No separate calibration applied",
        "hyperparameters": {
            "iterations": 600,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 10,
            "random_seed": 42,
        },
        "threshold_selection": {
            "data": (
                "Held-out training predictions from three profile-grouped "
                "folds"
            ),
            "method": "Maximize F1 for the historical DECLINED class",
            "training_f1_at_threshold": cutoff_f1,
        },
        "operating_points_on_held_out_training_scores": points,
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "validation_metrics": metrics,
        "validation_metrics_unseen_profiles": unseen_metrics,
        "validation_rows_with_seen_profile": int(seen.sum()),
        "validation_subgroup_rates": fairness,
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
            "Labels represent historical decisions, not repayment.",
            "Declined-class recall remains limited.",
            "Validation was reused during model development.",
            "Hidden evaluation performance is unknown.",
            "Fairness has not been established.",
            "Location (pincode) can act as a proxy for community traits.",
        ],
    }

    bundle = {"pipeline": model, "metadata": metadata}

    sample = X_validation.head(20)
    expected = predict_applications(bundle, sample)
    np.testing.assert_allclose(
        expected["probability_declined"],
        probabilities[: len(sample)],
        rtol=1e-12,
        atol=1e-15,
    )

    one_application = sample.head(1)
    durations_ms = []
    for _ in range(30):
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
            expected["decision"], actual["decision"]
        )
        metadata_path.write_text(metadata_json + "\n", encoding="utf-8")
    except Exception:
        artifact_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise

    print("\nModel version:", MODEL_VERSION)
    print("Fixed threshold:", threshold)
    print("Validation metrics:")
    print(json.dumps(metrics, indent=2))
    print("Validation metrics, unseen-profile rows only:")
    print(json.dumps(unseen_metrics, indent=2))
    print("Operating points (held-out training scores):")
    print(json.dumps(points, indent=2))
    print("Subgroup rates:")
    print(json.dumps(fairness, indent=2))
    print("Timing:")
    print(json.dumps(metadata["timing"], indent=2))
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()