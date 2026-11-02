"""Package the selected model without evaluating the test partition."""

import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from abc_credit.modeling.feature_analysis import (
    FEATURE_SETS,
    load_splits,
    make_pipeline,
)
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("models/random_forest_v1")
MODEL_VERSION = "random-forest-v1"

# Illustrative scenario, not measured financial losses.
FALSE_APPROVAL_COST = 5
FALSE_DECLINE_COST = 1


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    artifact_path = OUTPUT / "decision_model.joblib"
    metadata_path = OUTPUT / "metadata.json"

    if artifact_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "Version already exists. Use a new output directory/version "
            "rather than overwriting the packaged model."
        )

    splits = load_splits()
    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    features = FEATURE_SETS["without_gender"].copy()

    model = make_pipeline("without_gender")
    model.set_params(
        classifier=RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=5,
            class_weight=None,
            random_state=42,
            n_jobs=2,
        )
    )

    # Select inputs explicitly so gender is unnecessary at inference.
    X_train = splits.X_train[features]
    X_validation = splits.X_validation[features]

    print("Training selected Random Forest...", flush=True)
    model.fit(X_train, y_train)

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(
        X_validation
    )[:, positive_index]

    table = threshold_table(y_validation, probabilities)
    table["assumed_cost"] = (
        FALSE_APPROVAL_COST
        * table["historically_declined_predicted_approved"]
        + FALSE_DECLINE_COST
        * table["historically_approved_predicted_declined"]
    )

    selected = table.sort_values(
        ["assumed_cost", "threshold"],
        ascending=[True, False],
    ).iloc[0]

    threshold = float(selected["threshold"])
    metrics = evaluate(y_validation, probabilities, threshold)

    created_at = datetime.now(UTC).isoformat()

    metadata = {
        "model_version": MODEL_VERSION,
        "created_at_utc": created_at,
        "purpose": "Predict historical loan approval decisions",
        "status": "Academic prototype; declined recall remains limited",
        "model_type": "RandomForestClassifier",
        "feature_columns": features,
        "excluded_predictors": ["ID", "Decision", "Gender"],
        "label_mapping": {"APPROVED": 0, "DECLINED": 1},
        "positive_class": "DECLINED",
        "positive_class_index": positive_index,
        "threshold": threshold,
        "decision_rule": (
            "DECLINED if probability_declined >= threshold; "
            "otherwise APPROVED"
        ),
        "calibration": "None; sigmoid calibration evaluated but not selected",
        "preprocessing": {
            "numeric": (
                "Training median imputation with missing indicators; "
                "training-fitted StandardScaler"
            ),
            "categorical": (
                "Training most-frequent imputation; existing one-hot "
                "encoding with rare-category grouping"
            ),
            "feature_transform": (
                "Existing baseline transformation; no zero-LTV replacement"
            ),
        },
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 18,
            "min_samples_leaf": 5,
            "class_weight": None,
            "random_state": 42,
        },
        "threshold_selection": {
            "data": "Existing validation partition",
            "method": "Minimum cost over distinct score boundaries",
            "tie_break": "Highest threshold",
            "false_approval_cost": FALSE_APPROVAL_COST,
            "false_decline_cost": FALSE_DECLINE_COST,
            "assumption": (
                "Illustrative historical-decision disagreement costs; "
                "not measured default losses"
            ),
        },
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "validation_metrics": metrics,
        "versions": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in [
                    "scikit-learn",
                    "pandas",
                    "numpy",
                    "scipy",
                    "joblib",
                ]
            },
        },
        "limitations": [
            "Predicts historical decisions, not repayment/default.",
            "Validation data were reused for development and selection.",
            "Internal test partition was inspected in earlier work.",
            "No test evaluation performed by this packaging script.",
            "Excluding gender does not establish fairness.",
            "Artifact requires compatible dependencies and project source.",
        ],
    }

    bundle = {
        "pipeline": model,
        "metadata": metadata,
    }
    joblib.dump(bundle, artifact_path)

    # Verify that saving/loading preserves predictions.
    restored = joblib.load(artifact_path)
    before = model.predict_proba(X_validation.head(20))[:, positive_index]
    after = restored["pipeline"].predict_proba(
        X_validation.head(20)
    )[:, positive_index]

    np.testing.assert_allclose(
        before, after, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_array_equal(
        before >= threshold,
        after >= restored["metadata"]["threshold"],
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nModel version:", MODEL_VERSION)
    print("Selected threshold:", threshold)
    print("Validation metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nSerialization check passed.")
    print("Saved to:", OUTPUT)
    print("Test partition was not evaluated.")


if __name__ == "__main__":
    main()