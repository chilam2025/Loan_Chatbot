"""Select CatBoost parameters using training-only cross-validation."""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from abc_credit.modeling.compare_catboost import INPUT_FIELDS
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.nested_model_comparison import manual_search

OUTPUT = Path("reports/generated/catboost_selection")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    artifact_path = OUTPUT / "training_candidate.joblib"

    if artifact_path.exists():
        raise FileExistsError(
            "A candidate already exists in this output folder. "
            "Use another output folder for a separate experiment."
        )

    splits = load_splits()
    X_train = splits.X_train
    y_train = splits.y_train.eq("DECLINED").astype(int)

    parameter_grid = {
        "classifier__depth": [4, 6],
        "classifier__l2_leaf_reg": [3, 10],
    }

    folds = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    print(
        "Selecting settings using the original training partition only.",
        flush=True,
    )

    search = manual_search(
        X_train,
        y_train,
        parameter_grid,
        folds,
    )

    pd.DataFrame(search.cv_results_).to_csv(
        OUTPUT / "training_cv_results.csv",
        index=False,
    )

    # Derive the exact required inputs from the selected transformer.
    prepare = search.best_estimator_.named_steps["prepare"]
    

    metadata = {
        "status": "Training candidate; threshold and calibration not locked",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "training_rows": len(X_train),
        "feature_columns": INPUT_FIELDS,
        "positive_class": "DECLINED",
        "label_mapping": {"APPROVED": 0, "DECLINED": 1},
        "selection_metric": "average_precision",
        "cv_folds": 3,
        "selected_parameters": search.best_params_,
        "selected_mean_cv_average_precision": float(search.best_score_),
        "fixed_settings": {
            "iterations": 600,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "random_seed": 42,
        },
        "ltv_treatment": (
            "Zero/missing LTV replaced by training median, "
            "with an explicit indicator"
        ),
        "training_ltv_median": float(prepare.ltv_median_),
        "threshold": None,
        "versions": {
            package: version(package)
            for package in [
                "catboost",
                "scikit-learn",
                "pandas",
                "numpy",
                "joblib",
            ]
        },
        "limitations": [
            "CV score was used for selection, not independent evaluation.",
            "Predicts historical decisions, not repayment/default.",
            "Candidate is not ready for backend replacement.",
        ],
    }

    bundle = {
        "pipeline": search.best_estimator_,
        "metadata": metadata,
    }

    joblib.dump(bundle, artifact_path)

    # Verify serialization using training examples only.
    restored = joblib.load(artifact_path)
    sample = X_train.head(10)

    np.testing.assert_allclose(
        bundle["pipeline"].predict_proba(sample),
        restored["pipeline"].predict_proba(sample),
        rtol=1e-12,
        atol=1e-15,
    )

    (OUTPUT / "selected_settings.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSELECTED SETTINGS")
    print(json.dumps(metadata, indent=2))
    print("\nSerialization check passed.")
    print("Validation and test partitions were not evaluated.")
    print("Existing packaged forest was not modified.")


if __name__ == "__main__":
    main()