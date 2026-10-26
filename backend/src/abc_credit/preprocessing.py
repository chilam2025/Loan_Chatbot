"""Leakage-safe data splitting and preprocessing for model development."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from abc_credit.data_quality import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    load_raw_data,
    normalize_for_analysis,
)

TARGET_COLUMN = "Decision"
REFERENCE_COLUMNS = ["ID"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
DEFAULT_RANDOM_STATE = 42
MIN_CATEGORY_FREQUENCY = 10
MAX_CATEGORIES_PER_FEATURE = 100


@dataclass(frozen=True)
class DataSplits:
    """Feature and target partitions created before preprocessing is fitted."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class PreparedTrainingData:
    """Raw splits, fitted preprocessor, and transformed feature matrices."""

    splits: DataSplits
    preprocessor: ColumnTransformer
    X_train_transformed: Any
    X_validation_transformed: Any
    X_test_transformed: Any


def stratified_split(
    labelled: pd.DataFrame,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DataSplits:
    """Create train, validation, and test partitions with target stratification."""

    holdout_size = validation_size + test_size
    if validation_size <= 0 or test_size <= 0 or holdout_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1")
    if TARGET_COLUMN not in labelled:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    features = labelled[FEATURE_COLUMNS]
    target = labelled[TARGET_COLUMN]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=holdout_size,
        random_state=random_state,
        stratify=target,
    )

    relative_test_size = test_size / holdout_size
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_holdout,
    )
    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build imputers and encoders that must be fitted on training data only."""

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values=pd.NA)),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    max_categories=MAX_CATEGORIES_PER_FEATURE,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_training_data(
    labelled: pd.DataFrame,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> PreparedTrainingData:
    """Split first, then fit preprocessing exclusively on the training partition."""

    splits = stratified_split(
        labelled,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(splits.X_train)
    X_validation_transformed = preprocessor.transform(splits.X_validation)
    X_test_transformed = preprocessor.transform(splits.X_test)
    return PreparedTrainingData(
        splits=splits,
        preprocessor=preprocessor,
        X_train_transformed=X_train_transformed,
        X_validation_transformed=X_validation_transformed,
        X_test_transformed=X_test_transformed,
    )


def _class_summary(target: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = target.value_counts().sort_index()
    percentages = target.value_counts(normalize=True).mul(100).sort_index()
    return {
        label: {"count": int(counts[label]), "percent": round(float(percentages[label]), 4)}
        for label in counts.index
    }


def _all_finite(matrix: Any) -> bool:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    return bool(np.isfinite(values).all())


def build_preparation_summary(prepared: PreparedTrainingData) -> dict[str, Any]:
    """Describe split balance and fitted preprocessing without exposing applicant rows."""

    splits = prepared.splits
    return {
        "random_state": DEFAULT_RANDOM_STATE,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": REFERENCE_COLUMNS + [TARGET_COLUMN],
        "split_rows": {
            "train": len(splits.X_train),
            "validation": len(splits.X_validation),
            "test": len(splits.X_test),
        },
        "class_distribution": {
            "train": _class_summary(splits.y_train),
            "validation": _class_summary(splits.y_validation),
            "test": _class_summary(splits.y_test),
        },
        "preprocessing": {
            "numeric_imputation": "median fitted on training data",
            "numeric_missing_indicators": True,
            "categorical_imputation": "most frequent fitted on training data",
            "categorical_encoding": "one-hot with infrequent-category grouping",
            "minimum_category_frequency": MIN_CATEGORY_FREQUENCY,
            "maximum_categories_per_feature": MAX_CATEGORIES_PER_FEATURE,
            "unknown_categories": "mapped to infrequent bucket when available",
            "output_features": len(prepared.preprocessor.get_feature_names_out()),
        },
        "transformed_values_all_finite": {
            "train": _all_finite(prepared.X_train_transformed),
            "validation": _all_finite(prepared.X_validation_transformed),
            "test": _all_finite(prepared.X_test_transformed),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/training_data.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/preparation_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_raw_data(args.input)
    labelled = normalize_for_analysis(raw).labelled
    prepared = prepare_training_data(labelled)
    summary = build_preparation_summary(prepared)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "Preparation complete: "
        f"{summary['split_rows']['train']:,} train, "
        f"{summary['split_rows']['validation']:,} validation, "
        f"{summary['split_rows']['test']:,} test rows; "
        f"{summary['preprocessing']['output_features']:,} encoded features"
    )


if __name__ == "__main__":
    main()
