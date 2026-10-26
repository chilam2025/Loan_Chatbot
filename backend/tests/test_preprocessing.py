import numpy as np
import pandas as pd

from abc_credit.data_quality import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from abc_credit.preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    prepare_training_data,
    stratified_split,
)


def labelled_frame(rows: int = 100) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "ID": [str(index) for index in range(rows)],
        "Decision": ["DECLINED" if index % 5 == 0 else "APPROVED" for index in range(rows)],
    }
    for column in NUMERIC_COLUMNS:
        data[column] = [float(index + 1) for index in range(rows)]
    for column in CATEGORICAL_COLUMNS:
        data[column] = [f"{column}_{index % 3}" for index in range(rows)]
    return pd.DataFrame(data)


def test_stratified_split_is_disjoint_and_preserves_class_balance() -> None:
    splits = stratified_split(labelled_frame())

    assert (len(splits.X_train), len(splits.X_validation), len(splits.X_test)) == (70, 15, 15)
    assert set(splits.X_train.index).isdisjoint(splits.X_validation.index)
    assert set(splits.X_train.index).isdisjoint(splits.X_test.index)
    assert set(splits.X_validation.index).isdisjoint(splits.X_test.index)
    assert (splits.y_train == "DECLINED").mean() == 0.2
    assert (splits.y_validation == "DECLINED").mean() == 0.2
    assert (splits.y_test == "DECLINED").mean() == 0.2
    assert list(splits.X_train.columns) == FEATURE_COLUMNS


def test_imputers_are_fitted_only_from_supplied_training_rows() -> None:
    train = labelled_frame(6)[FEATURE_COLUMNS].copy()
    train["Age"] = [10.0, 20.0, np.nan, 30.0, 40.0, 50.0]
    train["Gender"] = ["Female", "Female", pd.NA, "Male", "Female", "Male"]
    preprocessor = build_preprocessor().fit(train)

    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    categorical_imputer = preprocessor.named_transformers_["categorical"].named_steps["imputer"]
    assert numeric_imputer.statistics_[NUMERIC_COLUMNS.index("Age")] == 30.0
    assert categorical_imputer.statistics_[CATEGORICAL_COLUMNS.index("Gender")] == "Female"


def test_unknown_categories_and_missing_values_transform_without_nan() -> None:
    train = labelled_frame(20)[FEATURE_COLUMNS].copy()
    validation = train.iloc[[0]].copy()
    validation["Gender"] = "UNSEEN_CATEGORY"
    validation["Age"] = np.nan
    preprocessor = build_preprocessor()

    train_transformed = preprocessor.fit_transform(train)
    validation_transformed = preprocessor.transform(validation)

    assert train_transformed.shape[1] == validation_transformed.shape[1]
    assert np.isfinite(validation_transformed).all()


def test_end_to_end_preparation_excludes_reference_and_target_columns() -> None:
    prepared = prepare_training_data(labelled_frame())

    assert "ID" not in prepared.splits.X_train
    assert "Decision" not in prepared.splits.X_train
    assert prepared.X_train_transformed.shape[0] == 70
