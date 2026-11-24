"""six-question model: the CatBoost pipeline behind a ten-field contract.

Applicants answer age, pincode, vehicle, loan amount and vehicle price, plus
whether they have an active loan. The backend derives the remaining inputs.
Fields the model does not use are filled with constants inside the pipeline,
so callers send only the contract fields below.
"""

import numpy as np
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from abc_credit.modeling.ablate_catboost_features import (
    PrepareAblation,
    make_model,
)
from abc_credit.modeling.compare_catboost import CATEGORICAL_FIELDS
from abc_credit.modeling.compare_pincode_signal import represent_pincode

MINIMAL_FIELDS = [
    "Age",
    "Loan_Amount",
    "LTV",
    "Pincode",
    "Final_Tier",
    "PAST_LOANS_ACTIVE",
    "Make_Code",
    "Model_Variant",
    "Product_Code",
    "Model_Description",
]

UNUSED_CATEGORICAL = ["Qualifications", "Employment_Type", "Resident_Type"]
UNUSED_NUMERIC = ["Net_salary"]
CONSTANT = "__CONSTANT__"


def add_unused_fields(frame):
    """Give the shared pipeline the columns it expects, without signal."""
    frame = frame.copy()
    for column in UNUSED_CATEGORICAL:
        frame[column] = CONSTANT
    for column in UNUSED_NUMERIC:
        frame[column] = np.nan
    return frame


def build_minimal_pipeline(seed=42):
    inner = make_model("vehicle_value")
    inner.set_params(classifier__depth=4, classifier__random_seed=seed)

    return Pipeline([
        ("unused_fields", FunctionTransformer(add_unused_fields)),
        (
            "pincode_prefix",
            FunctionTransformer(
                represent_pincode,
                kw_args={"variant": "prefix_3"},
            ),
        ),
        *inner.steps,
    ])


# Version 2: add coarser pincode prefixes as extra location features.
HIERARCHY_LENGTHS = (2, 4, 5)
HIERARCHY_COLUMNS = [f"pin{n}" for n in HIERARCHY_LENGTHS]
MISSING_PINCODE = "__MISSING__"


def add_pincode_hierarchy(frame):
    """Add 2-, 4- and 5-digit pincode prefixes before the 3-digit step."""
    frame = frame.copy()
    pincode = frame["Pincode"].astype("string").str.strip()
    valid = pincode.str.fullmatch(r"[0-9]{6}", na=False)
    for length, column in zip(
        HIERARCHY_LENGTHS, HIERARCHY_COLUMNS, strict=True
    ):
        frame[column] = (
            pincode.where(valid).str[:length].fillna(MISSING_PINCODE)
        )
    return frame


class PrepareWithPincodeHierarchy(PrepareAblation):
    """Shared preparation plus the pincode prefix columns."""

    def __init__(self):
        super().__init__(variant="vehicle_value")

    def transform(self, X):
        result = super().transform(X)
        for column in HIERARCHY_COLUMNS:
            result[column] = X[column].astype(str).to_numpy()
        return result


def build_minimal_v2_pipeline(seed=42):
    return Pipeline([
        ("unused_fields", FunctionTransformer(add_unused_fields)),
        ("pincode_hierarchy", FunctionTransformer(add_pincode_hierarchy)),
        (
            "pincode_prefix",
            FunctionTransformer(
                represent_pincode,
                kw_args={"variant": "prefix_3"},
            ),
        ),
        ("prepare", PrepareWithPincodeHierarchy()),
        (
            "classifier",
            CatBoostClassifier(
                iterations=600,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=10,
                # CatBoost 1.2.10 resolved its implicit CPU default to 4 for
                # this fitted pipeline. Keep it explicit for reproducibility.
                max_ctr_complexity=4,
                loss_function="Logloss",
                cat_features=[*CATEGORICAL_FIELDS, *HIERARCHY_COLUMNS],
                random_seed=seed,
                thread_count=2,
                verbose=False,
                allow_writing_files=False,
            ),
        ),
    ])
