"""Training-data audits and validation-only feature experiments."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from abc_credit.data_quality import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    load_raw_data,
    normalize_for_analysis,
)
from abc_credit.modeling.phase_b import choose_threshold, evaluate
from abc_credit.preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    stratified_split,
)

OUTPUT = Path("reports/generated/feature_selection")

ENGINEERED = [
    "zero_income",
    "income_missing",
    "ltv_zero_or_missing",
    "loan_to_monthly_income",
    "log_monthly_income",
]

# A candidate to test, not a final recommended question set.
REDUCED_FIELDS = [
    "Age",
    "Loan_Amount",
    "LTV",
    "Net_salary",
    "Employment_Type",
    "Product_Code",
    "PAST_LOANS_ACTIVE",
    "Final_Tier",
]
FEATURE_SETS = {
    "baseline": FEATURE_COLUMNS.copy(),
    "without_gender": [
        c for c in FEATURE_COLUMNS
        if c != "Gender"
    ],
    "without_low_contribution": [
        c for c in FEATURE_COLUMNS
        if c not in {
            "Gender",
            "Resident_Type",
            "Qualifications",
            "Net_salary",
            "Employment_Type",
        }
    ],
    "less_vehicle_redundancy": [
        c for c in FEATURE_COLUMNS
        if c not in {"Make_Code", "Model_Variant", "Product_Code"}
    ],
    "without_pincode": [
        c for c in FEATURE_COLUMNS
        if c != "Pincode"
    ],
}


class FeatureExperiment(BaseEstimator, TransformerMixin):
    """Deterministic transformations; no statistics learned here."""

    def __init__(self, mode="baseline"):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()

        if self.mode in {"zero_ltv", "engineered"}:
            result["LTV"] = result["LTV"].mask(result["LTV"].eq(0))

        if self.mode == "engineered":
            income = result["Net_salary"]
            positive_income = income.where(income > 0)

            result["zero_income"] = income.eq(0).astype(float)
            result["income_missing"] = income.isna().astype(float)
            result["ltv_zero_or_missing"] = result["LTV"].isna().astype(float)

            # Undefined for zero/missing income: retain NaN for train-only imputation.
            result["loan_to_monthly_income"] = (
                result["Loan_Amount"] / positive_income
            )
            result["log_monthly_income"] = np.log1p(income.where(income >= 0))

        return result


def load_splits():
    labelled = normalize_for_analysis(
        load_raw_data("data/raw/training_data.csv")
    ).labelled

    # Match the current Phase B procedure exactly.
    conflict = (
        labelled.groupby(FEATURE_COLUMNS, dropna=False)["Decision"]
        .transform("nunique")
        .gt(1)
    )
    usable = labelled.loc[~conflict].drop_duplicates(subset=FEATURE_COLUMNS)
    return stratified_split(usable)


def category_table(X, y, field):
    frame = pd.DataFrame({
        "category": X[field].fillna("<MISSING>"),
        "declined": y,
    })

    table = (
        frame.groupby("category", dropna=False)["declined"]
        .agg(count="size", declined_count="sum")
        .reset_index()
    )
    table["declined_percent"] = (
        100 * table["declined_count"] / table["count"]
    )
    return table.sort_values("count", ascending=False)


def audit_categories(X, y):
    """Vehicle summaries, possible label inconsistencies and pincode sparsity."""
    for field in ["Make_Code", "Model_Description", "Model_Variant"]:
        category_table(X, y, field).to_csv(
            OUTPUT / f"{field}_summary.csv", index=False
        )

    # Report possible aliases; do not automatically merge them.
    aliases = []

    for field in CATEGORICAL_COLUMNS:
        if field == "Pincode":
            continue

        original = X[field].dropna().astype(str).drop_duplicates()
        labels = pd.DataFrame({"original": original})
        labels["canonical_candidate"] = (
            labels["original"]
            .str.strip()
            .str.upper()
            .str.replace(r"\s+", " ", regex=True)
        )

        for candidate, group in labels.groupby("canonical_candidate"):
            if len(group) > 1:
                aliases.append({
                    "field": field,
                    "canonical_candidate": candidate,
                    "observed_labels": " | ".join(group["original"]),
                })

    pd.DataFrame(
        aliases,
        columns=["field", "canonical_candidate", "observed_labels"],
    ).to_csv(OUTPUT / "possible_category_aliases.csv", index=False)

    # Aggregate postal-code sparsity without exporting individual pincodes.
    counts = X["Pincode"].dropna().value_counts()
    summary = pd.DataFrame([{
        "distinct_training_pincodes": len(counts),
        "pincodes_seen_once": int(counts.eq(1).sum()),
        "pincodes_seen_fewer_than_10_times": int(counts.lt(10).sum()),
        "percent_training_rows_in_codes_below_10": (
            100 * counts[counts.lt(10)].sum() / len(X)
        ),
        "missing_training_pincodes": int(X["Pincode"].isna().sum()),
    }])
    summary.to_csv(OUTPUT / "pincode_sparsity.csv", index=False)


def audit_missingness(X, y):
    rows = []

    for field in FEATURE_COLUMNS:
        for missing in [False, True]:
            mask = X[field].isna().eq(missing)
            count = int(mask.sum())
            rows.append({
                "field": field,
                "group": "missing" if missing else "present",
                "count": count,
                "declined_count": int(y.loc[mask].sum()),
                "declined_percent": (
                    100 * y.loc[mask].mean() if count else np.nan
                ),
            })

    pd.DataFrame(rows).to_csv(
        OUTPUT / "missingness_by_decision.csv", index=False
    )


def audit_redundancy(X):
    X[NUMERIC_COLUMNS].corr(method="spearman").to_csv(
        OUTPUT / "numeric_spearman_correlations.csv"
    )

    # Example: does each model description identify a single vehicle make?
    pairs = [
        ("Model_Description", "Make_Code"),
        ("Model_Description", "Model_Variant"),
        ("Model_Description", "Product_Code"),
        ("Make_Code", "Product_Code"),
        ("Model_Variant", "Product_Code"),
    ]
    rows = []

    for source, destination in pairs:
        complete = X[[source, destination]].dropna()
        unique_destinations = complete.groupby(source)[destination].nunique()
        deterministic_sources = unique_destinations[
            unique_destinations.eq(1)
        ].index

        rows.append({
            "source": source,
            "destination": destination,
            "complete_rows": len(complete),
            "source_categories": len(unique_destinations),
            "source_categories_mapping_to_one_destination": int(
                unique_destinations.eq(1).sum()
            ),
            "percent_complete_rows_with_unambiguous_mapping": (
                100 * complete[source].isin(deterministic_sources).mean()
                if len(complete) else np.nan
            ),
        })

    pd.DataFrame(rows).to_csv(
        OUTPUT / "vehicle_redundancy.csv", index=False
    )


def make_pipeline(mode):
    if mode in FEATURE_SETS:
        selected = FEATURE_SETS[mode]
    elif mode == "reduced":
        selected = REDUCED_FIELDS
    else:
        selected = FEATURE_COLUMNS

    numeric = [c for c in NUMERIC_COLUMNS if c in selected]
    categorical = [c for c in CATEGORICAL_COLUMNS if c in selected]
    if mode == "engineered":
        numeric += ENGINEERED

    # Reuse the existing categorical recipe without changing its source file.
    existing = build_preprocessor()
    categorical_recipe = {
        name: transformer
        for name, transformer, _ in existing.transformers
    }["categorical"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median", add_indicator=True),
                    ),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            ),
            ("categorical", categorical_recipe, categorical),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("features", FeatureExperiment(mode=mode)),
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                solver="liblinear",
                max_iter=1500,
                random_state=42,
            ),
        ),
    ])


def run_experiments(splits):
    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    rows = []
    baseline = None

    for mode in FEATURE_SETS:
        pipeline = make_pipeline(mode)

        start = perf_counter()
        pipeline.fit(splits.X_train, y_train)
        training_seconds = perf_counter() - start

        start = perf_counter()
        probabilities = pipeline.predict_proba(splits.X_validation)[:, 1]
        prediction_seconds = perf_counter() - start

        # Keep the existing illustrative 5:1 assumption for comparability.
        threshold = choose_threshold(
            y_validation,
            probabilities,
            false_approval_cost=5.0,
            false_decline_cost=1.0,
        )
        metrics = evaluate(y_validation, probabilities, threshold)
        matrix = metrics.pop("confusion_matrix_approved_declined")

        rows.append({
            "experiment": mode,
            "threshold": threshold,
            **metrics,
            "historically_approved_predicted_declined": matrix[0][1],
            "historically_declined_predicted_approved": matrix[1][0],
            "validation_cost_5_to_1": 5 * matrix[1][0] + matrix[0][1],
            "training_seconds": training_seconds,
            "validation_batch_prediction_seconds": prediction_seconds,
        })

        if mode == "baseline":
            baseline = pipeline

        print(f"Completed: {mode}", flush=True)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT / "validation_experiments.csv", index=False)
    print("\nVALIDATION COMPARISON")
    print(comparison.round(4).to_string(index=False))

    # Shuffle original fields before preprocessing.
    # Use the same reproducible stratified subset for all importance measurements.
    from sklearn.model_selection import train_test_split

    X_importance = splits.X_validation
    y_importance = y_validation

    if len(X_importance) > 3000:
        X_importance, _, y_importance, _ = train_test_split(
            X_importance,
            y_importance,
            train_size=3000,
            stratify=y_importance,
            random_state=42,
        )

    importance = permutation_importance(
        baseline,
        X_importance,
        y_importance,
        scoring="average_precision",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )

    pd.DataFrame({
        "field": FEATURE_COLUMNS,
        "mean_pr_auc_decrease": importance.importances_mean,
        "std_across_shuffles": importance.importances_std,
    }).sort_values(
        "mean_pr_auc_decrease", ascending=False
    ).to_csv(OUTPUT / "baseline_permutation_importance.csv", index=False)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    y_train = splits.y_train.eq("DECLINED").astype(int)

    audit_categories(splits.X_train, y_train)
    audit_missingness(splits.X_train, y_train)
    audit_redundancy(splits.X_train)

    print("Training-data audits complete.", flush=True)
    run_experiments(splits)
    print(f"\nResults saved to: {OUTPUT}")
    print("No test-set evaluation was performed.")


if __name__ == "__main__":
    main()