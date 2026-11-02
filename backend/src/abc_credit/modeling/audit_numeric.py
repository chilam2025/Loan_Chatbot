"""Investigate numeric anomalies using the existing Phase B training split."""

from pathlib import Path

import pandas as pd

from abc_credit.data_quality import load_raw_data, normalize_for_analysis
from abc_credit.preprocessing import FEATURE_COLUMNS, stratified_split


def main():
    raw = load_raw_data("data/raw/training_data.csv")
    labelled = normalize_for_analysis(raw).labelled

    # Match the current Phase B record-selection and splitting procedure.
    conflict = (
        labelled.groupby(FEATURE_COLUMNS, dropna=False)["Decision"]
        .transform("nunique")
        .gt(1)
    )
    usable = labelled.loc[~conflict].drop_duplicates(subset=FEATURE_COLUMNS)
    splits = stratified_split(usable)

    train = splits.X_train.copy()
    train["Decision"] = splits.y_train

    flags = {
        "age_above_60": train["Age"] > 60,
        "zero_income": train["Net_salary"] == 0,
        "zero_ltv": train["LTV"] == 0,
    }

    output = Path("reports/generated/numeric_audit")
    output.mkdir(parents=True, exist_ok=True)

    summary = []

    for name, mask in flags.items():
        for group, selected in [
            ("flagged", train.loc[mask]),
            ("other_nonmissing", train.loc[
                ~mask & train[
                    {
                        "age_above_60": "Age",
                        "zero_income": "Net_salary",
                        "zero_ltv": "LTV",
                    }[name]
                ].notna()
            ]),
        ]:
            summary.append({
                "issue": name,
                "group": group,
                "count": len(selected),
                "declined_percent": (
                    100 * selected["Decision"].eq("DECLINED").mean()
                    if len(selected) else None
                ),
            })

        # Aggregate results only; no applicant-level records are exported.
        employment = (
            train.loc[mask]
            .groupby("Employment_Type", dropna=False)
            .agg(
                count=("Decision", "size"),
                declined_percent=(
                    "Decision",
                    lambda values: 100 * values.eq("DECLINED").mean(),
                ),
            )
            .reset_index()
        )
        employment.to_csv(output / f"{name}_by_employment.csv", index=False)

    summary = pd.DataFrame(summary)
    summary.to_csv(output / "anomaly_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nAggregate reports saved to: {output}")


if __name__ == "__main__":
    main()