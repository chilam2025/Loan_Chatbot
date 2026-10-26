"""Reproducible Phase A exploratory analysis for the ABC Credit dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from abc_credit.data_quality import (
    CATEGORICAL_COLUMNS,
    EXPECTED_COLUMNS,
    NUMERIC_COLUMNS,
    load_raw_data,
    normalize_for_analysis,
)


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_summary(raw: pd.DataFrame, labelled: pd.DataFrame, unlabelled: pd.DataFrame) -> dict:
    """Build a machine-readable audit summary."""

    target_counts = labelled["Decision"].value_counts()
    target_percent = labelled["Decision"].value_counts(normalize=True).mul(100)
    numeric = labelled[NUMERIC_COLUMNS].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])

    return {
        "rows_raw": len(raw),
        "columns": int(raw.shape[1]),
        "expected_columns_present": list(raw.columns) == EXPECTED_COLUMNS,
        "labelled_rows": len(labelled),
        "excluded_unlabelled_rows": len(unlabelled),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_non_null_ids": int(raw.loc[raw["ID"].notna(), "ID"].duplicated().sum()),
        "target_counts": {key: int(value) for key, value in target_counts.items()},
        "target_percent": {key: round(float(value), 4) for key, value in target_percent.items()},
        "missing_counts_raw": {key: int(value) for key, value in raw.isna().sum().items()},
        "business_rule_flags_labelled": {
            "age_below_18": int((labelled["Age"] < 18).sum()),
            "age_above_60": int((labelled["Age"] > 60).sum()),
            "ltv_not_between_0_and_100": int(
                ((labelled["LTV"] < 0) | (labelled["LTV"] > 100)).sum()
            ),
            "ltv_equal_zero": int((labelled["LTV"] == 0).sum()),
            "loan_amount_non_positive": int((labelled["Loan_Amount"] <= 0).sum()),
            "net_salary_non_positive": int((labelled["Net_salary"] <= 0).sum()),
            "pincode_not_six_digits": int(
                (~labelled["Pincode"].fillna("").str.fullmatch(r"\d{6}")).sum()
            ),
        },
        "cardinality_labelled": {
            column: int(labelled[column].nunique(dropna=True)) for column in CATEGORICAL_COLUMNS
        },
        "numeric_summary": {
            column: {stat: _json_value(value) for stat, value in numeric[column].items()}
            for column in NUMERIC_COLUMNS
        },
    }


def _write_tables(labelled: pd.DataFrame, output: Path) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    missing = labelled.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = missing["missing_count"].div(len(labelled)).mul(100)
    missing.sort_values("missing_count", ascending=False).to_csv(tables / "missingness.csv")

    target = labelled["Decision"].value_counts().rename("count").to_frame()
    target["percent"] = target["count"].div(len(labelled)).mul(100)
    target.to_csv(tables / "target_distribution.csv")

    labelled[NUMERIC_COLUMNS].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.to_csv(
        tables / "numeric_summary.csv"
    )

    category_rows: list[dict[str, Any]] = []
    for column in CATEGORICAL_COLUMNS:
        grouped = (
            labelled.assign(category=labelled[column].fillna("<MISSING>"))
            .groupby("category", dropna=False)["Decision"]
            .agg(count="size", decline_rate=lambda values: (values == "DECLINED").mean())
            .reset_index()
        )
        grouped["decline_rate_percent"] = grouped.pop("decline_rate").mul(100)
        grouped.insert(0, "feature", column)
        category_rows.extend(grouped.to_dict(orient="records"))
    pd.DataFrame(category_rows).to_csv(tables / "categorical_decline_rates.csv", index=False)

    by_target = labelled.groupby("Decision")[NUMERIC_COLUMNS].agg(
        ["count", "mean", "median", "min", "max"]
    )
    by_target.to_csv(tables / "numeric_by_decision.csv")


def _write_figures(labelled: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    counts = labelled["Decision"].value_counts().reindex(["APPROVED", "DECLINED"])
    ax = counts.plot(kind="bar", color=["#2878B5", "#D9534F"], figsize=(7, 4))
    ax.set(title="Loan decision class distribution", xlabel="Decision", ylabel="Applications")
    ax.tick_params(axis="x", rotation=0)
    for patch in ax.patches:
        ax.annotate(
            f"{int(patch.get_height()):,}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(figures / "target_distribution.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, axis in zip(NUMERIC_COLUMNS, axes.flat, strict=True):
        sns.boxplot(data=labelled, x="Decision", y=column, ax=axis, showfliers=False)
        axis.set_title(f"{column} by decision")
        axis.set_xlabel("")
    fig.tight_layout()
    fig.savefig(figures / "numeric_by_decision.png", dpi=160)
    plt.close(fig)


def run(input_path: Path, output_dir: Path) -> dict:
    """Run Phase A analysis and return the generated summary."""

    raw = load_raw_data(input_path)
    prepared = normalize_for_analysis(raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(raw, prepared.labelled, prepared.unlabelled)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_tables(prepared.labelled, output_dir)
    _write_figures(prepared.labelled, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/training_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.input, args.output)
    print(
        f"Phase A complete: {summary['labelled_rows']:,} labelled rows; "
        f"artifacts written to {args.output}"
    )


if __name__ == "__main__":
    main()
