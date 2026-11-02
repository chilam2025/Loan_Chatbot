"""Plot numeric distributions using the existing Phase B training split."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from abc_credit.data_quality import load_raw_data, normalize_for_analysis
from abc_credit.preprocessing import FEATURE_COLUMNS, NUMERIC_COLUMNS, stratified_split


def main():
    labelled = normalize_for_analysis(
        load_raw_data("data/raw/training_data.csv")
    ).labelled

    # Match the current Phase B record-selection procedure.
    conflict = (
        labelled.groupby(FEATURE_COLUMNS, dropna=False)["Decision"]
        .transform("nunique")
        .gt(1)
    )
    usable = labelled.loc[~conflict].drop_duplicates(subset=FEATURE_COLUMNS)
    splits = stratified_split(usable)

    train = splits.X_train.copy()
    train["Decision"] = splits.y_train

    output = Path("reports/generated/numeric_audit")
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for column, axis in zip(NUMERIC_COLUMNS, axes.flat, strict=True):
        sns.histplot(
            data=train,
            x=column,
            hue="Decision",
            hue_order=["APPROVED", "DECLINED"],
            stat="density",
            common_norm=False,
            common_bins=True,
            bins=40,
            element="step",
            fill=False,
            ax=axis,
        )
        axis.set_title(f"{column}: training distributions")

    fig.tight_layout()
    fig.savefig(output / "numeric_distributions.png", dpi=160)
    plt.close(fig)

    # Log view reveals positive-income patterns despite extreme values.
    positive = train.loc[train["Net_salary"] > 0].copy()
    positive["log10_monthly_income"] = np.log10(positive["Net_salary"])

    fig, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=positive,
        x="log10_monthly_income",
        hue="Decision",
        stat="density",
        common_norm=False,
        common_bins=True,
        bins=40,
        element="step",
        fill=False,
        ax=axis,
    )
    axis.set_title("Positive monthly income: logarithmic view")
    axis.set_xlabel("log10(monthly income in INR)")
    axis.set_xlim(
    positive["log10_monthly_income"].min() - 0.1,
    positive["log10_monthly_income"].max() + 0.1,
    )
    fig.tight_layout()
    fig.savefig(output / "positive_income_log_distribution.png", dpi=160)
    plt.close(fig)

    print("Training records:", len(train))
    print("Zero-income records:", train["Net_salary"].eq(0).sum())
    print("Missing-income records:", train["Net_salary"].isna().sum())
    print("Plots saved to:", output)


if __name__ == "__main__":
    main()