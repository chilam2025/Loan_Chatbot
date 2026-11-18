"""Analyse categorical fields using training data only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from abc_credit.data_quality import (
    CATEGORICAL_COLUMNS,
    load_raw_data,
    normalize_for_analysis,
)
from abc_credit.preprocessing import FEATURE_COLUMNS, stratified_split

FIELDS = CATEGORICAL_COLUMNS


def wilson_interval(declined, count):
    """Return lower and upper bounds for a 95% Wilson interval."""
    z = 1.959963984540054
    proportion = declined / count
    denominator = 1 + z**2 / count

    centre = (proportion + z**2 / (2 * count)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / count
            + z**2 / (4 * count**2)
        )
        / denominator
    )

    return np.clip(centre - margin, 0, 1), np.clip(centre + margin, 0, 1)


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
    train["is_declined"] = splits.y_train.eq("DECLINED").astype(int)

    output = Path("reports/generated/categorical_audit")
    output.mkdir(parents=True, exist_ok=True)

    overall_rate = train["is_declined"].mean() * 100
    all_tables = []

    for field in FIELDS:
        table = (
            train.assign(category=train[field].fillna("<MISSING>"))
            .groupby("category", dropna=False)["is_declined"]
            .agg(count="size", declined_count="sum")
            .reset_index()
        )

        table["approved_count"] = table["count"] - table["declined_count"]
        table["declined_percent"] = (
            100 * table["declined_count"] / table["count"]
        )

        lower, upper = wilson_interval(
            table["declined_count"], table["count"]
        )
        table["ci95_lower_percent"] = lower * 100
        table["ci95_upper_percent"] = upper * 100
        table["sample_below_100"] = table["count"] < 100

        table = table.sort_values("declined_percent").reset_index(drop=True)
        table.insert(0, "feature", field)

        table.to_csv(output / f"{field}_summary.csv", index=False)
        all_tables.append(table)
        if field == "Pincode":
            print("\nPincode: table saved; plot skipped.")
            continue

        # Point estimate with confidence interval and category sample sizes.
        positions = np.arange(len(table))
        rates = table["declined_percent"].to_numpy()
        lower_bounds = table["ci95_lower_percent"].to_numpy()
        upper_bounds = table["ci95_upper_percent"].to_numpy()

        fig, axis = plt.subplots(
            figsize=(10, max(4, len(table) * 0.55))
        )

        axis.hlines(
            positions, lower_bounds, upper_bounds,
            color="steelblue", linewidth=2,
        )
        axis.scatter(rates, positions, color="steelblue", zorder=3)

        labels = [
            f"{row.category} (n={row.count:,})"
            for row in table.itertuples()
        ]
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)

        axis.axvline(
            overall_rate,
            color="darkred",
            linestyle="--",
            label=f"Overall training decline rate: {overall_rate:.2f}%",
        )
        axis.set_xlim(
            0,
            min(100, max(float(upper_bounds.max()), overall_rate) + 2),
        )
        axis.set_xlabel("Declined applications (%)")
        axis.set_title(f"{field}: decline rates and 95% Wilson intervals")
        axis.grid(axis="x", alpha=0.3)
        axis.legend()

        fig.tight_layout()
        fig.savefig(output / f"{field}_decline_rates.png", dpi=160)
        plt.close(fig)

        print(f"\n--- {field} ---")
        print(table.round(3).to_string(index=False))

    pd.concat(all_tables, ignore_index=True).to_csv(
        output / "all_categorical_summaries.csv",
        index=False,
    )

    print(f"\nOverall training decline rate: {overall_rate:.2f}%")
    print(f"Reports saved to: {output}")


if __name__ == "__main__":
    main()