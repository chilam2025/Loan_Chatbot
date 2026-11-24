"""Audit prefix-model errors using held-out training predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.compare_pincode_signal import represent_pincode
from abc_credit.modeling.feature_analysis import FEATURE_SETS, load_splits
from abc_credit.modeling.threshold_analysis import threshold_table


def amount_bands(values):
    """Separate missing and nonpositive values before grouping positive values."""
    values = pd.to_numeric(values, errors="coerce")
    bands = pd.Series("Missing", index=values.index, dtype="string")
    bands.loc[values.le(0)] = "Zero or negative"

    positive = values.gt(0)
    bands.loc[positive] = pd.qcut(
        values.loc[positive], q=4, duplicates="drop"
    ).astype("string")
    return bands


def ltv_bands(values):
    values = pd.to_numeric(values, errors="coerce")
    bands = pd.cut(
        values,
        bins=[0, 60, 80, 90, 100],
        labels=["(0, 60]", "(60, 80]", "(80, 90]", "(90, 100]"],
    ).astype("string")
    bands.loc[values.isna()] = "Missing"
    bands.loc[values.eq(0)] = "Zero"
    bands.loc[values.lt(0) | values.gt(100)] = "Outside 0–100"
    return bands


def summarize(bands, actual, predicted):
    rows = []
    for band in bands.unique():
        mask = bands.eq(band).to_numpy(dtype=bool)
        declines = int((mask & actual).sum())
        approvals = int((mask & ~actual).sum())
        caught = int((mask & actual & predicted).sum())
        false_flags = int((mask & ~actual & predicted).sum())

        rows.append({
            "band": band,
            "historical_declines": declines,
            "historical_approvals": approvals,
            "caught": caught,
            "missed": declines - caught,
            "false_flags": false_flags,
            "recall_pct": 100 * caught / declines if declines else np.nan,
            "false_flag_pct": (
                100 * false_flags / approvals if approvals else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values("missed", ascending=False)


def main():
    splits = load_splits()
    X = splits.X_train[FEATURE_SETS["without_gender"]].reset_index(drop=True)
    y = splits.y_train.eq("DECLINED").astype(int).reset_index(drop=True)
    inputs = represent_pincode(X, "prefix_3")

    scores = np.empty(len(y))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for fold, (fit, held_out) in enumerate(cv.split(inputs, y), start=1):
        print(f"Training fold {fold}/3", flush=True)
        model = make_model("vehicle_value")
        model.fit(inputs.iloc[fit], y.iloc[fit])
        positive_index = list(model.classes_).index(1)
        scores[held_out] = model.predict_proba(
            inputs.iloc[held_out]
        )[:, positive_index]

    table = threshold_table(y, scores)
    threshold = float(
        table.loc[table["declined_recall"].ge(0.40), "threshold"].max()
    )
    actual = y.to_numpy(dtype=bool)
    predicted = scores >= threshold
    ltv_groups = ltv_bands(X["LTV"])

    print(f"\nPooled diagnostic recall: {predicted[actual].mean():.2%}")
    print(f"Historical declines caught: {int((actual & predicted).sum())}")

    for field, bands in (
        ("Loan_Amount", amount_bands(X["Loan_Amount"])),
        ("LTV", ltv_groups),
        ("Net_salary", amount_bands(X["Net_salary"])),
    ):
        print(f"\n{field}")
        print(summarize(bands, actual, predicted).round(2).to_string(index=False))

    print("\nRanking within LTV bands")
    print("\nExtra flags needed in the 80–90% LTV band")

    mask = ltv_groups.eq("(80, 90]").to_numpy(dtype=bool)
    band_actual = actual[mask]
    band_scores = scores[mask]
    current_flags = band_scores >= threshold

    current_caught = int((band_actual & current_flags).sum())
    target_caught = current_caught + 100

    decline_scores = np.sort(band_scores[band_actual])[::-1]
    if target_caught <= len(decline_scores):
        diagnostic_threshold = decline_scores[target_caught - 1]
        new_flags = band_scores >= diagnostic_threshold
        added_flags = new_flags & ~current_flags

        extra_caught = int((added_flags & band_actual).sum())
        extra_false_flags = int((added_flags & ~band_actual).sum())

        print(f"Current declines caught: {current_caught}")
        print(f"Additional declines caught: {extra_caught}")
        print(f"Additional historical approvals flagged: {extra_false_flags}")
        print(
            "Additional false flags per extra decline caught: "
            f"{extra_false_flags / extra_caught:.2f}"
        )
    for band in ("(60, 80]", "(80, 90]", "(90, 100]"):
        mask = ltv_groups.eq(band).to_numpy(dtype=bool)
        band_labels = actual[mask]

        if np.unique(band_labels).size < 2:
            print(f"{band} | Skipped because both classes are needed")
            continue

        ap = average_precision_score(band_labels, scores[mask])
        decline_rate = band_labels.mean()

        print(
            f"{band} | Declines={band_labels.sum()}/{len(band_labels)} | "
            f"AP={ap:.4f} | Decline-rate baseline={decline_rate:.4f}"
        )


if __name__ == "__main__":
    main()