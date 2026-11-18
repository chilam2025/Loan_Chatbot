"""Audit CatBoost errors using training-only outer-fold predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.nested_model_comparison import make_catboost
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/catboost_error_audit")

GROUP_FIELDS = [
    "Employment_Type",
    "Product_Code",
    "Make_Code",
    "Final_Tier",
    "ltv_status",
    "income_status",
    "pincode_coverage",
    "loan_amount_band",
]


def choose_recall_threshold(y, probabilities):
    table = threshold_table(y, probabilities)

    return float(
        table.loc[table["declined_recall"] >= 0.40]
        .sort_values("threshold", ascending=False)
        .iloc[0]["threshold"]
    )


def summarise(frame, group_field):
    rows = []

    for category, group in frame.groupby(
        group_field, dropna=False, observed=True
    ):
        actual = group["actual_declined"].to_numpy()
        predicted = group["predicted_declined"].to_numpy()

        tp = int(((actual == 1) & predicted).sum())
        fp = int(((actual == 0) & predicted).sum())
        fn = int(((actual == 1) & ~predicted).sum())
        tn = int(((actual == 0) & ~predicted).sum())

        rows.append({
            "field": group_field,
            "category": str(category),
            "count": len(group),
            "actual_declines": tp + fn,
            "historical_decline_rate": (tp + fn) / len(group),
            "mean_predicted_score": group["score"].mean(),
            "detected_declines": tp,
            "missed_declines": fn,
            "false_declines": fp,
            "correct_approvals": tn,
            "declined_precision": (
                tp / (tp + fp) if tp + fp else np.nan
            ),
            "declined_recall": (
                tp / (tp + fn) if tp + fn else np.nan
            ),
            "false_decline_rate": (
                fp / (fp + tn) if fp + tn else np.nan
            ),
            "sample_below_100": len(group) < 100,
            "fewer_than_20_actual_declines": tp + fn < 20,
        })

    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    X = splits.X_train
    y = splits.y_train.eq("DECLINED").astype(int)

    folds = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    evaluation_frames = []
    fold_rows = []

    for fold, (development_positions, evaluation_positions) in enumerate(
        folds.split(X, y), start=1
    ):
        X_development = X.iloc[development_positions]
        y_development = y.iloc[development_positions]
        X_evaluation = X.iloc[evaluation_positions]
        y_evaluation = y.iloc[evaluation_positions]

        X_fit, X_threshold, y_fit, y_threshold = train_test_split(
            X_development,
            y_development,
            test_size=0.20,
            stratify=y_development,
            random_state=42 + fold,
        )

        assert set(X_fit.index).isdisjoint(X_threshold.index)
        assert set(X_fit.index).isdisjoint(X_evaluation.index)
        assert set(X_threshold.index).isdisjoint(X_evaluation.index)

        model = make_catboost()
        model.set_params(
            classifier__depth=4,
            classifier__l2_leaf_reg=10,
            classifier__class_weights=[1.0, 1.0],
        )

        print(f"Training audit fold {fold}/3...", flush=True)
        model.fit(X_fit, y_fit)

        positive_index = list(model.classes_).index(1)

        threshold_scores = model.predict_proba(
            X_threshold
        )[:, positive_index]

        threshold = choose_recall_threshold(
            y_threshold, threshold_scores
        )

        scores = model.predict_proba(
            X_evaluation
        )[:, positive_index]

        audited = X_evaluation[
            [
                "Employment_Type",
                "Product_Code",
                "Make_Code",
                "Final_Tier",
            ]
        ].copy()

        for column in audited.columns:
            audited[column] = (
                audited[column].astype("string").fillna("<MISSING>")
            )

        audited["actual_declined"] = y_evaluation.to_numpy()
        audited["predicted_declined"] = scores >= threshold
        audited["score"] = scores

        audited["ltv_status"] = np.select(
            [
                X_evaluation["LTV"].isna(),
                X_evaluation["LTV"].eq(0),
            ],
            ["missing", "zero"],
            default="recorded_nonzero",
        )

        audited["income_status"] = np.select(
            [
                X_evaluation["Net_salary"].isna(),
                X_evaluation["Net_salary"].eq(0),
            ],
            ["missing", "zero"],
            default="recorded_nonzero",
        )

        # Coverage is determined using fitting records only.
        counts = X_fit["Pincode"].value_counts()
        code_frequency = X_evaluation["Pincode"].map(counts).fillna(0)

        audited["pincode_coverage"] = np.select(
            [
                X_evaluation["Pincode"].isna(),
                code_frequency.eq(0),
                code_frequency.lt(10),
            ],
            ["missing", "unseen_in_fit", "seen_1_to_9"],
            default="seen_10_or_more",
        )

        # Learn amount-band boundaries from fitting data only.
        boundaries = (
            X_fit["Loan_Amount"]
            .dropna()
            .quantile([0.20, 0.40, 0.60, 0.80])
            .to_numpy()
        )
        edges = np.r_[-np.inf, np.unique(boundaries), np.inf]
        labels = [
            f"fit_quantile_band_{index + 1}"
            for index in range(len(edges) - 1)
        ]

        audited["loan_amount_band"] = (
            pd.cut(
                X_evaluation["Loan_Amount"],
                bins=edges,
                labels=labels,
                include_lowest=True,
            )
            .astype("string")
            .fillna("<MISSING>")
        )

        fold_rows.append({
            "fold": fold,
            "threshold": threshold,
            "evaluation_records": len(audited),
            "actual_declines": int(audited["actual_declined"].sum()),
            "observed_recall": (
                audited.loc[
                    audited["actual_declined"].eq(1),
                    "predicted_declined",
                ].mean()
            ),
            **{
                f"loan_amount_boundary_{index + 1}": float(value)
                for index, value in enumerate(boundaries)
            },
        })

        evaluation_frames.append(audited)
        print(f"Completed audit fold {fold}", flush=True)

    # Row-level results stay in memory; only aggregate tables are saved.
    combined = pd.concat(evaluation_frames, ignore_index=True)

    tables = []

    for field in GROUP_FIELDS:
        table = summarise(combined, field)
        table.to_csv(OUTPUT / f"{field}_errors.csv", index=False)
        tables.append(table)

    all_groups = pd.concat(tables, ignore_index=True)
    all_groups.to_csv(OUTPUT / "all_group_errors.csv", index=False)

    pd.DataFrame(fold_rows).to_csv(
        OUTPUT / "fold_summary.csv", index=False
    )

    print("\nLTV STATUS")
    print(summarise(combined, "ltv_status").round(4).to_string(index=False))

    print("\nPINCODE COVERAGE")
    print(
        summarise(combined, "pincode_coverage")
        .round(4)
        .to_string(index=False)
    )

    print(f"\nAggregate results saved to: {OUTPUT}")
    print("No applicant-level predictions were exported.")
    print("Original validation and test partitions were not evaluated.")


if __name__ == "__main__":
    main()