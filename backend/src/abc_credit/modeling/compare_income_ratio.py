"""Compare the prefix model with and without a loan-to-income ratio."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from abc_credit.modeling.ablate_catboost_features import PrepareAblation, make_model
from abc_credit.modeling.compare_pincode_signal import represent_pincode
from abc_credit.modeling.feature_analysis import FEATURE_SETS, load_splits
from abc_credit.modeling.threshold_analysis import threshold_table


class PrepareIncomeRatio(PrepareAblation):
    """Add loan amount relative to positive monthly income."""

    def transform(self, X):
        prepared = super().transform(X)
        loan = pd.to_numeric(X["Loan_Amount"], errors="coerce").astype(float)
        income = pd.to_numeric(X["Net_salary"], errors="coerce").astype(float)

        ratio = loan / income.where(income.gt(0))
        prepared["loan_to_monthly_income"] = ratio.where(np.isfinite(ratio))
        return prepared


def report(name, labels, scores, min_caught):
    table = threshold_table(labels, scores)
    selected = (
        table.loc[table["correctly_detected_declines"].ge(min_caught)]
        .sort_values("threshold", ascending=False)
        .iloc[0]
    )

    caught = int(selected["correctly_detected_declines"])
    false_flags = int(selected["historically_approved_predicted_declined"])
    ap = average_precision_score(labels, scores)

    print(
        f"{name} | AP={ap:.4f} | "
        f"Declines caught={caught} | Historical approvals flagged={false_flags}"
    )


def main():
    splits = load_splits()
    X = splits.X_train[FEATURE_SETS["without_gender"]].reset_index(drop=True)
    y = splits.y_train.eq("DECLINED").astype(int).reset_index(drop=True)
    inputs = represent_pincode(X, "prefix_3")

    ltv = pd.to_numeric(X["LTV"], errors="coerce")
    in_band = ltv.between(80, 90, inclusive="right").to_numpy()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(cv.split(inputs, y))

    for variant in ("current", "with_income_ratio"):
        scores = np.empty(len(y))

        for fold, (fit, held_out) in enumerate(folds, start=1):
            print(f"{variant}: training fold {fold}/3", flush=True)
            model = make_model("vehicle_value")

            if variant == "with_income_ratio":
                model.set_params(prepare=PrepareIncomeRatio(variant="vehicle_value"))

            model.fit(inputs.iloc[fit], y.iloc[fit])
            positive_index = list(model.classes_).index(1)
            scores[held_out] = model.predict_proba(
                inputs.iloc[held_out]
            )[:, positive_index]

        print(f"\n{variant}: pooled held-out training predictions")
        report("Overall", y, scores, min_caught=1595)
        report("80–90% LTV band", y[in_band], scores[in_band], min_caught=822)
        print()


if __name__ == "__main__":
    main()