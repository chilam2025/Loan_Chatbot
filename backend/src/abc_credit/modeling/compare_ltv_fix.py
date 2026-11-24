"""Compare zero-LTV treatment using the existing development splits."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from abc_credit.modeling.feature_analysis import load_splits, make_pipeline
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/ltv_fix_comparison")


class CleanLTV(BaseEstimator, TransformerMixin):
    """Mark missing/zero LTV, then turn zero into missing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        invalid = result["LTV"].isna() | result["LTV"].eq(0)

        result["LTV_missing_or_zero"] = invalid.astype(float)
        result["LTV"] = result["LTV"].mask(result["LTV"].eq(0))

        return result


def build_model(apply_fix):
    pipeline = make_pipeline("without_gender")

    pipeline.set_params(
        classifier=RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=2,
        )
    )

    if apply_fix:
        pipeline.set_params(features=CleanLTV())

        # Preserve the existing transformations and explicitly include
        # the new indicator in the encoded model inputs.
        preprocessor = pipeline.named_steps["preprocessor"]
        pipeline.set_params(
            preprocessor=ColumnTransformer(
                transformers=[
                    *preprocessor.transformers,
                    (
                        "ltv_indicator",
                        "passthrough",
                        ["LTV_missing_or_zero"],
                    ),
                ],
                remainder="drop",
            )
        )

    return pipeline


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    results = []

    for name, apply_fix in [
        ("original_ltv", False),
        ("zero_ltv_imputed", True),
    ]:
        model = build_model(apply_fix)
        model.fit(splits.X_train, y_train)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(
            splits.X_validation
        )[:, positive_index]

        table = threshold_table(y_validation, probabilities)
        table["cost"] = (
            5 * table["historically_declined_predicted_approved"]
            + table["historically_approved_predicted_declined"]
        )

        threshold = float(
            table.sort_values(
                ["cost", "threshold"],
                ascending=[True, False],
            ).iloc[0]["threshold"]
        )

        metrics = evaluate(y_validation, probabilities, threshold)
        matrix = metrics.pop("confusion_matrix_approved_declined")

        results.append({
            "experiment": name,
            "threshold": threshold,
            **metrics,
            "false_declines": matrix[0][1],
            "missed_declines": matrix[1][0],
            "detected_declines": matrix[1][1],
            "assumed_cost_5_to_1": 5 * matrix[1][0] + matrix[0][1],
        })

        if apply_fix:
            valid_training_ltv = splits.X_train["LTV"].where(
                splits.X_train["LTV"].between(
                    0, 100, inclusive="right"
                )
            )
            expected_median = valid_training_ltv.median()

            numeric_imputer = (
                model.named_steps["preprocessor"]
                .named_transformers_["numeric"]
                .named_steps["imputer"]
            )
            numeric_columns = (
                model.named_steps["preprocessor"].transformers_[0][2]
            )
            fitted_median = numeric_imputer.statistics_[
                list(numeric_columns).index("LTV")
            ]

            np.testing.assert_allclose(
                fitted_median, expected_median
            )
            print("Training median used for LTV:", fitted_median)

        print(f"Completed: {name}", flush=True)

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT / "validation_comparison.csv", index=False)

    print("\nVALIDATION COMPARISON")
    print(comparison.round(5).to_string(index=False))
    print("Test partition was not evaluated.")
    print("Packaged model was not modified.")


if __name__ == "__main__":
    main()