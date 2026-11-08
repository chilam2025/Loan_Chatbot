"""Compare Random Forest and histogram boosting on validation data."""

from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from abc_credit.modeling.compare_ltv_fix import build_model
from abc_credit.modeling.feature_analysis import load_splits
from abc_credit.modeling.phase_b import evaluate
from abc_credit.modeling.threshold_analysis import threshold_table

OUTPUT = Path("reports/generated/boosting_comparison")


def make_candidate(family, apply_ltv_fix):
    pipeline = build_model(apply_ltv_fix)

    if family == "hist_gradient_boosting":
        pipeline.set_params(
            classifier=HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1.0,
                categorical_features=None,
                early_stopping=False,
                random_state=42,
            ),
            # Histogram boosting requires dense input.
            preprocessor__categorical__encoder__sparse_output=False,
            preprocessor__sparse_threshold=0,
        )

    return pipeline


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()

    y_train = splits.y_train.eq("DECLINED").astype(int)
    y_validation = splits.y_validation.eq("DECLINED").astype(int)

    summaries = []
    recall_results = []

    for family in ["random_forest", "hist_gradient_boosting"]:
        for apply_fix in [False, True]:
            treatment = "imputed_ltv" if apply_fix else "original_ltv"
            name = f"{family}_{treatment}"
            model = make_candidate(family, apply_fix)

            print(f"Training: {name}", flush=True)

            # Limit numerical-library threads on the laptop.
            with threadpool_limits(limits=2):
                start = perf_counter()
                model.fit(splits.X_train, y_train)
                training_seconds = perf_counter() - start

                positive_index = list(model.classes_).index(1)

                start = perf_counter()
                probabilities = model.predict_proba(
                    splits.X_validation
                )[:, positive_index]
                prediction_seconds = perf_counter() - start

            table = threshold_table(y_validation, probabilities)
            table["assumed_cost_5_to_1"] = (
                5 * table["historically_declined_predicted_approved"]
                + table["historically_approved_predicted_declined"]
            )

            best = table.sort_values(
                ["assumed_cost_5_to_1", "threshold"],
                ascending=[True, False],
            ).iloc[0]

            threshold = float(best["threshold"])
            metrics = evaluate(
                y_validation, probabilities, threshold
            )
            matrix = metrics.pop(
                "confusion_matrix_approved_declined"
            )

            summaries.append({
                "model": name,
                "threshold": threshold,
                **metrics,
                "false_declines": matrix[0][1],
                "missed_declines": matrix[1][0],
                "detected_declines": matrix[1][1],
                "assumed_cost_5_to_1": (
                    5 * matrix[1][0] + matrix[0][1]
                ),
                "training_seconds": training_seconds,
                "validation_batch_prediction_seconds": prediction_seconds,
            })

            for target in [0.20, 0.40, 0.60]:
                selected = (
                    table.loc[table["declined_recall"] >= target]
                    .sort_values("threshold", ascending=False)
                    .iloc[0]
                )

                recall_results.append({
                    "model": name,
                    "target_recall": target,
                    "actual_recall": float(
                        selected["declined_recall"]
                    ),
                    "threshold": float(selected["threshold"]),
                    "declined_precision": float(
                        selected["declined_precision"]
                    ),
                    "declined_f1": float(selected["declined_f1"]),
                    "detected_declines": int(
                        selected["correctly_detected_declines"]
                    ),
                    "false_declines": int(
                        selected[
                            "historically_approved_predicted_declined"
                        ]
                    ),
                    "missed_declines": int(
                        selected[
                            "historically_declined_predicted_approved"
                        ]
                    ),
                })

            # Save partial results after each model completes.
            pd.DataFrame(summaries).to_csv(
                OUTPUT / "validation_comparison.csv",
                index=False,
            )
            pd.DataFrame(recall_results).to_csv(
                OUTPUT / "matched_recall_comparison.csv",
                index=False,
            )

            print(f"Completed: {name}", flush=True)

    print("\nVALIDATION COMPARISON")
    print(pd.DataFrame(summaries).round(5).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT}")
    print("Test partition was not evaluated.")
    print("Existing packaged model was not modified.")


if __name__ == "__main__":
    main()