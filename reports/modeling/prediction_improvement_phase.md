# Prediction improvement phase

## Outcome

No new predictive artifact was promoted. `catboost-minimal-v2` remains the selected
model because the tested candidates did not produce a material, stable improvement
against its actual configuration. This is a successful model-governance result: a
slightly different score was not presented as an accuracy breakthrough.

The internal test partition and the instructor's hidden holdout were not evaluated in
this phase.

## Reproducibility correction

Inspection of the fitted v2 artifact showed CatBoost 1.2.10 had resolved the previously
implicit `max_ctr_complexity` setting to `4`. The pipeline now sets that value explicitly
so future training cannot silently change when library defaults change.

An initial screen incorrectly constrained the comparison baseline to `1`, which made a
less-constrained candidate appear better. The promotion run detected the mismatch and
rejected the candidate. The corrected experiment compared every candidate with the
actual v2 setting.

## Corrected profile-grouped training results

| Candidate | Pooled average precision | Precision at 40% recall | Approved cases flagged | Result |
|---|---:|---:|---:|---|
| Actual v2 (`depth=4`, CTR complexity 4) | 0.173748 | 0.181394 | 7,198 | Retain |
| Higher CTR complexity | 0.173748 | 0.181394 | 7,198 | Identical; reject |
| Depth 5 with current interactions | 0.173703 | 0.181044 | 7,215 | Worse overall; reject |

The deeper candidate's fold AP differences against v2 were `-0.001162`, `+0.000537`
and `-0.001247`, so it was not consistently better.

## Existing v2 operating-point trade-off

The thresholds below were selected from training-fold predictions and then measured on
the already-used development validation partition. They are not independent final
estimates.

| Training recall target | Validation accuracy | Decline precision | Decline recall | Decline F1 | False decline flags |
|---|---:|---:|---:|---:|---:|
| 30% | 89.67% | 19.41% | 30.68% | 0.2377 | 1,088 |
| Current 40% | 86.64% | 16.88% | 39.34% | 0.2362 | 1,655 |
| 50% | 82.93% | 15.21% | 49.18% | 0.2323 | 2,342 |
| 60% | 78.36% | 13.49% | 57.61% | 0.2186 | 3,156 |

The training-selected F1 cutoff would yield 88.44% validation accuracy, 18.54%
decline precision, 35.36% decline recall and 0.2433 decline F1. It was not adopted
because the current 40% threshold deliberately prioritizes decline recall. Changing
that policy requires an agreed lender cost or recall objective, not an engineering
preference.

## Why a real improvement needs better data

The supplied target records historical approvals and declines, not repayment. The
current fields do not contain delinquency/default outcomes, bureau history, verified
income, existing monthly obligations, loan tenure, interest rate, instalment burden or
fraud signals. Earlier controlled experiments with income ratios, calibration, class
weights, seed averaging, LightGBM, XGBoost, neural networks, EBM, stacking and feature
ablation also failed to produce a stable material gain.

For a production credit-risk model, collect consented and policy-approved data with:

- a defined outcome such as 90+ days past due within 12 months;
- observation and outcome windows that prevent future-data leakage;
- verified income and existing monthly debt obligations;
- requested tenure, interest rate and expected EMI;
- appropriately sourced credit-bureau attributes;
- fraud and identity-verification outcomes;
- lender policy overrides separated from model outcomes;
- application timestamps for out-of-time validation;
- protected-group or approved fairness-monitoring fields kept out of inference where
  required but available for governed evaluation.

PAN, OTP, raw documents and language-model text must never become predictive features.
Pincode should remain under fairness review because it can proxy community and
socioeconomic characteristics.

## Next model gate

Do not replace v2 until a candidate:

1. improves profile-grouped training AP by at least 0.003;
2. improves in every paired fold;
3. does not reduce precision at the agreed recall target;
4. passes calibration, subgroup, latency and explanation tests;
5. passes a never-used out-of-time or hidden holdout set;
6. is reviewed and approved by the regulated lender's model-risk owner.
