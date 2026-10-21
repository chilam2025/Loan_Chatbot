# ABC Credit Loan Approval Chatbot

This repository implements a loan application chatbot whose conversation layer collects and validates applicant information while a separate predictive model produces the approval probability. The language model must never make or override the credit decision.

## Current status

Phase A is implemented as a reproducible exploratory data analysis and data-quality pipeline. It validates the supplied schema, normalizes values for analysis, separates unlabelled records, measures class imbalance and missingness, checks business-rule anomalies, and generates tables and figures for review.

## Quick start

The raw CSV is deliberately excluded from Git because it may contain applicant information. Place `training_data.csv` in the repository root (as supplied), then run:

```powershell
uv sync --extra dev
uv run phase-a
uv run pytest
```

Generated artifacts are written to `reports/generated/`. The main conclusions and proposed preprocessing decisions are in `reports/phase_a_findings.md`.

## Repository layout

```text
src/abc_credit/        Reusable data validation and Phase A analysis code
tests/                 Automated tests for validation and normalization
reports/               Phase A findings and generated analysis artifacts
docs/                  Architecture and team working agreements
training_data.csv      Local raw data (ignored by Git)
```

## Design guardrails

- `Decision` is a historical approval outcome, not a loan-default label.
- `ID` is retained only for traceability and must not be used as a predictor.
- Preprocessing will be fitted on training folds only to prevent leakage.
- Unknown categories must be handled without crashing the inference pipeline.
- Model evaluation must emphasize precision, recall, F1 and the confusion matrix for `DECLINED`; raw accuracy is misleading because the data are highly imbalanced.
- Logs must mask or omit PII such as the full pincode and application identifier.

## Next phases

Phase B will establish leakage-safe train/validation splits, an interpretable logistic-regression baseline, a tree-based comparison model, probability calibration and business-informed threshold selection. Phase C will add the chatbot and deterministic model integration.

