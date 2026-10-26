# ABC Credit Loan Approval Chatbot

This repository implements a loan application chatbot whose conversation layer collects and validates applicant information while a separate predictive model produces the approval probability. The language model must never make or override the credit decision.

## Current status

Phase A is implemented as a reproducible exploratory data analysis and data-quality pipeline. It validates the supplied schema, normalizes values for analysis, separates unlabelled records, measures class imbalance and missingness, checks business-rule anomalies, and generates tables and figures for review.

## Quick start

The raw CSV is deliberately excluded from Git because it may contain applicant information. Place it at `data/raw/training_data.csv`, then run from the repository root:

```powershell
uv sync --project backend --extra dev
uv run --project backend phase-a
uv run --project backend pytest backend/tests
```

Generated artifacts are written to `reports/generated/`. The main conclusions and proposed preprocessing decisions are in `reports/phase_a_findings.md`.

## Repository layout

```text
backend/               Python services, data pipeline, and backend tests
frontend/              Chat client boundary and frontend implementation guide
data/raw/              Local sensitive source data (ignored by Git)
reports/               Findings and reproducible generated analysis artifacts
docs/                  Cross-cutting architecture and working agreements
```

The repository is organized as a monorepo. Backend dependencies stay isolated in
`backend/pyproject.toml`; the frontend can adopt its own toolchain without mixing
Python and JavaScript build files. Shared product contracts and architecture belong
in `docs/`, while generated analysis stays in `reports/generated/`.

## Design guardrails

- `Decision` is a historical approval outcome, not a loan-default label.
- `ID` is retained only for traceability and must not be used as a predictor.
- Preprocessing will be fitted on training folds only to prevent leakage.
- Unknown categories must be handled without crashing the inference pipeline.
- Model evaluation must emphasize precision, recall, F1 and the confusion matrix for `DECLINED`; raw accuracy is misleading because the data are highly imbalanced.
- Logs must mask or omit PII such as the full pincode and application identifier.

## Next phases

Phase B will establish leakage-safe train/validation splits, an interpretable logistic-regression baseline, a tree-based comparison model, probability calibration and business-informed threshold selection. Phase C will add the chatbot and deterministic model integration.

