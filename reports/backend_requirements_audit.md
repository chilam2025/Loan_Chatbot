# Project Brief Completion Audit

## Scope and conclusion

This audit compares the supplied ABC Credit project brief with the current repository.
The predictive decision remains separate from the conversation layer: the packaged
CatBoost model produces the score and the backend applies its locked threshold. The
backend now covers the requirements that can be completed with the supplied training
data and local model artifact. Final hidden-test predictions remain dependent on the
separate evaluator dataset, which was not included in the brief archive.

## Backend requirement status

| Brief requirement | Status | Implementation evidence |
|---|---|---|
| Collect a minimal set of applicant inputs | Complete | Six validated questions provide age, pincode, vehicle, vehicle price, loan amount, and active-loan status. LTV and catalogue fields are derived by the backend. |
| Dynamic question sequence | Complete | The next-question endpoint derives progress from persisted answers and returns the first unanswered localized question. |
| Predictive model owns Approve or Decline | Complete | Submission builds the ten-field model contract, obtains the decline probability, and applies the packaged threshold. Client-supplied outcomes are ignored. |
| Real-time response within Y seconds | Complete with measured target | Y is 2 seconds. Decision latency now covers prediction plus explanation. The final 30-run local benchmark measured 77.133 ms median and 82.885 ms p95, passing the 2,000 ms target. |
| Log every input and answer | Complete | Raw validated answers are persisted. Each accepted answer batch also creates an append-only audit event containing every supplied value, with residential pincode masked. |
| Log feature values and model score | Complete | Decision audit events contain the ten model fields, model version, provider, threshold, decline probability, decision, and latency. Pincode is reduced to its first three digits. |
| Mask PII | Complete for the academic prototype | Logs mask pincode, store only the mobile number's last four digits, hash OTPs, and do not store PAN. Validation errors do not echo submitted sensitive values. |
| Handle unseen categories | Complete | The trained pipeline tolerates unseen categories; unknown pincodes produce a null tier instead of crashing. Vehicle answers are constrained to the versioned catalogue. |
| Reproducible cleaning and training | Complete | Phase A, preprocessing, model packaging, evaluation, inference, and batch-scoring scripts are versioned with a locked dependency file. |
| Accuracy, precision, recall, F1, confusion matrix, and threshold justification | Complete on development validation | The model card reports both-class metrics, the confusion matrix, ranking diagnostics, imbalance baseline, and the locked threshold rationale. |
| Predictions on the reserved hold-out set | Externally pending | `predict_batch` produces the required ID, decision, score, threshold, and model-version output when the evaluator supplies the hold-out CSV. The hidden dataset was not included in the archive. |
| Offer next steps after the decision | Backend complete | The decision response returns `review_sanction_documents` for approval and `view_decision` for decline. Presentation belongs to the frontend. |

## Commands for final verification

```powershell
uv sync --project backend --extra dev
uv run --project backend alembic -c backend/alembic.ini upgrade head
uv run --project backend pytest backend/tests
uv run --project backend ruff check backend
uv run --project backend benchmark-decision --output reports/generated/decision_latency.json
uv run --project backend python -m abc_credit.modeling.predict_batch --input HOLDOUT.csv --output predictions.csv
```

## Frontend design and improvement report

No frontend implementation was changed during this brief-completion work. The frontend
should retain the earlier visual design while addressing the following before final
submission:

1. Check `/ready` before collecting consent or starting an application, and show a
   localized service-unavailable state when the model or reference tables are missing.
2. Run timed usability sessions and report the p95 completion time against X = 5 minutes.
3. Add browser end-to-end tests for approved, declined, manual-review, retry, offline,
   English, and Hindi paths, including explanation and next-step rendering.
4. Add refresh recovery so an in-progress application does not disappear when the page
   reloads.
5. Localize API error details instead of displaying English backend text in Hindi mode.
6. Complete keyboard, screen-reader, focus-order, error-announcement, contrast, and
   mobile-layout accessibility testing.
7. Make the mobile field consistent with its documented support for `+91`, spaces, and
   dashes; the current ten-character limit prevents those formats.
8. Perform the final Docker or deployed-browser walkthrough and capture evidence that
   the decision explanation and next-step message appear correctly without exposing the
   audit probability or threshold.
