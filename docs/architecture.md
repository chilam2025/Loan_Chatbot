# System Architecture and Workstreams

## Component boundary

The predictive and generative responsibilities remain deliberately separate:

```text
Applicant
   |
   v
Conversation layer ---- validation and clarification only
   |
   v
Deterministic feature contract
   |
   v
Fitted preprocessing pipeline --> predictive model --> probability threshold
   |                                                   |
   +---------------- audit event <---------------------+
                                                       |
                                                       v
                                             APPROVE or DECLINE
```

The conversation layer may explain the process and ask follow-up questions. It may not create, override or reinterpret the final credit decision.

## Workstream ownership

| Workstream | Primary responsibility | Initial deliverable |
|---|---|---|
| Data | Quality analysis, feature definitions and fitted preprocessing | Phase A report and reusable schema checks |
| Modelling | Baselines, imbalance-aware evaluation, calibration and threshold selection | Validation metrics and serialized model pipeline |
| Application | Conversation design, validation, privacy-safe logging and model integration | Tested chatbot flow and integration contract |

All members jointly approve the target definition, data split, minimal input set, evaluation metrics and integration schema.

## Proposed delivery targets

- Application completion target X: no more than 5 minutes, to be validated with user timing tests.
- Post-input decision latency target Y: no more than 2 seconds for local preprocessing and inference, measured at p95.

These are engineering targets, not claims of achieved performance.

