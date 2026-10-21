# Phase A Data Exploration and Quality Findings

## Executive summary

The supplied data contain 108,423 rows and 17 documented columns. One row has no valid target and must be excluded from supervised modelling, leaving 108,422 labelled applications. Of those, 102,724 are `APPROVED` and 5,698 are `DECLINED`, so a trivial all-approved classifier would score 94.74% accuracy while detecting no declined applications. Phase B must therefore emphasize declined-class recall, precision, F1 and the confusion matrix in addition to accuracy.

The dataset has low overall missingness but contains meaningful quality and policy issues. Several input fields have missing values; 1,105 labelled records have ages above the brief's stated 18–60 policy range; 626 records have zero income; 65 have zero LTV; and location and vehicle descriptions have high cardinality. These records should be flagged and investigated, not silently deleted or altered.

## Confirmed observations

- Raw shape: 108,423 rows by 17 columns.
- Labelled rows: 108,422; one row has no decision label.
- Target balance: 94.74% approved and 5.26% declined.
- Missing values include 28 in `Employment_Type`, 26 in `Qualifications`, 24 in `Net_salary`, 23 in `Final_Tier`, 20 in `Resident_Type`, and 19 in `Age`. Several other columns have one or four missing values. The unlabelled record is an otherwise empty row.
- `Age` ranges from 18 to 69 although the brief states that policy applies to ages 18 to 60.
- `LTV` ranges from 0.0 to 99.6. Zero LTV values require investigation because a zero loan-to-value ratio is unusual for a positive loan.
- `Net_salary` ranges from 0 to 980,000 INR and is strongly right-skewed; robust transformation or winsorization should be evaluated on training data only.
- `Resident_Type` is highly concentrated in owned residences, while `Final_Tier` is concentrated in rural locations.
- There are no exact duplicate rows and no duplicated non-null application IDs.
- `Pincode` has 13,999 observed categories, and `Model_Description` has 87, so naive one-hot encoding may create a brittle or unnecessarily large feature space.
- `ID` is a reference field and will be excluded from predictive features.

## Cleaning decisions for the modelling pipeline

1. Preserve the raw file unchanged and normalize whitespace/case only in a derived frame.
2. Exclude rows without a valid `APPROVED` or `DECLINED` target from supervised training and report them separately.
3. Convert numeric fields with explicit coercion and retain conversion failures as missing values for audit.
4. Fit imputers and encoders only after the train/validation split to prevent leakage.
5. Treat categorical fields as strings and configure encoders to tolerate unseen values.
6. Flag ages outside 18–60 and unusual zero values for domain review before deciding whether to exclude, cap or retain them.
7. Treat `Pincode` as a categorical/location feature, never as a continuous number, and assess whether a coarser privacy-preserving representation performs adequately.
8. Evaluate high-cardinality vehicle fields for consolidation or feature hashing; do not one-hot encode them blindly.
9. Keep `Decision` framed as historical approval prediction. It must not be described as default-risk prediction without repayment labels.

## Phase B handoff

Use a stratified split and keep a final internal validation set untouched during model selection. Establish an all-approved reference, then compare logistic regression with a tree-based model using the same leakage-safe pipeline. Report class-specific metrics and probability-quality diagnostics, and choose the operating threshold from explicit business costs rather than maximizing accuracy.

The generated summary, tables and plots can be reproduced with `uv run phase-a`.

