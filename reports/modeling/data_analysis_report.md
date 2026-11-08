ABC Credit: Data Analysis and Model Development Report
Status: academic prototype, based on the supplied project archives and experiment outputs shared during development. Prepared 2 November 2026. This report distinguishes full-data descriptive findings from training-only investigations and validation-based model comparisons. Later experiment results were supplied by the user; a fresh audit of the latest local repository has not been performed.
1. Problem and scope
The objective is to collect applicant and two-wheeler financing information through a chatbot and predict a historical APPROVED or DECLINED decision using a separate predictive model. The language model must not make or override that decision. This report covers Person 1's data analysis, modelling and model integration interface, not implementation of backend APIs or chatbot screens.

The target is a historical lending decision. There are no repayment outcomes, arrears or default labels. Therefore the model cannot establish repayment probability, and a historical decline does not prove that an applicant would default.
2. Data inventory
Item	Count
Original CSV rows	108,423
Columns	17
Valid APPROVED labels	102,724
Valid DECLINED labels	5,698
Unlabelled row excluded	1
Labelled applications	108,422


The unlabelled record was reported as otherwise empty. The original CSV is preserved. Approximately 94.7446% of labelled records are approved and 5.2554% declined. An all-approved predictor achieves 94.7446% full-data accuracy but zero declined recall.

The 17 columns comprise an application ID, 15 candidate predictors and Decision. ID and Decision are excluded from model inputs. Gender was subsequently excluded from the selected prototype, leaving 14 required input fields.
3. Data quality
3.1 Missing values
Full raw-data counts include Employment_Type 28, Qualifications 26, Net_salary 24, Final_Tier 23, Resident_Type 20 and Age 19. Other fields have smaller counts. Raw counts and labelled/normalised counts differ because the unlabelled row is excluded and normalisation can identify additional invalid numeric values.

The fitted baseline uses training medians for numeric missing values, numeric missing indicators, and training most-frequent categories for missing categorical values. Missing income is distinct from valid zero income. Mode imputation is a baseline choice and can obscure meaningful missingness; missing-as-category is a possible future experiment.

A script was supplied to compare missingness with Decision, but its resulting table has not been reviewed in the shared evidence. No final claim about informative missingness is made here.
3.2 Numeric anomalies
Full labelled-data EDA found 1,105 ages above 60, 626 zero-income records and 65 zero-LTV records. There were no non-positive loan amounts, LTV values outside 0–100, or malformed six-digit pincodes in the reported summary.

Training-only audit:
Condition	Flagged count	Flagged decline rate	Other nonmissing decline rate
Age above 60	776	3.74%	5.27%
Zero income	433	2.08%	5.27%
Zero LTV	43	6.98%	5.25%


The brief states: “Policy applies to ages 18 to 60.” It does not specify the exact action for applications outside that range. Ages above 60 are therefore retained without capping or relabelling. The interpretation needs domain clarification; no automatic decline policy was added.

Of 433 zero-income training cases, 421 were students, 5 non-earning members and 7 had missing employment. Zero income is plausible for these categories and is retained. These observations do not establish how applicants funded repayments.

Zero-LTV cases occurred in AGR (7), NREGI (6), SAL (13) and SEP (17). Positive loans with zero LTV are inconsistent with the ordinary loan/value formula for a finite positive vehicle value. A missing-value placeholder or entry error is possible but unconfirmed. An experiment converted zero LTV to missing, with train-fitted imputation and an indicator. The packaged prototype retains the original zero-LTV values.

Monthly income is right-skewed: the full-data median was INR 30,000 and maximum INR 980,000. High values were not automatically removed. Log-income features were tested; no separate fitted capping treatment was adopted.
3.3 Duplicate inputs and conflicting labels
The raw EDA reported no exact duplicate complete rows and no duplicated non-null IDs. Nevertheless, different IDs can have identical predictor values.

Current modelling preparation removes 8 records in identical-input groups with conflicting labels, then removes 11 repeated predictor records with matching labels. This leaves 108,403 modelling records: 75,882 training, 16,260 validation and 16,261 internal test.

This is a modelling choice, not proof of erroneous records. Different IDs may be genuine applications and conflicting outcomes can arise from omitted information or policy changes. Exclusion reduces representativeness. A team-reviewed alternative is retaining all applications while assigning identical-input groups to the same partition. Changing this policy requires new splits and renewed experiment comparisons.
3.4 Category consistency
Existing normalisation strips categorical whitespace, converts empty labels to missing and uppercases Decision. It does not automatically merge all case or spelling variants. A later audit script searches for case/spacing aliases; its output has not been supplied for interpretation. Semantic consistency is not established merely by passing schema tests.
4. Exploratory findings
Numeric histograms use density normalised separately within each class, so approved counts do not dominate. They show distributions conditional on the decision, not the probability of decline at a given feature value.
- LTV has the clearest visible numeric class difference: declines are concentrated at higher values.
- Loan amounts are generally higher among declines. Full-data mean amounts were INR 88,759 for approvals and INR 98,843 for declines.
- Full-data mean LTV was 77.24% for approvals and 85.84% for declines.
- Age distributions overlap substantially, with declines appearing somewhat younger.
- Income distributions overlap substantially, including in the positive-income logarithmic view. This does not establish that income is useless in interactions.
- Integer-age histograms can create binning spikes; spikes should not be interpreted as behavioural effects without age-count checks.
Categorical training-data summaries included counts, declined counts, rates and individual 95% Wilson intervals. Approximate visual findings:
Field	Observation
Employment	Most large groups are around the overall rate; students have lower observed declines; tiny groups have wide intervals.
Qualifications	PG/GRAD around 4.5–4.6%; OTHERS around 5.7%; several intervals overlap.
Residence	Owned is near overall; rented slightly lower; leased and missing groups have substantial uncertainty.
Active prior loan	Active loan about 4.2% declined versus about 6.2% for NO_PAST_LOANS.
Location tier	Rural/semi-urban around 5.3%; some metropolitan categories around 2.3–3.7%.
Gender	Female about 5.4%, male about 5.2%, with overlapping intervals.
Product	Motorcycle about 5.7%, scooter 5.3%, moped 4.5%, electric 2.3%.


These are unadjusted historical associations, not causal explanations. The prior-loan flag does not provide repayment history. No inference that taking another loan improves approval chances is justified. Overlapping intervals are not a formal group-difference test. The below-100 flag is a practical small-sample reminder, not a significance threshold. The initially clipped interval axis was corrected to use percentage units.
5. Vehicle redundancy and pincode sparsity
In 75,880 complete training records, every observed Model_Description mapped to exactly one Make_Code, Model_Variant and Product_Code (86 model descriptions). All 12 make categories and 21 variant categories each mapped to one product code.

This supports asking for one catalogue selection and deriving multiple model inputs. It does not guarantee the mapping for unseen models. A catalogue needs versioning and explicit unknown-model handling. Raw model text is currently treated as a categorical label, not processed with an NLP embedding or language model.

Training pincodes:
Measure	Result
Distinct codes	12,872
Codes seen once	3,296
Codes seen fewer than 10 times	10,636
Training rows in codes seen fewer than 10 times	44.91%


Pincode is categorical, never continuous. Existing encoding groups infrequent labels and caps each categorical field at 100 encoded categories. This discards detail but still produced measurable predictive contribution. Validation coverage of unseen pincodes remains an unreported check. Vehicle category tables were generated, but detailed make/variant rate findings were not supplied beyond redundancy and importance results.
6. Preprocessing and evaluation design
Stratified 70/15/15 splitting is used. Imputers, encoders and scalers learn from fitting records only. The later numeric/categorical audits reconstruct Phase B's record-selection policy and examine its training partition.

The separate prepare-data command originally split all labelled records without Phase B's duplicate/conflict filtering. Its partitions therefore differ despite the same random seed. The team should consolidate one authoritative split procedure or saved membership manifest before further changes.

Earlier full-data EDA included target relationships across all labelled data. That is descriptive exploration, but it weakens strict claims that the eventual internal holdout was never inspected even before its model score was printed.

Primary development diagnostics: declined precision, recall, F1, confusion matrix, average precision (labelled PR-AUC in outputs), Brier score and later log loss. Average precision is not necessarily identical to trapezoidal precision-recall area. Accuracy is secondary because approvals dominate.
7. Feature experiments
Later controlled Logistic Regression experiments scale numeric columns only. Their results should not be directly attributed to feature changes relative to the earliest script, which scaled the entire encoded matrix.
Experiment	Precision	Recall	F1	Average precision	Assumed 5:1 cost
Baseline	24.53%	13.82%	17.68%	0.1545	4,043
Zero LTV treated as missing	24.21%	15.22%	18.69%	0.1547	4,027
Engineered feature bundle	22.60%	17.68%	19.84%	0.1553	4,032
Initial eight-field subset	23.01%	17.92%	20.14%	0.1469	4,017


Engineered candidates were zero-income, missing-income and zero/missing-LTV indicators, log1p income, and loan amount divided by positive monthly income. Undefined ratios remain missing for train-fitted imputation. Loan/monthly-income is not EMI affordability. Bundled changes were not isolated in further ablations, and improvements were small. None was incorporated into the packaged forest.

Permutation importance on a reproducible stratified 3,000-record validation subset (3 shuffles) ranked LTV, Loan_Amount, Make_Code and Pincode highest. Mean decreases in average precision were 0.0885, 0.0792, 0.0309 and 0.0207 respectively. Some fields had near-zero or negative importance. Those values do not establish uselessness across models. Correlated vehicle fields and unrealistic shuffled combinations complicate interpretation; shuffle SD is not a confidence interval.

Removal experiments:
Experiment	Average precision	Assumed 5:1 cost
Baseline	0.1545	4,043
Without gender	0.1543	4,036
Without five low-contribution fields	0.1517	4,055
Only model description among vehicle fields	0.1556	4,039
Without pincode	0.1480	4,046


Gender exclusion had little observed impact and was adopted for later comparisons. Removing pincode weakened ranking. No final minimal feature set was established. The forest keeps all 14 non-gender predictors; Logistic Regression selection results do not establish the best subset for a forest.
8. Models and imbalance
Standard and balanced Logistic Regression and Random Forest were compared. At threshold 0.5, standard Logistic Regression predicted all approvals. Balanced Logistic Regression detected 657/854 declines but incorrectly declined 5,951 approvals. After threshold tuning, balanced Logistic Regression had average precision 0.1489 versus 0.1543 standard, and cost 4,053 versus 4,036. Weighting mainly shifted behaviour without improving this comparison.

Same-feature tree comparison:
Model	Precision	Recall	F1	Average precision	Cost (coarse grid)
Standard logistic	24.79%	13.93%	17.84%	0.1543	4,036
Standard forest	22.64%	17.10%	19.48%	0.1613	4,039
Balanced forest	23.70%	10.66%	14.70%	0.1509	4,108


No resampling was used. CatBoost was discussed but no completed CatBoost experiment was supplied; it must not be listed as trained.
9. Calibration
Sigmoid calibration used three training-only folds with the entire preprocessor inside each fold and ensemble=False. The model was refitted on all training data after out-of-fold calibration scores were obtained.
Candidate	Brier score	Log loss	Average precision
Logistic uncalibrated	0.047203	0.184503	0.1543
Logistic sigmoid	0.047232	0.184522	0.1543
Forest uncalibrated	0.047462	0.186524	0.1613
Forest sigmoid	0.047291	0.186716	0.1613


Calibration did not clearly improve both probability metrics. Uncalibrated models were retained. Sigmoid preserved score ordering in these runs, hence unchanged average precision. Reliability plots use quantile bins and can hide within-bin errors. Scores must not be described as established default probabilities or perfectly calibrated historical-decline probabilities.
10. Threshold and cost sensitivity
Positive class is DECLINED=1. A score at or above the threshold gives DECLINED. False approval means predicting APPROVED for a historically DECLINED record; false decline means the reverse. Costs represent assumed disagreement penalties, not measured repayment losses.

Later threshold selection evaluated all distinct validation-score boundaries, keeping ties together.
Model	Cost ratio	Threshold	Decline recall	False declines	Missed declines	Assumed cost
Logistic	1:1	0.45014	0.35%	0	851	851
Forest	1:1	0.24678	0.59%	0	849	849
Logistic	5:1	0.17503	13.93%	361	735	4,036
Forest	5:1	0.13052	16.98%	489	709	4,034
Logistic	10:1	0.09750	41.57%	2,104	499	7,094
Forest	10:1	0.07556	52.93%	3,088	402	7,108


At 1:1, 100% precision is based on only 3 logistic or 5 forest predictions. It does not indicate strong overall detection. Comparisons between models must be within the same cost scenario. The 5:1 choice requires written justification/team agreement; it was not supplied as a lender requirement.
11. Training-only stability check
Three outer stratified folds were used within the original training partition. Each development fold was split again, reserving 20% for threshold selection. Models fitted only the inner fit data; the same model and locked inner threshold were evaluated on the outer fold. Original validation and internal test records were not used in this script.
Metric	Logistic mean	Forest mean
Declined precision	20.67%	20.82%
Declined recall	13.42%	15.38%
Declined F1	16.20%	17.66%
Average precision	0.1417	0.1445
Cost per 1,000 applications	254.67	253.02
Cost reduction vs all-approved	3.04%	3.66%


Forest cost was lower in all three paired folds, by 0.83–2.45 units per 1,000. The average advantage was small; three folds do not establish significance. Development choices were already informed by earlier analyses, so this is a stability diagnostic, not a new independent final performance estimate.
12. Selected prototype and packaging
Selected candidate: standard Random Forest without gender. It has 150 trees, max_depth=18, min_samples_leaf=5, class_weight=None and random_state=42. Full training data were used for fitting. Its threshold was chosen on validation, not averaged from cross-validation.

Version: random-forest-v1. Threshold: 0.13052013989740333. No separate calibration or experimental numeric feature transformation selected.

Required inputs: Age, Loan_Amount, LTV, Net_salary, Qualifications, Employment_Type, Pincode, Product_Code, Resident_Type, Final_Tier, Make_Code, Model_Description, Model_Variant and PAST_LOANS_ACTIVE.

Validation confusion matrix:
Actual / predicted	APPROVED	DECLINED
APPROVED	14,917	489
DECLINED	709	145


Accuracy 92.63%; declined precision 22.87%; recall 16.98%; F1 19.49%; average precision 0.16127; Brier score 0.047462. These are repeatedly used development-validation results, not independent final estimates. The model detects 145/854 declines and misses 709. Its 5:1 cost is 4,034 versus 4,270 for approving everyone, a development reduction of approximately 5.53%.

Saved files: models/random_forest_v1/decision_model.joblib and metadata.json. The artifact contains the fitted pipeline and metadata including version, features, threshold, settings, training time stamp and package versions. It depends on compatible installed packages and project source; it is not a self-contained API. Only trusted joblib artifacts should be loaded.

The inference interface checks required fields, normalises types, rejects malformed/infinite numeric values and returns decision, score, threshold and version. Fields must exist, while missing values may be imputed. Extra fields are ignored. Full application validation is the backend's responsibility. Reported tests and lint passed (16 tests); artifact-independent CI tests remain outstanding because inference tests currently skip if the private model artifact is absent. A repeated-prediction test is not a reproducible-training test.
13. Question reduction recommendation
The deployed candidate has 14 predictor fields, but these need not become 14 separate questions. Recommend one vehicle-catalogue selection to supply model description, make, variant and product. If vehicle price and loan amount are known, the backend can calculate LTV consistently. Location tier may be derived only from a validated mapping, not guessed. Gender and application ID are not questions required by this model.

No empirically final minimum question set, completion-time target X, or decision-latency target Y has yet been established. Existing timings are batch timings and must not be claimed as single-request or end-to-end chatbot latency.
14. Limitations and remaining work
1. Declined detection is weak and the benefit over an all-approved cost baseline is modest.
2. The internal test score was inspected early; later work stopped using it. Full-data EDA also inspected distributions and target relationships. An internal test report cannot be described as completely untouched. No training fit on test rows was reported.
3. Repeated validation comparisons introduce selection optimism. The instructor's separate reserved holdout is the intended independent final evaluation; its delivery process needs clarification.
4. Duplicate/conflicting-label filtering is contestable and changes the modelling population.
5. Full group error evaluation and a model fairness assessment were not completed. Gender exclusion does not establish fairness, and location can still encode group information.
6. The 5:1 cost scenario needs explicit agreement. No financial default-cost estimates exist in this dataset.
7. Category-alias output, missingness-by-decision output and numeric-correlation output need review before closing the EDA checklist. Their scripts were supplied but the output findings were not shared.
8. An unseen-pincode coverage check, versioned vehicle catalogue, robust split manifest and full inference-artifact validation remain useful additions.
9. No training reproducibility test, private-artifact-independent CI suite, single-request latency benchmark or final hidden-set predictions have been completed in the shared evidence.
10. Backend integration, masked logging, verification mocks and chatbot validation are other team members' work, not implemented by this report.
15. Next actions before the pull request
- Review remaining audit tables and resolve the split/record-selection policy with the team.
- Complete docs/model/model_contract.md and docs/model/model_card.md using the actual packaged metadata.
- Add synthetic artifact-independent tests, including training reproducibility, serialization and inference checks.
- Measure repeated single-application inference latency with representative synthetic inputs; report environment and median/p95.
- Agree the prototype cost assumption and question flow with the team.
- Run tests and lint; inspect git status and diffs. Stage only intended owned source/docs and approved shared-file changes. Never commit raw applicant data, private generated tables or the trained artifact automatically.
- Request another person's review before merge. Do not claim completion of outstanding items or production suitability.
Evidence index
Original project brief and CSV; latest reviewed project archives; training numeric and categorical audit outputs; feature_analysis validation_experiments.csv, permutation importance, vehicle_redundancy.csv and pincode_sparsity.csv; removal experiments; class-weight and tree comparisons; calibration CSV and reliability plot; exact threshold scenarios; training-only stability summaries; packaging terminal output; reported pytest and Ruff results. No applicant-level personal records are reproduced in this report.
## Final audit and software verification update

### Category consistency

The category-alias audit returned no matching case/spacing variants
among the inspected training categories, excluding Pincode.

No category merging was required based on that check. This does not
rule out spelling errors or semantically equivalent labels.

### Missingness

Training-data missing values were uncommon:

| Field | Missing records |
|---|---:|
| Final_Tier | 18 |
| Net_salary | 15 |
| Employment_Type | 14 |
| Qualifications | 12 |
| Age | 7 |
| Resident_Type | 7 |
| Make_Code | 2 |
| Model_Description | 2 |
| Model_Variant | 2 |

None of these missing-value groups contained historical declines.
The samples are too small to conclude that missingness predicts
approval. Counts can overlap across fields.

The existing training-fitted imputation was retained.
Application-completeness validation remains a backend responsibility.

### Numeric redundancy

Spearman correlations between numeric inputs were:

| Pair | Correlation |
|---|---:|
| Loan_Amount and LTV | 0.378 |
| Age and Loan_Amount | -0.177 |
| Age and Net_salary | 0.101 |
| Loan_Amount and Net_salary | 0.083 |
| Age and LTV | 0.061 |
| LTV and Net_salary | -0.023 |

No strong pairwise rank relationship justified removing a numeric
field as redundant. All four were retained.

These correlations compare inputs with each other, not with Decision.
Low Spearman correlation does not exclude nonlinear relationships.

### Software verification

The reported local test run completed with:
- 25 tests passed
- Ruff: all checks passed

Nine synthetic tests run without private data or the trained artifact.
They cover the feature contract, prediction outputs, missing values,
unknown categories, invalid inputs, serialization, model version and
reproducible training on synthetic examples.

Seven integration tests depend on the local trained artifact and skip
if it is unavailable.

These checks verify software behaviour, not predictive effectiveness.

## Final development update: CatBoost candidate

Further experiments compared boosting approaches, CatBoost,
class weights, calibration and financing-feature combinations.

The selected candidate uses unweighted CatBoost with an implied
vehicle-value feature. In the three training-fold comparisons,
this feature increased average precision in every fold, with mean
average precision increasing from 0.15419 to 0.15588.

Development validation average precision increased from 0.16721
to 0.17007. These are modest improvements and do not establish
performance on unseen external data.

Sigmoid calibration produced negligible changes, so the simpler
uncalibrated pipeline was retained.

The packaged threshold uses an illustrative 5:1 disagreement-cost
assumption. Its validation precision is 27.56% and recall is 14.52%.
It favours fewer false declines at the expense of missing many
historical declines.

The artifact loads successfully, inference checks pass, all 66
project tests pass, and Ruff checks pass.
### CatBoost and Random Forest blend

Three stratified training folds selected CatBoost alone from five
candidate blend weights. Training-fold average precision was 0.15927
for CatBoost alone and 0.15925 for the 75% CatBoost blend.

On development validation, CatBoost achieved average precision of
0.1701 versus 0.1613 for Random Forest. At the same number of historical
declines caught, CatBoost flagged 59 fewer historical approvals at
approximately 20% recall and 192 fewer at approximately 40% recall.
The packaged candidate and threshold were retained.