# Data card: UCI Adult (Census Income)

| Field | Value |
|---|---|
| Source | UCI Machine Learning Repository, "Adult" (Becker & Kohavi, 1996), extracted from the 1994 US Census database. Loaded via OpenML `fetch_openml(name='adult', version=2)` (OpenML ID 1590). |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), as listed on the UCI repository. |
| Size | 48 842 rows × 15 columns (14 features + `income` label); 6 integer and 8 categorical features. |
| Task | Binary classification: does a person earn more than $50K per year? |
| Label balance | `<=50K`: 37 155 (76.1%), `>50K`: 11 687 (23.9%). |
| Missing values | `'?'` in `workclass` (2 799), `occupation` (2 809), `native-country` (857). |
| Duplicates | 52 exact duplicate rows; 5 groups of identical feature rows with conflicting labels (10 rows). |
| Processed version | 70/30 stratified split; train 34 165 rows (after dropping duplicates), test 14 653 rows; 107 processed features. SHA256 in `results/run_metadata.yaml`. |

## Known biases
- **Old data**: incomes are from 1994 and are not adjusted for inflation. $50K then is much more than $50K today.
- **Demographic imbalance**: 67% male and 85% white; 90% of people are from the United States. Many countries
  have fewer than 100 rows, so the model is unreliable for these groups.
- **Label correlates with protected attributes**: the share of `>50K` is 30.4% for men and only 10.9% for women, and
  it differs by race. A model trained on this data will copy those historical gaps.
- **Sampling weights**: `fnlwgt` is a census sampling weight, not a personal attribute. Using it as a feature
  mixes survey design with the prediction.
- **Proxy variables**: `relationship` (Husband/Wife) and `marital-status` encode sex and family status.

## Privacy concerns
- The records are anonymised, but they contain quasi-identifiers (age, sex, race, native country, occupation,
  education). Rare combinations (for example an older person from a small country with a rare occupation) could
  allow re-identification when linked with other data.
- Income, marital status and race are sensitive personal attributes. The model should not be used for
  decisions about individuals (credit, hiring) without a fairness review.

## Intended use
Teaching and benchmarking of data pipelines and tabular classifiers. Not intended for real decisions about people.
