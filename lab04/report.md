# Lab 4: Data pipeline and quality

## 1. Goal and problem definition
**Problem.** Predict from census data whether a person's yearly income is above $50K. The dataset is UCI Adult
(`fetch_openml(name='adult', version=2, as_frame=True)`).

- **Inputs**: 14 features. Numeric: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.
  Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country.
- **Output**: binary label `income` (`>50K` = 1, `<=50K` = 0). Classes are imbalanced (24% positive).
- **Success metrics**: F1 for the positive class (main metric, because of the imbalance) and accuracy. A model
  that always predicts `<=50K` already gets 76% accuracy, so accuracy alone is misleading.
- **Constraints**: (1) *no leakage*: the train/test split is done before any cleaning, and every statistic
  (median, mode, percentiles, scaling, categories) is fitted on the training set only; (2) *reproducibility*:
  fixed seed 42, all settings in `config.yaml`, raw data never modified, one command runs everything, and the
  processed output is identified by a SHA256 hash.

## 2. Method
Run everything with one command from the repository root:

```
python lab04/src/pipeline.py
```

Steps (`src/pipeline.py`, settings in `config.yaml`):
1. **Ingestion**: downloads once and saves an untouched copy `data/raw/adult_raw.csv`. All later steps read this
   file as strings and work on a copy. Logged: 48 842 rows, 15 columns.
2. **Validation**: schema (expected columns, integer columns are integers, categorical columns are non-numeric),
   ranges (age 17–90, hours 1–99, capital-gain 0–99 999, capital-loss 0–4 356), `'?'` counted as missing
   (OpenML gives empty values, which are mapped to `'?'`), duplicates, label values. Violations are only reported,
   no rows are removed. Output: `results/validation_report.md` / `.json`.
3. **Split**: 70/30 stratified by income, before any cleaning.
4. **Cleaning** (fitted on train only): drop exact duplicates from train only; numeric → median imputation →
   clip to train 1st/99th percentile → standard scaling; categorical → mode imputation → one-hot
   (unknown categories ignored). The fitted transformer is saved to `results/preprocessor.joblib`.
5. **Derived features**: `capital_net = capital-gain − capital-loss`, `hours_per_age = hours-per-week / (age − 16)`.
   Both are computed row by row (no fitted statistics), so they cannot leak. Compared with a
   `LogisticRegression(max_iter=1000)` baseline.
6. **Label quality**: duplicate-conflict audit on the full dataset, plus a rule-based check of 50 random rows
   (seed 42) against the data dictionary (`results/label_audit_sample.csv`).
7. **Reproducibility**: SHA256 of the processed train + test matrices and labels, stored with the full config,
   row counts and metrics in `results/run_metadata.yaml`. Data card: `data_card.md`.

## 3. Results

**Validation report** (`results/validation_report.md`)

| Check | Result |
|---|---|
| Rows / columns | 48 842 / 15, no missing or unexpected columns |
| Type violations | 0 in all columns |
| Range violations | 0 (age 17–90, hours 1–99, gain 0–99 999, loss 0–4 356, all exactly at the limits) |
| Missing (`'?'`) | workclass 2 799, occupation 2 809, native-country 857, others 0 |
| Exact duplicate rows | 52 (reported, kept in the raw data) |
| Invalid labels | 0 |

**Cleaning**: train 34 189 → 34 165 rows (24 duplicates removed); test kept at 14 653 rows. Processed matrix:
107 features.

**Baseline with and without derived features** (`results/baseline_metrics.csv`)

| Features | Accuracy | F1 (>50K) |
|---|---|---|
| Base (14) | 0.8540 | 0.6601 |
| Base + capital_net + hours_per_age | **0.8578** | **0.6696** |

**Label quality**
- Duplicate-conflict audit: **5** groups of identical feature rows with different income labels (10 rows in total).
- 50-row check against the data dictionary (ranges, education ↔ education-num match, Husband/Female or
  Wife/Male, Never-married with spouse relationship, capital-gain and loss both positive, valid label):
  **0 of 50** rows have impossible values.

**Reproducibility**: processed dataset SHA256
`9c730a08a49f02d121ad6272a65d52cac86777840c26b134cfa09b1d125b6819`. Two runs of the pipeline gave the same hash.

## 4. Conclusions
1. The raw data has no schema or range errors, but it is not clean: about 7% of rows have a missing category and
   there are 52 duplicates. Reporting these issues instead of silently dropping rows keeps the test set
   representative and makes every change traceable.
2. Splitting before cleaning matters: imputation values, clipping limits and one-hot categories come only from the
   training data, so the test score is a fair estimate. Duplicates are removed only from train, so the test set
   keeps the real distribution.
3. Simple domain features help a little (+0.4 points accuracy, +1.0 point F1) because a linear model cannot build
   `gain − loss` or a ratio by itself. The 5 conflicting duplicate groups show that the labels contain a small amount
   of noise that no model can remove.
