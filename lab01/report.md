# Lab 1: Environment and first system measurements

## 1. Goal
Set up a reproducible Python environment and measure not only the accuracy of two baseline models on
Breast Cancer Wisconsin, but also their system cost (training time, latency, size, memory), then decide on
which deployment targets (Cloud, Edge, Mobile, TinyML) they fit.

## 2. Method
- Environment: isolated `.venv`, packages from `requirements.txt`; `src/print_versions.py` writes
  `results/versions.txt`.
- Data: `load_breast_cancer(return_X_y=True)`, 70/30 stratified split, `random_state=42` (398 train / 171 test).
- Models: `LogisticRegression(max_iter=1000, random_state=42)` and
  `RandomForestClassifier(n_estimators=100, random_state=42)`; features are used raw, as in the manual.
- Training time: 1 warm-up + median of 5 fits. Inference: one test sample, median of 100 `predict` calls.
- Size: `joblib.dump` of each model (`results/model_*.joblib`; the more accurate one is also saved as
  `results/model.joblib`).
- Peak memory: a background thread samples `psutil.Process().memory_info().rss` every 1 ms during training
  and inference. This is the RSS of the whole Python process (interpreter + NumPy + scikit-learn + model).
- Run: `python lab01/src/run_all.py`.

## 3. Results

**Accuracy** (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| LogisticRegression | 0.9415 |
| RandomForest | 0.9357 |

**System cost** (`results/system_cost.csv`)

| Model | Train time (median, ms) | Single-sample latency (median, ms) | Size (bytes / KB) | Peak RSS (MB) | Memory added by training (MB) |
|---|---|---|---|---|---|
| LogisticRegression | 203.4 | 0.097 | 1 103 / 1.08 | 151.7 | 1.2 |
| RandomForest | 115.8 | 4.615 | 290 889 / 284.07 | 153.7 | 3.2 |

**Deployment fit** (`results/deployment_fit.csv`, memory = peak process RSS)

| Model | Cloud (≥1 GB, ≤100 ms, ≤500 MB) | Edge (256–1024 MB, ≤50 ms, ≤50 MB) | Mobile (64–256 MB, ≤20 ms, ≤10 MB) | TinyML (≤256 KB, ≤10 ms, ≤100 KB) |
|---|---|---|---|---|
| LogisticRegression | yes | yes | yes | no |
| RandomForest | yes | yes | yes | no |

Justification:
- **Cloud / Edge / Mobile**: both models need ~152 MB of RAM in total, have sub-5 ms latency and are far below
  10 MB, so they fit all three budgets. Mobile is the tightest one: 152 MB is inside 64–256 MB, but it leaves
  only ~100 MB of headroom, and almost all of it is the Python runtime, not the model.
- **TinyML**: neither model fits as deployed here, because a Python process needs ~150 MB, which is ~600× the
  256 KB limit. The random forest also fails on size (284 KB > 100 KB). Logistic regression itself is only
  31 weights + 1 bias (1.08 KB file, 0.1 ms), so if it were exported to C without Python it would easily fit
  TinyML. The random forest would still be too large unless we reduce the number or depth of trees.

Note: LogisticRegression gives a `ConvergenceWarning` because the features are not scaled and the manual fixes
`max_iter=1000`. The accuracy is still the highest of the two models.

## 4. Conclusions
1. Accuracy alone does not show the full picture: both models are nearly equal (94.2% vs 93.6%), but the forest
   is ~260× larger and ~47× slower per sample. For this task the linear model is clearly the better system.
2. The memory cost is dominated by the software stack, not by the model: training added only 1–3 MB on top of a
   ~150 MB Python process. When choosing a deployment target, the runtime matters as much as the model.
3. Measurements must be repeated and summarised by the median. Single runs of a 0.1 ms prediction are noisy,
   and the first (warm-up) run is always slower, so the measurement protocol strongly affects the numbers.
