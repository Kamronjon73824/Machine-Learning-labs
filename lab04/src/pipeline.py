"""Lab 4: reproducible, leakage-free data pipeline for UCI Adult.

Run from the repository root:  python lab04/src/pipeline.py
"""
import hashlib
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from transforms import PercentileClipper, add_derived_features

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parent
RESULTS = LAB / "results"
DATA = ROOT / "data"
RAW_CSV = DATA / "raw" / "adult_raw.csv"
PROCESSED = DATA / "processed"

CFG = yaml.safe_load((LAB / "config.yaml").read_text())
SEED = CFG["seed"]
TARGET = CFG["dataset"]["target"]
MISSING = CFG["missing_token"]
NUM = CFG["schema"]["int"]
CAT = [c for c in CFG["schema"]["category"] if c != TARGET]

random.seed(SEED)
np.random.seed(SEED)
RESULTS.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS / "pipeline.log", mode="w")])
log = logging.getLogger("lab04")


# ---------------------------------------------------------------- 2. ingestion
def ingest() -> pd.DataFrame:
    """Download once, store an untouched raw copy, always read from that copy."""
    if not RAW_CSV.exists():
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        # default sklearn cache (~/scikit_learn_data): OpenML's nested cache paths can
        # exceed the Windows path limit inside a deep repo folder
        ds = fetch_openml(name="adult", version=CFG["dataset"]["version"], as_frame=True, parser="auto")
        ds.frame.to_csv(RAW_CSV, index=False)
    raw = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False)
    log.info("Ingested raw data: %d rows, %d columns", *raw.shape)
    return raw


def standardise(raw: pd.DataFrame) -> pd.DataFrame:
    """Copy with target renamed and empty cells mapped to the '?' missing token."""
    df = raw.rename(columns={"class": TARGET}).copy()
    df = df.replace({"": MISSING, "nan": MISSING}).apply(lambda s: s.str.strip())
    return df


# --------------------------------------------------------------- 3. validation
def validate(df: pd.DataFrame) -> dict:
    report = {"rows": len(df), "columns": df.shape[1], "type_violations": {}, "range_violations": {},
              "missing_counts": {}, "exact_duplicate_rows": int(df.duplicated().sum()),
              "invalid_labels": 0}
    expected = set(NUM) | set(CAT) | {TARGET}
    report["missing_columns"] = sorted(expected - set(df.columns))
    report["unexpected_columns"] = sorted(set(df.columns) - expected)

    for col in NUM:
        vals = df[col][df[col] != MISSING]
        as_num = pd.to_numeric(vals, errors="coerce")
        bad = int(as_num.isna().sum() + (as_num.dropna() % 1 != 0).sum())
        report["type_violations"][col] = bad
    for col in CAT + [TARGET]:
        # categories must be non-numeric strings; '?' is an allowed category
        report["type_violations"][col] = int(pd.to_numeric(df[col], errors="coerce").notna().sum())

    for col, (lo, hi) in CFG["ranges"].items():
        v = pd.to_numeric(df[col], errors="coerce")
        report["range_violations"][col] = {"min": float(v.min()), "max": float(v.max()),
                                           "below": int((v < lo).sum()), "above": int((v > hi).sum())}
    for col in df.columns:
        report["missing_counts"][col] = int((df[col] == MISSING).sum())
    report["invalid_labels"] = int((~df[TARGET].isin([">50K", "<=50K"])).sum())
    report["label_distribution"] = df[TARGET].value_counts().to_dict()

    (RESULTS / "validation_report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Validation report", "", f"Rows: {report['rows']}, columns: {report['columns']}",
             f"Missing/unexpected columns: {report['missing_columns']} / {report['unexpected_columns']}",
             f"Exact duplicate rows: {report['exact_duplicate_rows']}",
             f"Invalid labels: {report['invalid_labels']}", "",
             "| column | type violations | missing ('?') |", "|---|---|---|"]
    lines += [f"| {c} | {report['type_violations'].get(c, 0)} | {report['missing_counts'][c]} |" for c in df.columns]
    lines += ["", "| column | min | max | below range | above range |", "|---|---|---|---|---|"]
    lines += [f"| {c} | {r['min']:g} | {r['max']:g} | {r['below']} | {r['above']} |"
              for c, r in report["range_violations"].items()]
    lines += ["", "No rows were dropped by validation; violations are only reported."]
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n")
    log.info("Validation: %d duplicates, missing=%s", report["exact_duplicate_rows"],
             {k: v for k, v in report["missing_counts"].items() if v})
    return report


# ------------------------------------------------------- 4/5. transformation
def build_preprocessor(derived: bool) -> Pipeline:
    num_cols = NUM + (CFG["features"]["derived"] if derived else [])
    lo, hi = CFG["cleaning"]["clip_percentiles"]
    num = make_pipeline(SimpleImputer(strategy=CFG["cleaning"]["numeric_impute"]),
                        PercentileClipper(lo, hi), StandardScaler())
    cat = make_pipeline(SimpleImputer(strategy=CFG["cleaning"]["categorical_impute"]),
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    cols = ColumnTransformer([("num", num, num_cols), ("cat", cat, CAT)])
    steps = [("derive", FunctionTransformer(add_derived_features))] if derived else []
    return Pipeline(steps + [("columns", cols)])


def to_model_frame(df: pd.DataFrame):
    X = df.drop(columns=[TARGET]).replace(MISSING, np.nan)
    X[NUM] = X[NUM].astype(float)
    y = (df[TARGET] == CFG["dataset"]["positive_label"]).astype(int)
    return X, y


def evaluate(preprocessor, X_tr, y_tr, X_te, y_te):
    model = Pipeline([("prep", preprocessor),
                      ("clf", LogisticRegression(max_iter=CFG["model"]["max_iter"], random_state=SEED))])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return accuracy_score(y_te, pred), f1_score(y_te, pred)


# ---------------------------------------------------------- 6. label quality
def label_audit(df: pd.DataFrame) -> dict:
    features = [c for c in df.columns if c != TARGET]
    groups = df.groupby(features, dropna=False)[TARGET].nunique()
    conflicts = groups[groups > 1]
    conflict_rows = df.set_index(features).index.isin(conflicts.index).sum()

    sample = df.sample(CFG["label_audit"]["sample_size"], random_state=SEED).copy()
    edu_map = df[df["education"] != MISSING].groupby("education")["education-num"].agg(lambda s: set(s))
    checks = {
        "age_out_of_range": ~pd.to_numeric(sample["age"]).between(17, 90),
        "hours_out_of_range": ~pd.to_numeric(sample["hours-per-week"]).between(1, 99),
        "gain_and_loss_both_positive": (pd.to_numeric(sample["capital-gain"]) > 0)
                                       & (pd.to_numeric(sample["capital-loss"]) > 0),
        "education_num_mismatch": sample.apply(lambda r: len(edu_map[r["education"]]) != 1, axis=1),
        "husband_female_or_wife_male": ((sample["relationship"] == "Husband") & (sample["sex"] == "Female"))
                                       | ((sample["relationship"] == "Wife") & (sample["sex"] == "Male")),
        "never_married_spouse": (sample["marital-status"] == "Never-married")
                                & sample["relationship"].isin(["Husband", "Wife"]),
        "invalid_label": ~sample[TARGET].isin([">50K", "<=50K"]),
    }
    for name, mask in checks.items():
        sample[name] = mask.values
    sample["any_issue"] = sample[list(checks)].any(axis=1)
    sample.to_csv(RESULTS / "label_audit_sample.csv", index=False)
    result = {"duplicate_feature_groups_with_conflicting_labels": int(len(conflicts)),
              "rows_in_conflicting_groups": int(conflict_rows),
              "sample_size": len(sample),
              "sample_rows_with_issues": int(sample["any_issue"].sum()),
              "sample_issue_counts": {k: int(v.sum()) for k, v in checks.items()}}
    log.info("Label audit: %s", result)
    return result


# --------------------------------------------------------------------- main
def sha256_of(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def main():
    subprocess.run([sys.executable, str(HERE / "print_versions.py"), str(RESULTS / "versions.txt")], check=True)
    raw = ingest()
    df = standardise(raw)
    report = validate(df)

    # split BEFORE any cleaning so the test set never influences fitted statistics
    train_df, test_df = train_test_split(df, test_size=CFG["split"]["test_size"],
                                         stratify=df[TARGET], random_state=SEED)
    n_before = len(train_df)
    train_df = train_df.drop_duplicates()
    log.info("Train %d -> %d rows after dropping exact duplicates; test kept at %d rows",
             n_before, len(train_df), len(test_df))

    X_tr, y_tr = to_model_frame(train_df)
    X_te, y_te = to_model_frame(test_df)

    rows = []
    for derived in (False, True):
        acc, f1 = evaluate(build_preprocessor(derived), X_tr, y_tr, X_te, y_te)
        rows.append({"features": "base + derived" if derived else "base (14)",
                     "accuracy": round(acc, 4), "f1": round(f1, 4)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "baseline_metrics.csv", index=False)
    log.info("\n%s", metrics.to_string(index=False))

    prep = build_preprocessor(derived=True).fit(X_tr, y_tr)
    joblib.dump(prep, RESULTS / "preprocessor.joblib")
    Xp_tr, Xp_te = prep.transform(X_tr), prep.transform(X_te)
    names = prep.named_steps["columns"].get_feature_names_out()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(Xp_tr, columns=names).assign(income=y_tr.values).to_csv(PROCESSED / "train.csv", index=False)
    pd.DataFrame(Xp_te, columns=names).assign(income=y_te.values).to_csv(PROCESSED / "test.csv", index=False)

    audit = label_audit(df)

    digest = sha256_of(Xp_tr, y_tr.values, Xp_te, y_te.values)
    meta = {"config": CFG, "processed_sha256": digest,
            "raw_rows": int(raw.shape[0]), "raw_columns": int(raw.shape[1]),
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
            "processed_features": int(Xp_tr.shape[1]),
            "validation": {"exact_duplicate_rows": report["exact_duplicate_rows"],
                           "invalid_labels": report["invalid_labels"]},
            "label_audit": audit, "metrics": rows}
    (RESULTS / "run_metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    log.info("Processed dataset SHA256: %s", digest)


if __name__ == "__main__":
    main()
