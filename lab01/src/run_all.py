"""Lab 1: baselines on Breast Cancer Wisconsin + system cost measurements."""
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

MB = 1024 ** 2
BUDGETS = {  # memory (MB, min-max), latency (ms), size (KB)
    "Cloud": dict(mem=(1024, float("inf")), lat=100, size=500 * 1024),
    "Edge": dict(mem=(256, 1024), lat=50, size=50 * 1024),
    "Mobile": dict(mem=(64, 256), lat=20, size=10 * 1024),
    "TinyML": dict(mem=(0, 256 / 1024), lat=10, size=100),
}


class PeakRSS:
    """Samples process RSS in a background thread and keeps the peak."""

    def __init__(self, interval=0.001):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def make_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }


def median_time(fn, repeats, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def fits(budget, mem_mb, lat_ms, size_kb):
    lo, hi = budget["mem"]
    return mem_mb <= hi and lat_ms <= budget["lat"] and size_kb <= budget["size"]


def main():
    # 1. versions
    subprocess.run([sys.executable, str(HERE / "print_versions.py"), str(RESULTS / "versions.txt")], check=True)

    # 2. data + baselines
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    acc_rows, cost_rows = [], []
    base_rss = psutil.Process().memory_info().rss
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        acc_rows.append({"model": name, "test_accuracy": round(acc, 4)})

        # 3. system cost
        train_s = median_time(lambda: make_models()[name].fit(X_tr, y_tr), repeats=5)
        sample = X_te[:1]
        infer_s = median_time(lambda: model.predict(sample), repeats=100)

        path = RESULTS / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_b = path.stat().st_size

        with PeakRSS() as m_train:
            make_models()[name].fit(X_tr, y_tr)
        with PeakRSS() as m_inf:
            for _ in range(100):
                model.predict(sample)

        cost_rows.append({
            "model": name,
            "train_time_ms": round(train_s * 1000, 2),
            "inference_latency_ms": round(infer_s * 1000, 4),
            "size_bytes": size_b,
            "size_KB": round(size_b / 1024, 2),
            "peak_rss_train_MB": round(m_train.peak / MB, 1),
            "peak_rss_infer_MB": round(m_inf.peak / MB, 1),
            "train_mem_delta_MB": round((m_train.peak - base_rss) / MB, 2),
        })

    # Manual asks for results/model.joblib: store the better model under that name.
    best = max(acc_rows, key=lambda r: r["test_accuracy"])["model"]
    joblib.dump(joblib.load(RESULTS / f"model_{best}.joblib"), RESULTS / "model.joblib")

    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(RESULTS / "baseline_accuracy.csv", index=False)
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(RESULTS / "system_cost.csv", index=False)

    # 4. budget table. Memory = peak process RSS (Python interpreter + libraries + model).
    budget_rows = []
    for r in cost_rows:
        mem = max(r["peak_rss_train_MB"], r["peak_rss_infer_MB"])
        row = {"model": r["model"]}
        for target, b in BUDGETS.items():
            row[target] = "yes" if fits(b, mem, r["inference_latency_ms"], r["size_KB"]) else "no"
        budget_rows.append(row)
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(RESULTS / "deployment_fit.csv", index=False)

    print("\n", acc_df.to_string(index=False))
    print("\n", cost_df.to_string(index=False))
    print("\n", budget_df.to_string(index=False))


if __name__ == "__main__":
    main()
