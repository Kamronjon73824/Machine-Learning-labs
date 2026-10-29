"""Lab 2: neural network from scratch on load_digits."""
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nn import accuracy, backward, cross_entropy, forward, init_params, loss_fn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


def load_data():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    return X_tr, X_te, y_tr, y_te, np.eye(10)[y_tr], np.eye(10)[y_te]


def sanity_checks(X, Y, log):
    rng = np.random.default_rng(SEED)
    for w in (16, 64):
        p = init_params(w, rng)
        probs, _ = forward(p, X)
        row_ok = np.allclose(probs.sum(1), 1, atol=1e-6)
        loss = cross_entropy(probs, Y)
        log.append(f"W={w}: output shape {probs.shape}, rows sum to 1: {row_ok}, "
                   f"untrained loss {loss:.4f} (ln10 = {np.log(10):.4f})")
        assert probs.shape == (X.shape[0], 10) and row_ok


def gradient_check(X, Y, width=16, n_weights=10, eps=1e-5):
    """Central differences on 10 random weights. Uses float64 and a small batch."""
    rng = np.random.default_rng(SEED)
    p = init_params(width, rng)
    Xb, Yb = X[:20], Y[:20]
    probs, cache = forward(p, Xb)
    g = backward(p, probs, Yb, cache)
    names = ["W1", "W2", "W3"]
    rows = []
    for _ in range(n_weights):
        name = names[rng.integers(3)]
        idx = tuple(int(rng.integers(s)) for s in p[name].shape)
        orig = p[name][idx]
        p[name][idx] = orig + eps
        lp = loss_fn(p, Xb, Yb)
        p[name][idx] = orig - eps
        lm = loss_fn(p, Xb, Yb)
        p[name][idx] = orig
        num = (lp - lm) / (2 * eps)
        ana = g[name][idx]
        rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-12)
        rows.append({"param": name, "index": str(idx), "analytic": ana, "numerical": num,
                     "rel_error": rel, "pass": rel < 1e-6})
    return pd.DataFrame(rows)


def train(X_tr, Y_tr, X_te, y_te, width, lr, epochs=50, batch=32):
    rng = np.random.default_rng(SEED)
    p = init_params(width, rng)
    losses = []
    n = X_tr.shape[0]
    for ep in range(epochs):
        perm = rng.permutation(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            probs, cache = forward(p, X_tr[idx])
            ep_loss += cross_entropy(probs, Y_tr[idx]) * len(idx)
            g = backward(p, probs, Y_tr[idx], cache)
            for k in p:
                p[k] -= lr * g[k]
        losses.append(ep_loss / n)
        print(f"  W={width} lr={lr} epoch {ep + 1:2d} loss {losses[-1]:.4f}")
    return p, losses, accuracy(p, X_te, y_te)


def time_passes(X, Y, width, repeats=100):
    p = init_params(width, np.random.default_rng(SEED))
    Xb, Yb = X[:32], Y[:32]
    fwd, bwd = [], []
    forward(p, Xb)  # warm-up
    for _ in range(repeats):
        t0 = time.perf_counter()
        probs, cache = forward(p, Xb)
        t1 = time.perf_counter()
        backward(p, probs, Yb, cache)
        t2 = time.perf_counter()
        fwd.append(t1 - t0)
        bwd.append(t2 - t1)
    return statistics.median(fwd) * 1e6, statistics.median(bwd) * 1e6


def main():
    subprocess.run([sys.executable, str(HERE / "print_versions.py"), str(RESULTS / "versions.txt")], check=True)
    X_tr, X_te, y_tr, y_te, Y_tr, Y_te = load_data()

    log = []
    sanity_checks(X_tr, Y_tr, log)
    (RESULTS / "sanity_checks.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))

    gc = gradient_check(X_tr, Y_tr)
    gc.to_csv(RESULTS / "gradient_check.csv", index=False)
    print(gc.to_string(index=False))
    print(f"Gradient check passed for {gc['pass'].sum()}/10 weights")

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, width in zip(axes, (16, 64)):
        for lr in (0.001, 0.01, 0.1):
            _, losses, acc = train(X_tr, Y_tr, X_te, y_te, width, lr)
            rows.append({"width": width, "lr": lr, "final_train_loss": round(losses[-1], 4),
                         "test_accuracy": round(acc, 4)})
            ax.plot(range(1, 51), losses, label=f"lr={lr}")
        ax.set_title(f"Hidden width W={width}")
        ax.set_xlabel("epoch")
        ax.legend()
    axes[0].set_ylabel("train cross-entropy")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=120)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "training_results.csv", index=False)
    print(res.to_string(index=False))

    trows = []
    for width in (16, 64):
        f_us, b_us = time_passes(X_tr, Y_tr, width)
        trows.append({"width": width, "batch": 32, "forward_us": round(f_us, 1),
                      "backward_us": round(b_us, 1), "backward/forward": round(b_us / f_us, 2)})
    tdf = pd.DataFrame(trows)
    tdf.to_csv(RESULTS / "timing.csv", index=False)
    print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()
