"""Lab 3: architectures and their costs on Fashion-MNIST."""
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
import torch
from thop import profile
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms

from models import build_models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)
DATA = ROOT / "data"
BATCH = 128


def loaders():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full = datasets.FashionMNIST(root=DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root=DATA, train=False, download=True, transform=tf)
    train, val = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(SEED))
    return (DataLoader(train, BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED)),
            DataLoader(val, BATCH), DataLoader(test, BATCH))


# ---------- hand-calculated FLOPs (Ch. 4 formulas, 1 MAC = 2 FLOPs) ----------
def linear_flops(i, o):
    return 2 * i * o


def conv_flops(k, cin, cout, h, w):
    return 2 * k * k * cin * cout * h * w


def encoder_layer_flops(seq=16, d=64, ff=128):
    return {
        "QKV projections": 2 * seq * d * 3 * d,
        "attention scores QK^T": 2 * seq ** 2 * d,
        "scores x V": 2 * seq ** 2 * d,
        "output projection": 2 * seq * d * d,
        "feed-forward": seq * (linear_flops(d, ff) + linear_flops(ff, d)),
    }


def hand_flops():
    enc = encoder_layer_flops()
    rows = [
        ("MLP", "Linear 784->256", linear_flops(784, 256)),
        ("CNN", "Conv2d 16->32 3x3 @14x14", conv_flops(3, 16, 32, 14, 14)),
    ]
    rows += [("Attention", f"Encoder layer: {part}", f) for part, f in enc.items()]
    rows.append(("Attention", "Encoder layer: TOTAL", sum(enc.values())))
    totals = {
        "MLP": linear_flops(784, 256) + linear_flops(256, 128) + linear_flops(128, 10),
        "CNN": conv_flops(3, 1, 16, 28, 28) + conv_flops(3, 16, 32, 14, 14) + linear_flops(32 * 49, 10),
        "Attention": 16 * linear_flops(49, 64) + 2 * sum(enc.values()) + linear_flops(64, 10),
    }
    return pd.DataFrame(rows, columns=["model", "layer", "hand_FLOPs"]), totals


def thop_flops(model):
    macs, _ = profile(model, inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    return 2 * macs  # thop reports MACs


def latency_ms(model, batch, repeats):
    x = torch.randn(batch, 1, 28, 28)
    model.eval()
    with torch.inference_mode():
        model(x)  # warm-up
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def train(model, train_dl, val_dl, epochs=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    log = []
    for ep in range(epochs):
        model.train()
        t0, running = time.perf_counter(), 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        val = evaluate(model, val_dl)
        log.append({"epoch": ep + 1, "train_loss": running / 55000, "val_acc": val,
                    "epoch_time_s": time.perf_counter() - t0})
        print(f"  epoch {ep + 1}: loss {log[-1]['train_loss']:.4f} val {val:.4f}")
    return log


def main():
    subprocess.run([sys.executable, str(HERE / "print_versions.py"), str(RESULTS / "versions.txt")], check=True)
    train_dl, val_dl, test_dl = loaders()
    models = build_models()

    layer_df, hand_totals = hand_flops()
    layer_df.to_csv(RESULTS / "hand_flops_layers.csv", index=False)

    rows, logs = [], []
    with open(RESULTS / "torchinfo_summary.txt", "w", encoding="utf-8") as f:
        for name, model in models.items():
            s = summary(model, input_size=(1, 1, 28, 28), verbose=0)
            f.write(f"===== {name} =====\n{s}\n\n")
            rows.append({
                "model": name,
                "params": s.total_params,
                "hand_FLOPs": hand_totals[name],
                "thop_FLOPs": thop_flops(build_models()[name]),
                "latency_b1_ms": latency_ms(model, 1, 100),
                "latency_b128_ms": latency_ms(model, 128, 20),
            })

    for r in rows:
        name = r["model"]
        print(f"Training {name}")
        torch.manual_seed(SEED)
        for ep in train(models[name], train_dl, val_dl):
            logs.append({"model": name, **ep})
        r["test_accuracy"] = evaluate(models[name], test_dl)

    df = pd.DataFrame(rows)
    df["thop/hand"] = (df["thop_FLOPs"] / df["hand_FLOPs"]).round(3)
    df = df.round({"latency_b1_ms": 3, "latency_b128_ms": 3, "test_accuracy": 4})
    df.to_csv(RESULTS / "architecture_costs.csv", index=False)
    pd.DataFrame(logs).round(4).to_csv(RESULTS / "training_log.csv", index=False)
    print(layer_df.to_string(index=False))
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, label in zip(axes, ["params", "latency_b1_ms", "latency_b128_ms"],
                              ["parameters", "latency, batch 1 (ms)", "latency, batch 128 (ms)"]):
        ax.scatter(df[col], df["test_accuracy"], s=60)
        for _, r in df.iterrows():
            ax.annotate(r["model"], (r[col], r["test_accuracy"]), textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel(label)
        ax.set_ylabel("test accuracy")
        ax.grid(alpha=0.3)
    fig.suptitle("Fashion-MNIST: accuracy vs cost (5 epochs, Adam 1e-3)")
    fig.tight_layout()
    fig.savefig(RESULTS / "tradeoff_plots.png", dpi=120)


if __name__ == "__main__":
    main()
