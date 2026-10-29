"""Two-hidden-layer MLP in NumPy: forward, cross-entropy, backward."""
import numpy as np


def init_params(width, rng, n_in=64, n_out=10, std=0.1):
    """N(0, 0.01) read as mean 0, variance 0.01, i.e. std 0.1."""
    return {
        "W1": rng.normal(0, std, (n_in, width)), "b1": np.zeros(width),
        "W2": rng.normal(0, std, (width, width)), "b2": np.zeros(width),
        "W3": rng.normal(0, std, (width, n_out)), "b3": np.zeros(n_out),
    }


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.maximum(z1, 0)
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = np.maximum(z2, 0)
    z3 = a2 @ p["W3"] + p["b3"]
    probs = softmax(z3)
    cache = (X, z1, a1, z2, a2)
    return probs, cache


def cross_entropy(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))


def backward(p, probs, Y, cache):
    """Chain rule. With softmax + CE, dL/dz3 = (probs - Y) / N."""
    X, z1, a1, z2, a2 = cache
    n = X.shape[0]
    dz3 = (probs - Y) / n
    g = {"W3": a2.T @ dz3, "b3": dz3.sum(0)}
    da2 = dz3 @ p["W3"].T
    dz2 = da2 * (z2 > 0)
    g["W2"], g["b2"] = a1.T @ dz2, dz2.sum(0)
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (z1 > 0)
    g["W1"], g["b1"] = X.T @ dz1, dz1.sum(0)
    return g


def loss_fn(p, X, Y):
    return cross_entropy(forward(p, X)[0], Y)


def accuracy(p, X, y):
    return float(np.mean(forward(p, X)[0].argmax(1) == y))
