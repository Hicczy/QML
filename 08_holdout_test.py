"""
Strict hold-out test evaluation.

Pipeline per cell:
    Train = full 3000 MNIST samples (the same ones the CV experiments used)
    Test  = sealed 1000 MNIST samples, never seen during any prior fit/tuning

For each (encoding × model × seed × test_condition):
    1. Fit the model on (X_train, y_train) — train features are clean.
    2. Evaluate on (X_test, y_test).
       Two test_conditions:
         - clean      : test features unchanged
         - noisy_0.2  : Gaussian σ=0.2 added to test features at predict time
                        (the "trained carefully, deployed messy" scenario)

Resumable: writes one row per cell to results_holdout.csv. Reruns skip
already-done cells. Default per-call time budget = 35s.

    python 08_holdout_test.py
    python 08_holdout_test.py --report
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from simple_kan import RBFKAN


HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CACHE = os.path.join(HERE, "features_mnist.npz")
TEST_CACHE  = os.path.join(HERE, "features_mnist_test.npz")
OUT_CSV     = os.path.join(HERE, "results_holdout.csv")

ENCODINGS  = ["raw", "angle", "amplitude"]
MODELS     = ["LR", "MLP", "SVM", "KAN"]
SEEDS      = [0, 1, 2]
CONDITIONS = [("clean", 0.0), ("noisy_0.2", 0.2)]


def build_model(name: str, seed: int):
    if name == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                               random_state=seed),
        )
    if name == "MLP":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu", solver="adam",
                max_iter=300, early_stopping=True, random_state=seed,
            ),
        )
    if name == "SVM":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale",
                cache_size=500, random_state=seed),
        )
    if name == "KAN":
        return make_pipeline(
            StandardScaler(),
            RBFKAN(n_basis=8, sigma=0.5, C=1.0, seed=seed),
        )
    raise ValueError(name)


def add_noise(X, sigma, rng):
    if sigma <= 0.0:
        return X
    return X + rng.normal(0.0, sigma, size=X.shape).astype(X.dtype)


def load_features():
    with np.load(TRAIN_CACHE) as f:
        train = {k: f[k] for k in f.files}
    with np.load(TEST_CACHE) as f:
        test = {k: f[k] for k in f.files}
    feature_train = {
        "raw":       train["X_raw"],
        "angle":     train["X_angle"],
        "amplitude": train["X_amplitude"],
    }
    feature_test = {
        "raw":       test["X_raw_test"],
        "angle":     test["X_angle_test"],
        "amplitude": test["X_amplitude_test"],
    }
    return feature_train, train["y"], feature_test, test["y_test"]


def load_done():
    if not os.path.exists(OUT_CSV):
        return set()
    df = pd.read_csv(OUT_CSV)
    return set(zip(df["encoding"], df["model"], df["seed"], df["condition"]))


def append_row(row):
    df = pd.DataFrame([row])
    header = not os.path.exists(OUT_CSV)
    df.to_csv(OUT_CSV, mode="a", header=header, index=False)


def evaluate_one(X_tr, y_tr, X_te, y_te, model_name, seed, sigma):
    rng = np.random.default_rng(seed)
    X_te_eval = add_noise(X_te, sigma, rng)
    m = build_model(model_name, seed)
    m.fit(X_tr, y_tr)
    yp = m.predict(X_te_eval)
    return (float(accuracy_score(y_te, yp)),
            float(f1_score(y_te, yp, average="macro")))


def report():
    if not os.path.exists(OUT_CSV):
        print("no results yet"); return
    df = pd.read_csv(OUT_CSV)
    agg = df.groupby(["encoding", "model", "condition"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        n_seeds=("seed", "nunique"),
    )

    for cond, _ in CONDITIONS:
        sub = agg[agg["condition"] == cond]
        if sub.empty:
            continue
        pivot_acc = sub.pivot(index="encoding", columns="model", values="accuracy")
        pivot_f1  = sub.pivot(index="encoding", columns="model", values="f1")
        pivot_std = sub.pivot(index="encoding", columns="model", values="accuracy_std")
        e_order = [e for e in ENCODINGS if e in pivot_acc.index]
        m_order = [m for m in MODELS    if m in pivot_acc.columns]
        print(f"\n=== HOLD-OUT TEST: {cond} ===")
        print("\nAccuracy (mean over seeds):")
        print(pivot_acc.loc[e_order, m_order].round(3).to_string())
        print("\nAccuracy std across seeds:")
        print(pivot_std.loc[e_order, m_order].round(3).to_string())
        print("\nMacro-F1 (mean over seeds):")
        print(pivot_f1.loc[e_order, m_order].round(3).to_string())


def run(budget_s):
    feature_train, y_train, feature_test, y_test = load_features()
    print(f"Train: {y_train.shape[0]} samples, Test: {y_test.shape[0]} samples")

    cells = [
        (enc, model, seed, cond_name, sigma)
        for enc in ENCODINGS
        for model in MODELS
        for seed in SEEDS
        for cond_name, sigma in CONDITIONS
    ]
    # Order: fast models first
    order = {"LR": 0, "KAN": 1, "SVM": 2, "MLP": 3}
    cells.sort(key=lambda t: (order[t[1]], ENCODINGS.index(t[0]), t[2]))

    done = load_done()
    remaining = [c for c in cells
                 if (c[0], c[1], c[2], c[3]) not in done]
    print(f"{len(done)}/{len(cells)} done, {len(remaining)} remaining "
          f"(budget={budget_s}s)")

    t0 = time.time()
    while remaining and (time.time() - t0) < budget_s:
        enc, model, seed, cond_name, sigma = remaining.pop(0)
        cell_t = time.time()
        acc, f1 = evaluate_one(
            feature_train[enc], y_train,
            feature_test[enc],  y_test,
            model, seed, sigma,
        )
        dt = time.time() - cell_t
        row = {
            "encoding": enc, "model": model, "seed": seed,
            "condition": cond_name, "noise_sigma": sigma,
            "accuracy": acc, "f1_macro": f1, "seconds": dt,
            "n_train": feature_train[enc].shape[0],
            "n_test":  feature_test[enc].shape[0],
        }
        append_row(row)
        done.add((enc, model, seed, cond_name))
        print(f"  {enc:>9s} + {model:<3s} seed={seed} {cond_name:<10s} "
              f"acc={acc:.3f}  F1={f1:.3f}  ({dt:.1f}s)")

    print(f"\nTotal: {len(done)}/{len(cells)}")
    if remaining:
        print("More to do — rerun.")
    else:
        print("ALL CELLS DONE.")
        report()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--budget", type=float, default=35.0)
    args = ap.parse_args()
    if args.report:
        report()
    else:
        run(args.budget)


if __name__ == "__main__":
    main()
