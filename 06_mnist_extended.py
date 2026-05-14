"""
Extended MNIST comparison — resumable, multi-seed, with optional input noise.

For each (encoding × model × seed) cell, runs 5-fold stratified CV on the
MNIST features cached by precompute_mnist.py. Writes one row per cell to
results_mnist_{tag}.csv. Reruns skip already-done cells.

The previous sklearn-digits results in results.csv are preserved.

Usage:
    # Clean (no noise) — call repeatedly until done
    python 06_mnist_extended.py --noise 0.0

    # Noisy variant — call repeatedly until done
    python 06_mnist_extended.py --noise 0.2

    # Summary table only, no compute
    python 06_mnist_extended.py --noise 0.0 --report

    # Time budget per call (default 35s, leaves headroom under the 45s bash cap)
    python 06_mnist_extended.py --noise 0.0 --budget 35
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from simple_kan import RBFKAN


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "features_mnist.npz")

ENCODINGS = ["raw", "angle", "amplitude"]
MODELS    = ["LR", "MLP", "SVM", "KAN"]
SEEDS     = [0, 1, 2]
N_FOLDS   = 5


def build_model(name: str, seed: int):
    if name == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=seed,
                               solver="lbfgs"),
        )
    if name == "MLP":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu", solver="adam",
                max_iter=300, early_stopping=True,
                random_state=seed,
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


def cv_eval(features, y, model_name, seed, noise_sigma):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for train_idx, test_idx in skf.split(features, y):
        X_tr = add_noise(features[train_idx], noise_sigma, rng)
        X_te = add_noise(features[test_idx],  noise_sigma, rng)
        m = build_model(model_name, seed)
        m.fit(X_tr, y[train_idx])
        yp = m.predict(X_te)
        accs.append(accuracy_score(y[test_idx], yp))
        f1s.append(f1_score(y[test_idx], yp, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), \
           float(np.mean(f1s)),  float(np.std(f1s))


def output_path(noise_sigma):
    tag = "clean" if noise_sigma == 0.0 else f"noisy_{noise_sigma:g}"
    return os.path.join(HERE, f"results_mnist_{tag}.csv"), tag


def load_done(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame(), set()
    df = pd.read_csv(csv_path)
    done = set(zip(df["encoding"], df["model"], df["seed"]))
    return df, done


def append_row(csv_path, row):
    df_new = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df_new.to_csv(csv_path, mode="a", header=header, index=False)


def summary(noise_sigma):
    csv_path, tag = output_path(noise_sigma)
    if not os.path.exists(csv_path):
        print(f"no results yet at {csv_path}")
        return
    df = pd.read_csv(csv_path)
    # Aggregate over seeds: mean ± std across seeds (each row already
    # is fold-averaged for one seed)
    agg = df.groupby(["encoding", "model"]).agg(
        accuracy=("accuracy_mean", "mean"),
        accuracy_std=("accuracy_mean", "std"),
        f1=("f1_macro_mean", "mean"),
        f1_std=("f1_macro_mean", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    print(f"\n=== MNIST results ({tag}, {len(df)} cells) ===")
    pivot_acc = agg.pivot(index="encoding", columns="model", values="accuracy")
    pivot_f1  = agg.pivot(index="encoding", columns="model", values="f1")
    order_e = [e for e in ENCODINGS if e in pivot_acc.index]
    order_m = [m for m in MODELS    if m in pivot_acc.columns]
    print("\nMean accuracy over seeds:")
    print(pivot_acc.loc[order_e, order_m].round(3).to_string())
    print("\nMean macro-F1 over seeds:")
    print(pivot_f1.loc[order_e, order_m].round(3).to_string())


def run(noise_sigma, budget_s):
    if not os.path.exists(CACHE_PATH):
        sys.exit(f"missing cache: {CACHE_PATH}\nRun precompute_mnist.py first.")
    with np.load(CACHE_PATH) as f:
        cache = {k: f[k] for k in f.files}
    feature_sets = {e: cache[f"X_{e}"] for e in ENCODINGS}
    y = cache["y"]

    csv_path, tag = output_path(noise_sigma)
    _, done = load_done(csv_path)

    # Order so faster cells go first (LR, KAN, SVM, MLP)
    order = {"LR": 0, "KAN": 1, "SVM": 2, "MLP": 3}
    cells = [(e, m, s) for e in ENCODINGS for m in MODELS for s in SEEDS]
    cells.sort(key=lambda t: (order[t[1]], ENCODINGS.index(t[0]), t[2]))

    remaining = [c for c in cells if c not in done]
    total = len(cells)
    print(f"MNIST {tag}: {len(done)}/{total} done, "
          f"{len(remaining)} remaining   (budget={budget_s}s)")

    t_start = time.time()
    while remaining and (time.time() - t_start) < budget_s:
        enc, model, seed = remaining.pop(0)
        t0 = time.time()
        acc_m, acc_s, f1_m, f1_s = cv_eval(
            feature_sets[enc], y, model, seed, noise_sigma
        )
        dt = time.time() - t0
        row = {
            "encoding": enc, "model": model, "seed": seed,
            "n_features": feature_sets[enc].shape[1],
            "noise_sigma": noise_sigma,
            "accuracy_mean": acc_m, "accuracy_std": acc_s,
            "f1_macro_mean": f1_m,  "f1_macro_std": f1_s,
            "seconds": dt,
        }
        append_row(csv_path, row)
        done.add((enc, model, seed))
        print(f"  {enc:>9s} + {model:<3s} seed={seed}  "
              f"acc={acc_m:.3f}  F1={f1_m:.3f}  ({dt:.1f}s)")

    print(f"\nCells done this call: {len(done) - (total - len(remaining)) + len(remaining)}")
    print(f"Total: {len(done)}/{total}")
    if remaining:
        print(f"More to do — rerun this script.")
    else:
        print("ALL CELLS DONE.")
        summary(noise_sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--budget", type=float, default=35.0,
                    help="seconds of compute before yielding back to caller")
    ap.add_argument("--report", action="store_true",
                    help="print summary table only, no compute")
    args = ap.parse_args()
    if args.report:
        summary(args.noise)
    else:
        run(args.noise, args.budget)


if __name__ == "__main__":
    main()
            "noise_sigma": noise_sigma,
            "accuracy_mean": acc_m, "accuracy_std": acc_s,
            "f1_macro_mean": f1_m,  "f1_macro_std": f1_s,
            "seconds": dt,
        }
        append_row(csv_path, row)
        done.add((enc, model, seed))
        print(f"  {enc:>9s} + {model:<3s} seed={seed}  "
              f"acc={acc_m:.3f}  F1={f1_m:.3f}  ({dt:.1f}s)")

    print(f"\nTotal: {len(done)}/{total}")
    if remaining:
        print("More to do — rerun this script.")
    else:
        print("ALL CELLS DONE.")
        summary(noise_sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--budget", type=float, default=35.0,
                    help="seconds of compute before yielding back to caller")
    ap.add_argument("--report", action="store_true",
                    help="print summary table only, no compute")
    args = ap.parse_args()
    if args.report:
        summary(args.noise)
    else:
        run(args.noise, args.budget)


if __name__ == "__main__":
    main()
oise)
    else:
        run(args.noise, args.budget)


if __name__ == "__main__":
    main()
