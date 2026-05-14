"""
Comparison study: 2 encodings × 3 classifier heads.

    Encodings:  angle (qubit-as-pixel), amplitude (flatten)
    Models:     MLP, SVM, KAN (RBF-basis single-layer)

Pipeline per cell:
    8x8 digit image  →  4x4 (avg-pool)  →  quantum encode + CNOT ring + measure
                                       →  feature vector  →  classifier  →  prediction

Run:  python 05_compare_models.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from quantum_encoding import encode_dataset
from simple_kan import RBFKAN


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
DOWNSAMPLE_TO = (4, 4)        # 8x8 -> 4x4
SUBSAMPLE = None              # set to an int for a quick run; None uses all 1797


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_and_downsample():
    """Load sklearn digits, normalize to [0,1], avg-pool 8x8 → 4x4."""
    data = load_digits()
    X = data.images.astype(float) / 16.0      # raw range is 0..16
    y = data.target

    # 2x2 average pooling to get 4x4
    h, w = DOWNSAMPLE_TO
    H, W = X.shape[1], X.shape[2]
    rh, rw = H // h, W // w
    X = X.reshape(-1, h, rh, w, rw).mean(axis=(2, 4))   # (N, 4, 4)
    X = X.reshape(-1, h * w)                            # (N, 16)
    return X, y


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def build_models(seed: int):
    """Return a dict of name → sklearn-compatible classifier (un-fit)."""
    return {
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=5000,
                early_stopping=True,
                random_state=seed,
            ),
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed),
        ),
        "KAN": make_pipeline(
            StandardScaler(),
            RBFKAN(n_basis=8, sigma=0.5, C=1.0, seed=seed),
        ),
    }


# ----------------------------------------------------------------------
# Cross-validated evaluation
# ----------------------------------------------------------------------
def cv_eval(features: np.ndarray, y: np.ndarray, model_factory, seed: int):
    """Run stratified K-fold CV. Returns mean accuracy and mean macro-F1."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for train_idx, test_idx in skf.split(features, y):
        model = model_factory(seed)
        model.fit(features[train_idx], y[train_idx])
        y_pred = model.predict(features[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), \
           float(np.mean(f1s)),  float(np.std(f1s))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    np.random.seed(SEED)

    print("=" * 72)
    print("QML model comparison: 2 encodings × 3 classifiers")
    print("=" * 72)

    print("\nLoading cached features ...")
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")
    if not os.path.exists(cache_path):
        print(f"  cache not found: {cache_path}")
        print("  Run `python precompute_features.py --encoding amplitude`")
        print("  Then `python precompute_angle.py` (repeat until done)")
        sys.exit(1)
    with np.load(cache_path) as f:
        cached = {k: f[k] for k in f.files}
    X, y = cached["X_raw"], cached["y"]
    if SUBSAMPLE is not None:
        idx = np.random.RandomState(SEED).choice(len(X), SUBSAMPLE, replace=False)
        X, y = X[idx], y[idx]
        angle_feat = cached["X_angle"][idx]
        amp_feat   = cached["X_amplitude"][idx]
    else:
        angle_feat = cached["X_angle"]
        amp_feat   = cached["X_amplitude"]
    print(f"  X shape: {X.shape}, classes: {sorted(set(y))}, samples: {len(X)}")

    feature_sets = {
        "raw":       X.copy(),
        "angle":     angle_feat,
        "amplitude": amp_feat,
    }
    for k, v in feature_sets.items():
        print(f"  {k:<10s} features shape: {v.shape}")

    # ------- 2. Loop over (encoding, model) and evaluate -------
    rows = []
    model_names = list(build_models(SEED).keys())

    for enc_name, features in feature_sets.items():
        for model_name in model_names:
            def factory(seed, name=model_name):
                return build_models(seed)[name]
            t0 = time.time()
            acc_m, acc_s, f1_m, f1_s = cv_eval(features, y, factory, SEED)
            dt = time.time() - t0
            print(f"  {enc_name:>9s} + {model_name:<3s}  "
                  f"acc={acc_m:.3f}±{acc_s:.3f}  "
                  f"F1={f1_m:.3f}±{f1_s:.3f}  ({dt:.1f}s)")
            rows.append({
                "encoding": enc_name,
                "model": model_name,
                "n_features": features.shape[1],
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "f1_macro_mean": f1_m,
                "f1_macro_std": f1_s,
                "seconds": dt,
            })

    # ------- 3. Save & print results -------
    df = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 72)
    print("Results")
    print("=" * 72)
    pivot_acc = df.pivot(index="encoding", columns="model", values="accuracy_mean")
    pivot_f1  = df.pivot(index="encoding", columns="model", values="f1_macro_mean")
    pivot_acc = pivot_acc.reindex(["raw", "angle", "amplitude"])
    pivot_f1  = pivot_f1.reindex(["raw", "angle", "amplitude"])

    print("\nMean accuracy (5-fold CV):")
    print(pivot_acc.round(3).to_string())
    print("\nMean macro-F1 (5-fold CV):")
    print(pivot_f1.round(3).to_string())
    print(f"\nFull results saved to: {out_csv}")


if __name__ == "__main__":
    main()
