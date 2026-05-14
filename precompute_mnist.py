"""
MNIST loader + chunked precompute, resumable.

Loads mnist_train.csv (Kaggle 'Digit Recognizer' format: label col + 784 pixel cols),
stratified-subsamples to N_SAMPLES, downsamples 28x28 -> 4x4 via 7x7 avg-pool,
and runs both quantum encodings. Saves to features_mnist.npz.

Why 4x4: angle encoding uses one qubit per pixel; 16 qubits is comfortably
simulable. 28x28 = 784 qubits is not. So we lose information by downsampling
— that's the point of the MNIST-at-low-resolution stress test.

Angle encoding for 3000 samples × 16 qubits doesn't fit in a single 45s bash
window, so it's chunked. Rerun until the script reports it's done:

    python precompute_mnist.py            # auto-resume
"""

import os
import sys
import time
import numpy as np

from quantum_encoding import encode_dataset

CSV_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_train.csv")
CACHE_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features_mnist.npz")
N_SAMPLES   = 3000
SEED        = 42
ANGLE_CHUNK = 700


def load_and_downsample():
    """Load Kaggle MNIST CSV, stratified sample, downsample to 4x4."""
    print(f"loading {CSV_PATH} ...")
    raw = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1, dtype=np.uint8)
    y_full = raw[:, 0]
    X_full = raw[:, 1:].astype(np.float32) / 255.0       # (N, 784) in [0,1]
    print(f"  loaded {X_full.shape[0]} samples")

    rng = np.random.RandomState(SEED)
    per_class = N_SAMPLES // 10
    idx = []
    for c in range(10):
        cls_idx = np.where(y_full == c)[0]
        rng.shuffle(cls_idx)
        idx.append(cls_idx[:per_class])
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    X = X_full[idx]                                       # (N_SAMPLES, 784)
    y = y_full[idx]
    print(f"  subsampled to {X.shape[0]} (per-class={per_class})")

    # 28x28 -> 4x4 via 7x7 average pooling
    X = X.reshape(-1, 4, 7, 4, 7).mean(axis=(2, 4))       # (N, 4, 4)
    X = X.reshape(-1, 16)                                 # (N, 16)
    return X, y


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"missing CSV: {CSV_PATH}")

    cache = {}
    if os.path.exists(CACHE_PATH):
        with np.load(CACHE_PATH) as f:
            cache = {k: f[k] for k in f.files}

    if "X_raw" not in cache or cache["X_raw"].shape != (N_SAMPLES, 16):
        X, y = load_and_downsample()
        cache["X_raw"] = X
        cache["y"] = y
        np.savez(CACHE_PATH, **cache)
        print(f"saved raw to {CACHE_PATH}")
    else:
        X = cache["X_raw"]
        y = cache["y"]
        print(f"using cached raw X={X.shape}, y={y.shape}")

    # -------- Amplitude (fast, one shot) --------
    if "X_amplitude" not in cache or cache["X_amplitude"].shape[0] != len(X):
        print("encoding amplitude ...")
        t0 = time.time()
        cache["X_amplitude"] = encode_dataset(X, kind="amplitude", verbose=False)
        print(f"  done in {time.time() - t0:.1f}s, shape {cache['X_amplitude'].shape}")
        np.savez(CACHE_PATH, **cache)
    else:
        print(f"amplitude already cached, shape {cache['X_amplitude'].shape}")

    # -------- Angle (chunked, resumable) --------
    N = len(X)
    if "X_angle" not in cache or cache["X_angle"].shape != (N, 16):
        cache["X_angle"] = np.full((N, 16), np.nan, dtype=np.float64)
        cache["X_angle_done"] = np.array(0)

    start = int(cache["X_angle_done"])
    if start >= N:
        print(f"angle encoding already complete ({N}/{N})")
        print("ALL DONE.")
        return

    end = min(start + ANGLE_CHUNK, N)
    print(f"encoding angle [{start}:{end}] of {N} ...")
    t0 = time.time()
    cache["X_angle"][start:end] = encode_dataset(X[start:end], kind="angle", verbose=True)
    cache["X_angle_done"] = np.array(end)
    np.savez(CACHE_PATH, **cache)
    print(f"  done {end}/{N} in {time.time() - t0:.1f}s")

    if end < N:
        print(f"  more to do: rerun this script ({N - end} remaining)")
    else:
        print("ALL DONE.")


if __name__ == "__main__":
    main()
