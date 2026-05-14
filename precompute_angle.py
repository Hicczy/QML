"""
Chunked + resumable angle encoding. Each invocation processes up to CHUNK
samples and saves; rerun until done.

    python precompute_angle.py
"""
import os
import sys
import time
import numpy as np
from sklearn.datasets import load_digits

from quantum_encoding import encode_dataset

CHUNK = 700
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")


def load_X():
    data = load_digits()
    X = data.images.astype(float) / 16.0
    X = X.reshape(-1, 4, 2, 4, 2).mean(axis=(2, 4)).reshape(-1, 16)
    return X, data.target


def main():
    X, y = load_X()
    N = X.shape[0]

    cache = {}
    if os.path.exists(CACHE_PATH):
        with np.load(CACHE_PATH) as f:
            cache = {k: f[k] for k in f.files}

    cache["X_raw"] = X
    cache["y"] = y

    existing = cache.get("X_angle")
    if existing is None or existing.shape[0] != N:
        # initialize a partially-filled array; use NaN as sentinel
        cache["X_angle"] = np.full((N, X.shape[1]), np.nan)
        cache["X_angle_done"] = np.array(0)

    start = int(cache["X_angle_done"])
    if start >= N:
        print(f"angle encoding already complete ({N}/{N})")
        return

    end = min(start + CHUNK, N)
    print(f"encoding angle [{start}:{end}] of {N} ...")
    t0 = time.time()
    cache["X_angle"][start:end] = encode_dataset(X[start:end], kind="angle", verbose=True)
    cache["X_angle_done"] = np.array(end)
    np.savez(CACHE_PATH, **cache)
    print(f"  done {end}/{N}  in {time.time() - t0:.1f}s, cache saved")

    if end < N:
        print(f"more to do: rerun this script ({N - end} samples remaining)")


if __name__ == "__main__":
    main()
