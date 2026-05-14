"""
One-shot feature precomputation. Encodes the whole digits dataset (4x4 form)
under both quantum encodings and caches the results to features.npz.

After this runs, 05_compare_models.py can reuse the cached features instantly.

Usage:
    python precompute_features.py [--encoding angle|amplitude|both]
"""
import os
import sys
import time
import argparse
import numpy as np
from sklearn.datasets import load_digits

from quantum_encoding import encode_dataset


CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")


def load_and_downsample():
    data = load_digits()
    X = data.images.astype(float) / 16.0
    X = X.reshape(-1, 4, 2, 4, 2).mean(axis=(2, 4)).reshape(-1, 16)
    return X, data.target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", choices=["angle", "amplitude", "both"], default="both")
    args = ap.parse_args()

    X, y = load_and_downsample()
    print(f"dataset: X={X.shape}, y={y.shape}")

    # Load existing cache if present so we can incrementally add encodings.
    cache = {}
    if os.path.exists(CACHE_PATH):
        with np.load(CACHE_PATH) as f:
            cache = {k: f[k] for k in f.files}
        print(f"loaded cache with keys: {list(cache.keys())}")

    cache["X_raw"] = X
    cache["y"] = y

    todo = ["angle", "amplitude"] if args.encoding == "both" else [args.encoding]
    for enc in todo:
        key = f"X_{enc}"
        if key in cache and cache[key].shape[0] == X.shape[0]:
            print(f"[{enc}] already cached -- skipping")
            continue
        print(f"[{enc}] encoding {X.shape[0]} samples ...")
        t0 = time.time()
        cache[key] = encode_dataset(X, kind=enc, verbose=True)
        print(f"[{enc}] done in {time.time() - t0:.1f}s, shape {cache[key].shape}")
        # Save incrementally so partial progress is durable.
        np.savez(CACHE_PATH, **cache)
        print(f"[{enc}] cache saved -> {CACHE_PATH}")

    print("all done.")


if __name__ == "__main__":
    main()
