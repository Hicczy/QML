"""
Pull a fresh hold-out test set from the unused MNIST samples and encode it.

Guarantees the test indices are DISJOINT from the train indices that
precompute_mnist.py picked. (Same RandomState(42) replay → identical train
selection → we know exactly which 3000 to avoid.)

Saves: features_mnist_test.npz with keys
    X_raw_test         (N_TEST, 16)         downsampled 4x4 pixels
    X_amplitude_test   (N_TEST, 4)          amplitude encoding output
    X_angle_test       (N_TEST, 16)         angle encoding output
    y_test             (N_TEST,)            ground-truth labels
    train_indices      (3000,)              for traceability
    test_indices       (N_TEST,)            for traceability
    X_angle_test_done  scalar               chunk-progress sentinel

Angle encoding is chunked + resumable. Rerun until ALL DONE.

    python precompute_mnist_test.py
"""

import os
import sys
import time
import numpy as np
from quantum_encoding import encode_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(HERE, "mnist_train.csv")
TEST_CACHE = os.path.join(HERE, "features_mnist_test.npz")
TRAIN_CACHE = os.path.join(HERE, "features_mnist.npz")

N_TEST_PER_CLASS = 100   # 1000 total
TRAIN_PER_CLASS  = 300   # same as precompute_mnist.py
SEED_TRAIN       = 42    # must match precompute_mnist.py
SEED_TEST        = 123
ANGLE_CHUNK      = 400


def select_indices(y_full):
    """Reproduce the train selection and pick disjoint test indices."""
    rng_train = np.random.RandomState(SEED_TRAIN)
    rng_test  = np.random.RandomState(SEED_TEST)

    train_idx_pieces = []
    test_idx_pieces  = []
    for c in range(10):
        cls_idx = np.where(y_full == c)[0].copy()
        # Train selection: same shuffle as precompute_mnist.py
        rng_train.shuffle(cls_idx)
        train_for_class = cls_idx[:TRAIN_PER_CLASS]
        train_idx_pieces.append(train_for_class)

        # Test selection: from the REMAINING samples, shuffle and take first N_TEST_PER_CLASS
        remaining = cls_idx[TRAIN_PER_CLASS:].copy()
        rng_test.shuffle(remaining)
        test_for_class = remaining[:N_TEST_PER_CLASS]
        test_idx_pieces.append(test_for_class)

    train_idx = np.concatenate(train_idx_pieces)
    test_idx  = np.concatenate(test_idx_pieces)
    # Same shuffle order as train (mirrors precompute_mnist.py)
    rng_train.shuffle(train_idx)
    rng_test.shuffle(test_idx)

    # Sanity check
    assert len(set(train_idx) & set(test_idx)) == 0, "train/test overlap!"
    return train_idx, test_idx


def load_and_downsample(csv_path):
    print(f"loading {csv_path} ...")
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.uint8)
    y_full = raw[:, 0]
    X_full = raw[:, 1:].astype(np.float32) / 255.0
    print(f"  loaded {X_full.shape[0]} samples")
    return X_full, y_full


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"missing CSV: {CSV_PATH}")

    cache = {}
    if os.path.exists(TEST_CACHE):
        with np.load(TEST_CACHE) as f:
            cache = {k: f[k] for k in f.files}

    # ---- Train/test index selection (cheap, redo every call) ----
    X_full, y_full = load_and_downsample(CSV_PATH)
    train_idx, test_idx = select_indices(y_full)
    print(f"  selected: train={len(train_idx)}, test={len(test_idx)}, "
          f"overlap={len(set(train_idx) & set(test_idx))}")

    # ---- Build the test set ----
    if "X_raw_test" not in cache or cache["X_raw_test"].shape[0] != len(test_idx):
        X_test_28 = X_full[test_idx]                                  # (N_TEST, 784)
        # 28x28 -> 4x4 by 7x7 average pooling
        X_test_4 = X_test_28.reshape(-1, 4, 7, 4, 7).mean(axis=(2, 4))
        X_test_4 = X_test_4.reshape(-1, 16)                           # (N_TEST, 16)
        y_test   = y_full[test_idx]

        cache["X_raw_test"] = X_test_4
        cache["y_test"] = y_test
        cache["train_indices"] = train_idx
        cache["test_indices"]  = test_idx
        np.savez(TEST_CACHE, **cache)
        print(f"  saved raw test features: shape {X_test_4.shape}")

    X_raw_test = cache["X_raw_test"]
    N_TEST = X_raw_test.shape[0]
    y_test = cache["y_test"]

    # ---- Amplitude encoding (fast) ----
    if "X_amplitude_test" not in cache or cache["X_amplitude_test"].shape[0] != N_TEST:
        print("encoding amplitude (test) ...")
        t0 = time.time()
        cache["X_amplitude_test"] = encode_dataset(X_raw_test, kind="amplitude",
                                                  verbose=False)
        print(f"  done in {time.time() - t0:.1f}s, "
              f"shape {cache['X_amplitude_test'].shape}")
        np.savez(TEST_CACHE, **cache)
    else:
        print(f"amplitude already done, shape {cache['X_amplitude_test'].shape}")

    # ---- Angle encoding (chunked + resumable) ----
    if "X_angle_test" not in cache or cache["X_angle_test"].shape != (N_TEST, 16):
        cache["X_angle_test"] = np.full((N_TEST, 16), np.nan, dtype=np.float64)
        cache["X_angle_test_done"] = np.array(0)

    start = int(cache["X_angle_test_done"])
    if start >= N_TEST:
        print(f"angle encoding (test) already complete ({N_TEST}/{N_TEST})")
        print("ALL DONE.")
        return

    end = min(start + ANGLE_CHUNK, N_TEST)
    print(f"encoding angle (test) [{start}:{end}] of {N_TEST} ...")
    t0 = time.time()
    cache["X_angle_test"][start:end] = encode_dataset(
        X_raw_test[start:end], kind="angle", verbose=True
    )
    cache["X_angle_test_done"] = np.array(end)
    np.savez(TEST_CACHE, **cache)
    print(f"  done {end}/{N_TEST}  in {time.time() - t0:.1f}s")

    if end < N_TEST:
        print(f"  more to do: rerun ({N_TEST - end} remaining)")
    else:
        print("ALL DONE.")


if __name__ == "__main__":
    main()
