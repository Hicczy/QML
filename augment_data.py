"""
Data augmentation for the comparison study.

We can't reach MNIST from this environment (openml.org egress blocked),
so we triple the sklearn digits dataset with classic image augmentations:
small rotations, small translations, and a touch of Gaussian noise.

Original: 1797 samples (8x8 grayscale, 10 classes).
After 3x: 5391 samples.

Augmentations happen on the 8x8 original. Downsampling to 4x4 happens
later inside precompute_features.

    python augment_data.py
"""
import os
import numpy as np
from scipy.ndimage import rotate as nd_rotate, shift as nd_shift
from sklearn.datasets import load_digits

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_augmented.npz")
N_AUG = 2          # 1 original + N_AUG augmented copies → (N_AUG + 1)x dataset
SEED  = 42


def augment_one(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a randomized rotation + shift + noise to a single 8x8 image."""
    angle = rng.uniform(-10, 10)                          # degrees
    rotated = nd_rotate(img, angle, reshape=False,
                        mode="constant", cval=0.0, order=1)
    sh = rng.uniform(-1.0, 1.0, size=2)                   # pixels (sub-px allowed)
    shifted = nd_shift(rotated, sh, mode="constant", cval=0.0, order=1)
    noise = rng.normal(0.0, 0.02, size=shifted.shape)
    noisy = shifted + noise
    return np.clip(noisy, 0.0, 1.0)


def main():
    data = load_digits()
    X = data.images.astype(float) / 16.0          # (1797, 8, 8) in [0, 1]
    y = data.target

    rng = np.random.default_rng(SEED)
    pieces_X = [X.copy()]
    pieces_y = [y.copy()]

    for k in range(N_AUG):
        print(f"Generating augmented copy {k + 1}/{N_AUG} ...")
        aug = np.empty_like(X)
        for i in range(len(X)):
            aug[i] = augment_one(X[i], rng)
        pieces_X.append(aug)
        pieces_y.append(y.copy())

    X_aug = np.concatenate(pieces_X, axis=0)      # (5391, 8, 8)
    y_aug = np.concatenate(pieces_y, axis=0)      # (5391,)
    print(f"\nFinal augmented dataset: X={X_aug.shape}, y={y_aug.shape}")

    # Shuffle so augmented copies are not contiguous in fold splits
    perm = rng.permutation(len(X_aug))
    X_aug = X_aug[perm]
    y_aug = y_aug[perm]

    np.savez(CACHE, X=X_aug, y=y_aug)
    print(f"Saved -> {CACHE}")


if __name__ == "__main__":
    main()
