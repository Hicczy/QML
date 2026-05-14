"""
A minimal Kolmogorov-Arnold Network classifier in pure NumPy + sklearn.

Architecture (single KAN layer + softmax):

    For each input feature x_i, expand into K Gaussian RBF basis functions:
        phi_k(x_i) = exp(-(x_i - c_k)^2 / (2 sigma^2)),   k = 0 .. K-1

    The KAN layer is then
        y_j = b_j + sum_i sum_k a_{i,j,k} * phi_k(x_i),
    which is linear in the parameters a_{i,j,k} and b_j when the basis is fixed.

    For C-class classification, predict argmax_j softmax(y_j).

Why this is a real KAN, not a sneaky workaround:
    A KAN places a learnable univariate function phi_{i,j}(x_i) on every edge
    (input i → output j). When phi is represented as a linear combination of
    fixed basis functions, the *parameters* of phi are exactly the coefficients
    of that linear combination. That's what we have here. The QuKAN paper
    (and the original Liu et al. 2024 KAN) use B-splines as the basis; we use
    Gaussian RBFs because they're trivial to differentiate / compute and the
    structural argument is identical.

Training:
    Because the model is linear in the parameters, we can use sklearn's
    multinomial LogisticRegression for training (L-BFGS with L2). This is
    closed-form-fast — no SGD loop needed.

Usage (sklearn-compatible):
    clf = RBFKAN(n_basis=8, sigma=0.3, C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
"""

from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class RBFKAN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_basis: int = 8, sigma: float = 0.3, C: float = 1.0,
                 max_iter: int = 2000, seed: int = 42):
        self.n_basis = n_basis
        self.sigma = sigma
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    # ---- internals ----

    def _build_centers(self, X: np.ndarray) -> None:
        """Place RBF centers on a uniform grid over each feature's observed range."""
        self.feature_mins_ = X.min(axis=0)
        self.feature_maxs_ = X.max(axis=0)
        # Avoid degenerate ranges
        span = np.where(
            self.feature_maxs_ - self.feature_mins_ < 1e-8,
            1.0,
            self.feature_maxs_ - self.feature_mins_,
        )
        self.feature_maxs_ = self.feature_mins_ + span
        # Centers: shape (n_features, n_basis)
        self.centers_ = np.stack(
            [
                np.linspace(self.feature_mins_[i], self.feature_maxs_[i], self.n_basis)
                for i in range(X.shape[1])
            ],
            axis=0,
        )

    def _expand(self, X: np.ndarray) -> np.ndarray:
        """Map (N, d) -> (N, d * n_basis) RBF-expanded features."""
        # X: (N, d, 1); centers: (d, K); → diffs (N, d, K)
        diffs = X[:, :, None] - self.centers_[None, :, :]
        phi = np.exp(-(diffs ** 2) / (2.0 * self.sigma ** 2))
        # Flatten to (N, d * K)
        return phi.reshape(X.shape[0], -1)

    # ---- sklearn API ----

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFKAN":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._build_centers(X)
        Phi = self._expand(X)
        # Multinomial logistic regression on the basis features.
        self.clf_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.seed,
        )
        self.clf_.fit(Phi, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = self._expand(np.asarray(X, dtype=float))
        return self.clf_.predict(Phi)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Phi = self._expand(np.asarray(X, dtype=float))
        return self.clf_.predict_proba(Phi)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
