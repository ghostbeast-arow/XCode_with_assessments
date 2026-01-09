"""Gaussian Mixture Model parameter learning via EM.

This module implements the EM procedure described in the coursework report
for fitting a K-component Gaussian Mixture Model to data.
"""

from __future__ import annotations

import numpy as np


class GMM:
    """Gaussian Mixture Model trained via Expectation-Maximization."""

    def __init__(self, n_components: int, max_iter: int = 100, tol: float = 1e-4, reg_covar: float = 1e-6):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.log_likelihood_: float | None = None

    def _initialize(self, x: np.ndarray) -> None:
        n_samples, n_features = x.shape
        rng = np.random.default_rng(42)
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = x[indices]
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
        base_cov = np.cov(x, rowvar=False) + self.reg_covar * np.eye(n_features)
        self.covariances_ = np.stack([base_cov.copy() for _ in range(self.n_components)], axis=0)

    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        n_features = x.shape[1]
        cov = cov + self.reg_covar * np.eye(n_features)
        inv_cov = np.linalg.inv(cov)
        diff = x - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        denom = np.sqrt(((2 * np.pi) ** n_features) * np.linalg.det(cov))
        return np.exp(exponent) / denom

    def _e_step(self, x: np.ndarray) -> np.ndarray:
        responsibilities = np.zeros((x.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self._gaussian_pdf(x, self.means_[k], self.covariances_[k])
        norm = responsibilities.sum(axis=1, keepdims=True)
        norm = np.clip(norm, 1e-12, None)
        return responsibilities / norm

    def _m_step(self, x: np.ndarray, responsibilities: np.ndarray) -> None:
        nk = responsibilities.sum(axis=0)
        self.weights_ = nk / x.shape[0]
        self.means_ = (responsibilities.T @ x) / nk[:, None]
        covariances = []
        for k in range(self.n_components):
            diff = x - self.means_[k]
            weighted = diff.T * responsibilities[:, k]
            cov = weighted @ diff / nk[k]
            covariances.append(cov + self.reg_covar * np.eye(x.shape[1]))
        self.covariances_ = np.stack(covariances, axis=0)

    def _compute_log_likelihood(self, x: np.ndarray) -> float:
        total = np.zeros(x.shape[0])
        for k in range(self.n_components):
            total += self.weights_[k] * self._gaussian_pdf(x, self.means_[k], self.covariances_[k])
        total = np.clip(total, 1e-12, None)
        return float(np.sum(np.log(total)))

    def fit(self, x: np.ndarray) -> "GMM":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("Input data must be 2D: (n_samples, n_features)")
        self._initialize(x)
        prev_ll = None
        for _ in range(self.max_iter):
            responsibilities = self._e_step(x)
            self._m_step(x, responsibilities)
            ll = self._compute_log_likelihood(x)
            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self.log_likelihood_ = prev_ll
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.weights_ is None:
            raise RuntimeError("Model must be fit before calling predict_proba.")
        return self._e_step(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = np.vstack([
        rng.normal(loc=(-2, -2), scale=0.5, size=(150, 2)),
        rng.normal(loc=(3, 3), scale=0.8, size=(150, 2)),
    ])
    gmm = GMM(n_components=2, max_iter=50).fit(data)
    labels = gmm.predict(data)
    print("GMM weights:", gmm.weights_)
    print("GMM means:\n", gmm.means_)
    print("Assigned labels shape:", labels.shape)
