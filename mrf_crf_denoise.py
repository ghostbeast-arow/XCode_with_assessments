"""MRF/CRF-inspired image restoration from noisy observations.

Implements an MRF-based denoiser using ICM and SA, plus a CRF-like
mean-field update where pairwise weights depend on observed intensities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class DenoiseConfig:
    lam: float = 0.8
    max_iter: int = 30
    step_size: float = 10.0
    temperature: float = 5.0
    temperature_decay: float = 0.95
    sigma: float = 15.0


def _neighbors(height: int, width: int, i: int, j: int) -> Iterable[Tuple[int, int]]:
    if i > 0:
        yield i - 1, j
    if i < height - 1:
        yield i + 1, j
    if j > 0:
        yield i, j - 1
    if j < width - 1:
        yield i, j + 1


def icm_denoise(noisy: np.ndarray, config: DenoiseConfig) -> np.ndarray:
    """Iterated Conditional Modes for quadratic MRF denoising."""
    clean = noisy.astype(float).copy()
    height, width = clean.shape
    for _ in range(config.max_iter):
        for i in range(height):
            for j in range(width):
                neigh_vals = [clean[ni, nj] for ni, nj in _neighbors(height, width, i, j)]
                degree = max(len(neigh_vals), 1)
                numerator = noisy[i, j] + config.lam * sum(neigh_vals)
                denominator = 1.0 + config.lam * degree
                clean[i, j] = numerator / denominator
    return np.clip(clean, 0, 255)


def simulated_annealing_denoise(noisy: np.ndarray, config: DenoiseConfig) -> np.ndarray:
    """Simulated annealing for MRF denoising to avoid local minima."""
    rng = np.random.default_rng(0)
    clean = noisy.astype(float).copy()
    height, width = clean.shape
    temperature = config.temperature
    for _ in range(config.max_iter):
        for _ in range(height * width):
            i = rng.integers(0, height)
            j = rng.integers(0, width)
            current = clean[i, j]
            proposal = current + rng.normal(0, config.step_size)
            proposal = float(np.clip(proposal, 0, 255))
            current_energy = _local_energy(noisy, clean, i, j, config.lam)
            clean[i, j] = proposal
            proposal_energy = _local_energy(noisy, clean, i, j, config.lam)
            delta = proposal_energy - current_energy
            if delta > 0 and rng.random() > math.exp(-delta / max(temperature, 1e-6)):
                clean[i, j] = current
        temperature *= config.temperature_decay
    return np.clip(clean, 0, 255)


def _local_energy(noisy: np.ndarray, clean: np.ndarray, i: int, j: int, lam: float) -> float:
    data_term = (clean[i, j] - noisy[i, j]) ** 2
    smooth_term = 0.0
    height, width = clean.shape
    for ni, nj in _neighbors(height, width, i, j):
        smooth_term += (clean[i, j] - clean[ni, nj]) ** 2
    return data_term + lam * smooth_term


def crf_mean_field_denoise(noisy: np.ndarray, config: DenoiseConfig) -> np.ndarray:
    """CRF-style mean-field update with bilateral weights based on observations."""
    clean = noisy.astype(float).copy()
    height, width = clean.shape
    for _ in range(config.max_iter):
        updated = clean.copy()
        for i in range(height):
            for j in range(width):
                weight_sum = 0.0
                value_sum = 0.0
                for ni, nj in _neighbors(height, width, i, j):
                    diff = noisy[i, j] - noisy[ni, nj]
                    weight = math.exp(-(diff ** 2) / (2 * config.sigma ** 2))
                    weight_sum += weight
                    value_sum += weight * clean[ni, nj]
                if weight_sum > 0:
                    updated[i, j] = (noisy[i, j] + config.lam * value_sum) / (1.0 + config.lam * weight_sum)
        clean = updated
    return np.clip(clean, 0, 255)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    base = np.tile(np.linspace(50, 200, 64), (64, 1))
    noisy_image = base + rng.normal(0, 15, size=base.shape)
    cfg = DenoiseConfig(lam=0.6, max_iter=10)
    denoised_icm = icm_denoise(noisy_image, cfg)
    denoised_sa = simulated_annealing_denoise(noisy_image, cfg)
    denoised_crf = crf_mean_field_denoise(noisy_image, cfg)
    print("ICM mean:", denoised_icm.mean())
    print("SA mean:", denoised_sa.mean())
    print("CRF mean:", denoised_crf.mean())
