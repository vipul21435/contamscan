"""Clean-versus-contaminated score comparison.

GPT-3 (Brown et al. 2020, Appendix C) compares a model's score on the clean subset with
its score on the full set; Llama 2 (Touvron et al. 2023, Appendix A.6) compares clean
and dirty subsets with their complements. This module reports the difference of means
``mean(contaminated) - mean(clean)`` with a percentile bootstrap interval that resamples
each group independently (the groups are disjoint sets of items).

It also gives the Llama 2 subset statistic ``Z_n = (mean(subset) - mu_n) / sigma_n``, where
``mu_n`` and ``sigma_n**2`` are the mean and variance of the mean of ``n`` items drawn from
the whole evaluation set (``sigma_n = s / sqrt(n)``), and the GPT-3 relative change of the
clean-only score against the score on the whole set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["Comparison", "compare", "SubsetZ", "subset_z", "relative_change"]


@dataclass(frozen=True)
class Comparison:
    n_clean: int
    n_contaminated: int
    mean_clean: float
    mean_contaminated: float
    difference: float
    low: float
    high: float
    confidence: float
    n_boot: int


def compare(
    clean: Sequence[float],
    contaminated: Sequence[float],
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison | None:
    """Return the difference ``mean(contaminated) - mean(clean)`` with a bootstrap interval.

    Returns ``None`` when either group is empty.
    """
    a = np.asarray(clean, dtype=float)
    b = np.asarray(contaminated, dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    mb = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    diffs = mb - ma
    alpha = (1 - confidence) / 2
    low, high = np.quantile(diffs, [alpha, 1 - alpha])
    return Comparison(
        n_clean=int(a.size),
        n_contaminated=int(b.size),
        mean_clean=float(a.mean()),
        mean_contaminated=float(b.mean()),
        difference=float(b.mean() - a.mean()),
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_boot=n_boot,
    )


@dataclass(frozen=True)
class SubsetZ:
    subset: str
    n: int
    mean: float
    mu: float
    sigma: float
    z: float | None


def subset_z(
    scores: Sequence[float], mask: Sequence[bool], subset: str = "", *, fpc: bool = False
) -> SubsetZ | None:
    """Llama 2 statistic for one subset of the evaluation set.

    ``mu_n`` is the mean over all ``N`` items and ``sigma_n**2 = s**2 / n`` the variance of
    the mean of ``n`` items drawn independently from them, with ``s**2`` the population
    variance. This is the form Table 51 of the paper is consistent with: for a subset and
    its complement, ``|Z_small| / |Z_large| = sqrt(n_large / n_small)`` in its rows. With
    ``fpc=True`` the finite-population correction ``(N - n) / (N - 1)`` (drawing without
    replacement) is applied, which makes a subset and its complement equal and opposite.
    ``z`` is ``None`` when ``sigma_n`` is zero. Returns ``None`` for an empty subset.
    """
    x = np.asarray(scores, dtype=float)
    m = np.asarray(mask, dtype=bool)
    n, big_n = int(m.sum()), int(x.size)
    if n == 0:
        return None
    mu = float(x.mean())
    var = float(x.var()) / n
    if fpc:
        var = var * (big_n - n) / (big_n - 1) if big_n > 1 else 0.0
    sigma = float(np.sqrt(var))
    mean = float(x[m].mean())
    z = (mean - mu) / sigma if sigma > 0 else None
    return SubsetZ(subset, n, mean, mu, sigma, z)


def relative_change(clean: Sequence[float], full: Sequence[float]) -> float | None:
    """GPT-3 relative change ``(mean(clean) - mean(full)) / mean(full)``.

    ``None`` when the clean subset is empty or the full-set mean is zero.
    """
    a = np.asarray(clean, dtype=float)
    b = np.asarray(full, dtype=float)
    if a.size == 0 or b.size == 0 or b.mean() == 0:
        return None
    return float((a.mean() - b.mean()) / b.mean())
