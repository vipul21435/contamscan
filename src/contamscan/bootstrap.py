"""Clean-versus-contaminated score comparison.

GPT-3 (Brown et al. 2020, Appendix C) compares a model's score on the clean subset with
its score on the full set; Llama 2 (Touvron et al. 2023, Appendix A.6) compares clean
and dirty subsets with their complements. This module reports the difference of means
``mean(contaminated) - mean(clean)`` with a percentile bootstrap interval that resamples
each group independently (the groups are disjoint sets of items).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["Comparison", "compare"]


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
