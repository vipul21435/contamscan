"""Contamination methods, their parameters and per-item labels.

Defaults and their sources:

* ``gpt3``: Brown et al. 2020 (arXiv 2005.14165, Appendix C). An example is *dirty* if it
  shares any N-gram with any training document, *clean* otherwise. The paper picks N per
  dataset as the 5th-percentile example length in words, clamped to [8, 13]; this tool
  uses one N for the whole set, default 13 (the paper's maximum and its headline value).
  Examples shorter than N are matched as a whole, as the paper does.
* ``palm``: Chowdhery et al. 2022 (arXiv 2204.02311, Section 8). An example is
  *contaminated* if at least 70 % of its 8-grams were seen in the training data.
* ``llama2``: Touvron et al. 2023 (arXiv 2307.09288, Appendix A.6). A token is contaminated
  if it lies in a matching n-gram longer than 10 tokens (default n = 11). Samples are
  *clean* below 20 % token contamination and *dirty* at or above 80 %; *not clean* and
  *not dirty* are the complements. The paper's skipgram budget of four mismatched
  positions is not implemented: matches here are exact.
* ``substring``: the longest span shared with one corpus document (a run of consecutive
  matching n-grams), flagged at or above ``substring_min_tokens`` tokens (default 50, a
  choice of this tool, not from a paper); plus exact containment of the whole item text
  in a document, raw and normalised.

``max_doc_freq`` (off by default): item n-grams found in more than this many corpus
documents are ignored by ``gpt3``, ``palm`` and ``llama2``. GPT-3 (Appendix C) ignored
13-grams that matched more than 10 training documents in its training-set filter, finding
them to be mostly common phrases and boilerplate; for code this removes stock idioms.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Params",
    "METHODS",
    "gpt3_auto_n",
    "label_gpt3",
    "label_palm",
    "label_llama2",
    "label_substring",
]

METHODS = ("gpt3", "palm", "llama2", "substring")


@dataclass(frozen=True)
class Params:
    gpt3_n: int = 13
    palm_n: int = 8
    palm_threshold: float = 0.70
    llama_n: int = 11
    llama_clean: float = 0.20
    llama_dirty: float = 0.80
    substring_min_tokens: int = 50
    max_doc_freq: int | None = None

    def ns(self) -> tuple[int, ...]:
        return tuple(sorted({self.gpt3_n, self.palm_n, self.llama_n}))

    def span_n(self) -> int:
        """n used to chain matches into longest shared spans (the smallest n)."""
        return min(self.ns())


def gpt3_auto_n(lengths: list[int], low: int = 8, high: int = 13) -> int:
    """GPT-3's per-dataset N: the 5th-percentile example length, clamped to [low, high].

    Brown et al. 2020 (Appendix C) use the 5th percentile of example lengths in words, a
    minimum of 8 "due to spurious collisions at lower values" and a maximum of 13. The
    percentile is the lower order statistic (no interpolation), so N is a real length.
    """
    if not lengths:
        return high
    ordered = sorted(lengths)
    k = int(0.05 * (len(ordered) - 1))
    return max(low, min(high, ordered[k]))


def label_gpt3(matched_grams: int) -> str:
    return "dirty" if matched_grams > 0 else "clean"


def label_palm(fraction: float | None, threshold: float) -> str:
    if fraction is None:
        return "too-short"
    return "contaminated" if fraction >= threshold else "clean"


def label_llama2(coverage: float, clean: float, dirty: float) -> str:
    """One of ``clean`` (< clean), ``dirty`` (>= dirty) or ``mixed`` (in between).

    ``clean`` items are also *not dirty*; ``dirty`` items are also *not clean*; ``mixed``
    items are both *not clean* and *not dirty*.
    """
    if coverage < clean:
        return "clean"
    if coverage >= dirty:
        return "dirty"
    return "mixed"


def label_substring(longest_tokens: int, exact: bool, min_tokens: int) -> str:
    if exact:
        return "exact"
    return "long-match" if longest_tokens >= min_tokens else "clean"
