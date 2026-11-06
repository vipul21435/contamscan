"""64-bit hashes of token n-grams.

Each token is hashed once with BLAKE2b (8-byte digest). An n-gram hash is the
polynomial ``sum(t_i * B**(n-1-i)) mod 2**64`` over the token hashes ``t_i``, computed
with a rolling update so that all n-grams of a sequence cost O(len) instead of O(len * n).
With independent 64-bit token hashes the chance that two different n-grams collide is
about 2**-64 per pair, so a set of a few million evaluation n-grams gives a negligible
false-positive rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from hashlib import blake2b

__all__ = ["token_hash", "ngram_hashes", "ngram_hash"]

_MASK = (1 << 64) - 1
_BASE = 0x100000001B3  # odd 64-bit multiplier


@lru_cache(maxsize=1 << 18)
def token_hash(token: str) -> int:
    return int.from_bytes(blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")


def ngram_hash(tokens: Sequence[str]) -> int:
    """Hash of one n-gram (the tokens in order)."""
    h = 0
    for t in tokens:
        h = (h * _BASE + token_hash(t)) & _MASK
    return h


def ngram_hashes(tokens: Sequence[str], n: int) -> list[int]:
    """Hashes of every contiguous n-gram of ``tokens``; entry i covers ``tokens[i:i+n]``."""
    if n <= 0:
        raise ValueError("n must be positive")
    m = len(tokens)
    if m < n:
        return []
    th = [token_hash(t) for t in tokens]
    top = pow(_BASE, n - 1, 1 << 64)
    h = 0
    for t in th[:n]:
        h = (h * _BASE + t) & _MASK
    out = [h]
    for i in range(n, m):
        h = ((h - th[i - n] * top) * _BASE + th[i]) & _MASK
        out.append(h)
    return out
