"""Normalisation and tokenisation with character offsets.

Two tokenisers are provided:

* ``word``: lower-cases, drops punctuation and splits on whitespace, following the
  GPT-3 overlap rule of comparing words "ignoring all punctuation, whitespace, and casing".
* ``code``: keeps identifiers, numbers and operator characters as separate tokens and
  keeps case, because in code ``Foo`` and ``foo`` are different names and punctuation
  such as ``(`` or ``:`` carries structure.

Every token carries the character span it came from, so a match can be reported as an
offset range in the original text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["Token", "normalise", "tokenize", "TOKENIZERS"]

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
# identifiers, numbers (int/float), then any single non-space character
_CODE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\S")
_PUNCT_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_WS_RE = re.compile(r"\s+")

TOKENIZERS = ("word", "code")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def normalise(
    text: str, *, case: bool = True, punctuation: bool = True, whitespace: bool = True
) -> str:
    """Return ``text`` with case folded, punctuation removed and whitespace collapsed.

    Each step can be switched off. Unicode is NFKC-normalised first so that visually
    identical strings compare equal.
    """
    out = unicodedata.normalize("NFKC", text)
    if case:
        out = out.casefold()
    if punctuation:
        out = _PUNCT_RE.sub(" ", out)
    if whitespace:
        out = _WS_RE.sub(" ", out).strip()
    return out


def tokenize(text: str, mode: str = "word") -> list[Token]:
    """Split ``text`` into tokens with character offsets into ``text``.

    ``word`` tokens are case-folded; ``code`` tokens keep case.
    """
    if mode == "word":
        return [Token(m.group().casefold(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if mode == "code":
        return [Token(m.group(), m.start(), m.end()) for m in _CODE_RE.finditer(text)]
    raise ValueError(f"unknown tokenizer {mode!r}; choose one of {TOKENIZERS}")
