"""Normalisation and tokenisation with character offsets.

Two tokenisers are provided:

* ``word``: lower-cases, drops punctuation and splits on whitespace, following the
  GPT-3 overlap rule of comparing words "ignoring all punctuation, whitespace, and casing".
* ``code``: keeps identifiers, numbers and operator characters as separate tokens and
  keeps case, because in code ``Foo`` and ``foo`` are different names and punctuation
  such as ``(`` or ``:`` carries structure.

In both, combining marks stay inside the word they follow, each token's text is
NFKC-normalised (so NFC, NFD and fullwidth copies give the same tokens), and every Han,
hiragana or katakana character is a token of its own, because Chinese and Japanese are
written without spaces between words.

Every token carries the character span it came from in the raw text, so a match can be
reported as an offset range in the original text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["Token", "normalise", "tokenize", "TOKENIZERS"]


def _mark_class() -> str:
    """Regex class body for the combining marks (Unicode category M) of the BMP.

    Marks outside the BMP (some historic scripts, variation selectors supplement) are left
    out: with them ``re`` cannot use a bitmap for the class and matching gets much slower.
    """
    ranges: list[list[int]] = []
    for c in range(0x10000):
        if unicodedata.category(chr(c)).startswith("M"):
            if ranges and ranges[-1][1] == c - 1:
                ranges[-1][1] = c
            else:
                ranges.append([c, c])
    return "".join(f"\\u{a:04x}-\\u{b:04x}" for a, b in ranges)


_MARK = _mark_class()
# Han (with extensions and compatibility ideographs), hiragana, katakana (not the
# katakana middle dot, which is punctuation)
_CJK = (
    "\\u3040-\\u30fa\\u30fc-\\u30ff\\u31f0-\\u31ff\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff"
    "\\uff66-\\uff9f\\U00020000-\\U0003134f"
)
_WORD_RE = re.compile(rf"[{_CJK}]|(?:[^\W_{_CJK}]|[{_MARK}])+")
# CJK characters, identifiers, numbers (int/float), then any single non-space character
_CODE_RE = re.compile(
    rf"[{_CJK}]|(?:[^\W\d{_CJK}]|[{_MARK}])(?:[^\W{_CJK}]|[{_MARK}])*|\d+(?:\.\d+)?|\S"
)
# The same rules for pure-ASCII text (no marks, no CJK, NFKC is the identity), which is
# most of a code corpus and is matched several times faster.
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_ASCII_CODE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\S")
_PUNCT_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_WS_RE = re.compile(r"\s+")

TOKENIZERS = ("word", "code")


@dataclass(frozen=True)
class Token:
    """One token: its normalised ``text`` and the ``start:end`` range it came from."""

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


def _nfkc(s: str) -> str:
    return s if s.isascii() else unicodedata.normalize("NFKC", s)


def tokenize(text: str, mode: str = "word") -> list[Token]:
    """Split ``text`` into tokens with character offsets into ``text``.

    Token texts are NFKC-normalised; ``word`` tokens are also case-folded, ``code`` tokens
    keep case. Offsets are those of the raw match.
    """
    if mode not in TOKENIZERS:
        raise ValueError(f"unknown tokenizer {mode!r}; choose one of {TOKENIZERS}")
    if text.isascii():
        if mode == "word":
            found = _ASCII_WORD_RE.finditer(text)
            return [Token(m.group().lower(), m.start(), m.end()) for m in found]
        return [Token(m.group(), m.start(), m.end()) for m in _ASCII_CODE_RE.finditer(text)]
    if mode == "word":
        found = _WORD_RE.finditer(text)
        return [Token(_nfkc(m.group()).casefold(), m.start(), m.end()) for m in found]
    return [Token(_nfkc(m.group()), m.start(), m.end()) for m in _CODE_RE.finditer(text)]
