import random

import pytest

from contamscan.hashing import ngram_hash, ngram_hashes, token_hash
from contamscan.text import normalise, tokenize


def test_normalise_case_punct_whitespace():
    assert normalise("Hello,   World!\n\tFoo_bar") == "hello world foo bar"
    assert normalise("A  B", case=False, punctuation=False) == "A B"
    assert normalise("a b") == "a b"  # NFKC turns NBSP into a space


def test_word_tokens_have_offsets():
    text = "Return the SUM, of x."
    toks = tokenize(text, "word")
    assert [t.text for t in toks] == ["return", "the", "sum", "of", "x"]
    for t in toks:
        assert text[t.start : t.end].casefold() == t.text


def test_code_tokens_keep_case_and_operators():
    toks = tokenize("def f(x): return x**2 + 1.5", "code")
    assert [t.text for t in toks] == [
        "def",
        "f",
        "(",
        "x",
        ")",
        ":",
        "return",
        "x",
        "*",
        "*",
        "2",
        "+",
        "1.5",
    ]
    assert tokenize("Foo foo", "code")[0].text == "Foo"


def test_unknown_tokenizer():
    with pytest.raises(ValueError):
        tokenize("x", "bpe")


def test_rolling_hash_matches_direct():
    rng = random.Random(1)
    vocab = [f"w{i}" for i in range(30)]
    toks = [rng.choice(vocab) for _ in range(200)]
    for n in (1, 3, 8, 13):
        rolled = ngram_hashes(toks, n)
        assert len(rolled) == len(toks) - n + 1
        assert rolled == [ngram_hash(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        assert all(0 <= h < 2**64 for h in rolled)


def test_hash_order_sensitive_and_short_input():
    assert ngram_hash(["a", "b"]) != ngram_hash(["b", "a"])
    assert ngram_hashes(["a"], 2) == []
    assert token_hash("a") != token_hash("b")
    with pytest.raises(ValueError):
        ngram_hashes(["a"], 0)
