"""Tokens are compared after NFKC normalisation; CJK text is split per character."""

import random
import unicodedata

import pytest

from contamscan.methods import Params
from contamscan.scan import EvalItem, scan
from contamscan.text import tokenize

ITEM = "Le café du coin sert un très bon thé à côté de la gare chaque matin depuis des années"
LETTERS = "aeiouyncsz" + "éèêàçñüöÅẛ"


def _fullwidth(s: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else "　" for c in s)


@pytest.mark.parametrize("mode", ["word", "code"])
@pytest.mark.parametrize("form", ["NFC", "NFD", "NFKC", "NFKD"])
def test_token_texts_do_not_depend_on_the_normalisation_form(mode, form):
    rng = random.Random(7)
    for _ in range(300):
        words = ["".join(rng.choice(LETTERS) for _ in range(rng.randint(1, 6))) for _ in range(8)]
        text = rng.choice([" ", "\t", " ", ", "]).join(words)
        base = [t.text for t in tokenize(unicodedata.normalize("NFC", text), mode)]
        other = unicodedata.normalize(form, text)
        toks = tokenize(other, mode)
        assert [t.text for t in toks] == base
        for t in toks:  # offsets index the raw text
            raw = unicodedata.normalize("NFKC", other[t.start : t.end])
            assert (raw.casefold() if mode == "word" else raw) == t.text


def test_fullwidth_copy_tokenizes_like_ascii():
    text = "THE QUICK BROWN FOX"
    assert [t.text for t in tokenize(_fullwidth(text))] == text.lower().split()
    assert [t.text for t in tokenize(_fullwidth("def f(x):"), "code")] == [
        "def", "f", "(", "x", ")", ":"
    ]  # fmt: skip


@pytest.mark.parametrize("tokenizer", ["word", "code"])
@pytest.mark.parametrize("variant", ["NFD", "fullwidth"])
def test_ngram_methods_find_a_copy_in_another_normalisation_form(tokenizer, variant):
    item = ITEM if variant == "NFD" else ITEM.upper().encode("ascii", "ignore").decode()
    copy = unicodedata.normalize("NFD", item) if variant == "NFD" else _fullwidth(item)
    (r,) = scan([EvalItem("a", item)], [("d", "intro " + copy + " outro")], tokenizer=tokenizer)
    assert r.gpt3_label == "dirty"
    assert r.palm_overlap == 1.0
    assert r.llama2_coverage == 1.0


def test_cjk_text_is_split_per_character():
    item = "下列关于中国古代科举制度的说法，正确的是哪一项？请从四个选项中选出最符合题意的答案。"
    toks = tokenize(item)
    assert len(toks) == sum(not unicodedata.category(c).startswith("P") for c in item)
    doc = "前言。" + item[:-5] + "其他内容。"
    (r,) = scan([EvalItem("a", item)], [("d", doc)], Params())
    assert r.gpt3_label == "dirty"
    assert r.palm_label == "contaminated"
    # Hiragana and katakana too; Latin words stay whole next to them
    assert [t.text for t in tokenize("テスト test です")] == ["テ", "ス", "ト", "test", "で", "す"]


@pytest.mark.parametrize("mode", ["word", "code"])
def test_ascii_fast_path_matches_the_general_tokenizer(mode):
    """Pure-ASCII text takes a faster regex; appending one non-ASCII character sends the
    same text through the general one, which must give the same tokens before it."""
    rng = random.Random(3)
    alphabet = "abcXYZ_019 .,:;()[]{}+-*/=<>!\t\n'\""
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        fast = tokenize(text, mode)
        general = tokenize(text + " é", mode)
        assert fast == general[:-1]
