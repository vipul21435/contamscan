from contamscan.methods import Params, label_llama2, label_palm, label_substring
from contamscan.scan import EvalItem, scan

WORDS = [f"w{i}" for i in range(200)]


def text(a: int, b: int) -> str:
    return " ".join(WORDS[a:b])


def run(item_text: str, docs: list[str], **kw):
    items = [EvalItem("q", item_text)]
    corpus = [(f"d{k}", d) for k, d in enumerate(docs)]
    return scan(items, corpus, Params(**kw))[0]


def test_gpt3_single_13gram_is_dirty_12_is_not():
    item = text(0, 40)
    hit = run(item, ["zz " + text(10, 23) + " zz"])  # exactly 13 shared words
    assert hit.gpt3_label == "dirty"
    assert hit.gpt3_overlap == 1 / 28
    miss = run(item, [text(10, 22)])  # 12 shared words
    assert miss.gpt3_label == "clean" and miss.gpt3_overlap == 0.0


def test_gpt3_short_item_matched_whole():
    r = run("alpha beta gamma", ["x alpha beta gamma y"])
    assert r.gpt3_label == "dirty" and r.gpt3_overlap == 1.0
    assert r.longest_match_tokens == 3
    r2 = run("alpha beta gamma", ["alpha beta delta"])
    assert r2.gpt3_label == "clean"


def test_palm_threshold_70_percent():
    item = text(0, 27)  # 20 eight-grams
    # 14 of 20 eight-grams = 70 % -> contaminated; doc holds words 0..20 (21 words)
    r = run(item, [text(0, 21)])
    assert r.palm_overlap == 14 / 20 and r.palm_label == "contaminated"
    r = run(item, [text(0, 20)])  # 13 of 20 = 65 %
    assert r.palm_overlap == 13 / 20 and r.palm_label == "clean"
    assert label_palm(None, 0.7) == "too-short"


def test_llama2_token_coverage_bands():
    item = text(0, 100)
    r = run(item, [text(0, 19)])  # 19 tokens covered: 19 %
    assert r.llama2_coverage == 0.19 and r.llama2_label == "clean"
    r = run(item, [text(0, 20)])
    assert r.llama2_coverage == 0.20 and r.llama2_label == "mixed"
    r = run(item, [text(0, 50), text(50, 80)])
    assert r.llama2_coverage == 0.80 and r.llama2_label == "dirty"
    # a 10-token match is not "longer than 10 tokens"
    r = run(item, [text(0, 10)])
    assert r.llama2_coverage == 0.0
    assert label_llama2(0.5, 0.2, 0.8) == "mixed"


def test_substring_longest_and_exact():
    item = "Write a function " + text(0, 60) + "."
    doc = "prefix words " + text(0, 60) + " suffix"
    r = run(item, [doc])
    assert r.longest_match_tokens == 60 and r.substring_label == "long-match"
    m = r.matches[0]
    assert doc[m.doc_start : m.doc_end] == text(0, 60)
    assert item[m.item_start : m.item_end] == text(0, 60)
    r = run(item, ["see: " + item.upper()])
    assert r.substring_label == "exact"
    assert r.matches[0].exact_normalised and not r.matches[0].exact
    assert label_substring(3, False, 50) == "clean"


def test_match_cap_and_doc_count():
    item = text(0, 30)
    docs = [text(0, k) for k in range(15, 30)]
    r = scan([EvalItem("q", item)], [(str(k), d) for k, d in enumerate(docs)], max_matches=3)[0]
    assert r.n_matching_docs == 15
    assert [m.tokens for m in r.matches] == [29, 28, 27]


def test_code_tokenizer_keeps_case():
    items = [EvalItem("c", "def Add(a, b):\n    return a + b\n")]
    corpus = [("d", "def add(a, b):\n    return a + b\n")]
    p = Params(gpt3_n=5, palm_n=5, llama_n=5)
    word = scan(items, corpus, p, tokenizer="word")[0]
    code = scan(items, corpus, p, tokenizer="code")[0]
    assert word.palm_overlap == 1.0
    assert code.palm_overlap < 1.0


def test_gpt3_auto_n_percentile_and_clamp():
    from contamscan.methods import gpt3_auto_n

    assert gpt3_auto_n([]) == 13
    assert gpt3_auto_n([3] * 50) == 8  # floor of 8
    assert gpt3_auto_n([100] * 50) == 13  # ceiling of 13
    lengths = list(range(1, 101))  # 5th percentile (lower) of 1..100 is 5 -> clamp to 8
    assert gpt3_auto_n(lengths) == 8
    assert gpt3_auto_n([10] * 5 + [50] * 95) == 10
    assert gpt3_auto_n([9, 11, 12] + [40] * 17) == 9


def test_max_doc_freq_ignores_common_ngrams():
    item = text(0, 40)
    common = text(0, 13)  # in three documents
    rare = text(20, 33)  # in one document
    docs = [common, "x " + common, common + " y", rare]
    r = run(item, docs)
    assert r.gpt3_overlap == 2 / 28
    r = run(item, docs, max_doc_freq=2)
    assert r.gpt3_overlap == 1 / 28 and r.gpt3_label == "dirty"
    r = run(item, docs[:3], max_doc_freq=2)
    assert r.gpt3_overlap == 0.0 and r.gpt3_label == "clean"
    # the longest-span report is not filtered
    assert r.longest_match_tokens == 13


def test_palm_per_field():
    question = text(0, 20)  # 13 eight-grams
    target = text(100, 140)  # 33 eight-grams
    it = EvalItem(
        "q",
        question + "\n" + target,
        field_spans=(
            ("q", 0, len(question)),
            ("t", len(question) + 1, len(question) + 1 + len(target)),
        ),
    )
    corpus = [("d", "zz " + question + " zz")]
    joined = scan([it], corpus, Params())[0]
    assert joined.palm_label == "clean" and joined.palm_fields is None
    per = scan([it], corpus, Params(palm_per_field=True))[0]
    assert per.palm_fields == {"q": 1.0, "t": 0.0}
    assert per.palm_overlap == 1.0 and per.palm_label == "contaminated"
    short = EvalItem("s", "a b", field_spans=(("q", 0, 3),))
    r = scan([short], corpus, Params(palm_per_field=True))[0]
    assert r.palm_fields == {"q": None} and r.palm_label == "too-short"


def test_llama2_skip_budget_joins_runs_across_substitutions():
    item = text(0, 30)
    words = item.split()
    words[12] = "X"  # runs 0-11 (12 tokens), 13-21 (9), 23-29 (7, below the 8-gram size)
    words[22] = "Y"
    doc = " ".join(words)
    exact = run(item, [doc])
    assert exact.llama2_coverage == 12 / 30
    skip = run(item, [doc], llama_skip_budget=4)
    assert skip.llama2_coverage == 22 / 30
    # budget too small for a gap of one: same as exact
    assert run(item, [doc], llama_skip_budget=0).llama2_coverage == 12 / 30


def test_skipgram_cover_rules():
    from contamscan.scan import skipgram_cover

    # first run shorter than 10 tokens may not be followed by mismatches
    assert skipgram_cover([(9, 0, 0), (12, 10, 10)], 4, 11) == set()
    # gaps add up against the budget
    runs = [(10, 0, 5), (8, 12, 17), (8, 23, 28)]  # gaps 2 and 3 on one diagonal
    assert skipgram_cover(runs, 5, 11) == set(range(0, 31))
    assert skipgram_cover(runs, 4, 11) == set(range(0, 20))
    # runs on different diagonals are never joined
    assert skipgram_cover([(10, 0, 0), (10, 11, 30)], 4, 11) == set()


def test_recased_code_item_is_exact_without_any_shared_ngram():
    # code tokens keep case, so the upper-cased copy shares no n-gram with the item
    item = "def add_two(a, b):\n    return a + b\n"
    doc = "notes\nDEF ADD_TWO(A, B):\n\tRETURN A + B\nend"
    r = scan([EvalItem("c", item)], [("d", doc)], Params(), tokenizer="code")[0]
    assert r.gpt3_label == "clean" and r.palm_overlap == 0.0
    assert r.substring_label == "exact" and r.n_matching_docs == 1
    m = r.matches[0]
    assert m.exact_normalised and not m.exact
    assert doc[m.doc_start : m.doc_end] == "DEF ADD_TWO(A, B):\n\tRETURN A + B"
    assert (m.item_start, m.item_end) == (0, len(item))


def test_short_item_contained_mid_text_and_raw_offsets():
    # two-word items must match whole words; raw hits report the raw offsets
    r = run("Alpha, beta", ["x alpha beta y"])
    assert r.substring_label == "exact" and r.matches[0].exact_normalised
    assert run("alpha beta", ["xalpha betay"]).substring_label == "clean"
    doc = "pre Alpha, beta post"
    m = run("Alpha, beta", [doc]).matches[0]
    assert m.exact and doc[m.doc_start : m.doc_end] == "Alpha, beta"


def test_normalised_hit_survives_the_match_cap():
    item = text(0, 30)
    long_docs = [text(0, 29)] * 4  # 29 shared tokens each, not contained
    docs = long_docs + ["see " + item.upper().replace(" ", "  ")]
    r = scan([EvalItem("q", item)], [(str(k), d) for k, d in enumerate(docs)], max_matches=2)[0]
    assert r.substring_label == "exact"
    assert r.matches[0].doc_id == "4" and r.matches[0].exact_normalised


def test_normalised_offsets_fall_back_to_whole_document(monkeypatch):
    import contamscan.scan as sc

    assert sc._norm_word_spans("a.b  c") == [(0, 3), (0, 3), (5, 6)]
    # if chunk-wise normalisation ever disagrees with the whole-text one, the match
    # covers the whole document instead of pointing at the wrong words
    monkeypatch.setattr(sc, "_norm_word_spans", lambda text: [])
    doc = "x ALPHA BETA GAMMA"
    m = run("alpha beta gamma", [doc]).matches[0]
    assert m.exact_normalised and (m.doc_start, m.doc_end) == (0, len(doc))
