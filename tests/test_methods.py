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
