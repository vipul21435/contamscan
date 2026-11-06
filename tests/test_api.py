import contamscan
from contamscan.scan import scan


def test_package_exports_and_python_example(tmp_path):
    ev = tmp_path / "eval.jsonl"
    ev.write_text('{"id": "q1", "question": "what is the sum of the first ten primes"}\n')
    items = contamscan.load_eval(ev, ["question"])
    corpus = [("doc-1", "Quiz: what is the sum of the first ten primes? Answer 129.")]
    (r,) = scan(items, corpus, contamscan.Params(gpt3_n=8))
    assert (r.gpt3_label, r.substring_label, r.matches[0].doc_id) == ("dirty", "exact", "doc-1")
    assert contamscan.summarise([r])["methods"]["gpt3"]["contaminated"] == 1
    assert set(contamscan.__all__) <= set(dir(contamscan))
