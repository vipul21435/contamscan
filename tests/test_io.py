import json

import pytest

from contamscan.io import iter_corpus, iter_records, load_eval


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")


def test_load_eval_joins_fields_and_scores(tmp_path):
    p = tmp_path / "e.jsonl"
    write_jsonl(p, [{"id": "a", "q": "x", "s": "y", "ok": True}, {"id": "b", "q": "z", "ok": 0}])
    items = load_eval(p, ["q", "s"], score_field="ok")
    assert [(i.id, i.text, i.score) for i in items] == [("a", "x\ny", 1.0), ("b", "z", 0.0)]


def test_duplicate_ids_and_bad_json(tmp_path):
    p = tmp_path / "e.jsonl"
    write_jsonl(p, [{"id": "a"}, {"id": "a"}])
    with pytest.raises(ValueError, match="duplicate"):
        load_eval(p, ["q"])
    p.write_text("{not json\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        list(iter_records(p))
    p.write_text("[1]\n")
    with pytest.raises(ValueError, match="JSON object"):
        list(iter_records(p))


def test_corpus_jsonl_text_dir_and_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    d = tmp_path / "corpus"
    d.mkdir()
    write_jsonl(d / "a.jsonl", [{"id": 7, "text": "hello"}, {"instruction": "i", "text": "t"}])
    (d / "b.txt").write_text("plain file")
    (d / "skip.bin").write_bytes(b"\x00")
    pq.write_table(pa.table({"text": ["pq row"]}), d / "c.parquet")
    docs = list(iter_corpus([d], text_fields=["instruction", "text"]))
    assert docs == [
        ("7", "hello"),
        ("a.jsonl:1", "i\nt"),
        ("b.txt", "plain file"),
        ("c.parquet:0", "pq row"),
    ]
    assert list(iter_corpus([d / "b.txt"])) == [("b.txt", "plain file")]
