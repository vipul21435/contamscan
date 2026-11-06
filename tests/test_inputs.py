"""Corpus file types, malformed input and the exit status of the CLI."""

import gzip
import json
from collections import Counter

import pytest

from contamscan.cli import main
from contamscan.io import iter_corpus, load_eval

ITEM = " ".join(f"w{i}" for i in range(30))


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def ev(tmp_path):
    return _write(tmp_path / "eval.jsonl", [{"id": "q", "text": ITEM, "s": 1}])


def test_gzip_and_upper_case_suffixes_are_read(tmp_path):
    rows = [{"id": "d", "text": ITEM}]
    with gzip.open(tmp_path / "c.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(rows[0]) + "\n")
    _write(tmp_path / "C.JSONL", rows)
    assert list(iter_corpus([tmp_path / "c.jsonl.gz", tmp_path / "C.JSONL"])) == [
        ("d", ITEM),
        ("d", ITEM),
    ]


def test_gzip_corpus_fails_the_gate(tmp_path, ev):
    with gzip.open(tmp_path / "c.jsonl.gz", "wt") as fh:
        fh.write(json.dumps({"id": "d", "text": ITEM}) + "\n")
    assert main(["scan", str(ev), str(tmp_path / "c.jsonl.gz"), "--fail-above", "0"]) == 1


def test_unknown_suffix_is_an_error_unless_text_files(tmp_path, ev, capsys):
    f = tmp_path / "c.json"
    f.write_text(json.dumps([{"text": ITEM}]))
    assert main(["scan", str(ev), str(f)]) == 2
    assert "unsupported corpus file type" in capsys.readouterr().err
    assert main(["scan", str(ev), str(f), "--text-files", "--fail-above", "0"]) == 1


def test_skipped_files_in_a_directory_are_counted(tmp_path, ev, capsys):
    d = tmp_path / "corpus"
    (d / "sub").mkdir(parents=True)
    _write(d / "a.jsonl", [{"text": ITEM}])
    (d / "b.json").write_text("[]")
    (d / "sub" / "c.bin").write_bytes(b"\x00")
    (d / "sub" / "README.md").write_text("readme")
    skipped: Counter[str] = Counter()
    docs = list(iter_corpus([d], skipped=skipped))
    assert docs == [("a.jsonl:0", ITEM), ("sub/README.md", "readme")]
    assert skipped == Counter({".json": 1, ".bin": 1})
    assert main(["scan", str(ev), str(d)]) == 0
    assert "skipped 2 files" in capsys.readouterr().err


def test_document_ids_are_relative_to_the_corpus_argument(tmp_path):
    for split in ("train", "test"):
        (tmp_path / split).mkdir()
        _write(tmp_path / split / "part-0.jsonl", [{"text": "x"}])
    ids = [i for i, _ in iter_corpus([tmp_path])]
    assert ids == ["test/part-0.jsonl:0", "train/part-0.jsonl:0"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"id": "d1", "text": "a"}\n{"id":"d2","text":"trunc', "c.jsonl:2: not valid JSON"),
        (b'{"id": "d1", "text": "a\xff"}\n', "c.jsonl:1: not valid UTF-8"),
        (b"[1, 2]\n", "c.jsonl:1: expected a JSON object"),
    ],
)
@pytest.mark.parametrize("workers", ["1", "2"])
def test_malformed_corpus_is_one_line_error_with_status_2(
    tmp_path, ev, capsys, content, message, workers
):
    f = tmp_path / "c.jsonl"
    f.write_bytes(content)
    assert main(["scan", str(ev), str(f), "--workers", workers]) == 2
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err
    assert err.startswith("contamscan: error:")


def test_report_directory_and_eval_path_are_checked_before_the_scan(tmp_path, ev, capsys):
    corpus = _write(tmp_path / "c.jsonl", [{"text": ITEM}])
    with pytest.raises(SystemExit) as e:
        main(["scan", str(ev), str(corpus), "--json", str(tmp_path / "nodir" / "r.json")])
    assert e.value.code == 2
    assert "nodir" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["scan", str(tmp_path), str(corpus)])
    assert e.value.code == 2
    assert "not a file" in capsys.readouterr().err


def test_empty_evaluation_set_is_an_error(tmp_path, capsys):
    ev = tmp_path / "e.jsonl"
    ev.write_text("")
    corpus = _write(tmp_path / "c.jsonl", [{"text": ITEM}])
    with pytest.raises(SystemExit) as e:
        main(["scan", str(ev), str(corpus), "--fail-above", "0"])
    assert e.value.code == 2
    assert "no records" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "[1]", '"x"', "{}"])
def test_scores_must_be_finite_numbers(tmp_path, bad):
    ev = tmp_path / "e.jsonl"
    ev.write_text('{"id": "a", "text": "x", "s": 1}\n{"id": "b", "text": "y", "s": ' + bad + "}\n")
    with pytest.raises(ValueError, match="e.jsonl:2: score .* is not a finite number"):
        load_eval(ev, ["text"], score_field="s")
    assert [i.score for i in load_eval(ev, ["text"])] == [None, None]


def test_numeric_string_scores_are_accepted(tmp_path):
    ev = tmp_path / "e.jsonl"
    ev.write_text('{"id": "a", "text": "x", "s": "0.5"}\n')
    assert load_eval(ev, ["text"], score_field="s")[0].score == 0.5


def test_parquet_reads_only_the_needed_columns(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    f = tmp_path / "c.parquet"
    pq.write_table(pa.table({"id": ["a"], "text": ["t"], "wide": ["x" * 100]}), f)
    seen = []
    orig = pq.ParquetFile.iter_batches

    def spy(self, *args, **kw):
        seen.append(kw.get("columns"))
        return orig(self, *args, **kw)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    assert list(iter_corpus([f], ["text", "missing"])) == [("a", "t")]
    assert sorted(seen[0]) == ["id", "text"]
