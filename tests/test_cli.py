import json

from contamscan.cli import main

WORDS = [f"w{i}" for i in range(300)]


def seg(a, b):
    return " ".join(WORDS[a:b])


def make(tmp_path, scores=True):
    ev = tmp_path / "eval.jsonl"
    rows = []
    for k in range(6):
        row = {"id": f"q{k}", "prompt": seg(40 * k, 40 * k + 30)}
        if scores:
            row["pass"] = k % 2
        rows.append(row)
    ev.write_text("\n".join(json.dumps(r) for r in rows))
    corpus = tmp_path / "corpus.jsonl"
    docs = [{"id": "d0", "text": "intro " + seg(0, 30)}, {"id": "d1", "text": seg(80, 95)}]
    corpus.write_text("\n".join(json.dumps(d) for d in docs))
    return ev, corpus


def test_scan_writes_reports(tmp_path, capsys):
    ev, corpus = make(tmp_path)
    out_json, out_md = tmp_path / "r.json", tmp_path / "r.md"
    rc = main(
        [
            "scan",
            str(ev),
            str(corpus),
            "--fields",
            "prompt",
            "--score-field",
            "pass",
            "--json",
            str(out_json),
            "--markdown",
            str(out_md),
            "--n-boot",
            "200",
        ]
    )
    assert rc == 0
    rep = json.loads(out_json.read_text())
    assert rep["summary"]["items"] == 6
    assert rep["summary"]["methods"]["llama2"]["contaminated"] == 1
    assert rep["summary"]["methods"]["gpt3"]["contaminated"] == 2
    q0 = rep["items"][0]
    assert q0["matches"][0]["doc_id"] == "d0" and q0["matches"][0]["exact"]
    comp = rep["comparison"]
    assert comp["differences"]["gpt3"]["n_contaminated"] == 2
    subsets = {z["subset"]: z for z in comp["llama2_subsets"]}
    assert subsets["dirty"]["n"] == 1 and subsets["not dirty"]["n"] == 5
    # q2 shares 15 of its 30 words with d1: not clean, not dirty
    assert subsets["clean"]["n"] == 4 and subsets["not clean"]["n"] == 2
    assert comp["gpt3_clean_vs_full"]["n_clean"] == 4
    assert rep["summary"]["llama2_subsets"] == {
        "clean": 4,
        "not clean": 2,
        "not dirty": 5,
        "dirty": 1,
    }
    md = out_md.read_text()
    assert "### q0" in md and "exact containment" in md and "Scores on clean" in md
    assert "| dirty | 1 |" in md and "relative change" in md
    assert "6 items, 2 documents" in capsys.readouterr().out


def test_fail_above_sets_exit_code(tmp_path):
    ev, corpus = make(tmp_path, scores=False)
    base = ["scan", str(ev), str(corpus), "--fields", "prompt"]
    assert main(base + ["--fail-above", "0.5", "--fail-method", "gpt3"]) == 0
    assert main(base + ["--fail-above", "0.2", "--fail-method", "gpt3"]) == 1


def test_gpt3_auto_n_and_bad_n(tmp_path):
    import pytest

    ev, corpus = make(tmp_path, scores=False)
    out = tmp_path / "r.json"
    args = ["scan", str(ev), str(corpus), "--fields", "prompt", "--json", str(out)]
    assert main(args + ["--gpt3-n", "auto"]) == 0
    assert json.loads(out.read_text())["params"]["gpt3_n"] == 13  # every item has 30 words
    for bad in ("x", "0"):
        with pytest.raises(SystemExit):
            main(args + ["--gpt3-n", bad])


def test_paper_flags_and_empty_groups(tmp_path):
    ev = tmp_path / "eval.jsonl"
    rows = [{"id": f"q{k}", "prompt": seg(20 * k, 20 * k + 15), "pass": k % 2} for k in range(4)]
    ev.write_text("\n".join(json.dumps(r) for r in rows))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"id": "d0", "text": "unrelated words only"}))
    out_json, out_md = tmp_path / "r.json", tmp_path / "r.md"
    rc = main(
        [
            "scan",
            str(ev),
            str(corpus),
            "--fields",
            "prompt",
            "--score-field",
            "pass",
            "--palm-per-field",
            "--max-doc-freq",
            "10",
            "--llama-skip-budget",
            "4",
            "--json",
            str(out_json),
            "--markdown",
            str(out_md),
            "--n-boot",
            "100",
        ]
    )
    assert rc == 0
    rep = json.loads(out_json.read_text())
    assert rep["params"]["palm_per_field"] and rep["params"]["llama_skip_budget"] == 4
    assert rep["items"][0]["palm_fields"] == {"prompt": 0.0}
    assert rep["comparison"]["differences"]["gpt3"] is None  # nothing contaminated
    dirty = [z for z in rep["comparison"]["llama2_subsets"] if z["subset"] == "dirty"][0]
    assert dirty["n"] == 0
    md = out_md.read_text()
    assert "one group empty" in md and "| dirty | 0 |" in md


def test_misspelt_eval_field_is_an_error(tmp_path, capsys):
    import pytest

    ev, corpus = make(tmp_path, scores=False)
    base = ["scan", str(ev), str(corpus), "--fail-above", "0.0", "--fail-method", "gpt3"]
    with pytest.raises(SystemExit) as e:
        main(base + ["--fields", "promt,soluton"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "'promt', 'soluton'" in err and "fields found: id, prompt" in err


def test_eval_field_missing_from_some_records_warns(tmp_path, capsys):
    ev, corpus = make(tmp_path, scores=False)
    rows = [json.loads(x) for x in ev.read_text().splitlines()]
    rows[0]["extra"] = "w0 w1"
    ev.write_text("\n".join(json.dumps(r) for r in rows))
    assert main(["scan", str(ev), str(corpus), "--fields", "prompt,extra"]) == 0
    assert "'extra' is missing from 5 of 6 records" in capsys.readouterr().err


def test_misspelt_corpus_field_fails_the_gate(tmp_path, capsys):
    ev, corpus = make(tmp_path, scores=False)
    args = ["scan", str(ev), str(corpus), "--fields", "prompt", "--corpus-fields", "txt"]
    assert main(args + ["--fail-above", "0.0", "--fail-method", "gpt3"]) == 2
    err = capsys.readouterr().err
    assert "corpus field 'txt' is missing from all 2 records" in err
    assert "none of the 2 corpus documents has any text" in err


def test_some_empty_corpus_documents_warn(tmp_path, capsys):
    ev, corpus = make(tmp_path, scores=False)
    corpus.write_text(corpus.read_text() + "\n" + json.dumps({"id": "d2", "text": "  "}))
    assert main(["scan", str(ev), str(corpus), "--fields", "prompt"]) == 0
    assert "1 of 3 corpus documents have no text" in capsys.readouterr().err
