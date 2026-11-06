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
    assert rep["comparison"]["gpt3"]["n_contaminated"] == 2
    md = out_md.read_text()
    assert "### q0" in md and "exact containment" in md and "Scores on clean" in md
    assert "6 items, 2 documents" in capsys.readouterr().out


def test_fail_above_sets_exit_code(tmp_path):
    ev, corpus = make(tmp_path, scores=False)
    base = ["scan", str(ev), str(corpus), "--fields", "prompt"]
    assert main(base + ["--fail-above", "0.5", "--fail-method", "gpt3"]) == 0
    assert main(base + ["--fail-above", "0.2", "--fail-method", "gpt3"]) == 1
