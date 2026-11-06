"""Plant edited HumanEval items in CodeAlpaca-20k and measure what each method finds.

Usage: python examples/planted_recall.py DATA_DIR OUT_DIR [SEED]

DATA_DIR/jsonl must hold humaneval.jsonl and code_alpaca_20k.jsonl (from
prepare_code_data.py). 100 of the 164 HumanEval items are planted, 20 per kind, each
appended to the output of a different, randomly chosen CodeAlpaca record:

* full: prompt and canonical solution, verbatim
* prompt: the prompt (signature and docstring) only
* solution: the canonical solution only
* recased: full text upper-cased and re-indented with tabs
* edited: full text with every 10th code token replaced by another identifier

The other 64 items are not planted; flags on them come from overlap that CodeAlpaca
already has. Prints a Markdown table of how many items of each kind every method flags.
"""

import json
import random
import sys
import time
from pathlib import Path

from contamscan.cli import main as cli_main
from contamscan.text import tokenize

KINDS = ("full", "prompt", "solution", "recased", "edited")


def edit_every(text: str, k: int) -> str:
    toks = tokenize(text, "code")
    out, last = [], 0
    for idx, t in enumerate(toks):
        if idx % k == k - 1:
            out.append(text[last : t.start] + "zq")
            last = t.end
    out.append(text[last:])
    return "".join(out)


def plant(item: dict, kind: str) -> str:
    full = item["prompt"] + item["solution"]
    if kind == "full":
        return full
    if kind == "prompt":
        return item["prompt"]
    if kind == "solution":
        return item["solution"]
    if kind == "recased":
        return full.upper().replace("    ", "\t")
    return edit_every(full, 10)


def main(data: Path, out: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    items = [json.loads(x) for x in (data / "jsonl/humaneval.jsonl").read_text().splitlines()]
    docs = [json.loads(x) for x in (data / "jsonl/code_alpaca_20k.jsonl").read_text().splitlines()]
    order = list(range(len(items)))
    rng.shuffle(order)
    kind_of = {}
    hosts = rng.sample(range(len(docs)), 100)
    for n, idx in enumerate(order[:100]):
        kind = KINDS[n // 20]
        kind_of[items[idx]["id"]] = kind
        doc = docs[hosts[n]]
        doc["output"] = (doc.get("output") or "") + "\n" + plant(items[idx], kind)
    out.mkdir(parents=True, exist_ok=True)
    corpus = out / "planted_corpus.jsonl"
    corpus.write_text("".join(json.dumps(d) + "\n" for d in docs))
    rows = []
    for tok in ("word", "code"):
        for extra in ([], ["--llama-skip-budget", "4"]):
            report = out / f"planted-{tok}{'-skip4' if extra else ''}.json"
            t0 = time.perf_counter()
            cli_main(
                ["scan", str(data / "jsonl/humaneval.jsonl"), str(corpus)]
                + ["--fields", "prompt,solution", "--corpus-fields", "instruction,input,output"]
                + ["--tokenizer", tok, "--workers", "4", "--json", str(report)]
                + extra
            )
            secs = time.perf_counter() - t0
            rep = json.loads(report.read_text())
            for kind in (*KINDS, "not planted"):
                sel = [it for it in rep["items"] if kind_of.get(it["id"], "not planted") == kind]
                rows.append(
                    (
                        tok + (" skip4" if extra else ""),
                        kind,
                        len(sel),
                        sum(it["gpt3_label"] == "dirty" for it in sel),
                        sum(it["palm_label"] == "contaminated" for it in sel),
                        sum(it["llama2_label"] == "dirty" for it in sel),
                        sum(it["llama2_label"] != "clean" for it in sel),
                        sum(it["substring_label"] != "clean" for it in sel),
                        secs,
                    )
                )
    print(
        "| tokenizer | planted as | items | gpt3 dirty | palm | llama2 dirty "
        "| llama2 not clean | substring | scan s |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r[:-1]) + f" | {r[-1]:.1f} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
