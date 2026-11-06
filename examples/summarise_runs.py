"""Print a Markdown table of the reports written by run_code_audit.sh.

Usage: python examples/summarise_runs.py OUT_DIR
"""

import json
import sys
from pathlib import Path


def main(out: Path) -> None:
    times = dict(line.split("\t") for line in (out / "times.tsv").read_text().splitlines())
    print(
        "| eval | corpus | tokenizer | flags | N | gpt3 dirty | palm | llama2 dirty "
        "| llama2 not clean | substring | any overlap | scan s | wall s |"
    )
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in times:
        rep = json.loads((out / f"{name}.json").read_text())
        ev, corpus, tok, variant = name.rsplit("-", 3)
        s = rep["summary"]
        m = s["methods"]
        n = s["items"]

        def cell(k: int, n: int = n) -> str:
            return f"{k} ({100 * k / n:.1f} %)" if k else "0"

        print(
            f"| {ev} | {corpus} | {tok} | {variant} | {rep['params']['gpt3_n']} "
            f"| {cell(m['gpt3']['contaminated'])} | {cell(m['palm']['contaminated'])} "
            f"| {cell(m['llama2']['contaminated'])} | {cell(s['llama2_subsets']['not clean'])} "
            f"| {cell(m['substring']['contaminated'])} | {s['any_overlap']} "
            f"| {rep['meta']['seconds']:.1f} | {times[name]} |"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
