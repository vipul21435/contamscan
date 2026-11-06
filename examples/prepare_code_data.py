"""Convert the downloaded HumanEval, MBPP and CodeAlpaca-20k files to JSONL.

Usage: python examples/prepare_code_data.py RAW_DIR OUT_DIR

RAW_DIR must hold the files fetched by examples/fetch_code_data.sh. Needs pyarrow.
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main(raw: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    he = pq.read_table(raw / "humaneval.parquet").to_pylist()
    with (out / "humaneval.jsonl").open("w") as fh:
        for r in he:
            row = {"id": r["task_id"], "prompt": r["prompt"], "solution": r["canonical_solution"]}
            fh.write(json.dumps(row) + "\n")
    mb = pq.read_table(raw / "mbpp_full_test.parquet").to_pylist()
    with (out / "mbpp_test.jsonl").open("w") as fh:
        for r in mb:
            row = {"id": f"mbpp/{r['task_id']}", "prompt": r["text"], "solution": r["code"]}
            fh.write(json.dumps(row) + "\n")
    rows = json.loads((raw / "code_alpaca_20k.json").read_text())
    with (out / "code_alpaca_20k.jsonl").open("w") as fh:
        for k, r in enumerate(rows):
            fh.write(json.dumps({"id": f"alpaca-{k}", **r}) + "\n")
    print(f"humaneval {len(he)}, mbpp test {len(mb)}, code_alpaca {len(rows)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
