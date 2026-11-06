"""Seeded synthetic demonstration of the clean-versus-contaminated comparison.

Builds 200 evaluation items of random words, plants 60 of them (whole or in part) in a
corpus of random documents, and gives every item a simulated pass/fail score whose pass
rate is 0.45 on items that were not planted and 0.75 on fully planted items. The scan
should recover the planted items and the comparison should show the score gap.

Usage: python examples/synthetic_comparison.py OUT_DIR
Nothing here is real model output.
"""

import json
import random
import sys
from pathlib import Path

from contamscan.cli import main as cli_main


def build(out: Path, seed: int = 0) -> tuple[Path, Path]:
    rng = random.Random(seed)
    vocab = [f"v{i}" for i in range(5000)]
    items, docs = [], []
    for k in range(200):
        words = [rng.choice(vocab) for _ in range(rng.randint(30, 80))]
        planted = "full" if k < 40 else "part" if k < 60 else "none"
        if planted == "full":
            docs.append(" ".join([rng.choice(vocab) for _ in range(20)] + words))
        elif planted == "part":
            docs.append(" ".join(words[: len(words) // 3]))
        p = 0.75 if planted == "full" else 0.45
        items.append(
            {"id": f"item-{k:03d}", "question": " ".join(words), "pass": int(rng.random() < p)}
        )
    for _ in range(2000):
        docs.append(" ".join(rng.choice(vocab) for _ in range(rng.randint(50, 300))))
    rng.shuffle(docs)
    out.mkdir(parents=True, exist_ok=True)
    ev, corpus = out / "eval.jsonl", out / "corpus.jsonl"
    ev.write_text("".join(json.dumps(r) + "\n" for r in items))
    corpus.write_text(
        "".join(json.dumps({"id": f"doc-{i}", "text": d}) + "\n" for i, d in enumerate(docs))
    )
    return ev, corpus


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/synthetic")
    ev, corpus = build(out)
    sys.exit(
        cli_main(
            [
                "scan",
                str(ev),
                str(corpus),
                "--fields",
                "question",
                "--score-field",
                "pass",
                "--json",
                str(out / "report.json"),
                "--markdown",
                str(out / "report.md"),
            ]
        )
    )
