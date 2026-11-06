"""Command-line interface: ``contamscan scan EVAL CORPUS... [options]``."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .io import iter_corpus, load_eval
from .methods import METHODS, Params
from .report import comparisons, summarise, to_json, to_markdown
from .scan import scan
from .text import TOKENIZERS


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="contamscan", description=__doc__)
    p.add_argument("--version", action="version", version=f"contamscan {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="scan a corpus for evaluation-set overlap")
    s.add_argument("eval", type=Path, help="evaluation set (.jsonl or .parquet)")
    s.add_argument("corpus", type=Path, nargs="+", help="corpus files or directories")
    g = s.add_argument_group("evaluation set")
    g.add_argument("--fields", type=_csv, default=["text"], help="text fields to check")
    g.add_argument("--id-field", default="id")
    g.add_argument("--score-field", default=None, help="per-item model score (e.g. pass@1)")
    g = s.add_argument_group("corpus")
    g.add_argument("--corpus-fields", type=_csv, default=["text"])
    g.add_argument("--corpus-id-field", default="id")
    g = s.add_argument_group("methods (defaults from the cited papers; see README)")
    g.add_argument("--tokenizer", choices=TOKENIZERS, default="word")
    g.add_argument("--gpt3-n", type=int, default=13, help="GPT-3 N (default 13)")
    g.add_argument("--palm-n", type=int, default=8, help="PaLM n (default 8)")
    g.add_argument("--palm-threshold", type=float, default=0.70, help="PaLM (default 0.70)")
    g.add_argument("--llama-n", type=int, default=11, help="Llama 2: n-grams longer than 10")
    g.add_argument("--llama-clean", type=float, default=0.20)
    g.add_argument("--llama-dirty", type=float, default=0.80)
    g.add_argument("--substring-min-tokens", type=int, default=50)
    g = s.add_argument_group("output")
    g.add_argument("--max-matches", type=int, default=5, help="matches kept per item")
    g.add_argument("--workers", type=int, default=1, help="processes for the corpus scan")
    g.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    g.add_argument("--markdown", type=Path, default=None, help="write the Markdown report here")
    g.add_argument("--examples", type=int, default=10, help="items shown in Markdown")
    g.add_argument("--n-boot", type=int, default=10_000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument(
        "--fail-above",
        type=float,
        default=None,
        help="exit 1 if the contaminated fraction under --fail-method exceeds this",
    )
    g.add_argument("--fail-method", choices=METHODS, default="palm")
    return p


def _counting(docs, counter):
    for d in docs:
        counter[0] += 1
        yield d


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = Params(
        gpt3_n=args.gpt3_n,
        palm_n=args.palm_n,
        palm_threshold=args.palm_threshold,
        llama_n=args.llama_n,
        llama_clean=args.llama_clean,
        llama_dirty=args.llama_dirty,
        substring_min_tokens=args.substring_min_tokens,
    )
    items = load_eval(args.eval, args.fields, args.id_field, args.score_field)
    count = [0]
    docs = _counting(iter_corpus(args.corpus, args.corpus_fields, args.corpus_id_field), count)
    t0 = time.perf_counter()
    results = scan(
        items,
        docs,
        params,
        tokenizer=args.tokenizer,
        max_matches=args.max_matches,
        workers=args.workers,
    )
    seconds = time.perf_counter() - t0
    meta = {"documents": count[0], "tokenizer": args.tokenizer, "seconds": round(seconds, 3)}
    comp = comparisons(results, n_boot=args.n_boot, seed=args.seed)
    if args.json:
        args.json.write_text(to_json(results, params, meta, comp))
    md = to_markdown(results, params, meta, comp, examples=args.examples)
    if args.markdown:
        args.markdown.write_text(md)
    summary = summarise(results)
    print(f"{summary['items']} items, {count[0]} documents, {seconds:.1f} s")
    for m, v in summary["methods"].items():
        print(f"  {m:<10} {v['contaminated']:>6} contaminated ({100 * v['fraction']:.1f}%)")
    if args.fail_above is not None:
        frac = summary["methods"][args.fail_method]["fraction"]
        if frac > args.fail_above:
            print(
                f"FAIL: {args.fail_method} fraction {frac:.3f} > {args.fail_above}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
