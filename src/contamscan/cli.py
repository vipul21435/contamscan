"""Command-line interface: ``contamscan scan EVAL CORPUS... [options]``."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from . import __version__
from .io import iter_corpus, load_eval
from .methods import METHODS, Params, gpt3_auto_n
from .report import comparisons, summarise, to_json, to_markdown
from .scan import EvalItem, scan
from .text import TOKENIZERS, tokenize


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _n_or_auto(s: str) -> int | str:
    if s == "auto":
        return s
    try:
        n = int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected an integer or 'auto', got {s!r}") from e
    if n <= 0:
        raise argparse.ArgumentTypeError("n must be positive")
    return n


def _int_at_least(low: int):
    def parse(s: str) -> int:
        try:
            n = int(s)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"expected an integer, got {s!r}") from e
        if n < low:
            raise argparse.ArgumentTypeError(f"must be at least {low}, got {n}")
        return n

    parse.__name__ = "integer"
    return parse


_positive = _int_at_least(1)
_non_negative = _int_at_least(0)


def _fraction(s: str) -> float:
    try:
        x = float(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected a number, got {s!r}") from e
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {x}")
    return x


DESCRIPTION = (
    "Measure how much of an evaluation set appears in a training corpus. "
    "Run 'contamscan scan EVAL CORPUS... [options]'."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="contamscan", description=DESCRIPTION)
    p.add_argument("--version", action="version", version=f"contamscan {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser(
        "scan",
        help="scan a corpus for evaluation-set overlap",
        description="Scan a corpus for overlap with an evaluation set under the GPT-3, PaLM, "
        "Llama 2 and substring rules.",
    )
    s.add_argument("eval", type=Path, help="evaluation set (.jsonl or .parquet)")
    s.add_argument("corpus", type=Path, nargs="+", help="corpus files or directories")
    g = s.add_argument_group("evaluation set")
    g.add_argument(
        "--fields",
        type=_csv,
        default=["text"],
        help="comma-separated text fields to check, joined by newlines (default: text)",
    )
    g.add_argument("--id-field", default="id", help="item id field (default: id; else row number)")
    g.add_argument(
        "--score-field", default=None, help="per-item model score, e.g. pass@1 (default: none)"
    )
    g = s.add_argument_group("corpus")
    g.add_argument(
        "--corpus-fields",
        type=_csv,
        default=["text"],
        help="comma-separated JSONL/Parquet text fields, joined by newlines (default: text)",
    )
    g.add_argument(
        "--text-files",
        action="store_true",
        help="read corpus files named directly with any other suffix whole, as one text "
        "document each (default: such files are an error)",
    )
    g.add_argument(
        "--corpus-id-field",
        default="id",
        help="document id field (default: id; else FILE:ROW)",
    )
    g = s.add_argument_group("methods (defaults from the cited papers; see README)")
    g.add_argument(
        "--tokenizer",
        choices=TOKENIZERS,
        default="word",
        help="word: case-folded words, the GPT-3 rule; code: identifiers, numbers and "
        "operators, case kept (default: word)",
    )
    g.add_argument(
        "--gpt3-n",
        type=_n_or_auto,
        default=13,
        help="GPT-3 N (default 13, Brown et al. 2020 App. C); 'auto' = 5th-percentile item "
        "length clamped to [8, 13]",
    )
    g.add_argument(
        "--palm-n", type=_positive, default=8, help="PaLM n (default 8, Chowdhery et al. 2022)"
    )
    g.add_argument(
        "--palm-threshold",
        type=_fraction,
        default=0.70,
        help="PaLM share of matched n-grams (default 0.70, Chowdhery et al. 2022)",
    )
    g.add_argument(
        "--llama-n",
        type=_positive,
        default=11,
        help="Llama 2 minimum match length in tokens (default 11: 'longer than 10', "
        "Touvron et al. 2023 App. A.6)",
    )
    g.add_argument(
        "--llama-clean",
        type=_fraction,
        default=0.20,
        help="Llama 2 clean below this token coverage (default 0.20, Touvron et al. 2023)",
    )
    g.add_argument(
        "--llama-dirty",
        type=_fraction,
        default=0.80,
        help="Llama 2 dirty at or above this token coverage (default 0.80, Touvron et al. 2023)",
    )
    g.add_argument(
        "--llama-skip-budget",
        type=_non_negative,
        default=0,
        help="mismatched positions allowed inside a Llama 2 span (default 0 = exact; the "
        "paper used 4; approximate)",
    )
    g.add_argument(
        "--substring-min-tokens",
        type=_positive,
        default=50,
        help="shared run that counts as long-match (default 50, this tool's choice)",
    )
    g.add_argument(
        "--palm-per-field",
        action="store_true",
        help="PaLM fraction per text field; contaminated if any field reaches the threshold",
    )
    g.add_argument(
        "--max-doc-freq",
        type=_positive,
        default=None,
        help="ignore item n-grams found in more than this many documents (off; GPT-3 used 10)",
    )
    g = s.add_argument_group("output")
    g.add_argument(
        "--max-matches", type=_positive, default=5, help="matched documents kept per item (5)"
    )
    g.add_argument("--workers", type=_positive, default=1, help="processes for the corpus scan (1)")
    g.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    g.add_argument("--markdown", type=Path, default=None, help="write the Markdown report here")
    g.add_argument(
        "--examples", type=_non_negative, default=10, help="items shown in Markdown (10)"
    )
    g.add_argument(
        "--n-boot",
        type=_positive,
        default=10_000,
        help="bootstrap resamples for the score difference intervals (10000)",
    )
    g.add_argument("--seed", type=int, default=0, help="bootstrap random seed (0)")
    g.add_argument(
        "--llama-z-fpc",
        action="store_true",
        help="apply the finite-population correction (N-n)/(N-1) to the Llama 2 sigma_n "
        "(off: sigma_n = s/sqrt(n), the form the paper's Table 51 is consistent with)",
    )
    g.add_argument(
        "--fail-above",
        type=_fraction,
        default=None,
        help="exit 1 if the contaminated fraction under --fail-method exceeds this (off)",
    )
    g.add_argument(
        "--fail-method",
        choices=METHODS,
        default="palm",
        help="method checked by --fail-above (default: palm)",
    )
    return p


def _counting(docs: Iterable[tuple[str, str]], counter: Counter[str]) -> Iterator[tuple[str, str]]:
    """Pass documents through, counting them (``docs``) and those with no text (``empty``)."""
    for d in docs:
        counter["docs"] += 1
        if not d[1].strip():
            counter["empty"] += 1
        yield d


def _warn(msg: str) -> None:
    print(f"contamscan: warning: {msg}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.llama_clean > args.llama_dirty:
        parser.error("--llama-clean must not be above --llama-dirty")
    for path in (args.eval, *args.corpus):
        if not path.exists():
            parser.error(f"no such file or directory: {path}")
    if not args.eval.is_file():
        parser.error(f"evaluation set is not a file: {args.eval}")
    for out in (args.json, args.markdown):
        if out is not None and not out.resolve().parent.is_dir():
            parser.error(f"directory for the report does not exist: {out.parent}")
    missing: Counter[str] = Counter()
    try:
        items = load_eval(args.eval, args.fields, args.id_field, args.score_field, missing)
    except (ValueError, OSError) as e:
        parser.error(str(e))
    if not items:
        parser.error(f"evaluation set has no records: {args.eval}")
    for f in args.fields:
        if missing[f]:
            _warn(f"eval field {f!r} is missing from {missing[f]} of {len(items)} records")
    if args.score_field:
        unscored = sum(it.score is None for it in items)
        if unscored:
            _warn(
                f"{unscored} of {len(items)} items have no {args.score_field!r}; "
                "the score comparison needs a score for every item and is skipped"
            )
    gpt3_n = args.gpt3_n
    if gpt3_n == "auto":
        gpt3_n = gpt3_auto_n([len(tokenize(it.text, args.tokenizer)) for it in items])
    params = Params(
        gpt3_n=gpt3_n,
        palm_n=args.palm_n,
        palm_threshold=args.palm_threshold,
        llama_n=args.llama_n,
        llama_clean=args.llama_clean,
        llama_dirty=args.llama_dirty,
        substring_min_tokens=args.substring_min_tokens,
        max_doc_freq=args.max_doc_freq,
        palm_per_field=args.palm_per_field,
        llama_skip_budget=args.llama_skip_budget,
    )
    try:
        return _run(args, items, params)
    except (ValueError, OSError) as e:  # malformed corpus records, unreadable files
        print(f"contamscan: error: {e}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, items: list[EvalItem], params: Params) -> int:
    count: Counter[str] = Counter()
    fields_seen: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    corpus = iter_corpus(
        args.corpus,
        args.corpus_fields,
        args.corpus_id_field,
        fields_seen,
        text_files=args.text_files,
        skipped=skipped,
    )
    docs = _counting(corpus, count)
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
    if skipped:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(skipped.items()))
        _warn(
            f"skipped {sum(skipped.values())} files of unsupported types in corpus "
            f"directories ({kinds})"
        )
    n_records = fields_seen[""]
    for f in args.corpus_fields:
        if n_records and not fields_seen[f]:
            _warn(f"corpus field {f!r} is missing from all {n_records} records")
    if count["docs"] == count["empty"]:
        print(
            f"contamscan: error: none of the {count['docs']} corpus documents has any text; "
            "check --corpus-fields and the corpus paths",
            file=sys.stderr,
        )
        return 2
    if count["empty"]:
        _warn(f"{count['empty']} of {count['docs']} corpus documents have no text")
    meta = {
        "documents": count["docs"],
        "tokenizer": args.tokenizer,
        "seconds": round(seconds, 3),
        "llama_z_fpc": args.llama_z_fpc,
    }
    comp = comparisons(results, n_boot=args.n_boot, seed=args.seed, z_fpc=args.llama_z_fpc)
    if args.json:
        args.json.write_text(to_json(results, params, meta, comp))
    md = to_markdown(results, params, meta, comp, examples=args.examples)
    if args.markdown:
        args.markdown.write_text(md)
    summary = summarise(results)
    print(f"{summary['items']} items, {count['docs']} documents, {seconds:.1f} s")
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
