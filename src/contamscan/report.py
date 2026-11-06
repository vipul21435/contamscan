"""Summaries and JSON / Markdown reports."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Any

from .bootstrap import compare
from .methods import Params
from .scan import ItemResult

__all__ = ["CONTAMINATED", "summarise", "comparisons", "to_json", "to_markdown"]

# which labels count as contaminated for each method
CONTAMINATED = {
    "gpt3": {"dirty"},
    "palm": {"contaminated"},
    "llama2": {"dirty"},
    "substring": {"exact", "long-match"},
}

_LABEL_ATTR = {
    "gpt3": "gpt3_label",
    "palm": "palm_label",
    "llama2": "llama2_label",
    "substring": "substring_label",
}


def is_contaminated(r: ItemResult, method: str) -> bool:
    return getattr(r, _LABEL_ATTR[method]) in CONTAMINATED[method]


def summarise(results: list[ItemResult]) -> dict[str, Any]:
    n = len(results)
    out: dict[str, Any] = {"items": n, "methods": {}}
    for m, attr in _LABEL_ATTR.items():
        counts = Counter(getattr(r, attr) for r in results)
        flagged = sum(is_contaminated(r, m) for r in results)
        out["methods"][m] = {
            "labels": dict(sorted(counts.items())),
            "contaminated": flagged,
            "fraction": flagged / n if n else 0.0,
        }
    out["any_overlap"] = sum(r.n_matching_docs > 0 for r in results)
    return out


def comparisons(results: list[ItemResult], *, n_boot: int = 10_000, seed: int = 0):
    """Score comparisons per method, when every item has a score.

    For ``llama2`` two splits are reported, as in the Llama 2 analysis: clean versus not
    clean, and dirty versus not dirty.
    """
    if not results or any(r.score is None for r in results):
        return {}
    out: dict[str, Any] = {}
    for m in _LABEL_ATTR:
        dirty = [r.score for r in results if is_contaminated(r, m)]
        clean = [r.score for r in results if not is_contaminated(r, m)]
        c = compare(clean, dirty, n_boot=n_boot, seed=seed)
        out[m] = asdict(c) if c else None
    clean = [r.score for r in results if r.llama2_label == "clean"]
    rest = [r.score for r in results if r.llama2_label != "clean"]
    c = compare(clean, rest, n_boot=n_boot, seed=seed)
    out["llama2_not_clean_vs_clean"] = asdict(c) if c else None
    return out


def to_json(
    results: list[ItemResult], params: Params, meta: dict[str, Any], comp: dict[str, Any]
) -> str:
    doc = {
        "params": asdict(params),
        "meta": meta,
        "summary": summarise(results),
        "comparison": comp,
        "items": [asdict(r) for r in results],
    }
    return json.dumps(doc, indent=2)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(
    results: list[ItemResult],
    params: Params,
    meta: dict[str, Any],
    comp: dict[str, Any],
    *,
    examples: int = 10,
) -> str:
    s = summarise(results)
    lines = [
        "# Contamination report",
        "",
        f"Evaluation items: {s['items']}; corpus documents scanned: {meta.get('documents', '?')}; "
        f"tokenizer: {meta.get('tokenizer', '?')}; scan time: {meta.get('seconds', 0):.1f} s.",
        "",
        "| method | rule | contaminated | fraction | labels |",
        "|---|---|---:|---:|---|",
    ]
    rules = {
        "gpt3": f"any {params.gpt3_n}-gram shared",
        "palm": f">= {params.palm_threshold:.0%} of {params.palm_n}-grams shared",
        "llama2": f">= {params.llama_dirty:.0%} of tokens in shared {params.llama_n}-grams",
        "substring": f"exact containment or a shared run >= {params.substring_min_tokens} tokens",
    }
    for m, v in s["methods"].items():
        labels = ", ".join(f"{k}={c}" for k, c in v["labels"].items())
        lines.append(
            f"| {m} | {rules[m]} | {v['contaminated']} | {_pct(v['fraction'])} | {labels} |"
        )
    lines += ["", f"Items with any shared n-gram: {s['any_overlap']}.", ""]
    if comp:
        lines += [
            "## Scores on clean and contaminated items",
            "",
            "| split | n clean | n contaminated | mean clean | mean contaminated | "
            "difference | 95% interval |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for m, c in comp.items():
            if c is None:
                lines.append(f"| {m} | - | - | - | - | - | one group empty |")
                continue
            lines.append(
                f"| {m} | {c['n_clean']} | {c['n_contaminated']} | {c['mean_clean']:.3f} | "
                f"{c['mean_contaminated']:.3f} | {c['difference']:+.3f} | "
                f"[{c['low']:+.3f}, {c['high']:+.3f}] |"
            )
        lines.append("")
    ranked = sorted(
        (r for r in results if r.matches),
        key=lambda r: (-r.llama2_coverage, -r.longest_match_tokens),
    )[:examples]
    if ranked:
        lines += ["## Items with the most overlap", ""]
        for r in ranked:
            m = r.matches[0]
            snippet = m.snippet.replace("`", "'").replace("\n", " ")
            lines += [
                f"### {r.id}",
                "",
                f"gpt3 {_pct(r.gpt3_overlap)} ({r.gpt3_label}), palm {_pct(r.palm_overlap)} "
                f"({r.palm_label}), llama2 {_pct(r.llama2_coverage)} ({r.llama2_label}), "
                f"longest shared run {r.longest_match_tokens} tokens, "
                f"{r.n_matching_docs} matching documents.",
                "",
                f"Best match: document `{m.doc_id}` characters {m.doc_start}-{m.doc_end}"
                + (" (exact containment)" if m.exact else "")
                + (" (normalised containment)" if m.exact_normalised and not m.exact else ""),
                "",
                f"> {snippet}",
                "",
            ]
    return "\n".join(lines)
