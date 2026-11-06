"""Summaries and JSON / Markdown reports."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Any

from .bootstrap import compare, relative_change, subset_z
from .methods import Params
from .scan import ItemResult

__all__ = [
    "CONTAMINATED",
    "LLAMA2_SUBSETS",
    "summarise",
    "comparisons",
    "to_json",
    "to_markdown",
]

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


# Llama 2 Appendix A.6: four non-disjoint subsets defined by token contamination
LLAMA2_SUBSETS = {
    "clean": {"clean"},
    "not clean": {"mixed", "dirty"},
    "not dirty": {"clean", "mixed"},
    "dirty": {"dirty"},
}


def is_contaminated(r: ItemResult, method: str) -> bool:
    """Whether ``r`` carries a contaminated label under ``method``."""
    return getattr(r, _LABEL_ATTR[method]) in CONTAMINATED[method]


def summarise(results: list[ItemResult]) -> dict[str, Any]:
    """Label counts and contaminated fraction per method, and the Llama 2 subset sizes."""
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
    out["llama2_subsets"] = {
        k: sum(r.llama2_label in labels for r in results) for k, labels in LLAMA2_SUBSETS.items()
    }
    out["any_overlap"] = sum(r.n_matching_docs > 0 for r in results)
    return out


def comparisons(
    results: list[ItemResult], *, n_boot: int = 10_000, seed: int = 0, z_fpc: bool = False
) -> dict[str, Any]:
    """Score comparisons, when every item has a score; ``{}`` otherwise.

    * ``differences``: for each method, ``mean(contaminated) - mean(clean)`` with a bootstrap
      interval. For ``llama2`` this is dirty versus not dirty; ``llama2_not_clean_vs_clean``
      adds the other Llama 2 split.
    * ``llama2_subsets``: mean score and ``Z_n`` of the four Llama 2 subsets against the
      whole set (the paper calls a set affected when all four have ``|Z_n| > 2``);
      ``z_fpc`` applies the finite-population correction to ``sigma_n``.
    * ``gpt3_clean_vs_full``: the GPT-3 relative change of the clean-only score against the
      score on all items.
    """
    scored = [(r, r.score) for r in results if r.score is not None]
    if not results or len(scored) < len(results):
        return {}
    scores = [x for _, x in scored]
    diffs: dict[str, Any] = {}
    for m in _LABEL_ATTR:
        dirty = [x for r, x in scored if is_contaminated(r, m)]
        clean = [x for r, x in scored if not is_contaminated(r, m)]
        c = compare(clean, dirty, n_boot=n_boot, seed=seed)
        diffs[m] = asdict(c) if c else None
    clean = [x for r, x in scored if r.llama2_label == "clean"]
    rest = [x for r, x in scored if r.llama2_label != "clean"]
    c = compare(clean, rest, n_boot=n_boot, seed=seed)
    diffs["llama2_not_clean_vs_clean"] = asdict(c) if c else None
    subsets = []
    for name, labels in LLAMA2_SUBSETS.items():
        z = subset_z(scores, [r.llama2_label in labels for r in results], name, fpc=z_fpc)
        subsets.append(asdict(z) if z else {"subset": name, "n": 0})
    gpt3_clean = [x for r, x in scored if not is_contaminated(r, "gpt3")]
    full_mean = sum(scores) / len(scores)
    gpt3 = {
        "n_clean": len(gpt3_clean),
        "n_full": len(scores),
        "mean_clean": sum(gpt3_clean) / len(gpt3_clean) if gpt3_clean else None,
        "mean_full": full_mean,
        "relative_change": relative_change(gpt3_clean, scores),
    }
    return {"differences": diffs, "llama2_subsets": subsets, "gpt3_clean_vs_full": gpt3}


def to_json(
    results: list[ItemResult], params: Params, meta: dict[str, Any], comp: dict[str, Any]
) -> str:
    """JSON report: parameters, run metadata, summary, score comparison and every item."""
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
    """Markdown report: summary tables, score comparison and the ``examples`` items with the
    most overlap and their matched documents."""
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
    lines += [
        "",
        f"Items with any shared n-gram or whole-item containment: {s['any_overlap']}.",
        "",
    ]
    ls = s["llama2_subsets"]
    lines += [
        f"Llama 2 subsets: clean {ls['clean']}, not clean {ls['not clean']}, "
        f"not dirty {ls['not dirty']}, dirty {ls['dirty']}.",
        "",
    ]
    if comp:
        lines += _markdown_scores(comp)
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


def _f(x: float | None, fmt: str = ".3f") -> str:
    return "n/a" if x is None else format(x, fmt)


def _markdown_scores(comp: dict[str, Any]) -> list[str]:
    lines = [
        "## Scores on clean and contaminated items",
        "",
        "| split | n clean | n contaminated | mean clean | mean contaminated | "
        "difference | 95% interval |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for m, c in comp["differences"].items():
        if c is None:
            lines.append(f"| {m} | - | - | - | - | - | one group empty |")
            continue
        lines.append(
            f"| {m} | {c['n_clean']} | {c['n_contaminated']} | {c['mean_clean']:.3f} | "
            f"{c['mean_contaminated']:.3f} | {c['difference']:+.3f} | "
            f"[{c['low']:+.3f}, {c['high']:+.3f}] |"
        )
    lines += [
        "",
        "Llama 2 subsets against the whole set (Z = (mean - mu_n) / sigma_n):",
        "",
        "| subset | n | mean | mu_n | sigma_n | Z |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for z in comp["llama2_subsets"]:
        if not z["n"]:
            lines.append(f"| {z['subset']} | 0 | - | - | - | - |")
            continue
        lines.append(
            f"| {z['subset']} | {z['n']} | {z['mean']:.3f} | {z['mu']:.3f} | "
            f"{z['sigma']:.3f} | {_f(z['z'], '+.2f')} |"
        )
    g = comp["gpt3_clean_vs_full"]
    lines += [
        "",
        f"GPT-3 clean-only score {_f(g['mean_clean'])} on {g['n_clean']} items against "
        f"{g['mean_full']:.3f} on all {g['n_full']}: relative change "
        f"{_f(None if g['relative_change'] is None else 100 * g['relative_change'], '+.1f')} %.",
        "",
    ]
    return lines
