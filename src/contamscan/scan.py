"""Index the evaluation set and stream a corpus against it once.

The index maps each 64-bit n-gram hash of the evaluation set to the (item, position)
pairs where it occurs, one table per n. Each corpus document is tokenised, its n-gram
hashes are looked up, and the hits are folded into per-item accumulators. Memory is
proportional to the evaluation set, not the corpus.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .hashing import ngram_hash, ngram_hashes
from .methods import Params, label_gpt3, label_llama2, label_palm, label_substring
from .text import normalise, tokenize

__all__ = ["EvalItem", "Match", "ItemResult", "EvalIndex", "scan"]


@dataclass(frozen=True)
class EvalItem:
    id: str
    text: str
    score: float | None = None


@dataclass
class Match:
    """Longest span one corpus document shares with one item."""

    doc_id: str
    tokens: int
    item_start: int
    item_end: int
    doc_start: int
    doc_end: int
    exact: bool = False
    exact_normalised: bool = False
    snippet: str = ""


@dataclass
class ItemResult:
    id: str
    n_tokens: int
    score: float | None
    gpt3_overlap: float
    gpt3_label: str
    palm_overlap: float | None
    palm_label: str
    llama2_coverage: float
    llama2_label: str
    longest_match_tokens: int
    substring_label: str
    n_matching_docs: int
    matches: list[Match] = field(default_factory=list)


@dataclass
class _DocHits:
    # item -> n -> set of matched n-gram start positions in the item
    positions: dict[int, dict[int, set[int]]]
    # item -> longest span (tokens, item_tok_start, doc_tok_start)
    spans: dict[int, tuple[int, int, int]]
    # item -> (exact, exact_normalised)
    exact: dict[int, tuple[bool, bool]]


class EvalIndex:
    def __init__(self, items: list[EvalItem], params: Params, tokenizer: str = "word"):
        self.items = items
        self.params = params
        self.tokenizer = tokenizer
        self.tokens = [tokenize(it.text, tokenizer) for it in items]
        self.norm_texts = [normalise(it.text) for it in items]
        self.tables: dict[int, dict[int, list[tuple[int, int]]]] = {}
        for n in params.ns():
            table: dict[int, list[tuple[int, int]]] = {}
            for i, toks in enumerate(self.tokens):
                for p, h in enumerate(ngram_hashes([t.text for t in toks], n)):
                    table.setdefault(h, []).append((i, p))
            self.tables[n] = table
        # GPT-3 rule: an item shorter than N is matched as a whole sequence.
        self.short: dict[int, dict[int, list[int]]] = {}
        for i, toks in enumerate(self.tokens):
            k = len(toks)
            if 0 < k < params.gpt3_n:
                h = ngram_hash([t.text for t in toks])
                self.short.setdefault(k, {}).setdefault(h, []).append(i)

    def n_grams(self, i: int, n: int) -> int:
        return max(0, len(self.tokens[i]) - n + 1)

    def scan_doc(self, text: str) -> _DocHits:
        toks = tokenize(text, self.tokenizer)
        words = [t.text for t in toks]
        positions: dict[int, dict[int, set[int]]] = {}
        span_n = self.params.span_n()
        pairs: dict[int, set[tuple[int, int]]] = {}
        for n, table in self.tables.items():
            for j, h in enumerate(ngram_hashes(words, n)):
                hits = table.get(h)
                if not hits:
                    continue
                for i, p in hits:
                    positions.setdefault(i, {}).setdefault(n, set()).add(p)
                    if n == span_n:
                        pairs.setdefault(i, set()).add((p, j))
        for k, table in self.short.items():
            for j, h in enumerate(ngram_hashes(words, k)):
                for i in table.get(h, ()):
                    positions.setdefault(i, {}).setdefault(-1, set()).add(0)
                    pairs.setdefault(i, set()).add((-1 - j, j))  # marker for whole-item hit
        spans: dict[int, tuple[int, int, int]] = {}
        for i, ps in pairs.items():
            best = (0, 0, 0)
            for p, j in ps:
                if p < 0:  # whole short item matched at doc position j
                    cand = (len(self.tokens[i]), 0, j)
                elif (p - 1, j - 1) in ps:
                    continue  # not the start of a run
                else:
                    r = 1
                    while (p + r, j + r) in ps:
                        r += 1
                    cand = (r + span_n - 1, p, j)
                if cand[0] > best[0]:
                    best = cand
            spans[i] = best
        exact: dict[int, tuple[bool, bool]] = {}
        if positions:
            norm_doc = normalise(text)
            for i in positions:
                raw = self.items[i].text
                nt = self.norm_texts[i]
                exact[i] = (bool(raw) and raw in text, bool(nt) and nt in norm_doc)
        self._last_doc_tokens = toks
        return _DocHits(positions, spans, exact)


_WORKER_INDEX: EvalIndex | None = None


def _init_worker(index: EvalIndex) -> None:
    global _WORKER_INDEX
    _WORKER_INDEX = index


def _work(doc: tuple[str, str]) -> tuple[str, _DocHits, list[tuple[int, int]]]:
    assert _WORKER_INDEX is not None
    hits = _WORKER_INDEX.scan_doc(doc[1])
    toks = _WORKER_INDEX._last_doc_tokens
    offsets = [(t.start, t.end) for t in toks] if hits.spans else []
    return doc[0], hits, offsets


def scan(
    items: list[EvalItem],
    corpus: Iterable[tuple[str, str]],
    params: Params | None = None,
    *,
    tokenizer: str = "word",
    max_matches: int = 5,
    workers: int = 1,
    snippet_chars: int = 240,
) -> list[ItemResult]:
    """Scan ``corpus`` (an iterable of ``(doc_id, text)``) for overlap with ``items``."""
    params = params or Params()
    index = EvalIndex(items, params, tokenizer)
    n_items = len(items)
    matched: dict[int, list[set[int]]] = {n: [set() for _ in range(n_items)] for n in params.ns()}
    short_hit = [False] * n_items
    doc_count = [0] * n_items
    best: list[list[Match]] = [[] for _ in range(n_items)]

    def results() -> Iterator[tuple[str, _DocHits, list[tuple[int, int]]]]:
        if workers <= 1:
            _init_worker(index)
            for doc in corpus:
                yield _work(doc)
        else:
            with mp.get_context().Pool(workers, _init_worker, (index,)) as pool:
                yield from pool.imap(_work, corpus, chunksize=64)

    for doc_id, hits, offsets in results():
        for i, by_n in hits.positions.items():
            doc_count[i] += 1
            for n, ps in by_n.items():
                if n == -1:
                    short_hit[i] = True
                else:
                    matched[n][i].update(ps)
            span = hits.spans.get(i)
            ex, exn = hits.exact.get(i, (False, False))
            if span is None or span[0] == 0:
                continue
            length, p, j = span
            itoks = index.tokens[i]
            m = Match(
                doc_id=doc_id,
                tokens=length,
                item_start=itoks[p].start,
                item_end=itoks[min(p + length, len(itoks)) - 1].end,
                doc_start=offsets[j][0],
                doc_end=offsets[min(j + length, len(offsets)) - 1][1],
                exact=ex,
                exact_normalised=exn,
            )
            m.snippet = items[i].text[m.item_start : m.item_end][:snippet_chars]
            lst = best[i]
            lst.append(m)
            lst.sort(key=lambda x: (-x.exact, -x.tokens))
            del lst[max_matches:]

    out: list[ItemResult] = []
    for i, it in enumerate(items):
        k = len(index.tokens[i])
        g_total = index.n_grams(i, params.gpt3_n)
        g_hit = len(matched[params.gpt3_n][i])
        if g_total == 0:
            g_frac = 1.0 if short_hit[i] else 0.0
            g_hit = int(short_hit[i])
        else:
            g_frac = g_hit / g_total
        p_total = index.n_grams(i, params.palm_n)
        p_frac = len(matched[params.palm_n][i]) / p_total if p_total else None
        covered: set[int] = set()
        for p in matched[params.llama_n][i]:
            covered.update(range(p, p + params.llama_n))
        cov = len(covered) / k if k else 0.0
        longest = max((m.tokens for m in best[i]), default=0)
        exact = any(m.exact or m.exact_normalised for m in best[i])
        out.append(
            ItemResult(
                id=it.id,
                n_tokens=k,
                score=it.score,
                gpt3_overlap=g_frac,
                gpt3_label=label_gpt3(g_hit),
                palm_overlap=p_frac,
                palm_label=label_palm(p_frac, params.palm_threshold),
                llama2_coverage=cov,
                llama2_label=label_llama2(cov, params.llama_clean, params.llama_dirty),
                longest_match_tokens=longest,
                substring_label=label_substring(longest, exact, params.substring_min_tokens),
                n_matching_docs=doc_count[i],
                matches=best[i],
            )
        )
    return out
