"""Index the evaluation set and stream a corpus against it once.

The index maps each 64-bit n-gram hash of the evaluation set to the (item, position)
pairs where it occurs, one table per n. Each corpus document is tokenised, its n-gram
hashes are looked up, and the hits are folded into per-item accumulators. Memory is
proportional to the evaluation set, not the corpus.
"""

from __future__ import annotations

import multiprocessing as mp
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .hashing import ngram_hash, ngram_hashes
from .methods import Params, label_gpt3, label_llama2, label_palm, label_substring
from .text import Token, normalise, tokenize

__all__ = ["EvalItem", "Match", "ItemResult", "EvalIndex", "scan"]


@dataclass(frozen=True)
class EvalItem:
    id: str
    text: str
    score: float | None = None
    # (field name, start, end) character ranges of the fields joined into ``text``
    field_spans: tuple[tuple[str, int, int], ...] = ()


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
    palm_fields: dict[str, float | None] | None = None


@dataclass
class _DocHits:
    # item -> n -> set of matched n-gram start positions in the item
    positions: dict[int, dict[int, set[int]]]
    # item -> longest span (tokens, item_tok_start, doc_tok_start)
    spans: dict[int, tuple[int, int, int]]
    # item -> (exact, exact_normalised, doc char start, doc char end) for contained items
    contained: dict[int, tuple[bool, bool, int, int]]
    # character offsets of the document's tokens (filled only when ``spans`` is not empty)
    offsets: list[tuple[int, int]]


# Containment prefilter: the item's normalised words, minus the first and last (which may
# be cut mid-word inside a document), give an anchor of up to this many words that must
# occur as consecutive whole words of the normalised document.
ANCHOR_WORDS = 8
_CHUNK_RE = re.compile(r"\S+")


def _anchor(words: list[str]) -> list[str]:
    inner = words[1:-1] if len(words) >= 3 else words
    return inner[:ANCHOR_WORDS]


def _norm_word_spans(text: str) -> list[tuple[int, int]]:
    """Character range in ``text`` of every word of ``normalise(text)``, in order.

    Each whitespace-free chunk of ``text`` is normalised on its own; a chunk such as
    ``a.b`` that yields several words gives each of them the range of the whole chunk.
    """
    out = []
    for m in _CHUNK_RE.finditer(text):
        out.extend((m.start(), m.end()) for _ in normalise(m.group()).split())
    return out


class EvalIndex:
    def __init__(self, items: list[EvalItem], params: Params, tokenizer: str = "word"):
        self.items = items
        self.params = params
        self.tokenizer = tokenizer
        self.tokens = [tokenize(it.text, tokenizer) for it in items]
        self.norm_texts = [normalise(it.text) for it in items]
        self.tables: dict[int, dict[int, list[tuple[int, int]]]] = {}
        # n -> item -> hash of the n-gram starting at each position
        self.item_hashes: dict[int, list[list[int]]] = {}
        for n in params.ns():
            table: dict[int, list[tuple[int, int]]] = {}
            per_item = []
            for i, toks in enumerate(self.tokens):
                hs = ngram_hashes([t.text for t in toks], n)
                per_item.append(hs)
                for p, h in enumerate(hs):
                    table.setdefault(h, []).append((i, p))
            self.tables[n] = table
            self.item_hashes[n] = per_item
        # GPT-3 rule: an item shorter than N is matched as a whole sequence.
        self.short: dict[int, dict[int, list[int]]] = {}
        for i, toks in enumerate(self.tokens):
            k = len(toks)
            if 0 < k < params.gpt3_n:
                h = ngram_hash([t.text for t in toks])
                self.short.setdefault(k, {}).setdefault(h, []).append(i)
        # Exact-containment prefilter over normalised words, independent of the tokenizer:
        # anchor length -> anchor hash -> items.
        self.anchors: dict[int, dict[int, list[int]]] = {}
        for i, nt in enumerate(self.norm_texts):
            a = _anchor(nt.split())
            if a:
                self.anchors.setdefault(len(a), {}).setdefault(ngram_hash(a), []).append(i)

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
        budget = self.params.llama_skip_budget
        for i, ps in pairs.items():
            best = (0, 0, 0)
            runs: list[tuple[int, int, int]] = []  # (tokens, item start, doc start)
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
                    runs.append(cand)
                if cand[0] > best[0]:
                    best = cand
            spans[i] = best
            if budget > 0 and runs:
                covered = skipgram_cover(runs, budget, self.params.llama_n)
                if covered:
                    positions.setdefault(i, {})[SKIP_KEY] = covered
        contained = self._containment(text)
        offsets = [(t.start, t.end) for t in toks] if spans else []
        return _DocHits(positions, spans, contained, offsets)

    def _containment(self, text: str) -> dict[int, tuple[bool, bool, int, int]]:
        """Items whose whole text occurs in ``text``, raw or after ``normalise``.

        Every item is checked against every document, whether or not it shares an n-gram
        with it; the anchor lookup only narrows down which items to compare in full.
        """
        norm_doc = normalise(text)
        words = norm_doc.split()
        cands: set[int] = set()
        for k, table in self.anchors.items():
            for h in ngram_hashes(words, k):
                cands.update(table.get(h, ()))
        out: dict[int, tuple[bool, bool, int, int]] = {}
        spans: list[tuple[int, int]] | None = None
        for i in sorted(cands):
            raw, nt = self.items[i].text, self.norm_texts[i]
            q = text.find(raw) if raw else -1
            exn = nt in norm_doc
            if q >= 0:
                out[i] = (True, exn, q, q + len(raw))
            elif exn:
                if spans is None:
                    spans = _norm_word_spans(text)
                if len(spans) == len(words):
                    qn = norm_doc.find(nt)
                    w0 = norm_doc.count(" ", 0, qn)
                    w1 = norm_doc.count(" ", 0, qn + len(nt) - 1)
                    out[i] = (False, True, spans[w0][0], spans[w1][1])
                else:  # chunk-wise normalisation disagrees: report the whole document
                    out[i] = (False, True, 0, len(text))
        return out


SKIP_KEY = -2  # positions entry holding token positions covered by skipgram spans


def skipgram_cover(runs: list[tuple[int, int, int]], budget: int, n: int) -> set[int]:
    """Item token positions covered by Llama 2-style skipgram spans.

    ``runs`` are maximal exact runs ``(tokens, item start, doc start)`` between one item and
    one document. Runs on the same diagonal (same ``doc start - item start``) are joined
    while the tokens between them, counted as mismatches, total at most ``budget``. A span
    may not start with mismatches in its first ``n - 1`` tokens (its first run must be at
    least that long), ends on a run (no trailing mismatches) and counts when it has at
    least ``n`` tokens. Mismatches are only seen between runs of the chaining n-gram size,
    so this is an approximation of the paper's skipgrams: two mismatches separated by fewer
    matching tokens than that size use up the whole gap between them.
    """
    by_diag: dict[int, list[tuple[int, int]]] = {}
    for length, p, j in runs:
        by_diag.setdefault(j - p, []).append((p, p + length))
    head = n - 1
    covered: set[int] = set()
    for segs in by_diag.values():
        segs.sort()
        start, end = segs[0]
        first, used = end - start, 0
        spans = []
        for a, b in segs[1:]:
            gap = a - end
            if first >= head and used + gap <= budget:
                end, used = b, used + gap
            else:
                spans.append((start, end, used))
                start, end, first, used = a, b, b - a, 0
        spans.append((start, end, used))
        for a, b, used in spans:
            if used and b - a >= n:
                covered.update(range(a, b))
    return covered


_WORKER_INDEX: EvalIndex | None = None


def _init_worker(index: EvalIndex) -> None:
    global _WORKER_INDEX
    _WORKER_INDEX = index


def _work(doc: tuple[str, str]) -> tuple[str, _DocHits]:
    if _WORKER_INDEX is None:
        raise RuntimeError("worker index not initialised")
    return doc[0], _WORKER_INDEX.scan_doc(doc[1])


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
    skip_cov: list[set[int]] = [set() for _ in range(n_items)]
    doc_count = [0] * n_items
    best: list[list[Match]] = [[] for _ in range(n_items)]
    # (n, hash) -> number of corpus documents containing that item n-gram
    doc_freq: Counter[tuple[int, int]] = Counter()
    count_freq = params.max_doc_freq is not None

    def results() -> Iterator[tuple[str, _DocHits]]:
        if workers <= 1:
            _init_worker(index)
            for doc in corpus:
                yield _work(doc)
        else:
            with mp.get_context().Pool(workers, _init_worker, (index,)) as pool:
                yield from pool.imap(_work, corpus, chunksize=64)

    for doc_id, hits in results():
        seen: set[tuple[int, int]] = set()
        for i in hits.positions.keys() | hits.contained.keys():
            doc_count[i] += 1
            for n, ps in hits.positions.get(i, {}).items():
                if n == -1:
                    short_hit[i] = True
                elif n == SKIP_KEY:
                    skip_cov[i].update(ps)
                else:
                    matched[n][i].update(ps)
                    if count_freq:
                        hs = index.item_hashes[n][i]
                        seen.update((n, hs[p]) for p in ps)
            itoks = index.tokens[i]
            c = hits.contained.get(i)
            if c is not None:  # the whole item: all its tokens are shared
                ex, exn, d0, d1 = c
                m = Match(doc_id, len(itoks), 0, len(items[i].text), d0, d1, ex, exn)
            else:
                span = hits.spans.get(i)
                if span is None or span[0] == 0:
                    continue
                length, p, j = span
                offsets = hits.offsets
                m = Match(
                    doc_id=doc_id,
                    tokens=length,
                    item_start=itoks[p].start,
                    item_end=itoks[min(p + length, len(itoks)) - 1].end,
                    doc_start=offsets[j][0],
                    doc_end=offsets[min(j + length, len(offsets)) - 1][1],
                )
            m.snippet = items[i].text[m.item_start : m.item_end][:snippet_chars]
            lst = best[i]
            lst.append(m)
            lst.sort(key=lambda x: (-(x.exact or x.exact_normalised), -x.tokens))
            del lst[max_matches:]
        doc_freq.update(seen)

    if count_freq:
        k_max = params.max_doc_freq
        for n, per_item in matched.items():
            hs_n = index.item_hashes[n]
            for i, ps in enumerate(per_item):
                if ps:
                    per_item[i] = {p for p in ps if doc_freq[(n, hs_n[i][p])] <= k_max}

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
        p_fields = None
        if params.palm_per_field and it.field_spans:
            p_fields = _field_fractions(
                index.tokens[i], it.field_spans, matched[params.palm_n][i], params.palm_n
            )
            present = [v for v in p_fields.values() if v is not None]
            p_frac = max(present) if present else None
        covered: set[int] = set()
        for p in matched[params.llama_n][i]:
            covered.update(range(p, p + params.llama_n))
        covered |= skip_cov[i]
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
                palm_fields=p_fields,
            )
        )
    return out


def _field_fractions(
    toks: list[Token], spans: tuple[tuple[str, int, int], ...], hit: set[int], n: int
) -> dict[str, float | None]:
    """Fraction of matched n-grams among the n-grams lying wholly inside each field."""
    out: dict[str, float | None] = {}
    for name, start, end in spans:
        idx = [k for k, t in enumerate(toks) if start <= t.start and t.end <= end]
        if len(idx) < n:
            out[name] = None
            continue
        a, b = idx[0], idx[-1] + 1  # fields are contiguous token ranges
        total = b - a - n + 1
        out[name] = sum(1 for p in range(a, b - n + 1) if p in hit) / total
    return out
