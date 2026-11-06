"""Streaming readers for evaluation sets and corpora.

Corpora are read one record at a time: JSONL line by line, Parquet one row group batch
at a time (needs the optional ``pyarrow`` dependency), and plain-text files one file per
document. Nothing is loaded whole.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .scan import EvalItem

__all__ = ["iter_records", "iter_corpus", "load_eval"]

TEXT_SUFFIXES = {".txt", ".md", ".py", ".rst"}


def _iter_parquet(path: Path, batch_size: int = 1024) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit("reading Parquet needs pyarrow: pip install 'contamscan[parquet]'") from e
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield dict records from a ``.jsonl`` or ``.parquet`` file."""
    path = Path(path)
    if path.suffix == ".parquet":
        yield from _iter_parquet(path)
        return
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not valid JSON: {e}") from e
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            yield rec


def _join(rec: dict[str, Any], fields: Sequence[str]) -> str:
    parts = []
    for f in fields:
        v = rec.get(f)
        if v is None:
            continue
        parts.append(v if isinstance(v, str) else json.dumps(v))
    return "\n".join(parts)


def iter_corpus(
    paths: Sequence[str | Path],
    text_fields: Sequence[str] = ("text",),
    id_field: str = "id",
) -> Iterator[tuple[str, str]]:
    """Yield ``(doc_id, text)`` for every document in ``paths``.

    Directories are walked for ``.jsonl``, ``.parquet`` and plain-text files. A record
    without ``id_field`` gets the id ``<file name>:<record number>``.
    """
    for p in paths:
        p = Path(p)
        files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
        for f in files:
            if f.suffix in (".jsonl", ".parquet"):
                for k, rec in enumerate(iter_records(f)):
                    doc_id = rec.get(id_field)
                    yield (
                        (str(doc_id) if doc_id is not None else f"{f.name}:{k}"),
                        _join(rec, text_fields),
                    )
            elif f.suffix in TEXT_SUFFIXES or not p.is_dir():
                yield f.name, f.read_text(encoding="utf-8", errors="replace")


def load_eval(
    path: str | Path,
    fields: Sequence[str],
    id_field: str = "id",
    score_field: str | None = None,
) -> list[EvalItem]:
    """Load an evaluation set; the checked text is ``fields`` joined by newlines."""
    items = []
    seen: set[str] = set()
    for k, rec in enumerate(iter_records(path)):
        item_id = str(rec.get(id_field, k))
        if item_id in seen:
            raise ValueError(f"duplicate item id {item_id!r} in {path}")
        seen.add(item_id)
        score = rec.get(score_field) if score_field else None
        if isinstance(score, bool):
            score = float(score)
        items.append(EvalItem(item_id, _join(rec, fields), None if score is None else float(score)))
    return items
