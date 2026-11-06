"""Streaming readers for evaluation sets and corpora.

Corpora are read one record at a time: JSONL (plain or gzip-compressed) line by line,
Parquet one batch at a time with only the needed columns (needs the optional ``pyarrow``
dependency), and plain-text files one file per document. Nothing is loaded whole.
Suffixes are compared case-insensitively.
"""

from __future__ import annotations

import gzip
import json
import math
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from .scan import EvalItem

__all__ = ["iter_records", "iter_corpus", "load_eval"]

TEXT_SUFFIXES = {".txt", ".md", ".py", ".rst"}
RECORD_SUFFIXES = (".jsonl", ".jsonl.gz", ".parquet")


def record_kind(path: Path) -> str | None:
    """``".jsonl"``, ``".jsonl.gz"`` or ``".parquet"`` for record files, else ``None``."""
    name = path.name.lower()
    return next((x for x in RECORD_SUFFIXES if name.endswith(x)), None)


def _iter_parquet(
    path: Path, columns: Sequence[str] | None = None, batch_size: int = 1024
) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ValueError(
            f"{path}: reading Parquet needs pyarrow: pip install 'contamscan[parquet]'"
        ) from e
    pf = pq.ParquetFile(path)
    cols = None
    if columns is not None:
        names = set(pf.schema_arrow.names)
        cols = [c for c in dict.fromkeys(columns) if c in names]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        if cols == []:  # none of the fields exist: keep the row count
            yield from ({} for _ in range(batch.num_rows))
        else:
            yield from batch.to_pylist()


def iter_records(
    path: str | Path, columns: Sequence[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Yield dict records from a ``.jsonl``, ``.jsonl.gz`` or ``.parquet`` file.

    ``columns`` limits the Parquet columns read (JSONL lines are parsed whole). Malformed
    input raises ``ValueError`` naming the file and line.
    """
    path = Path(path)
    kind = record_kind(path)
    if kind == ".parquet":
        yield from _iter_parquet(path, columns)
        return
    if kind != ".jsonl.gz":
        with open(path, "rb") as fh:
            yield from _parse_lines(path, fh)
        return
    try:
        with gzip.open(path, "rb") as gz:
            yield from _parse_lines(path, gz)
    except (EOFError, zlib.error, gzip.BadGzipFile) as e:
        raise ValueError(f"{path}: truncated or corrupt gzip: {e}") from e


def _parse_lines(path: Path, lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    for lineno, raw in enumerate(lines, 1):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"{path}:{lineno}: not valid UTF-8: {e.reason}") from e
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: not valid JSON: {e}") from e
        if not isinstance(rec, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        yield rec


def _leaves(v: Any) -> Iterator[str]:
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _leaves(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _leaves(x)
    elif v is not None:
        yield json.dumps(v, ensure_ascii=False, default=str)


def _flatten(v: Any) -> str:
    """A field value as text: a string as is, a list or dict as its leaves, one per line.

    A chat ``messages`` list thus becomes its roles and turns, and a list of answer choices
    its choices, rather than JSON text with escapes and quotes that would hide a verbatim
    copy from every method. Numbers and booleans are written as JSON (``3``, ``true``).
    """
    return v if isinstance(v, str) else "\n".join(_leaves(v))


def _join(rec: dict[str, Any], fields: Sequence[str]) -> str:
    parts = []
    for f in fields:
        v = rec.get(f)
        if v is None:
            continue
        parts.append(_flatten(v))
    return "\n".join(parts)


def _join_with_spans(
    rec: dict[str, Any], fields: Sequence[str]
) -> tuple[str, tuple[tuple[str, int, int], ...]]:
    """Like ``_join``, also returning each present field's character range in the text."""
    parts: list[str] = []
    spans = []
    pos = 0
    for f in fields:
        v = rec.get(f)
        if v is None:
            continue
        s = _flatten(v)
        if parts:
            pos += 1  # the joining newline
        spans.append((f, pos, pos + len(s)))
        parts.append(s)
        pos += len(s)
    return "\n".join(parts), tuple(spans)


def iter_corpus(
    paths: Sequence[str | Path],
    text_fields: Sequence[str] = ("text",),
    id_field: str = "id",
    field_counts: Counter[str] | None = None,
    *,
    text_files: bool = False,
    skipped: Counter[str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(doc_id, text)`` for every document in ``paths``.

    Directories are walked for ``.jsonl``, ``.jsonl.gz``, ``.parquet`` and plain-text files
    (``.txt``, ``.md``, ``.py``, ``.rst``); other files are skipped and, if ``skipped`` is
    given, counted by suffix. A file named directly must be one of those types, or any
    file when ``text_files`` is true (read whole as one document); otherwise ``ValueError``.
    Ids are relative to the path argument: a record without ``id_field`` gets
    ``<relative path>:<record number>``, a text file its relative path. If ``field_counts``
    is given, it counts the records (key ``""``) and, for each of ``text_fields``, the
    records that have it.
    """
    columns = [*text_fields, id_field]
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = sorted(x for x in p.rglob("*") if x.is_file())
        else:
            files = [p]
        for f in files:
            name = f.relative_to(p).as_posix() if p.is_dir() else f.name
            if record_kind(f):
                for k, rec in enumerate(iter_records(f, columns)):
                    if field_counts is not None:
                        field_counts[""] += 1
                        field_counts.update(x for x in text_fields if rec.get(x) is not None)
                    doc_id = rec.get(id_field)
                    yield (
                        (str(doc_id) if doc_id is not None else f"{name}:{k}"),
                        _join(rec, text_fields),
                    )
            elif f.suffix.lower() in TEXT_SUFFIXES or (text_files and not p.is_dir()):
                yield name, f.read_text(encoding="utf-8", errors="replace")
            elif p.is_dir():
                if skipped is not None:
                    skipped[f.suffix.lower() or f.name] += 1
            else:
                raise ValueError(
                    f"{f}: unsupported corpus file type {f.suffix or '(no suffix)'!r}; "
                    "use .jsonl, .jsonl.gz, .parquet or a text file "
                    f"({', '.join(sorted(TEXT_SUFFIXES))}), or pass --text-files to read "
                    "it whole as one document"
                )


def _score(v: Any, path: str | Path, record: int) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = math.nan
    if isinstance(v, (list, dict)) or not math.isfinite(x):
        raise ValueError(f"{path}:{record}: score {v!r} is not a finite number")
    return x


def load_eval(
    path: str | Path,
    fields: Sequence[str],
    id_field: str = "id",
    score_field: str | None = None,
    missing: Counter[str] | None = None,
) -> list[EvalItem]:
    """Load an evaluation set; the checked text is ``fields`` joined by newlines.

    Raises ``ValueError`` if one of ``fields`` is absent from every record, which is almost
    always a misspelt field name, or if a score is not a finite number (``NaN``, a list).
    If ``missing`` is given, it receives, for each of ``fields``, the number of records
    without it.
    """
    items = []
    seen: set[str] = set()
    absent: Counter[str] = Counter()
    for k, rec in enumerate(iter_records(path)):
        absent.update(f for f in fields if rec.get(f) is None)
        item_id = str(rec.get(id_field, k))
        if item_id in seen:
            raise ValueError(f"duplicate item id {item_id!r} in {path}")
        seen.add(item_id)
        score = _score(rec.get(score_field), path, k + 1) if score_field else None
        text, spans = _join_with_spans(rec, fields)
        items.append(EvalItem(item_id, text, score, field_spans=spans))
    gone = [f for f in fields if items and absent[f] == len(items)]
    if gone:
        keys = sorted({key for rec in iter_records(path) for key in rec})
        raise ValueError(
            f"field(s) {', '.join(map(repr, gone))} missing from all {len(items)} records of "
            f"{path}; fields found: {', '.join(keys[:20])}"
        )
    if missing is not None:
        missing.update({f: absent[f] for f in fields})
    return items
