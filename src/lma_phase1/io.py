"""Streaming JSONL helpers used throughout the data pipeline."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from .schema import Document


@contextmanager
def open_text(path: Path, mode: str = "rt") -> Iterator[IO[str]]:
    """Open plain or gzip-compressed UTF-8 text based on the filename suffix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, mode, encoding="utf-8") as handle:
            yield handle
    else:
        with path.open(mode, encoding="utf-8") as handle:
            yield handle


def iter_jsonl(path: Path) -> Iterator[Document]:
    """Stream validated ``Document`` records from a JSONL file."""

    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                yield Document.from_dict(value)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, documents: Iterable[Document], append: bool = False) -> int:
    """Write validated documents and return the number of records written."""

    mode = "at" if append else "wt"
    count = 0
    with open_text(path, mode) as handle:
        for document in documents:
            json.dump(document.to_dict(), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            count += 1
    return count


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a small JSON state or statistics file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning ``default`` when the file does not exist."""

    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
