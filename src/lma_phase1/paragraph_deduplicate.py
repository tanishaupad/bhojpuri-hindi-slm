"""Corpus-wide exact paragraph removal followed by exact document deduplication."""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path

from lma_phase1.io import atomic_write_json, iter_jsonl, write_jsonl
from lma_phase1.schema import Document
from lma_phase1.text import canonical_text, stable_text_hash


def paragraph_units(text: str) -> list[str]:
    """Return non-empty line-preserving units used for conservative paragraph deduplication."""

    return [line.strip() for line in text.splitlines() if line.strip()]


def eligible_paragraph(text: str, minimum_characters: int, minimum_words: int) -> bool:
    """Return whether a unit is long enough for safe corpus-wide comparison."""

    canonical = canonical_text(text)
    return len(canonical) >= minimum_characters and len(canonical.split()) >= minimum_words


class ParagraphIndex:
    """Disk-backed index for paragraph hashes, cleaned document hashes, and resume state."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            "PRAGMA journal_mode=WAL;"
            "PRAGMA synchronous=NORMAL;"
            "CREATE TABLE IF NOT EXISTS paragraph_hashes ("
            "paragraph_hash TEXT PRIMARY KEY, document_id TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS document_hashes ("
            "text_hash TEXT PRIMARY KEY, document_id TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS processed_ids ("
            "document_id TEXT PRIMARY KEY, status TEXT NOT NULL);"
        )
        self.connection.commit()

    def was_processed(self, document_id: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM processed_ids WHERE document_id=?", (document_id,)
            ).fetchone()
            is not None
        )

    def paragraph_match(self, paragraph_hash: str) -> str | None:
        row = self.connection.execute(
            "SELECT document_id FROM paragraph_hashes WHERE paragraph_hash=?",
            (paragraph_hash,),
        ).fetchone()
        return row[0] if row else None

    def add_paragraph(self, paragraph_hash: str, document_id: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO paragraph_hashes(paragraph_hash, document_id) VALUES (?, ?)",
            (paragraph_hash, document_id),
        )

    def document_match(self, text_hash: str) -> str | None:
        row = self.connection.execute(
            "SELECT document_id FROM document_hashes WHERE text_hash=?", (text_hash,)
        ).fetchone()
        return row[0] if row else None

    def add_document(self, text_hash: str, document_id: str) -> None:
        self.connection.execute(
            "INSERT INTO document_hashes(text_hash, document_id) VALUES (?, ?)",
            (text_hash, document_id),
        )
        self.connection.execute(
            "INSERT INTO processed_ids(document_id, status) VALUES (?, 'kept')", (document_id,)
        )

    def reject(self, document_id: str, status: str) -> None:
        self.connection.execute(
            "INSERT INTO processed_ids(document_id, status) VALUES (?, ?)",
            (document_id, status),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def clean_document_paragraphs(
    document: Document,
    index: ParagraphIndex,
    minimum_paragraph_characters: int,
    minimum_paragraph_words: int,
) -> tuple[str, dict[str, int], list[str]]:
    """Remove eligible paragraphs already retained from an earlier document."""

    kept: list[str] = []
    pending_hashes: list[str] = []
    pending_hash_set: set[str] = set()
    counters: Counter[str] = Counter()
    for unit in paragraph_units(document.text):
        counters["paragraphs_examined"] += 1
        if not eligible_paragraph(
            unit, minimum_paragraph_characters, minimum_paragraph_words
        ):
            counters["paragraphs_ineligible_short"] += 1
            kept.append(unit)
            continue
        counters["paragraphs_eligible"] += 1
        unit_hash = stable_text_hash(unit)
        if index.paragraph_match(unit_hash) or unit_hash in pending_hash_set:
            counters["paragraph_duplicates_removed"] += 1
            counters["duplicate_characters_removed"] += len(unit)
            counters["duplicate_words_removed"] += len(unit.split())
            continue
        pending_hashes.append(unit_hash)
        pending_hash_set.add(unit_hash)
        kept.append(unit)
    return "\n".join(kept).strip(), dict(counters), pending_hashes


def run(args: argparse.Namespace) -> Counter[str]:
    """Stream the corpus, preserving input priority and safe resume behavior."""

    output = Path(args.output)
    database = Path(args.database)
    if output.exists() != database.exists():
        raise SystemExit(
            "Output and paragraph database must either both exist (resume) or both be absent"
        )
    index = ParagraphIndex(database)
    counters: Counter[str] = Counter()
    batch: list[Document] = []

    for input_path in [Path(path) for path in args.input]:
        for document in iter_jsonl(input_path):
            if index.was_processed(document.document_id):
                continue
            counters["documents_examined"] += 1
            cleaned_text, paragraph_counts, pending_hashes = clean_document_paragraphs(
                document,
                index,
                args.minimum_paragraph_characters,
                args.minimum_paragraph_words,
            )
            counters.update(paragraph_counts)
            if (
                len(cleaned_text) < args.minimum_document_characters
                or len(cleaned_text.split()) < args.minimum_document_words
            ):
                counters["documents_removed_after_paragraph_deduplication"] += 1
                index.reject(document.document_id, "paragraph_duplicate_empty")
                continue
            cleaned_hash = stable_text_hash(cleaned_text)
            if index.document_match(cleaned_hash):
                counters["exact_duplicates_after_paragraph_cleanup"] += 1
                index.reject(document.document_id, "exact_duplicate_after_paragraph_cleanup")
                continue
            document.text = cleaned_text
            document.metadata = {
                **document.metadata,
                "paragraph_deduplication": {
                    "method": "corpus_wide_exact_line_or_paragraph_hash",
                    **paragraph_counts,
                },
            }
            for paragraph_hash in pending_hashes:
                index.add_paragraph(paragraph_hash, document.document_id)
            index.add_document(cleaned_hash, document.document_id)
            batch.append(document)
            counters["documents_kept"] += 1
            counters[f"documents_kept_{document.collection_type}"] += 1
            if len(batch) >= args.write_batch_size:
                write_jsonl(output, batch, append=True)
                batch.clear()
                index.commit()
    if batch:
        write_jsonl(output, batch, append=True)
    index.close()
    return counters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--minimum-paragraph-characters", type=int, default=100)
    parser.add_argument("--minimum-paragraph-words", type=int, default=15)
    parser.add_argument("--minimum-document-characters", type=int, default=30)
    parser.add_argument("--minimum-document-words", type=int, default=5)
    parser.add_argument("--write-batch-size", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counters = run(args)
    atomic_write_json(
        Path(args.report),
        {
            "inputs": args.input,
            "output": args.output,
            "settings": {
                "input_priority_preserved": True,
                "minimum_paragraph_characters": args.minimum_paragraph_characters,
                "minimum_paragraph_words": args.minimum_paragraph_words,
                "minimum_document_characters": args.minimum_document_characters,
                "minimum_document_words": args.minimum_document_words,
                "exact_document_deduplication_after_paragraph_cleanup": True,
            },
            "counts_this_run": dict(sorted(counters.items())),
        },
    )
    print(" ".join(f"{key}={value}" for key, value in sorted(counters.items())))


if __name__ == "__main__":
    main()
