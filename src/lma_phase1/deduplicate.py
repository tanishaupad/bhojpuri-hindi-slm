"""Streaming exact and near-duplicate removal with manual-source priority."""

from __future__ import annotations

import argparse
import hashlib
import math
import sqlite3
from collections import Counter
from pathlib import Path

from lma_phase1.io import atomic_write_json, iter_jsonl, write_jsonl
from lma_phase1.schema import Document
from lma_phase1.text import canonical_text, stable_text_hash

MASK_64 = (1 << 64) - 1


def simhash(text: str, shingle_size: int = 5, maximum_shingles: int = 2048) -> int:
    """Compute a deterministic 64-bit SimHash over word shingles."""

    words = canonical_text(text).split()
    if not words:
        return 0
    if len(words) < shingle_size:
        shingles = [" ".join(words)]
    else:
        count = len(words) - shingle_size + 1
        stride = max(1, math.ceil(count / maximum_shingles))
        shingles = [
            " ".join(words[index : index + shingle_size]) for index in range(0, count, stride)
        ]
    weights = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint & MASK_64


def hamming_distance(left: int, right: int) -> int:
    """Return the number of differing bits between two fingerprints."""

    return (left ^ right).bit_count()


class DuplicateIndex:
    """Disk-backed duplicate index suitable for resumable large-corpus runs."""

    def __init__(self, path: Path, bands: int = 4):
        self.path = path
        self.bands = bands
        self.band_width = 64 // bands
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            "PRAGMA journal_mode=WAL;"
            "CREATE TABLE IF NOT EXISTS exact_hashes ("
            "text_hash TEXT PRIMARY KEY, document_id TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS near_hashes ("
            "band INTEGER NOT NULL, band_value INTEGER NOT NULL, "
            "fingerprint TEXT NOT NULL, document_id TEXT NOT NULL);"
            "CREATE INDEX IF NOT EXISTS near_band_idx ON near_hashes(band, band_value);"
            "CREATE TABLE IF NOT EXISTS processed_ids ("
            "document_id TEXT PRIMARY KEY, status TEXT NOT NULL);"
        )
        self.connection.commit()

    def was_processed(self, document_id: str) -> bool:
        """Return whether this record was handled in a previous/resumed pass."""

        return (
            self.connection.execute(
                "SELECT 1 FROM processed_ids WHERE document_id=?", (document_id,)
            ).fetchone()
            is not None
        )

    def exact_match(self, text_hash: str) -> str | None:
        """Return the kept document ID with the same canonical text, if any."""

        row = self.connection.execute(
            "SELECT document_id FROM exact_hashes WHERE text_hash=?", (text_hash,)
        ).fetchone()
        return row[0] if row else None

    def near_match(self, fingerprint: int, threshold: int) -> str | None:
        """Return one approximate duplicate within the Hamming threshold."""

        for band in range(self.bands):
            band_value = (fingerprint >> (band * self.band_width)) & ((1 << self.band_width) - 1)
            rows = self.connection.execute(
                "SELECT fingerprint, document_id FROM near_hashes "
                "WHERE band=? AND band_value=? LIMIT 5000",
                (band, band_value),
            )
            for stored, document_id in rows:
                if hamming_distance(fingerprint, int(stored, 16)) <= threshold:
                    return document_id
        return None

    def add(self, document: Document, text_hash: str, fingerprint: int | None) -> None:
        """Add one retained record to exact, near, and processed indexes."""

        self.connection.execute(
            "INSERT INTO exact_hashes(text_hash, document_id) VALUES (?, ?)",
            (text_hash, document.document_id),
        )
        if fingerprint is not None:
            for band in range(self.bands):
                band_value = (fingerprint >> (band * self.band_width)) & (
                    (1 << self.band_width) - 1
                )
                self.connection.execute(
                    "INSERT INTO near_hashes(band, band_value, fingerprint, document_id) "
                    "VALUES (?, ?, ?, ?)",
                    (band, band_value, f"{fingerprint:016x}", document.document_id),
                )
        self.connection.execute(
            "INSERT INTO processed_ids(document_id, status) VALUES (?, 'kept')",
            (document.document_id,),
        )

    def reject(self, document_id: str, status: str) -> None:
        """Mark a rejected record as processed for safe resumption."""

        self.connection.execute(
            "INSERT INTO processed_ids(document_id, status) VALUES (?, ?)",
            (document_id, status),
        )

    def commit(self) -> None:
        """Commit pending index changes."""

        self.connection.commit()

    def close(self) -> None:
        """Commit and close the index."""

        self.connection.commit()
        self.connection.close()


def deduplicate(args: argparse.Namespace) -> Counter[str]:
    """Process manual documents first so overlap is charged to manual collection."""

    output = Path(args.output)
    database = Path(args.database)
    if output.exists() != database.exists():
        raise SystemExit(
            "Output and duplicate database must either both exist (resume) or both be absent"
        )
    index = DuplicateIndex(database, bands=args.bands)
    counters: Counter[str] = Counter()
    batch: list[Document] = []
    input_paths = [Path(path) for path in args.input]

    for collection_type in ("manual", "downloaded"):
        for input_path in input_paths:
            for document in iter_jsonl(input_path):
                if document.collection_type != collection_type or index.was_processed(
                    document.document_id
                ):
                    continue
                counters["examined"] += 1
                text_hash = stable_text_hash(document.text)
                exact_document = index.exact_match(text_hash)
                if exact_document:
                    counters["exact_duplicates"] += 1
                    index.reject(document.document_id, "exact_duplicate")
                    continue
                fingerprint = None
                near_document = None
                if not args.exact_only:
                    fingerprint = simhash(
                        document.text, args.shingle_size, args.maximum_shingles
                    )
                    near_document = index.near_match(fingerprint, args.hamming_threshold)
                if near_document:
                    counters["near_duplicates"] += 1
                    index.reject(document.document_id, "near_duplicate")
                    continue
                index.add(document, text_hash, fingerprint)
                batch.append(document)
                counters["kept"] += 1
                counters[f"kept_{collection_type}"] += 1
                if len(batch) >= args.write_batch_size:
                    write_jsonl(output, batch, append=True)
                    batch.clear()
                    index.commit()
    if batch:
        write_jsonl(output, batch, append=True)
    index.close()
    return counters


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--hamming-threshold", type=int, default=3)
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Skip near-duplicate rejection while measuring exact overlap",
    )
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--maximum-shingles", type=int, default=2048)
    parser.add_argument("--write-batch-size", type=int, default=1000)
    return parser


def main() -> None:
    """Run the deduplication CLI."""

    args = build_parser().parse_args()
    if 64 % args.bands:
        raise SystemExit("--bands must divide 64 exactly")
    counters = deduplicate(args)
    atomic_write_json(
        Path(args.report),
        {
            "inputs": args.input,
            "output": args.output,
            "settings": {
                "manual_records_processed_first": True,
                "exact_only": args.exact_only,
                "hamming_threshold": args.hamming_threshold,
                "bands": args.bands,
                "shingle_size": args.shingle_size,
                "maximum_shingles": args.maximum_shingles,
            },
            "counts_this_run": dict(sorted(counters.items())),
        },
    )
    print(" ".join(f"{key}={value}" for key, value in sorted(counters.items())))


if __name__ == "__main__":
    main()
