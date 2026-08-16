"""Bottom-k MinHash candidate search for auditable near-duplicate removal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sqlite3
import struct
from collections import Counter
from pathlib import Path

from lma_phase1.io import atomic_write_json, iter_jsonl, write_jsonl
from lma_phase1.schema import Document
from lma_phase1.text import canonical_text


def shingle_hashes(text: str, shingle_size: int, maximum_shingles: int) -> set[int]:
    """Return deterministic sampled word-shingle hashes."""

    words = canonical_text(text).split()
    if not words:
        return set()
    if len(words) < shingle_size:
        shingles = [" ".join(words)]
    else:
        count = len(words) - shingle_size + 1
        stride = max(1, math.ceil(count / maximum_shingles))
        shingles = (
            " ".join(words[index : index + shingle_size])
            for index in range(0, count, stride)
        )
    return {
        int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for shingle in shingles
    }


def bottom_k_sketch(
    text: str,
    sketch_size: int = 64,
    shingle_size: int = 5,
    maximum_shingles: int = 2048,
) -> tuple[int, ...]:
    """Return the smallest unique shingle hashes as a bottom-k MinHash sketch."""

    return tuple(sorted(shingle_hashes(text, shingle_size, maximum_shingles))[:sketch_size])


def estimate_jaccard(left: tuple[int, ...], right: tuple[int, ...], sketch_size: int) -> float:
    """Estimate Jaccard similarity from the bottom-k sample of the combined hash set."""

    left_set = set(left)
    right_set = set(right)
    sampled_union = sorted(left_set | right_set)[:sketch_size]
    if not sampled_union:
        return 0.0
    shared = sum(value in left_set and value in right_set for value in sampled_union)
    return shared / len(sampled_union)


def pack_sketch(sketch: tuple[int, ...]) -> bytes:
    """Serialize unsigned 64-bit sketch values."""

    return struct.pack(f">{len(sketch)}Q", *sketch) if sketch else b""


def unpack_sketch(value: bytes) -> tuple[int, ...]:
    """Deserialize unsigned 64-bit sketch values."""

    if not value:
        return ()
    return struct.unpack(f">{len(value) // 8}Q", value)


def audit_sample(text: str, maximum_characters: int = 600) -> str:
    """Return a compact single-line document sample for pair review."""

    collapsed = " ".join(text.split())
    return collapsed[:maximum_characters] + ("…" if len(collapsed) > maximum_characters else "")


class MinHashIndex:
    """SQLite-backed bottom-k sketch and inverted-prefix index."""

    def __init__(self, path: Path, index_size: int):
        self.index_size = index_size
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            "PRAGMA journal_mode=WAL;"
            "PRAGMA synchronous=NORMAL;"
            "CREATE TABLE IF NOT EXISTS signatures ("
            "document_id TEXT PRIMARY KEY, sketch BLOB NOT NULL, source_name TEXT NOT NULL, "
            "source_url TEXT, text_sample TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS buckets ("
            "value TEXT NOT NULL, document_id TEXT NOT NULL, "
            "PRIMARY KEY(value, document_id));"
            "CREATE INDEX IF NOT EXISTS bucket_value_idx ON buckets(value);"
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

    def candidates(
        self,
        sketch: tuple[int, ...],
        minimum_shared_index_hashes: int,
        maximum_candidates: int,
    ) -> list[str]:
        values = [f"{value:016x}" for value in sketch[: self.index_size]]
        if not values:
            return []
        placeholders = ",".join("?" for _ in values)
        rows = self.connection.execute(
            f"SELECT document_id, COUNT(*) AS shared FROM buckets "
            f"WHERE value IN ({placeholders}) GROUP BY document_id "
            "HAVING shared >= ? ORDER BY shared DESC, document_id LIMIT ?",
            (*values, minimum_shared_index_hashes, maximum_candidates),
        )
        return [row[0] for row in rows]

    def signature(self, document_id: str) -> tuple[int, ...]:
        row = self.connection.execute(
            "SELECT sketch FROM signatures WHERE document_id=?", (document_id,)
        ).fetchone()
        if not row:
            raise KeyError(document_id)
        return unpack_sketch(row[0])

    def audit_fields(self, document_id: str) -> tuple[str, str | None, str]:
        row = self.connection.execute(
            "SELECT source_name, source_url, text_sample FROM signatures WHERE document_id=?",
            (document_id,),
        ).fetchone()
        if not row:
            raise KeyError(document_id)
        return row[0], row[1], row[2]

    def best_match(
        self,
        sketch: tuple[int, ...],
        sketch_size: int,
        minimum_shared_index_hashes: int,
        maximum_candidates: int,
    ) -> tuple[str | None, float]:
        best_document = None
        best_similarity = 0.0
        for document_id in self.candidates(
            sketch, minimum_shared_index_hashes, maximum_candidates
        ):
            similarity = estimate_jaccard(sketch, self.signature(document_id), sketch_size)
            if similarity > best_similarity:
                best_document = document_id
                best_similarity = similarity
        return best_document, best_similarity

    def add(self, document: Document, sketch: tuple[int, ...]) -> None:
        self.connection.execute(
            "INSERT INTO signatures(document_id, sketch, source_name, source_url, text_sample) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                document.document_id,
                pack_sketch(sketch),
                document.source_name,
                document.source_url,
                audit_sample(document.text),
            ),
        )
        self.connection.executemany(
            "INSERT INTO buckets(value, document_id) VALUES (?, ?)",
            [
                (f"{value:016x}", document.document_id)
                for value in sketch[: self.index_size]
            ],
        )
        self.connection.execute(
            "INSERT INTO processed_ids(document_id, status) VALUES (?, 'kept')",
            (document.document_id,),
        )

    def reject(self, document_id: str, status: str = "near_duplicate") -> None:
        self.connection.execute(
            "INSERT INTO processed_ids(document_id, status) VALUES (?, ?)",
            (document_id, status),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


AUDIT_COLUMNS = [
    "pair_id",
    "similarity",
    "candidate_document_id",
    "candidate_source_name",
    "candidate_source_url",
    "candidate_text_sample",
    "matched_document_id",
    "matched_source_name",
    "matched_source_url",
    "matched_text_sample",
    "duplicate_label",
    "reviewer",
    "notes",
]


def open_audit_writer(path: Path) -> tuple[object, csv.DictWriter]:
    """Open an append-safe audit CSV and write its header when new."""

    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
    if not exists:
        writer.writeheader()
    return handle, writer


def run(args: argparse.Namespace) -> Counter[str]:
    """Audit or apply near-duplicate decisions while preserving input order."""

    if args.mode == "apply" and not args.output:
        raise SystemExit("--output is required in apply mode")
    if args.mode == "audit" and not args.audit_output:
        raise SystemExit("--audit-output is required in audit mode")
    database = Path(args.database)
    output = Path(args.output) if args.output else None
    audit_output = Path(args.audit_output) if args.audit_output else None
    if args.mode == "apply" and output and output.exists() != database.exists():
        raise SystemExit("Output and MinHash database must both exist for resume or both be absent")
    if args.mode == "audit" and database.exists() != bool(audit_output and audit_output.exists()):
        raise SystemExit("Audit CSV and MinHash database must both exist for resume or both be absent")

    index = MinHashIndex(database, args.index_size)
    audit_handle = audit_writer = None
    if audit_output:
        audit_handle, audit_writer = open_audit_writer(audit_output)
    counters: Counter[str] = Counter()
    batch: list[Document] = []
    indexed_collection_types = set(args.index_collection_type or [])

    try:
        for input_path in [Path(path) for path in args.input]:
            for document in iter_jsonl(input_path):
                if args.maximum_documents and counters["documents_examined"] >= args.maximum_documents:
                    break
                if index.was_processed(document.document_id):
                    continue
                counters["documents_examined"] += 1
                sketch = bottom_k_sketch(
                    document.text,
                    args.sketch_size,
                    args.shingle_size,
                    args.maximum_shingles,
                )
                if len(sketch) < args.minimum_shingles:
                    counters["documents_below_minimum_shingles"] += 1
                    matched_document, similarity = None, 0.0
                else:
                    matched_document, similarity = index.best_match(
                        sketch,
                        args.sketch_size,
                        args.minimum_shared_index_hashes,
                        args.maximum_candidates,
                    )
                is_candidate = bool(
                    matched_document and similarity >= args.similarity_threshold
                )
                if is_candidate:
                    counters["near_duplicate_candidates"] += 1
                    if audit_writer:
                        matched_source, matched_url, matched_sample = index.audit_fields(
                            matched_document
                        )
                        audit_writer.writerow(
                            {
                                "pair_id": f"pair_{counters['near_duplicate_candidates']:06d}",
                                "similarity": f"{similarity:.6f}",
                                "candidate_document_id": document.document_id,
                                "candidate_source_name": document.source_name,
                                "candidate_source_url": document.source_url or "",
                                "candidate_text_sample": audit_sample(document.text),
                                "matched_document_id": matched_document,
                                "matched_source_name": matched_source,
                                "matched_source_url": matched_url or "",
                                "matched_text_sample": matched_sample,
                                "duplicate_label": "",
                                "reviewer": "",
                                "notes": "",
                            }
                        )
                    if args.mode == "apply":
                        counters["near_duplicates_removed"] += 1
                        index.reject(document.document_id)
                        continue
                should_index = (
                    not indexed_collection_types
                    or document.collection_type in indexed_collection_types
                )
                if should_index:
                    index.add(document, sketch)
                else:
                    index.reject(document.document_id, "kept_not_indexed")
                    counters["documents_kept_not_indexed"] += 1
                counters["documents_kept"] += 1
                if output:
                    batch.append(document)
                    if len(batch) >= args.write_batch_size:
                        write_jsonl(output, batch, append=True)
                        batch.clear()
                if counters["documents_examined"] % args.commit_interval == 0:
                    if batch and output:
                        write_jsonl(output, batch, append=True)
                        batch.clear()
                    index.commit()
                    if audit_handle:
                        audit_handle.flush()
            else:
                continue
            break
        if batch and output:
            write_jsonl(output, batch, append=True)
    finally:
        if audit_handle:
            audit_handle.close()
        index.close()
    return counters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--mode", choices=("audit", "apply"), default="audit")
    parser.add_argument("--output")
    parser.add_argument("--audit-output")
    parser.add_argument("--database", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--sketch-size", type=int, default=64)
    parser.add_argument("--index-size", type=int, default=16)
    parser.add_argument("--minimum-shared-index-hashes", type=int, default=2)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--minimum-shingles", type=int, default=20)
    parser.add_argument("--maximum-shingles", type=int, default=2048)
    parser.add_argument("--maximum-candidates", type=int, default=200)
    parser.add_argument(
        "--index-collection-type",
        action="append",
        choices=("manual", "downloaded"),
        help=(
            "Only retained documents of this collection type become future candidates; "
            "repeat to select both. By default every retained document is indexed."
        ),
    )
    parser.add_argument("--maximum-documents", type=int)
    parser.add_argument("--write-batch-size", type=int, default=1000)
    parser.add_argument("--commit-interval", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.similarity_threshold <= 1.0:
        raise SystemExit("--similarity-threshold must be between 0 and 1")
    if not 1 <= args.index_size <= args.sketch_size:
        raise SystemExit("--index-size must be between 1 and --sketch-size")
    counters = run(args)
    atomic_write_json(
        Path(args.report),
        {
            "inputs": args.input,
            "output": args.output,
            "audit_output": args.audit_output,
            "mode": args.mode,
            "settings": {
                "method": "bottom_k_minhash_with_inverted_prefix_candidates",
                "similarity_threshold": args.similarity_threshold,
                "sketch_size": args.sketch_size,
                "index_size": args.index_size,
                "minimum_shared_index_hashes": args.minimum_shared_index_hashes,
                "shingle_size": args.shingle_size,
                "minimum_shingles": args.minimum_shingles,
                "maximum_shingles": args.maximum_shingles,
                "maximum_candidates": args.maximum_candidates,
                "indexed_collection_types": args.index_collection_type
                or ["manual", "downloaded"],
                "input_priority_preserved": True,
            },
            "counts_this_run": dict(sorted(counters.items())),
        },
    )
    print(" ".join(f"{key}={value}" for key, value in sorted(counters.items())))


if __name__ == "__main__":
    main()
