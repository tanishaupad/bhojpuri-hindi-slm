"""Audit training splits against protected validation and test documents.

The audit is intentionally asymmetric: validation and test are protected, and
only training documents are flagged. It checks canonical exact-document
matches, exact paragraph matches, and conservative word-shingle near matches.
Results are stored in SQLite so an interrupted scan can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from lma_phase1.io import atomic_write_json, iter_jsonl, write_jsonl
from lma_phase1.minhash_deduplicate import (
    audit_sample,
    estimate_jaccard,
    pack_sketch,
    unpack_sketch,
)
from lma_phase1.paragraph_deduplicate import paragraph_units
from lma_phase1.schema import Document
from lma_phase1.text import canonical_text

CHECKS = ("exact_document", "exact_paragraph", "near_document")
AUDIT_COLUMNS = [
    "match_type",
    "similarity",
    "train_document_id",
    "train_source_name",
    "train_source_url",
    "train_text_sample",
    "protected_split",
    "protected_document_id",
    "protected_source_name",
    "protected_source_url",
    "protected_text_sample",
    "matching_paragraph_sample",
    "review_label",
    "reviewer",
    "notes",
]


@dataclass(frozen=True, slots=True)
class ProtectedReference:
    """Compact reference to a validation or test document."""

    key: str
    split: str
    document_id: str


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    """Pure feature-extraction settings shared with worker processes."""

    checks: tuple[str, ...]
    sketch_size: int
    shingle_size: int
    maximum_shingles: int
    minimum_paragraph_characters: int
    minimum_paragraph_words: int


@dataclass(slots=True)
class TrainFeatures:
    """Deterministic document features computed without database access."""

    document: Document
    exact_hash: str | None
    paragraphs: list[tuple[str, str]]
    sketch: tuple[int, ...] | None


_WORKER_FEATURE_SETTINGS: FeatureSettings | None = None


class AuditIndex:
    """Persistent protected signatures, scan state, and audit matches."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            "PRAGMA journal_mode=WAL;"
            "PRAGMA synchronous=NORMAL;"
            "CREATE TABLE IF NOT EXISTS metadata ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS protected ("
            "protected_key TEXT PRIMARY KEY, split TEXT NOT NULL, "
            "document_id TEXT NOT NULL, source_name TEXT NOT NULL, source_url TEXT, "
            "text_sample TEXT NOT NULL, sketch BLOB);"
            "CREATE TABLE IF NOT EXISTS buckets ("
            "value TEXT NOT NULL, protected_key TEXT NOT NULL, "
            "PRIMARY KEY(value, protected_key));"
            "CREATE INDEX IF NOT EXISTS bucket_value_idx ON buckets(value);"
            "CREATE TABLE IF NOT EXISTS processed_train ("
            "document_id TEXT PRIMARY KEY, flagged INTEGER NOT NULL);"
            "CREATE TABLE IF NOT EXISTS matches ("
            "match_type TEXT NOT NULL, train_document_id TEXT NOT NULL, "
            "protected_key TEXT NOT NULL, similarity REAL, "
            "train_source_name TEXT NOT NULL, train_source_url TEXT, "
            "train_text_sample TEXT NOT NULL, matching_paragraph_sample TEXT, "
            "PRIMARY KEY(match_type, train_document_id, protected_key));"
        )
        self.connection.commit()

    def metadata(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def add_protected(
        self,
        reference: ProtectedReference,
        document: Document,
        sketch: tuple[int, ...] | None,
        index_size: int,
    ) -> None:
        if self.connection.execute(
            "SELECT 1 FROM protected WHERE protected_key=?", (reference.key,)
        ).fetchone():
            return
        self.connection.execute(
            "INSERT INTO protected("
            "protected_key, split, document_id, source_name, source_url, text_sample, sketch"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                reference.key,
                reference.split,
                reference.document_id,
                document.source_name,
                document.source_url,
                audit_sample(document.text),
                pack_sketch(sketch) if sketch is not None else None,
            ),
        )
        if sketch:
            self.connection.executemany(
                "INSERT INTO buckets(value, protected_key) VALUES (?, ?)",
                [(f"{value:016x}", reference.key) for value in sketch[:index_size]],
            )

    def best_near_match(
        self,
        sketch: tuple[int, ...],
        sketch_size: int,
        index_size: int,
        minimum_shared_index_hashes: int,
        maximum_candidates: int,
    ) -> tuple[str | None, float]:
        values = [f"{value:016x}" for value in sketch[:index_size]]
        if not values:
            return None, 0.0
        placeholders = ",".join("?" for _ in values)
        rows = self.connection.execute(
            f"SELECT b.protected_key, p.sketch, COUNT(*) AS shared "
            f"FROM buckets b JOIN protected p USING(protected_key) "
            f"WHERE b.value IN ({placeholders}) GROUP BY b.protected_key "
            "HAVING shared >= ? ORDER BY shared DESC, b.protected_key LIMIT ?",
            (*values, minimum_shared_index_hashes, maximum_candidates),
        )
        best_key = None
        best_similarity = 0.0
        for protected_key, packed, _shared in rows:
            similarity = estimate_jaccard(sketch, unpack_sketch(packed), sketch_size)
            if similarity > best_similarity:
                best_key = protected_key
                best_similarity = similarity
        return best_key, best_similarity

    def was_train_processed(self, document_id: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM processed_train WHERE document_id=?", (document_id,)
            ).fetchone()
            is not None
        )

    def add_match(
        self,
        match_type: str,
        document: Document,
        protected_key: str,
        similarity: float | None = None,
        matching_paragraph_sample: str | None = None,
    ) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO matches("
            "match_type, train_document_id, protected_key, similarity, train_source_name, "
            "train_source_url, train_text_sample, matching_paragraph_sample"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                match_type,
                document.document_id,
                protected_key,
                similarity,
                document.source_name,
                document.source_url,
                audit_sample(document.text),
                matching_paragraph_sample,
            ),
        )

    def mark_train_processed(self, document_id: str, flagged: bool) -> None:
        self.connection.execute(
            "INSERT INTO processed_train(document_id, flagged) VALUES (?, ?)",
            (document_id, int(flagged)),
        )

    def cumulative_counts(self) -> dict[str, int]:
        counts = {
            "protected_documents": self.connection.execute(
                "SELECT COUNT(*) FROM protected"
            ).fetchone()[0],
            "train_documents_processed": self.connection.execute(
                "SELECT COUNT(*) FROM processed_train"
            ).fetchone()[0],
            "train_documents_flagged": self.connection.execute(
                "SELECT COUNT(*) FROM processed_train WHERE flagged=1"
            ).fetchone()[0],
        }
        for check in CHECKS:
            counts[f"matches_{check}"] = self.connection.execute(
                "SELECT COUNT(*) FROM matches WHERE match_type=?", (check,)
            ).fetchone()[0]
        counts["train_documents_clear"] = (
            counts["train_documents_processed"] - counts["train_documents_flagged"]
        )
        return counts

    def flagged_document_ids(self) -> set[str]:
        return {
            row[0]
            for row in self.connection.execute(
                "SELECT document_id FROM processed_train WHERE flagged=1"
            )
        }

    def export_matches(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_reviews: dict[tuple[str, str, str, str], dict[str, str]] = {}
        if path.exists():
            with path.open(encoding="utf-8", newline="") as existing_handle:
                for row in csv.DictReader(existing_handle):
                    key = (
                        row["match_type"],
                        row["train_document_id"],
                        row["protected_split"],
                        row["protected_document_id"],
                    )
                    existing_reviews[key] = {
                        "review_label": row.get("review_label", ""),
                        "reviewer": row.get("reviewer", ""),
                        "notes": row.get("notes", ""),
                    }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
            writer.writeheader()
            rows = self.connection.execute(
                "SELECT m.match_type, m.similarity, m.train_document_id, "
                "m.train_source_name, m.train_source_url, m.train_text_sample, "
                "p.split, p.document_id, p.source_name, p.source_url, p.text_sample, "
                "m.matching_paragraph_sample "
                "FROM matches m JOIN protected p USING(protected_key) "
                "ORDER BY m.train_document_id, m.match_type, p.split, p.document_id"
            )
            for row in rows:
                output_row = {
                    "match_type": row[0],
                    "similarity": "" if row[1] is None else f"{row[1]:.6f}",
                    "train_document_id": row[2],
                    "train_source_name": row[3],
                    "train_source_url": row[4] or "",
                    "train_text_sample": row[5],
                    "protected_split": row[6],
                    "protected_document_id": row[7],
                    "protected_source_name": row[8],
                    "protected_source_url": row[9] or "",
                    "protected_text_sample": row[10],
                    "matching_paragraph_sample": row[11] or "",
                    "review_label": "",
                    "reviewer": "",
                    "notes": "",
                }
                review_key = (row[0], row[2], row[6], row[7])
                output_row.update(existing_reviews.get(review_key, {}))
                writer.writerow(output_row)
        temporary.replace(path)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def protected_key(split: str, document_id: str) -> str:
    """Build a collision-safe key without changing the source document ID."""

    return f"{split}\0{document_id}"


def canonical_hash(canonical: str) -> str:
    """Hash text that has already been canonicalized."""

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bottom_k_from_canonical(
    canonical: str,
    sketch_size: int,
    shingle_size: int,
    maximum_shingles: int,
) -> tuple[int, ...]:
    """Build the Phase 1 bottom-k sketch without normalizing the text twice."""

    words = canonical.split()
    if not words:
        return ()
    if len(words) < shingle_size:
        shingles = [" ".join(words)]
    else:
        count = len(words) - shingle_size + 1
        stride = max(1, math.ceil(count / maximum_shingles))
        shingles = (
            " ".join(words[index : index + shingle_size]) for index in range(0, count, stride)
        )
    hashes = {
        int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for shingle in shingles
    }
    return tuple(sorted(hashes)[:sketch_size])


def canonical_paragraphs(text: str, canonical_document: str) -> list[tuple[str, str]]:
    """Return original/canonical paragraph pairs, reusing one-line normalization."""

    units = paragraph_units(text)
    if len(units) == 1:
        return [(units[0], canonical_document)]
    return [(unit, canonical_text(unit)) for unit in units]


def paragraph_is_eligible(canonical: str, minimum_characters: int, minimum_words: int) -> bool:
    return len(canonical) >= minimum_characters and len(canonical.split()) >= minimum_words


def compute_train_features(document: Document, settings: FeatureSettings) -> TrainFeatures:
    """Compute matching features without touching shared state."""

    canonical_document = canonical_text(document.text)
    exact_hash = canonical_hash(canonical_document) if "exact_document" in settings.checks else None
    paragraphs = []
    if "exact_paragraph" in settings.checks:
        for paragraph, canonical_paragraph in canonical_paragraphs(
            document.text, canonical_document
        ):
            if paragraph_is_eligible(
                canonical_paragraph,
                settings.minimum_paragraph_characters,
                settings.minimum_paragraph_words,
            ):
                paragraphs.append((canonical_hash(canonical_paragraph), paragraph))
    sketch = None
    if "near_document" in settings.checks:
        sketch = bottom_k_from_canonical(
            canonical_document,
            settings.sketch_size,
            settings.shingle_size,
            settings.maximum_shingles,
        )
    return TrainFeatures(document, exact_hash, paragraphs, sketch)


def initialize_feature_worker(settings: FeatureSettings) -> None:
    """Set immutable settings once in each process-pool worker."""

    global _WORKER_FEATURE_SETTINGS
    _WORKER_FEATURE_SETTINGS = settings


def compute_features_in_worker(document: Document) -> TrainFeatures:
    """Process-pool adapter using settings installed by the initializer."""

    if _WORKER_FEATURE_SETTINGS is None:
        raise RuntimeError("Feature worker was not initialized")
    return compute_train_features(document, _WORKER_FEATURE_SETTINGS)


def settings_for(args: argparse.Namespace, checks: tuple[str, ...]) -> dict[str, object]:
    """Return the scan settings whose identity must remain stable on resume."""

    def file_identity(value: str) -> dict[str, int | str]:
        path = Path(value).resolve()
        stat = path.stat()
        return {"path": str(path), "size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}

    return {
        "train": file_identity(args.train),
        "validation": file_identity(args.validation),
        "test": file_identity(args.test),
        "checks": list(checks),
        "similarity_threshold": args.similarity_threshold,
        "sketch_size": args.sketch_size,
        "index_size": args.index_size,
        "minimum_shared_index_hashes": args.minimum_shared_index_hashes,
        "shingle_size": args.shingle_size,
        "minimum_shingles": args.minimum_shingles,
        "maximum_shingles": args.maximum_shingles,
        "maximum_candidates": args.maximum_candidates,
        "minimum_paragraph_characters": args.minimum_paragraph_characters,
        "minimum_paragraph_words": args.minimum_paragraph_words,
    }


def index_protected_splits(
    args: argparse.Namespace,
    checks: tuple[str, ...],
    index: AuditIndex,
) -> tuple[
    dict[str, ProtectedReference],
    dict[str, tuple[ProtectedReference, str]],
    set[int],
    Counter[str],
]:
    """Index protected documents and return exact hashes held in memory."""

    exact_documents: dict[str, ProtectedReference] = {}
    exact_paragraphs: dict[str, tuple[ProtectedReference, str]] = {}
    protected_prefix_hashes: set[int] = set()
    counters: Counter[str] = Counter()
    for split, input_path in (("validation", args.validation), ("test", args.test)):
        for document in iter_jsonl(Path(input_path)):
            canonical_document = canonical_text(document.text)
            reference = ProtectedReference(
                protected_key(split, document.document_id), split, document.document_id
            )
            counters[f"protected_{split}_documents"] += 1
            if "exact_document" in checks:
                exact_documents.setdefault(canonical_hash(canonical_document), reference)
            if "exact_paragraph" in checks:
                for paragraph, canonical_paragraph in canonical_paragraphs(
                    document.text, canonical_document
                ):
                    counters["protected_paragraphs_examined"] += 1
                    if paragraph_is_eligible(
                        canonical_paragraph,
                        args.minimum_paragraph_characters,
                        args.minimum_paragraph_words,
                    ):
                        counters["protected_paragraphs_eligible"] += 1
                        exact_paragraphs.setdefault(
                            canonical_hash(canonical_paragraph), (reference, paragraph)
                        )
            sketch = None
            if "near_document" in checks:
                sketch = bottom_k_from_canonical(
                    canonical_document,
                    args.sketch_size,
                    args.shingle_size,
                    args.maximum_shingles,
                )
                if len(sketch) < args.minimum_shingles:
                    counters["protected_documents_below_minimum_shingles"] += 1
                    sketch = None
                else:
                    counters["protected_documents_near_indexed"] += 1
                    protected_prefix_hashes.update(sketch[: args.index_size])
            index.add_protected(reference, document, sketch, args.index_size)
    counters["protected_documents_total"] = (
        counters["protected_validation_documents"] + counters["protected_test_documents"]
    )
    index.commit()
    return exact_documents, exact_paragraphs, protected_prefix_hashes, counters


def matching_paragraph_feature(
    document_paragraphs: list[tuple[str, str]],
    paragraphs: dict[str, tuple[ProtectedReference, str]],
) -> tuple[ProtectedReference | None, str | None]:
    """Return the first eligible protected paragraph copied into training."""

    for paragraph_hash, paragraph in document_paragraphs:
        match = paragraphs.get(paragraph_hash)
        if match:
            return match
    return None, None


def scan_training(
    args: argparse.Namespace,
    checks: tuple[str, ...],
    index: AuditIndex,
    exact_documents: dict[str, ProtectedReference],
    exact_paragraphs: dict[str, tuple[ProtectedReference, str]],
    protected_prefix_hashes: set[int],
) -> tuple[Counter[str], bool]:
    """Scan training documents, recording one strongest match per document."""

    counters: Counter[str] = Counter()
    reached_end = True
    feature_settings = FeatureSettings(
        checks,
        args.sketch_size,
        args.shingle_size,
        args.maximum_shingles,
        args.minimum_paragraph_characters,
        args.minimum_paragraph_words,
    )
    executor = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_feature_worker,
            initargs=(feature_settings,),
        )

    def inspect(features: TrainFeatures) -> None:
        document = features.document
        counters["train_documents_inspected"] += 1
        match_type = None
        match_reference = None
        similarity = None
        paragraph_sample = None

        if "exact_document" in checks:
            match_reference = exact_documents.get(features.exact_hash or "")
            if match_reference:
                match_type = "exact_document"

        paragraph_reference = None
        if not match_type and "exact_paragraph" in checks:
            paragraph_reference, paragraph_sample = matching_paragraph_feature(
                features.paragraphs, exact_paragraphs
            )

        if not match_type and "near_document" in checks:
            sketch = features.sketch or ()
            if len(sketch) < args.minimum_shingles:
                counters["train_documents_below_minimum_shingles"] += 1
            elif (
                sum(value in protected_prefix_hashes for value in sketch[: args.index_size])
                < args.minimum_shared_index_hashes
            ):
                counters["train_documents_without_near_candidates"] += 1
            else:
                near_key, near_similarity = index.best_near_match(
                    sketch,
                    args.sketch_size,
                    args.index_size,
                    args.minimum_shared_index_hashes,
                    args.maximum_candidates,
                )
                if near_key and near_similarity >= args.similarity_threshold:
                    match_type = "near_document"
                    match_reference = ProtectedReference(near_key, "", "")
                    similarity = near_similarity

        if not match_type and paragraph_reference:
            match_type = "exact_paragraph"
            match_reference = paragraph_reference

        flagged = match_type is not None and match_reference is not None
        if flagged:
            index.add_match(
                match_type,
                document,
                match_reference.key,
                similarity,
                audit_sample(paragraph_sample) if paragraph_sample else None,
            )
            counters[f"matches_{match_type}"] += 1
            counters["train_documents_flagged"] += 1
        else:
            counters["train_documents_clear"] += 1
        index.mark_train_processed(document.document_id, flagged)

        if counters["train_documents_inspected"] % args.commit_interval == 0:
            index.commit()

    batch: list[Document] = []
    try:
        for document in iter_jsonl(Path(args.train)):
            if index.was_train_processed(document.document_id):
                counters["train_documents_already_processed"] += 1
                continue
            if (
                args.maximum_train_documents
                and counters["train_documents_examined"] >= args.maximum_train_documents
            ):
                reached_end = False
                break
            counters["train_documents_examined"] += 1
            batch.append(document)
            if len(batch) < args.feature_batch_size:
                continue
            features = (
                executor.map(compute_features_in_worker, batch, chunksize=25)
                if executor
                else (compute_train_features(item, feature_settings) for item in batch)
            )
            for item in features:
                inspect(item)
            batch.clear()
        if batch:
            features = (
                executor.map(compute_features_in_worker, batch, chunksize=25)
                if executor
                else (compute_train_features(item, feature_settings) for item in batch)
            )
            for item in features:
                inspect(item)
    finally:
        if executor:
            executor.shutdown(cancel_futures=True)
    index.set_metadata("audit_complete", "1" if reached_end else "0")
    index.commit()
    return counters, reached_end


def reviewed_removal_ids(audit_path: Path) -> set[str]:
    """Return approved removals, refusing incomplete or unknown review labels."""

    removals: set[str] = set()
    incomplete = 0
    invalid: Counter[str] = Counter()
    with audit_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("review_label", "").strip().casefold()
            if not label:
                incomplete += 1
            elif label == "remove":
                removals.add(row["train_document_id"])
            elif label != "keep":
                invalid[label] += 1
    if incomplete or invalid:
        details = []
        if incomplete:
            details.append(f"{incomplete} unlabeled rows")
        if invalid:
            details.append("invalid labels: " + ", ".join(sorted(invalid)))
        raise SystemExit(
            "Review is incomplete; label every audit row 'remove' or 'keep' ("
            + "; ".join(details)
            + ")"
        )
    return removals


def write_clean_training(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    index: AuditIndex,
) -> int:
    """Write a new training file using explicit remove/keep review decisions."""

    if index.metadata("audit_complete") != "1":
        raise SystemExit("Cannot write cleaned training data before the audit is complete")
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing cleaned training file: {output_path}")
    flagged_ids = reviewed_removal_ids(audit_path)
    return write_jsonl(
        output_path,
        (
            document
            for document in iter_jsonl(input_path)
            if document.document_id not in flagged_ids
        ),
    )


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 checksum for an artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run or resume a decontamination audit and return its report."""

    checks = tuple(args.check or CHECKS)
    settings = settings_for(args, checks)
    settings_json = json.dumps(settings, ensure_ascii=False, sort_keys=True)
    index = AuditIndex(Path(args.database))
    try:
        previous_settings = index.metadata("settings")
        if previous_settings and previous_settings != settings_json:
            raise SystemExit(
                "Audit database settings do not match this run; use a new database or "
                "restore the original arguments"
            )
        index.set_metadata("settings", settings_json)
        (
            exact_documents,
            exact_paragraphs,
            protected_prefix_hashes,
            protected_counts,
        ) = index_protected_splits(args, checks, index)
        if index.metadata("audit_complete") == "1":
            run_counts: Counter[str] = Counter()
            reached_end = True
        else:
            run_counts, reached_end = scan_training(
                args,
                checks,
                index,
                exact_documents,
                exact_paragraphs,
                protected_prefix_hashes,
            )
        index.export_matches(Path(args.audit_output))
        cleaned_documents = None
        if args.clean_train_output:
            cleaned_documents = write_clean_training(
                Path(args.train),
                Path(args.clean_train_output),
                Path(args.audit_output),
                index,
            )
        cumulative_counts = index.cumulative_counts()
        report: dict[str, object] = {
            "status": "complete" if reached_end else "partial",
            "inputs": {
                "train": args.train,
                "validation": args.validation,
                "test": args.test,
            },
            "audit_output": args.audit_output,
            "audit_output_sha256": file_sha256(Path(args.audit_output)),
            "database": args.database,
            "clean_train_output": args.clean_train_output,
            "execution": {
                "workers": args.workers,
                "feature_batch_size": args.feature_batch_size,
                "commit_interval": args.commit_interval,
            },
            "settings": settings,
            "protected_counts": dict(sorted(protected_counts.items())),
            "counts_this_run": dict(sorted(run_counts.items())),
            "counts_cumulative": dict(sorted(cumulative_counts.items())),
            "policy": {
                "protected_splits": ["validation", "test"],
                "only_training_documents_are_flagged": True,
                "match_precedence": [
                    "exact_document",
                    "near_document",
                    "exact_paragraph",
                ],
                "official_inputs_modified": False,
                "manual_review_required_before_replacing_training_split": True,
            },
        }
        if cleaned_documents is not None:
            report["clean_train_documents_written"] = cleaned_documents
            report["clean_train_output_sha256"] = file_sha256(Path(args.clean_train_output))
        return report
    finally:
        index.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--clean-train-output")
    parser.add_argument("--check", action="append", choices=CHECKS)
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    parser.add_argument("--sketch-size", type=int, default=64)
    parser.add_argument("--index-size", type=int, default=16)
    parser.add_argument("--minimum-shared-index-hashes", type=int, default=2)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--minimum-shingles", type=int, default=20)
    parser.add_argument("--maximum-shingles", type=int, default=2048)
    parser.add_argument("--maximum-candidates", type=int, default=200)
    parser.add_argument("--minimum-paragraph-characters", type=int, default=100)
    parser.add_argument("--minimum-paragraph-words", type=int, default=15)
    parser.add_argument("--maximum-train-documents", type=int)
    parser.add_argument("--commit-interval", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--feature-batch-size", type=int, default=1000)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.similarity_threshold <= 1.0:
        raise SystemExit("--similarity-threshold must be between 0 and 1")
    if not 1 <= args.index_size <= args.sketch_size:
        raise SystemExit("--index-size must be between 1 and --sketch-size")
    if args.minimum_shared_index_hashes > args.index_size:
        raise SystemExit("--minimum-shared-index-hashes cannot exceed --index-size")
    if args.commit_interval <= 0:
        raise SystemExit("--commit-interval must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.feature_batch_size <= 0:
        raise SystemExit("--feature-batch-size must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    report = run(args)
    atomic_write_json(Path(args.report), report)
    counts = report["counts_cumulative"]
    print(
        f"status={report['status']} "
        + " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )


if __name__ == "__main__":
    main()
