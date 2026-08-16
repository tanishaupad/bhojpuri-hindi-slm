"""Print a deterministic sample of corpus-wide exact paragraph duplicate pairs."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sqlite3
from pathlib import Path

from lma_phase1.io import iter_jsonl
from lma_phase1.paragraph_deduplicate import eligible_paragraph, paragraph_units
from lma_phase1.text import stable_text_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--minimum-paragraph-characters", type=int, default=100)
    parser.add_argument("--minimum-paragraph-words", type=int, default=15)
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    sample: list[tuple[int, dict[str, object]]] = []
    for document in iter_jsonl(args.input):
        status_row = connection.execute(
            "SELECT status FROM processed_ids WHERE document_id=?", (document.document_id,)
        ).fetchone()
        status = status_row[0] if status_row else "unknown"
        for paragraph in paragraph_units(document.text):
            if not eligible_paragraph(
                paragraph,
                args.minimum_paragraph_characters,
                args.minimum_paragraph_words,
            ):
                continue
            paragraph_hash = stable_text_hash(paragraph)
            match = connection.execute(
                "SELECT document_id FROM paragraph_hashes WHERE paragraph_hash=?",
                (paragraph_hash,),
            ).fetchone()
            if not match or match[0] == document.document_id:
                continue
            key = int.from_bytes(
                hashlib.sha256(
                    f"{document.document_id}:{match[0]}:{paragraph_hash}".encode()
                ).digest()[:8],
                "big",
            )
            item: dict[str, object] = {
                "candidate_document_id": document.document_id,
                "candidate_source_name": document.source_name,
                "candidate_collection_type": document.collection_type,
                "candidate_status": status,
                "matched_document_id": match[0],
                "paragraph_characters": len(paragraph),
                "paragraph_words": len(paragraph.split()),
                "paragraph_sample": " ".join(paragraph.split())[:700],
            }
            if len(sample) < args.sample_size:
                heapq.heappush(sample, (-key, item))
            elif key < -sample[0][0]:
                heapq.heapreplace(sample, (-key, item))

    matched_ids = {str(item["matched_document_id"]) for _, item in sample}
    matched_metadata: dict[str, tuple[str, str]] = {}
    for document in iter_jsonl(args.input):
        if document.document_id in matched_ids:
            matched_metadata[document.document_id] = (
                document.source_name,
                document.collection_type,
            )
            if len(matched_metadata) == len(matched_ids):
                break

    for _, item in sorted(sample, key=lambda entry: -entry[0]):
        matched_id = str(item["matched_document_id"])
        source_name, collection_type = matched_metadata.get(
            matched_id, ("unknown", "unknown")
        )
        item["matched_source_name"] = source_name
        item["matched_collection_type"] = collection_type
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))
    connection.close()


if __name__ == "__main__":
    main()
