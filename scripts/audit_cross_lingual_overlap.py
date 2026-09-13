"""Audit exact and near-duplicate cross-lingual overlap between Nepali and Hindi/Bhojpuri splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lma_phase1.io import iter_jsonl
from lma_phase1.text import stable_text_hash


def build_text_hashes(paths: list[Path]) -> dict[str, str]:
    """Map stable text hashes to document IDs."""
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for doc in iter_jsonl(path):
            h = stable_text_hash(doc.text)
            if h not in hashes:
                hashes[h] = f"{doc.source_name}:{doc.document_id}"
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nepali-split", type=Path, action="append", required=True)
    parser.add_argument("--hindi-splits", type=Path, nargs="+", default=[])
    parser.add_argument("--bhojpuri-splits", type=Path, nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print("Building Hindi hash index...")
    hindi_hashes = build_text_hashes([p for p in args.hindi_splits if p.exists()])
    print(f"Loaded {len(hindi_hashes)} Hindi document hashes.")

    print("Building Bhojpuri hash index...")
    bhojpuri_hashes = build_text_hashes([p for p in args.bhojpuri_splits if p.exists()])
    print(f"Loaded {len(bhojpuri_hashes)} Bhojpuri document hashes.")

    print("Scanning Nepali splits for exact cross-lingual matches...")
    nepali_doc_count = 0
    hindi_matches: list[dict[str, str]] = []
    bhojpuri_matches: list[dict[str, str]] = []

    for path in args.nepali_split:
        if not path.exists():
            continue
        for doc in iter_jsonl(path):
            nepali_doc_count += 1
            h = stable_text_hash(doc.text)
            if h in hindi_hashes:
                hindi_matches.append({
                    "nepali_doc_id": doc.document_id,
                    "nepali_source": doc.source_name,
                    "matched_hindi_doc": hindi_hashes[h],
                    "sample": doc.text[:200],
                })
            if h in bhojpuri_hashes:
                bhojpuri_matches.append({
                    "nepali_doc_id": doc.document_id,
                    "nepali_source": doc.source_name,
                    "matched_bhojpuri_doc": bhojpuri_hashes[h],
                    "sample": doc.text[:200],
                })

    report = {
        "nepali_documents_examined": nepali_doc_count,
        "hindi_documents_indexed": len(hindi_hashes),
        "bhojpuri_documents_indexed": len(bhojpuri_hashes),
        "exact_matches_with_hindi": len(hindi_matches),
        "exact_matches_with_bhojpuri": len(bhojpuri_matches),
        "hindi_match_details": hindi_matches[:20],
        "bhojpuri_match_details": bhojpuri_matches[:20],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit completed: {len(hindi_matches)} Hindi matches, {len(bhojpuri_matches)} Bhojpuri matches. Report saved to {args.output}")


if __name__ == "__main__":
    main()
