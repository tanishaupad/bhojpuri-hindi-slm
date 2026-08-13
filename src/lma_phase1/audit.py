"""Create deterministic, human-reviewable corpus audit samples as CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from lma_phase1.io import iter_jsonl


def sample_score(seed: str, document_id: str) -> bytes:
    """Return a deterministic pseudorandom ordering key."""

    return hashlib.sha256(f"{seed}\0{document_id}".encode()).digest()


def create_sample(args: argparse.Namespace) -> int:
    """Choose the lowest stable hash scores and write a review worksheet."""

    candidates = []
    for document in iter_jsonl(Path(args.input)):
        candidates.append((sample_score(args.seed, document.document_id), document))
    chosen = [document for _, document in sorted(candidates, key=lambda item: item[0])[: args.size]]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing audit worksheet: {output}")
    fields = [
        "document_id",
        "source_name",
        "collection_type",
        "source_url",
        "text_sample",
        "language_label",
        "quality_label",
        "duplicate_label",
        "reviewer",
        "notes",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for document in chosen:
            writer.writerow(
                {
                    "document_id": document.document_id,
                    "source_name": document.source_name,
                    "collection_type": document.collection_type,
                    "source_url": document.source_url or "",
                    "text_sample": " ".join(document.text.split())[: args.sample_characters],
                    "language_label": "",
                    "quality_label": "",
                    "duplicate_label": "",
                    "reviewer": "",
                    "notes": "",
                }
            )
    return len(chosen)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--sample-characters", type=int, default=500)
    parser.add_argument("--seed", default="lma-phase-1-audit-2026")
    return parser


def main() -> None:
    """Run the audit worksheet generator."""

    args = build_parser().parse_args()
    if args.size <= 0 or args.sample_characters <= 0:
        raise SystemExit("--size and --sample-characters must be positive")
    count = create_sample(args)
    print(f"wrote {count} audit rows to {args.output}")


if __name__ == "__main__":
    main()
