"""Deterministically downsample downloaded documents while preserving all manual text."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lma_phase1.io import atomic_write_json, iter_jsonl, write_jsonl


@dataclass(frozen=True, slots=True)
class Candidate:
    """Minimal downloaded-document information needed for stable selection."""

    score: bytes
    document_id: str
    words: int


def selection_score(seed: str, document_id: str) -> bytes:
    """Return a stable pseudorandom ordering score."""

    return hashlib.sha256(f"{seed}\0{document_id}".encode()).digest()


def downloaded_word_budget(manual_words: int, target_fraction: float) -> int:
    """Return the largest downloaded word count compatible with the target."""

    if not 0 < target_fraction < 1:
        raise ValueError("target manual fraction must be between zero and one")
    return int(manual_words * (1 - target_fraction) / target_fraction)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Select downloaded records by stable hash and write the balanced corpus."""

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output_path}")

    manual_words = 0
    manual_documents = 0
    downloaded_words = 0
    candidates: list[Candidate] = []
    for document in iter_jsonl(input_path):
        words = len(document.text.split())
        if document.collection_type == "manual":
            manual_words += words
            manual_documents += 1
        elif document.collection_type == "downloaded":
            downloaded_words += words
            candidates.append(
                Candidate(selection_score(args.seed, document.document_id), document.document_id, words)
            )
        else:
            raise SystemExit(f"Unsupported collection_type: {document.collection_type}")

    budget = min(
        downloaded_words,
        downloaded_word_budget(manual_words, args.target_manual_word_fraction),
    )
    selected_ids: set[str] = set()
    selected_downloaded_words = 0
    for candidate in sorted(candidates, key=lambda item: (item.score, item.document_id)):
        if selected_downloaded_words + candidate.words <= budget:
            selected_ids.add(candidate.document_id)
            selected_downloaded_words += candidate.words

    counters: Counter[str] = Counter()
    batch = []
    for document in iter_jsonl(input_path):
        words = len(document.text.split())
        keep = document.collection_type == "manual" or document.document_id in selected_ids
        decision = "kept" if keep else "dropped"
        counters[f"{decision}_{document.collection_type}_documents"] += 1
        counters[f"{decision}_{document.collection_type}_words"] += words
        if keep:
            batch.append(document)
            if len(batch) >= args.write_batch_size:
                write_jsonl(output_path, batch, append=True)
                batch.clear()
    if batch:
        write_jsonl(output_path, batch, append=True)

    kept_words = manual_words + selected_downloaded_words
    actual_fraction = manual_words / kept_words if kept_words else None
    report = {
        "input": args.input,
        "output": args.output,
        "seed": args.seed,
        "target_manual_word_fraction": args.target_manual_word_fraction,
        "downloaded_word_budget": budget,
        "actual_manual_word_fraction": actual_fraction,
        "counts": dict(sorted(counters.items())),
    }
    atomic_write_json(Path(args.report), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-manual-word-fraction", type=float, default=0.22)
    parser.add_argument("--seed", default="lma-downsample-2026")
    parser.add_argument("--write-batch-size", type=int, default=1000)
    return parser


def main() -> None:
    """Run the deterministic downsampler CLI."""

    args = build_parser().parse_args()
    if not 0 < args.target_manual_word_fraction < 1:
        raise SystemExit("--target-manual-word-fraction must be between zero and one")
    report = run(args)
    print(
        f"manual_word_fraction={report['actual_manual_word_fraction']:.6f} "
        f"downloaded_word_budget={report['downloaded_word_budget']}"
    )


if __name__ == "__main__":
    main()
