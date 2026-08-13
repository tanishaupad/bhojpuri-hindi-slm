"""Create deterministic, document-grouped train/validation/test JSONL splits."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from lma_phase1.io import atomic_write_json, iter_jsonl, write_jsonl
from lma_phase1.schema import Document


def document_group(document: Document) -> str:
    """Choose a stable group so pages from one work cannot cross splits."""

    for key in ("group_id", "work_id", "book_id", "parent_id"):
        value = document.metadata.get(key)
        if value:
            return str(value)
    return document.document_id


def assign_split(document: Document, seed: str, train_ratio: float, validation_ratio: float) -> str:
    """Assign a source-stratified group using a stable cryptographic hash."""

    key = f"{seed}\0{document.source_name}\0{document_group(document)}"
    value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def split_corpus(args: argparse.Namespace) -> Counter[str]:
    """Write three split files and return record-level statistics."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"{name}.jsonl.gz" for name in ("train", "validation", "test")}
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite existing split files: {existing}")
    buffers: dict[str, list[Document]] = {name: [] for name in paths}
    counters: Counter[str] = Counter()

    for document in iter_jsonl(Path(args.input)):
        split_name = assign_split(document, args.seed, args.train_ratio, args.validation_ratio)
        buffers[split_name].append(document)
        counters[f"{split_name}_documents"] += 1
        counters[f"{split_name}_{document.collection_type}_documents"] += 1
        if len(buffers[split_name]) >= args.write_batch_size:
            write_jsonl(paths[split_name], buffers[split_name], append=True)
            buffers[split_name].clear()
    for split_name, buffer in buffers.items():
        write_jsonl(paths[split_name], buffer, append=True)
    return counters


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--seed", default="lma-phase-1-2026")
    parser.add_argument("--write-batch-size", type=int, default=1000)
    return parser


def main() -> None:
    """Run the deterministic splitter CLI."""

    args = build_parser().parse_args()
    total = args.train_ratio + args.validation_ratio + args.test_ratio
    if (
        abs(total - 1.0) > 1e-9
        or min(args.train_ratio, args.validation_ratio, args.test_ratio) <= 0
    ):
        raise SystemExit("Split ratios must be positive and sum to 1.0")
    counters = split_corpus(args)
    atomic_write_json(
        Path(args.report),
        {
            "input": args.input,
            "output_dir": args.output_dir,
            "ratios": {
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": args.test_ratio,
            },
            "seed": args.seed,
            "counts": dict(sorted(counters.items())),
        },
    )
    print(" ".join(f"{key}={value}" for key, value in sorted(counters.items())))


if __name__ == "__main__":
    main()
