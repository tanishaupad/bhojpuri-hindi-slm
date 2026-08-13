"""Calculate per-source corpus statistics and the required manual-token fraction."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from lma_phase1.io import atomic_write_json, iter_jsonl
from lma_phase1.text import text_metrics


def empty_counts() -> dict[str, int]:
    """Return a new mutable aggregate counter mapping."""

    return {"documents": 0, "characters": 0, "words": 0, "tokens": 0}


def calculate(args: argparse.Namespace) -> dict[str, Any]:
    """Stream all split documents and aggregate source/provenance statistics."""

    processor = None
    if args.tokenizer_model:
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Install project dependencies first: pip install -e .") from exc
        processor = spm.SentencePieceProcessor(model_file=args.tokenizer_model)

    overall = empty_counts()
    by_source: dict[str, dict[str, int]] = defaultdict(empty_counts)
    by_collection_type: dict[str, dict[str, int]] = defaultdict(empty_counts)
    by_split: dict[str, dict[str, int]] = defaultdict(empty_counts)
    by_split_collection: dict[str, dict[str, int]] = defaultdict(empty_counts)

    for item in args.input:
        split_name, raw_path = item.split("=", 1)
        for document in iter_jsonl(Path(raw_path)):
            metrics = text_metrics(document.text)
            token_count = len(processor.encode(document.text, out_type=int)) if processor else 0
            values = {
                "documents": 1,
                "characters": metrics.characters,
                "words": metrics.words,
                "tokens": token_count,
            }
            groups = (
                overall,
                by_source[document.source_name],
                by_collection_type[document.collection_type],
                by_split[split_name],
                by_split_collection[f"{split_name}:{document.collection_type}"],
            )
            for group in groups:
                for key, value in values.items():
                    group[key] += value

    train_manual = by_split_collection["train:manual"]["tokens"]
    train_downloaded = by_split_collection["train:downloaded"]["tokens"]
    denominator = train_manual + train_downloaded
    manual_fraction = train_manual / denominator if denominator else None
    return {
        "tokenizer_model": args.tokenizer_model,
        "overall": overall,
        "by_source": dict(sorted(by_source.items())),
        "by_collection_type": dict(sorted(by_collection_type.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_split_and_collection_type": dict(sorted(by_split_collection.items())),
        "training_manual_token_fraction": manual_fraction,
        "manual_requirement_met": manual_fraction >= 0.20 if manual_fraction is not None else None,
    }


def write_source_csv(path: Path, sources: dict[str, dict[str, int]]) -> None:
    """Write a report-ready per-source CSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "documents", "characters", "words", "tokens"],
            lineterminator="\n",
        )
        writer.writeheader()
        for source, counts in sources.items():
            writer.writerow({"source": source, **counts})


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="SPLIT=PATH",
        help="Repeat for train, validation, and test",
    )
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-csv", required=True)
    return parser


def main() -> None:
    """Run the corpus statistics CLI."""

    args = build_parser().parse_args()
    if any("=" not in item for item in args.input):
        raise SystemExit("Every --input must use SPLIT=PATH syntax")
    report = calculate(args)
    atomic_write_json(Path(args.output), report)
    write_source_csv(Path(args.source_csv), report["by_source"])
    print(
        f"documents={report['overall']['documents']} tokens={report['overall']['tokens']} "
        f"manual_fraction={report['training_manual_token_fraction']}"
    )


if __name__ == "__main__":
    main()
