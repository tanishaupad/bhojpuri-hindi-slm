"""Streaming Hugging Face dataset collector for downloaded corpus sources."""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lma_phase1.io import iter_jsonl, write_jsonl
from lma_phase1.schema import Document


def value_or_default(row: dict[str, Any], field: str | None, default: Any) -> Any:
    """Read an optional field from a dataset row."""

    if not field:
        return default
    value = row.get(field)
    return default if value is None or value == "" else value


def stable_row_id(dataset: str, row: dict[str, Any], index: int, id_field: str | None) -> str:
    """Construct a stable identifier without depending solely on stream position."""

    if id_field and row.get(id_field) is not None:
        return str(row[id_field]).replace(" ", "_")
    text = str(row.get("text", ""))
    digest = hashlib.sha256(f"{dataset}\0{text}".encode()).hexdigest()[:20]
    return f"{index}_{digest}"


def collect(args: argparse.Namespace) -> int:
    """Stream dataset rows into the common schema, resuming by row count."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency error is user-facing
        raise SystemExit("Install project dependencies first: pip install -e .") from exc

    output = Path(args.output)
    already_written = sum(1 for _ in iter_jsonl(output)) if output.exists() else 0
    load_kwargs: dict[str, Any] = {
        "path": args.dataset,
        "split": args.split,
        "streaming": True,
    }
    if args.subset:
        load_kwargs["name"] = args.subset
    if args.data_dir:
        load_kwargs["data_dir"] = args.data_dir
    # Parquet builders reject the CSV/text-only ``encoding_errors`` option.
    # The default strict mode needs no explicit argument.
    if args.encoding_errors and args.encoding_errors != "strict":
        load_kwargs["encoding_errors"] = args.encoding_errors
    data_files = getattr(args, "data_file", None)
    if data_files:
        load_kwargs["data_files"] = data_files
    dataset = load_dataset(**load_kwargs)
    if already_written:
        dataset = dataset.skip(already_written)

    collected_at = datetime.now(UTC).date().isoformat()
    batch: list[Document] = []
    written = 0
    for offset, row in enumerate(dataset, start=already_written):
        text = str(row.get(args.text_field, "")).strip()
        if not text:
            continue
        row_id = stable_row_id(args.dataset, row, offset, args.id_field)
        original_source = value_or_default(row, args.source_field, args.source_name)
        source_name = (
            f"{args.source_name}:{str(original_source).replace(' ', '_')}"
            if args.source_field
            else args.source_name
        )
        document = Document(
            document_id=f"{args.source_name}_{row_id}",
            language=args.language,
            text=text,
            source_name=source_name,
            source_url=value_or_default(row, args.url_field, None),
            collection_type="downloaded",
            collection_method="huggingface_datasets_streaming",
            license=str(value_or_default(row, args.license_field, args.license)),
            collected_at=collected_at,
            source_document_id=row_id,
            metadata={"dataset": args.dataset, "dataset_split": args.split},
        )
        batch.append(document)
        if len(batch) >= args.write_batch_size:
            written += write_jsonl(output, batch, append=True)
            batch.clear()
        if args.max_documents and written + len(batch) >= args.max_documents:
            break
    written += write_jsonl(output, batch, append=True)
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--data-file",
        action="append",
        help="Repository-relative data file; repeat to pin multiple files",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field")
    parser.add_argument("--source-field")
    parser.add_argument("--license-field")
    parser.add_argument("--url-field")
    parser.add_argument(
        "--encoding-errors",
        choices=("strict", "ignore", "replace"),
        default="strict",
        help="How text-backed datasets handle malformed byte sequences",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--write-batch-size", type=int, default=1000)
    return parser


def main() -> None:
    """Run the Hugging Face collector CLI."""

    args = build_parser().parse_args()
    written = collect(args)
    print(f"{args.source_name}: wrote {written} new records to {args.output}")


if __name__ == "__main__":
    main()
