"""Tokenize JSONL corpora into compact resumable uint16 streams for pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm

from lma_phase1.io import atomic_write_json, iter_jsonl, read_json

DTYPE = np.dtype("<u2")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def checkpoint_state(
    path: Path,
    input_path: Path,
    tokenizer_path: Path,
    documents: int,
    tokens: int,
    utf8_bytes: int,
    last_document_id: str,
) -> None:
    atomic_write_json(
        path,
        {
            "input": input_identity(input_path),
            "tokenizer": input_identity(tokenizer_path),
            "documents_written": documents,
            "tokens_written": tokens,
            "utf8_bytes_written": utf8_bytes,
            "bytes_written": tokens * DTYPE.itemsize,
            "last_document_id": last_document_id,
        },
    )


def prepare_split(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    processor: spm.SentencePieceProcessor,
    language: str,
    split: str,
    checkpoint_interval: int = 10_000,
) -> dict[str, Any]:
    """Tokenize one split, resuming from the last durable document boundary."""

    if output_path.exists() or manifest_path.exists():
        if output_path.exists() and manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as handle:
                return json.load(handle)
        raise SystemExit(
            f"Completed token file and manifest must both exist or both be absent: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    state_path = output_path.with_suffix(output_path.suffix + ".state.json")
    state = read_json(state_path, default={}) or {}
    documents_written = int(state.get("documents_written", 0))
    tokens_written = int(state.get("tokens_written", 0))
    utf8_bytes_written = int(state.get("utf8_bytes_written", 0))
    last_document_id = state.get("last_document_id")
    expected_bytes = tokens_written * DTYPE.itemsize
    if partial_path.exists() != bool(state):
        raise SystemExit(
            "Partial token file and state must both exist for resume or both be absent"
        )
    if state:
        if state.get("input") != input_identity(input_path):
            raise SystemExit("Input corpus changed since the tokenization checkpoint")
        if state.get("tokenizer") != input_identity(tokenizer_path):
            raise SystemExit("Tokenizer changed since the tokenization checkpoint")
        with partial_path.open("r+b") as handle:
            handle.truncate(expected_bytes)

    eos_id = processor.eos_id()
    if eos_id < 0:
        raise SystemExit("Tokenizer must define an EOS token for packed document boundaries")
    if processor.get_piece_size() > np.iinfo(DTYPE).max + 1:
        raise SystemExit("Vocabulary does not fit in uint16")

    skipping = bool(state)
    counters: Counter[str] = Counter()
    mode = "ab" if state else "wb"
    with partial_path.open(mode) as output_handle:
        for document in iter_jsonl(input_path):
            if skipping:
                if document.document_id == last_document_id:
                    skipping = False
                continue
            token_ids = processor.encode(document.text, out_type=int)
            token_ids.append(eos_id)
            np.asarray(token_ids, dtype=DTYPE).tofile(output_handle)
            documents_written += 1
            tokens_written += len(token_ids)
            utf8_bytes_written += len(document.text.encode("utf-8"))
            counters["documents_this_run"] += 1
            counters["tokens_this_run"] += len(token_ids)
            last_document_id = document.document_id
            if documents_written % checkpoint_interval == 0:
                output_handle.flush()
                os.fsync(output_handle.fileno())
                checkpoint_state(
                    state_path,
                    input_path,
                    tokenizer_path,
                    documents_written,
                    tokens_written,
                    utf8_bytes_written,
                    last_document_id,
                )
    if skipping:
        raise SystemExit("Resume document ID was not found in the input corpus")
    partial_path.replace(output_path)
    if state_path.exists():
        state_path.unlink()
    manifest = {
        "schema_version": "1.0",
        "language": language,
        "split": split,
        "input": input_identity(input_path),
        "input_sha256": file_sha256(input_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "token_file": output_path.name,
        "token_file_sha256": file_sha256(output_path),
        "dtype": DTYPE.str,
        "document_count": documents_written,
        "token_count": tokens_written,
        "utf8_bytes": utf8_bytes_written,
        "eos_id": eos_id,
        "vocabulary_size": processor.get_piece_size(),
        "counts_this_run": dict(sorted(counters.items())),
        "document_boundaries": "eos_token",
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_split(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SPLIT=/path/to/file.jsonl.gz")
    split, path = value.split("=", 1)
    if split not in {"train", "validation", "test"}:
        raise argparse.ArgumentTypeError("Split must be train, validation, or test")
    return split, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_split)
    parser.add_argument("--tokenizer-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint_interval <= 0:
        raise SystemExit("--checkpoint-interval must be positive")
    processor = spm.SentencePieceProcessor(model_file=str(args.tokenizer_model))
    for split, input_path in args.input:
        manifest = prepare_split(
            input_path=input_path,
            output_path=args.output_dir / f"{split}.tokens.bin",
            manifest_path=args.output_dir / f"{split}.manifest.json",
            tokenizer_path=args.tokenizer_model,
            processor=processor,
            language=args.language,
            split=split,
            checkpoint_interval=args.checkpoint_interval,
        )
        print(
            f"split={split} documents={manifest['document_count']} "
            f"tokens={manifest['token_count']} output={manifest['token_file']}"
        )


if __name__ == "__main__":
    main()
