"""Audit final training provenance with normalized, raw and packed token counts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from array import array
from collections import Counter
from pathlib import Path

import sentencepiece as spm

from lma_phase1.io import atomic_write_json
from lma_phase1.tokenizer.common import normalize_tokenizer_input


def recount(path: Path, tokenizer: Path, expected: dict, threads: int) -> dict:
    """Re-encode every document and verify the packed stream without storing it."""
    with path.open("rb") as handle:
        input_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if input_hash != expected["input_sha256"]:
        raise ValueError("Input differs from the final Phase 2 training split")
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    counts = {kind: Counter() for kind in ("manual", "downloaded")}
    packed_hash = hashlib.sha256()

    def consume(rows: list[dict]) -> None:
        texts = [row["text"] for row in rows]
        raw = processor.encode(texts, num_threads=threads)
        normalized = processor.encode(
            [normalize_tokenizer_input(text) for text in texts], num_threads=threads
        )
        for row, ids, norm in zip(rows, raw, normalized, strict=True):
            c = counts[row["collection_type"]]
            c.update(
                {
                    "documents": 1,
                    "raw_tokens": len(ids),
                    "normalized_tokens": len(norm),
                    "packed_tokens": len(ids) + 1,
                    "utf8_bytes": len(row["text"].encode("utf-8")),
                }
            )
            packed = array("H", [*ids, processor.eos_id()])
            import sys

            if sys.byteorder != "little":
                packed.byteswap()
            packed_hash.update(packed.tobytes())

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        batch = []
        for line in handle:
            row = json.loads(line)
            if row["collection_type"] not in counts:
                raise ValueError("Unknown collection type")
            batch.append(row)
            if len(batch) == 512:
                consume(batch)
                batch.clear()
                n = sum(c["documents"] for c in counts.values())
                if n % 51200 == 0:
                    print(path.parent.parent.parent.name, n, "documents", flush=True)
        if batch:
            consume(batch)
    totals = counts["manual"] + counts["downloaded"]
    for actual, key in [
        ("documents", "documents"),
        ("packed_tokens", "tokens"),
        ("utf8_bytes", "utf8_bytes"),
    ]:
        if totals[actual] != expected[key]:
            raise ValueError(f"Manifest mismatch: {actual}")
    if packed_hash.hexdigest() != expected["token_file_sha256"]:
        raise ValueError("Re-encoded stream differs from the training token-stream hash")
    return {
        "input_sha256": input_hash,
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        "reencoded_packed_sha256": packed_hash.hexdigest(),
        "matches_phase2_manifest": True,
        "by_collection_type": {k: dict(v) for k, v in counts.items()},
        "totals": dict(totals),
        "manual_fractions": {
            key: counts["manual"][key] / totals[key]
            for key in ("normalized_tokens", "raw_tokens", "packed_tokens")
        },
        "manual_requirement_met_excluding_eos": counts["manual"]["raw_tokens"]
        / totals["raw_tokens"]
        >= 0.20,
    }


def main() -> None:
    """Read Drive inputs; write only the requested local audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "report/audits/generated/phase2_data_manifest.json").read_text())
    result = {
        "protocol": "Final decontaminated training splits; raw content tokens exclude "
        "EOS, packed counts assign one EOS to the source document. Normalized counts "
        "are a separate diagnostic. Packed SHA-256 is reproduced in document order.",
        "languages": {},
    }
    for language in ("bhojpuri", "hindi"):
        result["languages"][language] = recount(
            args.artifact_root / language / "splits/phase2_decontaminated_v1/train.jsonl.gz",
            root / language / f"tokenizer/{language}.model",
            manifest["languages"][language]["splits"]["train"],
            args.threads,
        )
        atomic_write_json(args.output, result)
        print(language, result["languages"][language]["manual_fractions"], flush=True)


if __name__ == "__main__":
    main()
