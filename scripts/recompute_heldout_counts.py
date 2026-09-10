"""Compare normalized Phase 1 counts with raw-text Phase 2 packing on held-out splits."""

import argparse
import hashlib
import json
from pathlib import Path

import sentencepiece as sp

from lma_phase1.io import iter_jsonl
from lma_phase1.tokenizer.common import normalize_tokenizer_input

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifact-root", type=Path, required=True)
root = parser.parse_args().artifact_root
out = {}
for lang in ["hindi", "bhojpuri"]:
    tok = sp.SentencePieceProcessor(model_file=f"{lang}/tokenizer/{lang}.model")
    for split in ["validation", "test"]:
        p = root / lang / "splits/phase2_decontaminated_v1" / f"{split}.jsonl.gz"
        original = (
            root
            / lang
            / "splits"
            / ("downsampled22_v1" if lang == "bhojpuri" else "")
            / f"{split}.jsonl.gz"
        )
        row = {
            "documents": 0,
            "normalized_tokens": 0,
            "raw_tokens": 0,
            "raw_unk": 0,
            "normalized_unk": 0,
            "utf8_bytes": 0,
        }
        for d in iter_jsonl(p):
            a = tok.encode(d.text)
            b = tok.encode(normalize_tokenizer_input(d.text))
            row["documents"] += 1
            row["normalized_tokens"] += len(b)
            row["raw_tokens"] += len(a)
            row["raw_unk"] += a.count(tok.unk_id())
            row["normalized_unk"] += b.count(tok.unk_id())
            row["utf8_bytes"] += len(d.text.encode())
        row["packed_tokens"] = row["raw_tokens"] + row["documents"]
        row["sha256"] = hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        row["original_sha256"] = hashlib.file_digest(original.open("rb"), "sha256").hexdigest()
        out[f"{lang}_{split}"] = row
        print(lang, split, row, flush=True)
Path("report/audits/generated/heldout_recomputation.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
