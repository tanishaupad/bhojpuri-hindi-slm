"""Recompute full validation metrics from the original independent checkpoints."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lma_phase2.data import PackedTokenDataset
from lma_phase2.evaluate import load_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifact-root", type=Path, required=True)
parser.add_argument("--language", choices=("both", "hindi", "bhojpuri"), default="both")
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()
manifest = json.loads(Path("report/results/generated/phase2/checkpoint_manifest.json").read_text())
data_manifest = json.loads(Path("report/audits/generated/phase2_data_manifest.json").read_text())
torch.set_num_threads(8)
for language in ("bhojpuri", "hindi") if args.language == "both" else (args.language,):
    started = time.monotonic()
    checkpoint = (
        args.artifact_root
        / language
        / "phase2/runs/pretrain_v1"
        / manifest["checkpoints"][language]["file"]
    )
    checkpoint_hash = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
    assert checkpoint_hash == manifest["checkpoints"][language]["sha256"]
    experiment, model = load_model(
        Path(language) / "configs/phase2.toml", checkpoint, torch.device("cpu")
    )
    dataset = PackedTokenDataset(
        args.artifact_root / language / "phase2/tokenized_v1/validation.manifest.json"
    )
    token_hash = hashlib.file_digest(dataset.path.open("rb"), "sha256").hexdigest()
    assert (
        token_hash
        == data_manifest["languages"][language]["splits"]["validation"]["token_file_sha256"]
    )
    total_nll = 0.0
    total_tokens = 0
    context = experiment.model.context_length
    # Same independent 256-token windows as intrinsic_metrics, batched for CPU throughput.
    with torch.inference_mode():
        for inputs, targets in dataset.sequential_batches(args.batch_size, context, torch.device("cpu")):
            loss = model(inputs, targets=targets).loss
            total_nll += float(loss) * targets.numel()
            total_tokens += targets.numel()
            if total_tokens % (2048 * 100) == 0:
                print(
                    language,
                    total_tokens,
                    "/",
                    dataset.token_count - 1,
                    "elapsed",
                    round(time.monotonic() - started),
                    flush=True,
                )
        if total_tokens < dataset.token_count - 1:
            values = torch.from_numpy(
                np.asarray(dataset.tokens[total_tokens:], dtype=np.int64)
            ).unsqueeze(0)
            targets = values[:, 1:]
            total_nll += float(model(values[:, :-1], targets=targets).loss) * targets.numel()
            total_tokens += targets.numel()
    assert total_tokens == dataset.token_count - 1
    utf8_bytes = dataset.manifest["utf8_bytes"]
    ce = total_nll / total_tokens
    result = {
        "language": language,
        "split": "full_validation",
        "cross_entropy": ce,
        "perplexity": math.exp(ce),
        "bits_per_byte": total_nll / (math.log(2) * utf8_bytes),
        "tokens_scored": total_tokens,
        "utf8_bytes": utf8_bytes,
        "checkpoint_sha256": checkpoint_hash,
        "token_file_sha256": token_hash,
        "precision": "float32",
        "device": "cpu",
        "torch_version": torch.__version__,
        "context_length": context,
        "batch_size": args.batch_size,
        "protocol": "Non-overlapping 256-target windows, including final partial window; same boundaries as phase2 intrinsic_metrics. BPB includes EOS likelihood and divides by original document UTF-8 bytes.",
        "elapsed_seconds": time.monotonic() - started,
    }
    output = Path("report/results/generated/phase2") / f"{language}_full_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
