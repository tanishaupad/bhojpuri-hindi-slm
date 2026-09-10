"""Compare generic-text language modeling before and after SFT on identical tokens."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from lma_phase1.io import atomic_write_json
from lma_phase2.config import load_experiment_config
from lma_phase2.data import PackedTokenDataset
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase2.training import select_device
from lma_phase3.evaluate import file_hash


@torch.inference_mode()
def evaluate_stream(model, dataset, device, batch_size=8):
    """Score all targets, including the tail, with the original Phase 2 boundaries."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    context = model.config.context_length
    dataset.validate_context(context)
    byte_count = int(dataset.manifest.get("utf8_bytes", 0))
    if byte_count <= 0:
        raise ValueError("Positive utf8_bytes required for BPB")
    model.eval()
    total_nll, total_tokens = 0.0, 0
    started = time.monotonic()
    for index, (inputs, targets) in enumerate(
        dataset.sequential_batches(batch_size, context, device)
    ):
        loss = float(model(inputs, targets=targets).loss)
        if not math.isfinite(loss):
            raise ValueError("Non-finite evaluation loss")
        total_nll += loss * targets.numel()
        total_tokens += targets.numel()
        if index % 100 == 0:
            print(f"Scored {total_tokens}/{dataset.token_count - 1} tokens", flush=True)
    if total_tokens < dataset.token_count - 1:
        tail = torch.from_numpy(
            np.asarray(dataset.tokens[total_tokens:], dtype=np.int64)
        ).unsqueeze(0).to(device)
        targets = tail[:, 1:]
        total_nll += float(model(tail[:, :-1], targets=targets).loss) * targets.numel()
        total_tokens += targets.numel()
    if total_tokens != dataset.token_count - 1 or not math.isfinite(total_nll):
        raise ValueError("Incomplete or non-finite held-out evaluation")
    ce = total_nll / total_tokens
    return {
        "cross_entropy": ce,
        "perplexity": math.exp(ce),
        "bits_per_byte": total_nll / (math.log(2) * byte_count),
        "tokens_scored": total_tokens,
        "utf8_bytes": byte_count,
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    """Validate artifact identities and evaluate both checkpoints in float32."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-config", type=Path, required=True)
    parser.add_argument("--finetuned-config", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = select_device(args.device)
    torch.set_num_threads(8)
    before = load_experiment_config(args.pretrained_config)
    after = load_experiment_config(args.finetuned_config)
    if before.model != after.model or before.identity.language != after.identity.language:
        raise ValueError("Comparison requires the same architecture and language")
    if before.identity.tokenizer != after.identity.tokenizer:
        raise ValueError("Tokenizer must remain fixed during SFT")
    dataset = PackedTokenDataset(args.test_manifest)
    root = Path(__file__).resolve().parents[2]
    import json

    data_record = json.loads((root / "report/audits/generated/phase2_data_manifest.json").read_text())
    language = before.identity.language_name.lower()
    expected = data_record["languages"][language]
    token_hash = file_hash(dataset.path)
    tokenizer_hash = file_hash(root / before.identity.tokenizer)
    if token_hash != expected["splits"]["test"]["token_file_sha256"]:
        raise ValueError("Test stream differs from the recorded Phase 2 test stream")
    if tokenizer_hash != expected["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from Phase 2")
    for key, value in [("token_count", "tokens"), ("utf8_bytes", "utf8_bytes")]:
        if dataset.manifest[key] != expected["splits"]["test"][value]:
            raise ValueError(f"Test manifest mismatch: {key}")
    result = {
        "language": language,
        "split": "full_phase2_test",
        "protocol": "Non-overlapping 256-target windows including the final partial window; "
        "original raw-text packing and EOS likelihood retained. BPB divides by original UTF-8 bytes.",
        "device": str(device),
        "precision": "float32",
        "torch_version": torch.__version__,
        "context_length": before.model.context_length,
        "batch_size": args.batch_size,
        "token_file_sha256": token_hash,
        "test_manifest_sha256": file_hash(args.test_manifest),
        "tokenizer_sha256": tokenizer_hash,
        "evaluator_sha256": file_hash(Path(__file__)),
    }
    for label, experiment, path in [
        ("pretrained", before, args.pretrained_checkpoint),
        ("finetuned", after, args.finetuned_checkpoint),
    ]:
        print(f"Evaluating {language}: {label}", flush=True)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != experiment.to_dict():
            raise ValueError(f"Checkpoint/config mismatch: {label}")
        model = DecoderOnlyTransformer(experiment.model).to(device)
        model.load_state_dict(checkpoint["model_state"])
        del checkpoint
        result[label] = evaluate_stream(model, dataset, device, args.batch_size)
        result[label]["checkpoint_sha256"] = file_hash(path)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        atomic_write_json(args.output, result)
    result["delta_after_minus_before"] = {
        key: result["finetuned"][key] - result["pretrained"][key]
        for key in ("cross_entropy", "perplexity", "bits_per_byte")
    }
    result["perplexity_ratio_after_over_before"] = (
        result["finetuned"]["perplexity"] / result["pretrained"]["perplexity"]
    )
    atomic_write_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
