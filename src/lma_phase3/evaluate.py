"""Frozen exact-match reasoning evaluation of independent pretrained models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import sentencepiece as spm
import torch

from lma_phase1.io import atomic_write_json
from lma_phase1.tokenizer.common import normalize_tokenizer_input
from lma_phase2.config import load_experiment_config
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase3.generate import FAMILIES


def normalize_answer(text: str) -> str:
    """Apply only the predeclared NFC and whitespace normalization."""

    return " ".join(unicodedata.normalize("NFC", text).split())


def file_hash(path: Path) -> str:
    """Stream SHA-256 without loading a checkpoint into memory twice."""

    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@torch.inference_mode()
def next_token_logits(model: DecoderOnlyTransformer, tokens: torch.Tensor) -> torch.Tensor:
    """Use the unchanged Transformer, projecting only the last position to vocabulary.

    Earlier-position vocabulary logits cannot affect the final-position prediction.
    Avoiding those unused projections reduces evaluation memory and CPU/GPU work.
    """

    length = tokens.shape[1]
    if length > model.config.context_length:
        raise ValueError("Generation exceeds model context")
    positions = torch.arange(length, device=tokens.device)
    hidden = model.embedding_dropout(
        model.token_embeddings(tokens) + model.position_embeddings(positions)[None, :, :]
    )
    for block in model.blocks:
        hidden, _ = block(hidden)
    return model.language_model_head(model.final_norm(hidden[:, -1]))


@torch.inference_mode()
def greedy_batch(model, prefixes: list[list[int]], eos_id: int, device, limit: int = 16):
    """Decode equal-length prefixes without padding or batch-dependent positions."""

    if not prefixes or not prefixes[0] or len({len(p) for p in prefixes}) != 1:
        raise ValueError("Batch requires nonempty equal-length prefixes")
    if len(prefixes[0]) + limit > model.config.context_length:
        raise ValueError("Full generation does not fit context")
    model.eval()
    tokens = torch.tensor(prefixes, dtype=torch.long, device=device)
    continuations = [[] for _ in prefixes]
    active = [True] * len(prefixes)
    for _ in range(limit):
        next_ids = next_token_logits(model, tokens).argmax(dim=-1)
        for i, token in enumerate(next_ids.tolist()):
            if active[i]:
                continuations[i].append(token)
                if token == eos_id:
                    active[i] = False
        if not any(active):
            break
        tokens = torch.cat((tokens, next_ids[:, None]), dim=1)
    return continuations


def score_prediction(row: dict, generated: str, resource: dict) -> dict:
    """Score the full continuation; classify errors without changing the primary metric."""

    prediction = normalize_answer(generated)
    reference = normalize_answer(row["answer_text"])
    if row["task_family"] == "equality":
        choices = [resource["yes"], resource["no"]]
    else:
        problem = row["problem"]
        choices = [problem["entities"][i] for i in problem["query"]]
    correct = prediction == reference
    category = (
        "correct"
        if correct
        else "empty_output"
        if not prediction
        else "wrong_canonical_answer"
        if prediction in choices
        else "noncanonical_output"
    )
    return {
        "correct": correct,
        "error_category": category,
        "normalized_prediction": prediction,
        "normalized_reference": reference,
    }


def summarize(predictions: list[dict]) -> dict:
    """Aggregate accuracy and error categories over all examples and each task family."""

    def metrics(rows):
        correct = sum(r["correct"] for r in rows)
        return {
            "examples": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
            "errors": dict(Counter(r["error_category"] for r in rows)),
            "stopped_at_eos": sum(r["stop_reason"] == "eos" for r in rows),
        }

    return {
        "overall": metrics(predictions),
        "per_task": {
            family: metrics([r for r in predictions if r["task_family"] == family])
            for family in FAMILIES
        },
    }


def evaluate_split(
    model, rows: list[dict], tokenizer, resource: dict, device, batch_size: int, progress_name: str
) -> list[dict]:
    """Bucket by prompt length for padding-free generation and restore original order."""

    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        ids = tokenizer.encode(normalize_tokenizer_input(row["prompt"]), out_type=int)
        if tokenizer.unk_id() in ids:
            raise ValueError("Unknown token in release prompt")
        buckets[len(ids)].append((index, ids))
    predictions = [None] * len(rows)
    completed = 0
    last_update = time.monotonic()
    for length in sorted(buckets):
        bucket = buckets[length]
        for start in range(0, len(bucket), batch_size):
            batch = bucket[start : start + batch_size]
            generated = greedy_batch(model, [ids for _, ids in batch], tokenizer.eos_id(), device)
            for (index, _), token_ids in zip(batch, generated, strict=True):
                row = rows[index]
                text = tokenizer.decode(token_ids)
                predictions[index] = {
                    "id": row["id"],
                    "split": row["split"],
                    "task_family": row["task_family"],
                    "domain": row["domain"],
                    "prompt": row["prompt"],
                    "reference": row["answer_text"],
                    "generated": text,
                    "generated_ids": token_ids,
                    "stop_reason": "eos" if token_ids[-1] == tokenizer.eos_id() else "token_limit",
                    **score_prediction(row, text, resource),
                }
            completed += len(batch)
            if time.monotonic() - last_update >= 20:
                print(progress_name, completed, "/", len(rows), "scored", flush=True)
                last_update = time.monotonic()
    return predictions


def main() -> None:
    """Verify all input hashes and save full predictions and baseline summary JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Project Drive root containing each language checkpoint folder",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Release root with hindi/ and bhojpuri/ folders",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", choices=["both", "hindi", "bhojpuri"], default="both")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    root = Path(__file__).resolve().parents[2]
    checkpoint_manifest = json.loads(
        (root / "report/results/generated/phase2/checkpoint_manifest.json").read_text()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    device = torch.device(args.device)
    for lang in ("hindi", "bhojpuri") if args.language == "both" else (args.language,):
        directory = args.output_dir / lang
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Refusing to overwrite existing evaluation: {directory}")
        checkpoint_info = checkpoint_manifest["checkpoints"][lang]
        checkpoint_path = (
            args.artifact_root / lang / "phase2/runs/pretrain_v1" / checkpoint_info["file"]
        )
        print(lang, "verifying checkpoint and release hashes", flush=True)
        checkpoint_hash = file_hash(checkpoint_path)
        if checkpoint_hash != checkpoint_info["sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        experiment = load_experiment_config(root / lang / "configs/phase2.toml")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != experiment.to_dict():
            raise ValueError("Checkpoint configuration mismatch")
        model = DecoderOnlyTransformer(experiment.model)
        model.load_state_dict(checkpoint["model_state"])
        del checkpoint
        model.to(device).eval()
        dataset_dir = args.dataset_root / lang
        manifest = json.loads((dataset_dir / "manifest.json").read_text())
        recorded = json.loads(
            (root / f"report/audits/generated/phase3/reasoning_v1/{lang}_manifest.json").read_text()
        )
        if manifest != recorded:
            raise ValueError("Dataset manifest differs from frozen release")
        tokenizer_path = root / lang / "tokenizer" / f"{lang}.model"
        if file_hash(tokenizer_path) != manifest["tokenizer_sha256"]:
            raise ValueError("Tokenizer hash mismatch")
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        resource = json.loads((root / lang / "configs/phase3_templates.json").read_text())
        directory.mkdir(exist_ok=True)
        results = {}
        for split in ("validation", "test", "challenge_names", "challenge_wording"):
            source = dataset_dir / manifest["split_summaries"][split]["file"]
            if file_hash(source) != manifest["split_summaries"][split]["sha256"]:
                raise ValueError("Split hash mismatch")
            with gzip.open(source, "rt") as handle:
                rows = [json.loads(line) for line in handle]
            if len(rows) != manifest["split_summaries"][split]["examples"]:
                raise ValueError("Incomplete split")
            started = time.monotonic()
            predictions = evaluate_split(
                model, rows, tokenizer, resource, device, args.batch_size, f"{lang} {split}"
            )
            prediction_path = directory / f"{split}_predictions.jsonl"
            prediction_path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions)
            )
            result = summarize(predictions)
            result.update(
                elapsed_seconds=time.monotonic() - started,
                split_sha256=file_hash(source),
                predictions_sha256=file_hash(prediction_path),
            )
            results[split] = result
            atomic_write_json(directory / f"{split}_metrics.json", result)
            print(lang, split, json.dumps(result["overall"]), flush=True)
        summary = {
            "language": lang,
            "model_stage": "pretrained",
            "checkpoint_sha256": checkpoint_hash,
            "tokenizer_sha256": file_hash(tokenizer_path),
            "dataset_version": manifest["version"],
            "dataset_manifest_sha256": file_hash(dataset_dir / "manifest.json"),
            "protocol": manifest["scoring_protocol"],
            "device": str(device),
            "precision": "float32",
            "torch_version": torch.__version__,
            "batch_size": args.batch_size,
            "evaluator_sha256": file_hash(Path(__file__)),
            "splits": results,
        }
        atomic_write_json(directory / "summary.json", summary)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
