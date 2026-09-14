"""Re-evaluate pretrained models with explicit answer-and-explanation requests."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import sentencepiece as spm
import torch

from lma_phase1.io import atomic_write_json
from lma_phase1.tokenizer.common import normalize_tokenizer_input
from lma_phase2.config import load_experiment_config
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase3.cached_decode import greedy_cached
from lma_phase3.evaluate import file_hash, greedy_batch
from lma_phase3.generate import FAMILIES
from lma_phase3.proof_checker import check_output, reference_explanation
from lma_phase3.release import write_gzip

INSTRUCTIONS = {
    "hi": "पहले उत्तर लिखें। फिर कारण: लिखकर तुलना का छोटा कारण बताएँ।",
    "bho": "पहिले जवाब लिखीं। फेर कारण: लिख के तुलना के छोट कारण बताईं।",
    "ne": "पहिले उत्तर लेख्नुहोस्। त्यसपछि कारण: लेखेर छोटो तुलना दिनुहोस्।",
}
SPLITS = ("validation", "test", "challenge_names", "challenge_wording")
LIMIT = 96


def explanation_prompt(row: dict, base: dict) -> str:
    """Append a target-language format request, without showing the solution or proof."""

    cue = base["answer_cue"]
    if not row["prompt"].endswith(cue):
        raise ValueError("Missing answer cue")
    return row["prompt"][: -len(cue)] + INSTRUCTIONS[base["language"]] + " " + cue


def aggregate(rows: list[dict]) -> dict:
    """Report coverage and verified correctness on the full denominator."""

    count = len(rows)
    fields = (
        "answer_parsed",
        "answer_correct",
        "explanation_present",
        "explanation_parseable",
        "explanation_verified",
        "joint_correct",
    )
    totals = {field: sum(r["checks"][field] for r in rows) for field in fields}
    return {
        "examples": count,
        **totals,
        "rates": {field: total / count for field, total in totals.items()},
        "statuses": dict(Counter(r["checks"]["status"] for r in rows)),
        "stopped_at_eos": sum(r["stop_reason"] == "eos" for r in rows),
    }


def read_progress(path: Path) -> list[dict]:
    """Read durable records, tolerating only a torn final JSON line after interruption."""

    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("Corrupt progress record before the last line") from None
    return rows


def append_progress(path: Path, rows: list[dict]) -> None:
    """Durably save each finished batch before beginning the next one."""

    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    """Freeze/check the new protocol and run all original held-out sets without training."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("Use an empty output directory; old evaluations are preserved")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    root = Path(__file__).resolve().parents[2]
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(20260910)
    code = {
        p.name: file_hash(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("proof_checker.py"),
            Path(__file__).with_name("cached_decode.py"),
        ]
    }
    prepared = {}
    preflight = {}
    for lang in ("hindi", "bhojpuri"):
        base = json.loads((root / lang / "configs/phase3_templates.json").read_text())
        tokenizer_path = root / lang / "tokenizer" / f"{lang}.model"
        tok = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        manifest = json.loads((args.dataset_root / lang / "manifest.json").read_text())
        frozen = json.loads(
            (root / f"report/audits/generated/phase3/reasoning_v1/{lang}_manifest.json").read_text()
        )
        if manifest != frozen or file_hash(tokenizer_path) != manifest["tokenizer_sha256"]:
            raise ValueError("Release or tokenizer mismatch")
        if file_hash(root / lang / "configs/phase3_templates.json") != manifest["templates_sha256"]:
            raise ValueError("Templates changed since release")
        groups = {}
        max_prompt = 0
        max_target = 0
        for split in SPLITS:
            path = args.dataset_root / lang / f"{split}.jsonl.gz"
            if file_hash(path) != manifest["split_summaries"][split]["sha256"]:
                raise ValueError("Dataset hash mismatch")
            with gzip.open(path, "rt") as f:
                rows = [json.loads(line) for line in f]
            if len(rows) != manifest["split_summaries"][split]["examples"]:
                raise ValueError("Wrong split size")
            for row in rows:
                prompt = explanation_prompt(row, base)
                reference = reference_explanation(row, base)
                ids = tok.encode(normalize_tokenizer_input(prompt))
                reference_ids = tok.encode(normalize_tokenizer_input(reference)) + [tok.eos_id()]
                if not check_output(row, reference, base)["joint_correct"]:
                    raise ValueError("Constructed reference does not pass checker")
                if len(reference_ids) > LIMIT or len(ids) + LIMIT > 256:
                    raise ValueError("Output limit or context would truncate a valid reference")
                if tok.unk_id() in ids + reference_ids:
                    raise ValueError("Unknown token in explanation protocol")
                max_prompt = max(max_prompt, len(ids))
                max_target = max(max_target, len(reference_ids))
                row["evaluation_prompt"] = prompt
                row["reference_explanation_output"] = reference
                row["prompt_ids"] = ids
            groups[split] = rows
        prepared[lang] = (base, tok, manifest, groups)
        preflight[lang] = {
            "reference_proofs_verified": sum(len(r) for r in groups.values()),
            "maximum_prompt_tokens": max_prompt,
            "maximum_reference_tokens_including_eos": max_target,
        }
        print(lang, "preflight passed", preflight[lang], flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": "pretrained-proof-v2",
        "status": "supplemental protocol introduced after answer-only baseline",
        "instructions": INSTRUCTIONS,
        "generation": {
            "decoding": "greedy",
            "maximum_new_tokens": LIMIT,
            "stop": "EOS or token limit",
            "precision": "float32",
            "device": str(device),
            "batch_size": args.batch_size,
        },
        "code_sha256": code,
        "preflight": preflight,
        "checker": "Controlled complete-sentence parser; supported facts plus task-specific sufficient proof.",
        "coverage": "Unparsed sentences are unverified, not proof of factual error.",
        "comparison": "Prompt and token budget differ from the frozen 16-token answer-only baseline; report separately.",
    }
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Resume protocol or code changed; use a new output directory")
    else:
        atomic_write_json(protocol_path, protocol)
    checkpoints = json.loads(
        (root / "report/results/generated/phase2/checkpoint_manifest.json").read_text()
    )
    for lang, (base, tok, manifest, groups) in prepared.items():
        print(lang, "loading original checkpoint", flush=True)
        info = checkpoints["checkpoints"][lang]
        path = args.artifact_root / lang / "phase2/runs/pretrain_v1" / info["file"]
        checkpoint_hash = file_hash(path)
        if checkpoint_hash != info["sha256"]:
            raise ValueError("Wrong pretrained checkpoint")
        config = load_experiment_config(root / lang / "configs/phase2.toml")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["config"] != config.to_dict():
            raise ValueError("Checkpoint config mismatch")
        model = DecoderOnlyTransformer(config.model)
        model.load_state_dict(saved["model_state"])
        del saved
        model.to(device).eval()
        crosschecks = []
        # Check real trained weights and new prompts before using the cache in the full evaluation.
        for family in FAMILIES:
            row = next(r for r in groups["validation"] if r["task_family"] == family)
            a = greedy_cached(model, [row["prompt_ids"]], tok.eos_id(), device, 16)[0]
            b = greedy_batch(model, [row["prompt_ids"]], tok.eos_id(), device, 16)[0]
            if a != b:
                raise ValueError("Cached/uncached trained-model generation differs")
            crosschecks.append(
                {"id": row["id"], "cached_equals_uncached": True, "comparison_limit": 16}
            )
        print(lang, "real-checkpoint cache cross-checks passed", flush=True)
        directory = args.output_dir / lang
        directory.mkdir(exist_ok=True)
        metrics = {}
        for split, rows in groups.items():
            started = time.monotonic()
            last_update = started
            metrics_path = directory / f"{split}_metrics.json"
            final_path = directory / f"{split}_predictions.jsonl.gz"
            if metrics_path.exists():
                finished = json.loads(metrics_path.read_text())
                if (
                    file_hash(final_path) != finished["predictions_sha256"]
                    or finished["dataset_sha256"] != manifest["split_summaries"][split]["sha256"]
                ):
                    raise ValueError("Completed split hash mismatch on resume")
                metrics[split] = finished
                print(lang, split, "already complete", flush=True)
                continue
            progress_path = directory / f"{split}.partial.jsonl"
            prior = read_progress(progress_path)
            positions = {row["id"]: i for i, row in enumerate(rows)}
            predictions = [None] * len(rows)
            for prediction in prior:
                index = positions[prediction["id"]]
                row = rows[index]
                if predictions[index] is not None:
                    raise ValueError("Duplicate progress prediction")
                if (
                    prediction["prompt"] != row["evaluation_prompt"]
                    or prediction["reference_answer"] != row["answer_text"]
                    or tok.decode(prediction["generated_ids"]) != prediction["generated"]
                    or check_output(row, prediction["generated"], base) != prediction["checks"]
                ):
                    raise ValueError("Progress prediction/source mismatch")
                predictions[index] = prediction
            # Rewrite valid records to remove a torn trailing line, if any.
            progress_path.write_text("")
            append_progress(progress_path, prior)
            completed = len(prior)
            buckets = defaultdict(list)
            for index, row in enumerate(rows):
                if predictions[index] is None:
                    buckets[len(row["prompt_ids"])].append(index)
            print(lang, split, "starting from", completed, "/", len(rows), flush=True)
            for length in sorted(buckets):
                indices = buckets[length]
                for start in range(0, len(indices), args.batch_size):
                    batch = indices[start : start + args.batch_size]
                    generations = greedy_cached(
                        model, [rows[i]["prompt_ids"] for i in batch], tok.eos_id(), device, LIMIT
                    )
                    for index, ids in zip(batch, generations, strict=True):
                        row = rows[index]
                        text = tok.decode(ids)
                        predictions[index] = {
                            "id": row["id"],
                            "task_family": row["task_family"],
                            "split": split,
                            "prompt": row["evaluation_prompt"],
                            "reference_answer": row["answer_text"],
                            "reference_explanation_output": row["reference_explanation_output"],
                            "generated": text,
                            "generated_ids": ids,
                            "stop_reason": "eos" if ids[-1] == tok.eos_id() else "token_limit",
                            "checks": check_output(row, text, base),
                        }
                    append_progress(progress_path, [predictions[i] for i in batch])
                    completed += len(batch)
                    if time.monotonic() - last_update >= 20:
                        print(lang, split, completed, "/", len(rows), "checked", flush=True)
                        last_update = time.monotonic()
            prediction_path = directory / f"{split}_predictions.jsonl.gz"
            write_gzip(prediction_path, predictions)
            result = {
                "overall": aggregate(predictions),
                "per_task": {
                    f: aggregate([r for r in predictions if r["task_family"] == f])
                    for f in FAMILIES
                },
                "elapsed_seconds": time.monotonic() - started,
                "predictions_sha256": file_hash(prediction_path),
                "dataset_sha256": manifest["split_summaries"][split]["sha256"],
            }
            metrics[split] = result
            atomic_write_json(directory / f"{split}_metrics.json", result)
            progress_path.unlink()
            print(lang, split, json.dumps(result["overall"]), flush=True)
        atomic_write_json(
            directory / "summary.json",
            {
                "language": lang,
                "model_stage": "pretrained",
                "checkpoint_sha256": checkpoint_hash,
                "tokenizer_sha256": manifest["tokenizer_sha256"],
                "protocol_sha256": file_hash(args.output_dir / "protocol.json"),
                "cache_crosschecks": crosschecks,
                "torch_version": torch.__version__,
                "metrics": metrics,
            },
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
