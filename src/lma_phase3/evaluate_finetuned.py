"""Comprehensive evaluation of finetuned reasoning models across both protocols.

Evaluates on:
1. Original short-answer prompt (16 tokens)
2. Explanation prompt (96 tokens)
Computes exact match, whole-word answer presence, and semantic verification
(supported facts, unsupported claims, sufficient proofs, unresolved text).
Also reports explicit-answer accuracy and explicit-answer-plus-explanation success.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

from lma_phase1.io import atomic_write_json
from lma_phase1.tokenizer.common import normalize_tokenizer_input
from lma_phase2.config import load_experiment_config
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase2.training import select_device
from lma_phase3.cached_decode import greedy_cached
from lma_phase3.evaluate import normalize_answer
from lma_phase3.evaluate_proofs import explanation_prompt
from lma_phase3.explicit_answer import SCORING_VERSION, score_explicit_answer
from lma_phase3.score_answer_presence import score_row
from lma_phase3.semantic_check import score_semantics

SPLITS = ("validation", "test", "challenge_names", "challenge_wording")


def evaluate_split(
    model: DecoderOnlyTransformer,
    tokenizer: spm.SentencePieceProcessor,
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    protocol: str,
    device: torch.device,
    batch_size: int = 32,
    limit: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode and score an entire split under the given protocol."""
    predictions = []

    # Sort rows by prompt length for efficient batched decoding
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]

    for batch in batches:
        prompts = []
        for r in batch:
            if protocol == "explanation":
                p = explanation_prompt(r, base)
            else:
                p = r["prompt"]
            prompts.append(p)

        prefix_ids_list = [
            tokenizer.encode(normalize_tokenizer_input(p)) for p in prompts
        ]

        # Pad prefixes on the left if lengths differ, or decode item by item / bucketed
        # Since greedy_cached supports equal length batches, we group by length
        # For simplicity and robustness with small batches, run through greedy_cached
        generated_token_ids_list = []
        for prefix_ids in prefix_ids_list:
            gen_ids = greedy_cached(model, [prefix_ids], tokenizer.eos_id(), device, limit)[0]
            # Strip trailing EOS if present
            if gen_ids and gen_ids[-1] == tokenizer.eos_id():
                gen_ids = gen_ids[:-1]
            generated_token_ids_list.append(gen_ids)

        for r, p, gen_ids in zip(batch, prompts, generated_token_ids_list, strict=True):
            gen_text = tokenizer.decode(gen_ids)
            norm_gen = normalize_answer(gen_text)
            norm_ref = normalize_answer(r["answer_text"])

            p_check = score_row({"generated": norm_gen}, r, base)
            s_check = score_semantics(r, norm_gen, base)

            pred_record = {
                "id": r["id"],
                "task_family": r["task_family"],
                "prompt": p,
                "reference_answer": r["answer_text"],
                "generated": norm_gen,
                "generated_token_ids": gen_ids,
                "strict_exact_match": norm_gen == norm_ref,
                "answer_present": p_check["answer_present"],
                "semantic_checks": s_check,
                **score_explicit_answer(norm_gen, r["answer_text"], s_check),
            }
            predictions.append(pred_record)

    # Aggregate summary metrics
    total = len(predictions)
    exact_matches = sum(p["strict_exact_match"] for p in predictions)
    answer_presents = sum(p["answer_present"] for p in predictions)
    supported_claims = sum(p["semantic_checks"]["supported_claim_found"] for p in predictions)
    unsupported_claims = sum(p["semantic_checks"]["has_unsupported_claim"] for p in predictions)
    sufficient_proofs = sum(p["semantic_checks"]["sufficient_supported_justification"] for p in predictions)
    verified_explanations = sum(p["semantic_checks"]["explanation_fully_verified"] for p in predictions)
    joint_verified = sum(p["semantic_checks"]["joint_fully_verified"] for p in predictions)
    explicit_correct = sum(p["explicit_answer_correct"] for p in predictions)
    explicit_joint = sum(p["explicit_answer_and_explanation_verified"] for p in predictions)

    task_breakdown = defaultdict(lambda: {
        "count": 0, "exact_match": 0, "answer_present": 0, "joint_verified": 0,
        "explicit_answer_correct": 0, "explicit_answer_and_explanation_verified": 0,
    })
    for p in predictions:
        tf = p["task_family"]
        task_breakdown[tf]["count"] += 1
        task_breakdown[tf]["exact_match"] += int(p["strict_exact_match"])
        task_breakdown[tf]["answer_present"] += int(p["answer_present"])
        task_breakdown[tf]["joint_verified"] += int(p["semantic_checks"]["joint_fully_verified"])
        task_breakdown[tf]["explicit_answer_correct"] += int(p["explicit_answer_correct"])
        task_breakdown[tf]["explicit_answer_and_explanation_verified"] += int(
            p["explicit_answer_and_explanation_verified"]
        )

    for tf, d in task_breakdown.items():
        c = d["count"]
        d["exact_match_rate"] = d["exact_match"] / c
        d["answer_present_rate"] = d["answer_present"] / c
        d["joint_verified_rate"] = d["joint_verified"] / c
        d["explicit_answer_accuracy"] = d["explicit_answer_correct"] / c
        d["explicit_answer_and_explanation_verified_rate"] = (
            d["explicit_answer_and_explanation_verified"] / c
        )

    summary = {
        "protocol": protocol,
        "token_limit": limit,
        "total_examples": total,
        "answer_scoring_version": SCORING_VERSION,
        "explicit_answer_correct_count": explicit_correct,
        "explicit_answer_accuracy": explicit_correct / total,
        "explicit_answer_and_explanation_verified_count": explicit_joint,
        "explicit_answer_and_explanation_verified_rate": explicit_joint / total,
        "exact_match_count": exact_matches,
        "exact_match_rate": exact_matches / total,
        "answer_present_count": answer_presents,
        "answer_present_rate": answer_presents / total,
        "supported_claim_count": supported_claims,
        "supported_claim_rate": supported_claims / total,
        "unsupported_claim_count": unsupported_claims,
        "unsupported_claim_rate": unsupported_claims / total,
        "sufficient_proof_count": sufficient_proofs,
        "sufficient_proof_rate": sufficient_proofs / total,
        "verified_explanation_count": verified_explanations,
        "verified_explanation_rate": verified_explanations / total,
        "joint_verified_count": joint_verified,
        "joint_verified_rate": joint_verified / total,
        "task_breakdown": dict(task_breakdown),
    }

    return predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to config.toml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to reasoning_v1 root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--protocols", choices=("both", "original", "explanation"), default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = select_device(args.device)
    experiment = load_experiment_config(args.config)
    language_dirs = {"hi": "hindi", "bho": "bhojpuri", "ne": "nepali"}
    try:
        lang = language_dirs[experiment.identity.language]
    except KeyError as exc:
        raise ValueError(f"Unsupported language code: {experiment.identity.language}") from exc
    root = Path(__file__).resolve().parents[2]
    template_path = root / lang / "configs/phase3_templates.json"
    tokenizer_path = root / experiment.identity.tokenizer

    base = json.loads(template_path.read_text(encoding="utf-8"))
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    print(f"Loading checkpoint {args.checkpoint}...")
    model = DecoderOnlyTransformer(experiment.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    protocols = ["original", "explanation"] if args.protocols == "both" else [args.protocols]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, Any] = {}

    for proto in protocols:
        limit = 16 if proto == "original" else 96
        proto_dir = args.output_dir / f"protocol_{proto}_{limit}tokens"
        proto_dir.mkdir(parents=True, exist_ok=True)
        proto_summaries: dict[str, Any] = {}

        for split in SPLITS:
            data_file = args.dataset_root / lang / f"{split}.jsonl.gz"
            print(f"Evaluating {lang} {split} under {proto} protocol ({limit} tokens)...")
            with gzip.open(data_file, "rt", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]

            preds, summary = evaluate_split(
                model=model,
                tokenizer=tokenizer,
                rows=rows,
                base=base,
                protocol=proto,
                device=device,
                batch_size=args.batch_size,
                limit=limit,
            )

            # Save predictions
            preds_file = proto_dir / f"{split}_predictions.jsonl.gz"
            with gzip.open(preds_file, "wt", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

            proto_summaries[split] = summary
            print(
                f"  -> Exact Match: {summary['exact_match_rate']*100:.2f}% | "
                f"Explicit Answer: {summary['explicit_answer_accuracy']*100:.2f}% | "
                f"Answer Present: {summary['answer_present_rate']*100:.2f}% | "
                f"Joint Verified: {summary['joint_verified_rate']*100:.2f}%"
            )

        summary_file = proto_dir / "summary.json"
        atomic_write_json(summary_file, proto_summaries)
        all_summaries[proto] = proto_summaries

    atomic_write_json(args.output_dir / "evaluation_summary.json", all_summaries)
    print("All evaluations complete!")


if __name__ == "__main__":
    main()
