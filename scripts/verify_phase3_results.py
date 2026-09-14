"""Verify all saved Phase 3 finetuned predictions against released examples and scoring rules.

Run from an installed project environment. Download/release the reasoning dataset first
if it is not staged locally. Input predictions and model weights are never modified.
This verifies artifact consistency, not arbitrary-prose semantic accuracy or model provenance.
"""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

import sentencepiece as spm

from lma_phase3.evaluate import normalize_answer
from lma_phase3.evaluate_proofs import explanation_prompt
from lma_phase3.explicit_answer import score_explicit_answer
from lma_phase3.score_answer_presence import score_row
from lma_phase3.semantic_check import score_semantics


def main():
    """Recompute saved scores; this does not regenerate outputs or judge arbitrary prose."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=root / "tmp/phase3/reasoning_v1")
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=root / "report/results/generated/phase3/finetuned_v1",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=("hindi", "nepali", "bhojpuri"),
        default=["hindi", "nepali"],
        help="Languages to verify (default: hindi nepali)",
    )
    parser.add_argument(
        "--model-directories",
        nargs="+",
        help="Optional result subdirectories, one per language (defaults to language names)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON evidence file; use a new path to preserve earlier audits",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model_directories = args.model_directories or args.languages
    if len(model_directories) != len(args.languages):
        parser.error("--model-directories must contain one value per --languages entry")
    results = {}
    total = 0
    for lang, model_directory in zip(args.languages, model_directories, strict=True):
        base = json.loads((root / lang / "configs/phase3_templates.json").read_text())
        sp = spm.SentencePieceProcessor(model_file=str(root / lang / f"tokenizer/{lang}.model"))
        for proto, limit in [("original", 16), ("explanation", 96)]:
            folder = args.predictions_root / model_directory / f"protocol_{proto}_{limit}tokens"
            summary = json.loads((folder / "summary.json").read_text())
            for split in ["validation", "test", "challenge_names", "challenge_wording"]:
                with gzip.open(args.dataset_root / lang / f"{split}.jsonl.gz", "rt") as f:
                    src = {r["id"]: r for r in map(json.loads, f)}
                with gzip.open(folder / f"{split}_predictions.jsonl.gz", "rt") as f:
                    preds = list(map(json.loads, f))
                errors = collections.Counter()
                examples = []
                counts = collections.Counter()
                tasks = {}
                if len(preds) != len(src) or {p["id"] for p in preds} != set(src):
                    errors["coverage"] += 1
                for p in preds:
                    r = src[p["id"]]
                    expected_prompt = (
                        explanation_prompt(r, base) if proto == "explanation" else r["prompt"]
                    )
                    checks = {
                        "prompt": p["prompt"] == expected_prompt,
                        "reference": p["reference_answer"] == r["answer_text"],
                        "decode": normalize_answer(sp.decode(p["generated_token_ids"]))
                        == p["generated"],
                        "token_limit": len(p["generated_token_ids"]) <= limit,
                    }
                    s = score_semantics(r, p["generated"], base)
                    a = score_row(p, r, base)
                    checks["semantic"] = s == p["semantic_checks"]
                    checks["presence"] = a["answer_present"] == p["answer_present"]
                    checks["strict"] = a["strict_exact_match"] == p["strict_exact_match"]
                    explicit = score_explicit_answer(p["generated"], r["answer_text"], s)
                    checks["explicit"] = all(p[k] == v for k, v in explicit.items() if k in p)
                    counts["explicit_answer_correct_count"] += explicit["explicit_answer_correct"]
                    counts["explicit_answer_and_explanation_verified_count"] += explicit[
                        "explicit_answer_and_explanation_verified"
                    ]
                    for k, ok in checks.items():
                        if not ok:
                            errors[k] += 1
                    if not all(checks.values()) and len(examples) < 3:
                        examples.append(
                            {"id": p["id"], "failed": [k for k, v in checks.items() if not v]}
                        )
                    counts["exact_match_count"] += a["strict_exact_match"]
                    counts["answer_present_count"] += a["answer_present"]
                    mapping = {
                        "supported_claim_count": "supported_claim_found",
                        "unsupported_claim_count": "has_unsupported_claim",
                        "sufficient_proof_count": "sufficient_supported_justification",
                        "verified_explanation_count": "explanation_fully_verified",
                        "joint_verified_count": "joint_fully_verified",
                    }
                    for k, v in mapping.items():
                        counts[k] += s[v]
                    counts["answer_correct_count"] += s["answer_correct"]
                    counts["answer_resolved_count"] += s["answer_resolved"]
                    task = tasks.setdefault(r["task_family"], collections.Counter())
                    task["n"] += 1
                    task["joint"] += s["joint_fully_verified"]
                    task["answer_correct"] += s["answer_correct"]
                for k, v in counts.items():
                    if k in summary[split] and summary[split][k] != v:
                        errors["aggregate_" + k] += 1
                results[f"{lang}/{proto}/{split}"] = {
                    "n": len(preds),
                    "errors": dict(errors),
                    "error_examples": examples,
                    "counts": dict(counts),
                    "tasks": {k: dict(v) for k, v in tasks.items()},
                    "predictions_sha256": hashlib.sha256(
                        (folder / f"{split}_predictions.jsonl.gz").read_bytes()
                    ).hexdigest(),
                }
                total += len(preds)
                print(lang, proto, split, len(preds), dict(errors), dict(counts), flush=True)
    document = {"predictions_checked": total, "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n")
    if any(r["errors"] for r in results.values()):
        raise SystemExit("FAIL: saved predictions or summaries disagree with recomputed checks")
    print(
        f"PASS: {total} saved predictions agree with their source prompts, decoded text, scores and summaries."
    )


if __name__ == "__main__":
    main()
