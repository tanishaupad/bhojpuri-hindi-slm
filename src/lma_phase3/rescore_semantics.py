"""Re-score existing outputs by stated meaning without requiring answer/reason headers."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from lma_phase3.evaluate import file_hash
from lma_phase3.generate import FAMILIES
from lma_phase3.proof_checker import reference_explanation
from lma_phase3.release import write_gzip
from lma_phase3.score_answer_presence import score_row, summarize
from lma_phase3.semantic_check import score_semantics

FIELDS = (
    "answer_correct",
    "answer_resolved",
    "claim_coverage",
    "supported_claim_found",
    "has_unsupported_claim",
    "full_text_coverage",
    "sufficient_supported_justification",
    "explanation_fully_verified",
    "joint_supported",
    "joint_fully_verified",
)


def aggregate(rows):
    """Keep answer coverage, claim correctness and proof verification separate."""
    totals = {field: sum(r["semantic_checks"][field] for r in rows) for field in FIELDS}
    return {
        "examples": len(rows),
        **totals,
        "rates": {f: n / len(rows) for f, n in totals.items()},
        "answer_statuses": dict(Counter(r["semantic_checks"]["answer_status"] for r in rows)),
        "answer_presence": summarize([r["presence_checks"] for r in rows]),
        "answer_accuracy_among_resolved": totals["answer_correct"] / totals["answer_resolved"]
        if totals["answer_resolved"]
        else None,
    }


def main():
    """Audit reference acceptance and source hashes, then save per-output semantic checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new output directory")
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    references_checked = 0
    for lang in ("hindi", "bhojpuri"):
        base = json.loads((root / lang / "configs/phase3_templates.json").read_text())
        source_summary = json.loads((args.predictions_root / lang / "summary.json").read_text())
        source_metrics = source_summary.get("splits", source_summary.get("metrics"))
        metrics = {}
        for split in ("validation", "test", "challenge_names", "challenge_wording"):
            data_path = args.dataset_root / lang / f"{split}.jsonl.gz"
            sm = source_metrics[split]
            if file_hash(data_path) != sm.get("split_sha256", sm.get("dataset_sha256")):
                raise ValueError("Dataset hash mismatch")
            with gzip.open(data_path, "rt") as f:
                sources = {r["id"]: r for r in map(json.loads, f)}
            plain = args.predictions_root / lang / f"{split}_predictions.jsonl"
            path = plain if plain.exists() else plain.with_suffix(".jsonl.gz")
            if file_hash(path) != sm["predictions_sha256"]:
                raise ValueError("Prediction hash mismatch")
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt") as f:
                predictions = list(map(json.loads, f))
            if len(predictions) != len(sources) or {r["id"] for r in predictions} != set(sources):
                raise ValueError("Incomplete coverage")
            rows = []
            for p in predictions:
                source = sources[p["id"]]
                reference = reference_explanation(source, base).replace("कारण: ", "")
                if not score_semantics(source, reference, base)["joint_fully_verified"]:
                    raise ValueError("Marker-free reference proof failed")
                references_checked += 1
                rows.append(
                    {
                        "id": p["id"],
                        "task_family": source["task_family"],
                        "reference_answer": source["answer_text"],
                        "generated": p["generated"],
                        "presence_checks": score_row(p, source, base),
                        "semantic_checks": score_semantics(source, p["generated"], base),
                    }
                )
            out = args.output_dir / f"{lang}_{split}.jsonl.gz"
            write_gzip(out, rows)
            metrics[split] = {
                "overall": aggregate(rows),
                "per_task": {
                    f: aggregate([r for r in rows if r["task_family"] == f]) for f in FAMILIES
                },
                "input_predictions_sha256": file_hash(path),
                "scores_sha256": file_hash(out),
            }
            print(lang, split, json.dumps(metrics[split]["overall"]), flush=True)
        all_metrics[lang] = metrics
    summary = {
        "protocol": "format-independent semantic rescore v1",
        "predictions_root": str(args.predictions_root),
        "reference_proofs_passed_without_markers": references_checked,
        "languages": all_metrics,
        "code_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "semantic_check.py",
                "rescore_semantics.py",
                "proof_checker.py",
                "score_answer_presence.py",
            )
        },
        "limitations": [
            "Finite supported grammar, not an unrestricted natural-language judge.",
            "Unresolved answers are not counted as established incorrect conclusions.",
            "A supported claim or copied premise is not a sufficient explanation.",
            "Supported-subset proof metrics do not validate uninterpreted extra clauses.",
            "These post-hoc checks preserve original generation and frozen baseline results.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
