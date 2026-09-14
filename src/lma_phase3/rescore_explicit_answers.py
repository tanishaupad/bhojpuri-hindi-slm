"""Retrospectively score explicit answers without regenerating or changing v1 outputs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from lma_phase1.io import atomic_write_json
from lma_phase3.evaluate import file_hash, normalize_answer
from lma_phase3.explicit_answer import SCORING_VERSION, score_explicit_answer
from lma_phase3.release import write_gzip


def read_rows(path):
    """Load saved gzip JSONL predictions or dataset rows."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def rescore_rows(predictions, sources):
    """Require complete unique IDs and matching references before applying the rule."""
    reference_by_id = {row["id"]: row for row in sources}
    ids = [p["id"] for p in predictions]
    if len(reference_by_id) != len(sources) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs")
    if set(ids) != set(reference_by_id):
        raise ValueError("Prediction/dataset coverage mismatch")
    rescored = []
    for prediction in predictions:
        source = reference_by_id[prediction["id"]]
        if normalize_answer(prediction["reference_answer"]) != normalize_answer(source["answer_text"]):
            raise ValueError("Prediction reference differs from dataset")
        semantics = prediction["semantic_checks"]
        presence = prediction.get("answer_present")
        if presence is None:
            presence = prediction["presence_checks"]["answer_present"]
        rescored.append({
            "id": prediction["id"],
            "task_family": source["task_family"],
            "generated": prediction["generated"],
            "reference_answer": source["answer_text"],
            "answer_present": presence,
            "strict_exact_match": normalize_answer(prediction["generated"])
            == normalize_answer(source["answer_text"]),
            "explanation_fully_verified": semantics["explanation_fully_verified"],
            "legacy_joint_fully_verified": semantics["joint_fully_verified"],
            **score_explicit_answer(prediction["generated"], source["answer_text"], semantics),
        })
    return rescored


def summarize(rows):
    """Use every example as denominator, including absent or malformed answers."""
    fields = (
        "explicit_answer_correct", "explanation_fully_verified",
        "explicit_answer_and_explanation_verified", "answer_present",
        "strict_exact_match", "legacy_joint_fully_verified",
    )
    if not rows:
        raise ValueError("Cannot score an empty split")
    counts = {field: sum(row[field] for row in rows) for field in fields}
    return {
        "examples": len(rows),
        "counts": counts,
        "rates": {key: count / len(rows) for key, count in counts.items()},
    }


def main():
    """Read historical semantic scores and write a separately versioned rescore."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("report/results/generated/phase3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test", "challenge_names", "challenge_wording"],
                        choices=("validation", "test", "challenge_names", "challenge_wording"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory; preserve existing rescores")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for language in ("hindi", "bhojpuri"):
        for protocol, limit in (("original", 16), ("explanation", 96)):
            for stage in ("pretrained", "finetuned"):
                for split in args.splits:
                    if stage == "pretrained":
                        directory = "semantic_original_v1" if protocol == "original" else "semantic_explanations_v1"
                        path = args.results_root / directory / f"{language}_{split}.jsonl.gz"
                    else:
                        path = args.results_root / "finetuned_v1" / language / f"protocol_{protocol}_{limit}tokens" / f"{split}_predictions.jsonl.gz"
                    data_path = args.dataset_root / language / f"{split}.jsonl.gz"
                    rows = rescore_rows(read_rows(path), read_rows(data_path))
                    key = f"{language}/{protocol}/{stage}/{split}"
                    out = args.output_dir / f"{language}_{protocol}_{stage}_{split}.jsonl.gz"
                    write_gzip(out, rows)
                    results[key] = {
                        **summarize(rows),
                        "per_task": {
                            family: summarize([row for row in rows if row["task_family"] == family])
                            for family in sorted({row["task_family"] for row in rows})
                        },
                        "input_predictions_sha256": file_hash(path),
                        "dataset_sha256": file_hash(data_path),
                        "scores_sha256": file_hash(out),
                    }
                    print(key, results[key]["rates"], flush=True)
    atomic_write_json(args.output_dir / "summary.json", {
        "version": SCORING_VERSION,
        "status": "Retrospective diagnostic; historical outputs informed earlier analysis. "
        "Extraction rule checked on validation only, then frozen before this held-out rescore. "
        "Existing explanation-verification scores reused without parser changes.",
        "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in (
            "explicit_answer.py", "rescore_explicit_answers.py",
        )},
        "results": results,
    })


if __name__ == "__main__":
    main()
