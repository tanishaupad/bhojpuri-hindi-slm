"""Inventory historical auxiliary artifacts without changing weights or predictions."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from lma_phase1.io import atomic_write_json
from lma_phase2.config import load_experiment_config


def file_hash(path: Path) -> str:
    """Hash an artifact incrementally."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    """Record actual schedules separately from legacy base-config fields."""
    root = Path(__file__).resolve().parents[1]
    result = {
        "inventory_date": "2026-09-11",
        "provenance": "Retrospective local inventory; not original training-time attestation.",
        "historical_script_commit": "912c20a",
        "accuracy_protocol": "First 100 validation rows, all direct comparisons; original prompt, "
        "greedy generation, 16-token limit, explicit_answer_v1. Loss uses all 2000 validation rows.",
        "selection": "Exploratory, after consulting generic test PPL; no optimality claim.",
        "runs": [],
        "predictions": {},
    }
    for language in ("hindi", "bhojpuri"):
        base = load_experiment_config(root / language / "configs/phase3.toml")
        for size in (1000, 3000, 5000, 8000, 12000):
            checkpoint = (
                root / f"tmp/phase3/ablation_runs/{language}/run_{size}/checkpoint_final.pt"
            )
            data = root / f"tmp/phase3/ablation_data/{language}/train_{size}.jsonl.gz"
            with gzip.open(data, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            steps = size // 8
            assert len(rows) == len({row["id"] for row in rows}) == size
            assert payload["training_step"] == steps
            assert payload["samples"] == size
            run = {
                "language": language,
                "samples": size,
                "optimizer_steps": steps,
                "warmup_steps": max(10, min(100, round(steps * 0.1))),
                "epochs": 2,
                "effective_batch_size": 16,
                "learning_rate": base.training.learning_rate,
                "minimum_learning_rate": base.training.minimum_learning_rate,
                "checkpoint_sha256": file_hash(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_fields": sorted(payload),
                "historical_checkpoint_resume_capable": False,
                "original_per_step_logs": "not retained",
                "embedded_config_warning": "Base config retains maximum_steps=2500 and warmup=100; "
                "use optimizer_steps and warmup_steps in this record for the actual schedule.",
                "subset_sha256": file_hash(data),
                "ordered_example_ids_sha256": hashlib.sha256(
                    "\n".join(row["id"] for row in rows).encode()
                ).hexdigest(),
                "task_counts": dict(Counter(row["task_family"] for row in rows)),
                "tokenizer_sha256": file_hash(root / base.identity.tokenizer),
                "templates_sha256": file_hash(root / language / "configs/phase3_templates.json"),
                "training_release_sha256": file_hash(
                    root / f"tmp/phase3/reasoning_v1/{language}/train.jsonl.gz"
                ),
            }
            if (language, size) in (("hindi", 5000), ("bhojpuri", 3000)):
                run["local_checkpoint"] = str(checkpoint.relative_to(root))
                run["drive_upload_verified"] = False
            result["runs"].append(run)
            del payload
    for path in sorted(
        (root / "report/results/generated/phase3/optimal_eval").glob(
            "*/protocol*/*predictions.jsonl.gz"
        )
    ):
        result["predictions"][str(path.relative_to(root))] = file_hash(path)
    atomic_write_json(
        root / "report/results/generated/phase3/ablation_artifact_manifest.json", result
    )


if __name__ == "__main__":
    main()
