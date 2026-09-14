#!/usr/bin/env python3
"""Run Nepali Phase 3 training, evaluation, and sample-size ablations on CUDA."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("tmp/phase3/reasoning_v1"))
    parser.add_argument("--work-root", type=Path, default=Path("/kaggle/working/phase3_nepali"))
    parser.add_argument(
        "--mode", choices=("train", "evaluate", "ablate", "all"), default="all"
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "mps", "auto"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    dataset_root = (root / args.dataset_root).resolve() if not args.dataset_root.is_absolute() else args.dataset_root
    work_root = args.work_root.resolve()
    pretrain = (
        args.artifact_root.resolve()
        / "nepali/phase2/runs/pretrain_v1/checkpoint_step_0016720.pt"
    )
    test_manifest = (
        args.artifact_root.resolve() / "nepali/phase2/tokenized_v1/test.manifest.json"
    )
    run_dir = work_root / "runs/nepali/finetune_v1"
    best = run_dir / "best_checkpoint.pt"
    python = sys.executable

    if args.mode in ("train", "all"):
        command = [
            python,
            "-m",
            "lma_phase3.finetune",
            "--config",
            "nepali/configs/phase3.toml",
            "--pretrained-checkpoint",
            str(pretrain),
            "--train-data",
            str(dataset_root / "nepali/train.jsonl.gz"),
            "--val-data",
            str(dataset_root / "nepali/validation.jsonl.gz"),
            "--template-config",
            "nepali/configs/phase3_templates.json",
            "--output-dir",
            str(run_dir),
            "--device",
            args.device,
        ]
        if args.resume:
            command.append("--resume")
        run(command, root)

    if args.mode in ("evaluate", "all"):
        run(
            [
                python,
                "-m",
                "lma_phase3.evaluate_finetuned",
                "--config",
                "nepali/configs/phase3.toml",
                "--checkpoint",
                str(best),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(work_root / "evaluation/nepali"),
                "--device",
                args.device,
                "--protocols",
                "both",
            ],
            root,
        )

    if args.mode in ("ablate", "all"):
        run(
            [
                python,
                "scripts/ablation_samples_vs_ppl.py",
                "--languages",
                "nepali",
                "--pretrained-checkpoint",
                str(pretrain),
                "--main-checkpoint",
                str(best),
                "--test-manifest",
                str(test_manifest),
                "--dataset-root",
                str(dataset_root),
                "--work-dir",
                str(work_root / "ablation"),
                "--output-dir",
                str(work_root / "ablation/results"),
                "--device",
                args.device,
            ],
            root,
        )


if __name__ == "__main__":
    main()
