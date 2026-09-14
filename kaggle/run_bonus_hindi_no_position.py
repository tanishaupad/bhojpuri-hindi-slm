"""Train, resume, and evaluate the Hindi no-position bonus on Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], repo: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=repo, env=environment)


def evaluate(
    repo: Path,
    config: Path,
    checkpoint: Path,
    data: Path,
    output: Path,
    name: str,
    device: str,
) -> None:
    run(
        [
            sys.executable,
            "-m",
            "lma_phase2.evaluate",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--test-manifest",
            str(data / "test.manifest.json"),
            "--test-jsonl",
            str(data / "test.jsonl.gz"),
            "--tokenizer-model",
            str(repo / "hindi/tokenizer/hindi.model"),
            "--output",
            str(output / f"{name}_evaluation.json"),
            "--samples-output",
            str(output / f"{name}_samples.jsonl"),
            "--attention-csv",
            str(output / f"{name}_attention.csv"),
            "--figure-dir",
            str(output / "attention" / name),
            "--device",
            device,
        ],
        repo,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "evaluate", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--work-root", type=Path, default=Path("/kaggle/working/hindi_no_position"))
    args = parser.parse_args()

    root = args.bundle_root.resolve()
    repo = root / "repo"
    data = root / "data/hindi"
    baseline = root / "baseline/checkpoint_step_0020300.pt"
    run_dir = args.work_root / "run"
    report_dir = args.work_root / "report"
    ablation_config = repo / "hindi/configs/phase2_no_position.toml"

    manifest = json.loads((root / "bonus_bundle_manifest.json").read_text())
    print(f"Bundle: {manifest['bundle_name']} ({manifest['source_commit']})", flush=True)
    if args.resume_root and not run_dir.exists():
        source = args.resume_root / "run"
        if (source / "latest.json").exists():
            shutil.copytree(source, run_dir)

    if args.mode in {"train", "all"}:
        command = [
            sys.executable,
            "-m",
            "lma_phase2.train",
            "--config",
            str(ablation_config),
            "--train-manifest",
            str(data / "train.manifest.json"),
            "--validation-manifest",
            str(data / "validation.manifest.json"),
            "--output-dir",
            str(run_dir),
            "--device",
            args.device,
        ]
        if (run_dir / "latest.json").exists():
            command.extend(["--resume", "latest"])
        run(command, repo)

    if args.mode in {"evaluate", "all"}:
        latest = json.loads((run_dir / "latest.json").read_text())
        ablated_checkpoint = run_dir / latest["checkpoint"]
        report_dir.mkdir(parents=True, exist_ok=True)
        evaluate(
            repo,
            repo / "hindi/configs/phase2.toml",
            baseline,
            data,
            report_dir,
            "baseline",
            args.device,
        )
        evaluate(
            repo,
            ablation_config,
            ablated_checkpoint,
            data,
            report_dir,
            "no_position",
            args.device,
        )
        run(
            [
                sys.executable,
                "-m",
                "lma_phase2.plot_training",
                "--log",
                f"Hindi NoPE={run_dir / 'training_log.csv'}",
                "--output",
                str(report_dir / "training_curve.png"),
            ],
            repo,
        )


if __name__ == "__main__":
    main()
