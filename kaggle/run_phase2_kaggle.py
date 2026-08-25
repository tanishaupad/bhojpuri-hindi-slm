"""Train, resume, evaluate, and package Phase 2 runs on Kaggle.

This runner expects the self-contained bundle produced for this repository. It never writes to
Kaggle's read-only input directory; checkpoints and reports go under /kaggle/working by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

LANGUAGES = {
    "hindi": {
        "config": "hindi/configs/phase2.toml",
        "tokenizer": "hindi/tokenizer/hindi.model",
        "label": "Hindi",
    },
    "bhojpuri": {
        "config": "bhojpuri/configs/phase2.toml",
        "tokenizer": "bhojpuri/tokenizer/bhojpuri.model",
        "label": "Bhojpuri",
    },
}


def find_bundle_root(explicit: str | None) -> Path:
    """Locate an extracted bundle, or extract the uploaded zip into Kaggle working storage."""

    if explicit:
        root = Path(explicit).resolve()
        if not (root / "lma_phase2_bundle_manifest.json").exists():
            raise SystemExit(f"Bundle manifest not found under {root}")
        return root

    search_roots = [Path("/kaggle/input"), Path.cwd()]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        manifests = sorted(search_root.rglob("lma_phase2_bundle_manifest.json"))
        if manifests:
            return manifests[0].parent

    archives: list[Path] = []
    for search_root in search_roots:
        if search_root.exists():
            archives.extend(search_root.rglob("lma_phase2_kaggle_upload.zip"))
    if not archives:
        raise SystemExit(
            "Could not find lma_phase2_bundle_manifest.json or "
            "lma_phase2_kaggle_upload.zip under /kaggle/input"
        )
    extraction_root = Path("/kaggle/working/lma_phase2_bundle")
    extraction_root.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(min(archives)), extraction_root)
    manifests = sorted(extraction_root.rglob("lma_phase2_bundle_manifest.json"))
    if not manifests:
        raise SystemExit("Uploaded archive did not contain the expected bundle manifest")
    return manifests[0].parent


def validate_bundle(root: Path, languages: list[str]) -> None:
    """Check required code and data files before consuming GPU time."""

    repo = root / "repo"
    for language in languages:
        values = LANGUAGES[language]
        required = [
            repo / values["config"],
            repo / values["tokenizer"],
            root / "data" / language / "train.tokens.bin",
            root / "data" / language / "train.manifest.json",
            root / "data" / language / "validation.tokens.bin",
            root / "data" / language / "validation.manifest.json",
            root / "data" / language / "test.tokens.bin",
            root / "data" / language / "test.manifest.json",
            root / "data" / language / "validation.jsonl.gz",
            root / "data" / language / "test.jsonl.gz",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit("Bundle is incomplete:\n" + "\n".join(missing))
        for split in ("train", "validation", "test"):
            manifest_path = root / "data" / language / f"{split}.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            token_path = manifest_path.parent / manifest["token_file"]
            expected_bytes = int(manifest["token_count"]) * 2
            if token_path.stat().st_size != expected_bytes:
                raise SystemExit(
                    f"Packed data size mismatch for {language}/{split}: "
                    f"expected {expected_bytes}, got {token_path.stat().st_size}"
                )


def command_environment(repo: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repo / "src") + (os.pathsep + existing if existing else "")
    return environment


def restore_previous_run(language: str, output_dir: Path, resume_root: Path | None) -> None:
    """Copy a prior Kaggle output into writable storage so logging can continue."""

    if output_dir.exists() or resume_root is None:
        return
    candidates = [
        resume_root / language,
        resume_root / "phase2_runs" / language,
        resume_root / "runs" / language,
    ]
    source = next((path for path in candidates if (path / "latest.json").exists()), None)
    if source is not None:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, output_dir)
        print(f"Restored prior {language} run from {source}", flush=True)


def run_training(
    root: Path,
    language: str,
    output_root: Path,
    resume_root: Path | None,
    device: str,
) -> Path:
    repo = root / "repo"
    data = root / "data" / language
    output_dir = output_root / language
    restore_previous_run(language, output_dir, resume_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "lma_phase2.train",
        "--config",
        str(repo / LANGUAGES[language]["config"]),
        "--train-manifest",
        str(data / "train.manifest.json"),
        "--validation-manifest",
        str(data / "validation.manifest.json"),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
    ]
    if (output_dir / "latest.json").exists():
        command.extend(["--resume", "latest"])
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=repo, env=command_environment(repo))
    return output_dir


def latest_checkpoint(run_dir: Path) -> Path:
    latest = json.loads((run_dir / "latest.json").read_text(encoding="utf-8"))
    return run_dir / latest["checkpoint"]


def run_evaluation(root: Path, language: str, output_root: Path, report_root: Path) -> None:
    repo = root / "repo"
    data = root / "data" / language
    run_dir = output_root / language
    if not (run_dir / "latest.json").exists():
        raise SystemExit(f"No completed checkpoint is available for {language}")
    report_root.mkdir(parents=True, exist_ok=True)
    figure_dir = report_root / "attention" / language
    command = [
        sys.executable,
        "-m",
        "lma_phase2.evaluate",
        "--config",
        str(repo / LANGUAGES[language]["config"]),
        "--checkpoint",
        str(latest_checkpoint(run_dir)),
        "--test-manifest",
        str(data / "test.manifest.json"),
        "--test-jsonl",
        str(data / "test.jsonl.gz"),
        "--tokenizer-model",
        str(repo / LANGUAGES[language]["tokenizer"]),
        "--output",
        str(report_root / f"{language}_evaluation.json"),
        "--samples-output",
        str(report_root / f"{language}_samples.jsonl"),
        "--attention-csv",
        str(report_root / f"{language}_attention_summary.csv"),
        "--figure-dir",
        str(figure_dir),
        "--device",
        "cuda",
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=repo, env=command_environment(repo))


def plot_training(root: Path, languages: list[str], output_root: Path, report_root: Path) -> None:
    repo = root / "repo"
    logs = [
        (LANGUAGES[language]["label"], output_root / language / "training_log.csv")
        for language in languages
        if (output_root / language / "training_log.csv").exists()
    ]
    if not logs:
        return
    report_root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "lma_phase2.plot_training"]
    for label, path in logs:
        command.extend(["--log", f"{label}={path}"])
    command.extend(["--output", str(report_root / "pretraining_curves.png")])
    subprocess.run(command, check=True, cwd=repo, env=command_environment(repo))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root")
    parser.add_argument("--language", choices=["hindi", "bhojpuri", "both"], default="both")
    parser.add_argument("--mode", choices=["train", "evaluate", "all"], default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="/kaggle/working/phase2_runs")
    parser.add_argument("--report-root", default="/kaggle/working/phase2_report")
    parser.add_argument(
        "--resume-root",
        help="Optional attached prior Kaggle output containing phase2_runs/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_bundle_root(args.bundle_root)
    languages = list(LANGUAGES) if args.language == "both" else [args.language]
    validate_bundle(root, languages)
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    resume_root = Path(args.resume_root) if args.resume_root else None
    print(f"Bundle root: {root}", flush=True)
    print(f"Languages: {', '.join(languages)}", flush=True)
    if args.mode in {"train", "all"}:
        for language in languages:
            run_training(root, language, output_root, resume_root, args.device)
        plot_training(root, languages, output_root, report_root)
    if args.mode in {"evaluate", "all"}:
        for language in languages:
            run_evaluation(root, language, output_root, report_root)


if __name__ == "__main__":
    main()
