"""Build the self-contained Hindi no-position Kaggle upload archive."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "external_artifacts/phase2/hindi"
OUTPUT = ROOT / "output/kaggle/bonus_hindi_no_position_kaggle.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_tree(archive: zipfile.ZipFile, source: Path, destination: str) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            archive.write(path, f"lma_bonus_bundle/{destination}/{path.relative_to(source)}")


def main() -> None:
    required = [
        "checkpoint_step_0020300.pt",
        "train.tokens.bin",
        "train.manifest.json",
        "validation.tokens.bin",
        "validation.manifest.json",
        "test.tokens.bin",
        "test.manifest.json",
        "test.jsonl.gz",
    ]
    missing = [name for name in required if not (ARTIFACTS / name).is_file()]
    if missing:
        raise SystemExit(f"Missing artifacts: {', '.join(missing)}")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "bundle_name": "Hindi no-position Phase 2 bonus",
        "source_commit": source_commit,
        "files": {
            name: {"bytes": (ARTIFACTS / name).stat().st_size, "sha256": sha256(ARTIFACTS / name)}
            for name in required
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("lma_bonus_bundle/bonus_bundle_manifest.json", json.dumps(manifest, indent=2))
        add_tree(archive, ROOT / "src", "repo/src")
        add_tree(archive, ROOT / "hindi/configs", "repo/hindi/configs")
        archive.write(ROOT / "hindi/tokenizer/hindi.model", "lma_bonus_bundle/repo/hindi/tokenizer/hindi.model")
        archive.write(ROOT / "pyproject.toml", "lma_bonus_bundle/repo/pyproject.toml")
        archive.write(
            ROOT / "kaggle/run_bonus_hindi_no_position.py",
            "lma_bonus_bundle/repo/kaggle/run_bonus_hindi_no_position.py",
        )
        for name in required:
            destination = "baseline" if name.endswith(".pt") else "data/hindi"
            archive.write(ARTIFACTS / name, f"lma_bonus_bundle/{destination}/{name}")
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
