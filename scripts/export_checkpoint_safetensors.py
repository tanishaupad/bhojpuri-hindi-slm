"""Export a trusted resume checkpoint as non-executable SafeTensors model weights."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model

from lma_phase1.io import atomic_write_json
from lma_phase2.config import load_experiment_config
from lma_phase2.model import DecoderOnlyTransformer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.suffix != ".safetensors":
        raise SystemExit("Output filename must end in .safetensors")
    experiment = load_experiment_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["config"] != experiment.to_dict():
        raise ValueError("Checkpoint and export config differ")

    model = DecoderOnlyTransformer(experiment.model)
    model.load_state_dict(checkpoint["model_state"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.output, metadata={"format": "pt"})

    restored = DecoderOnlyTransformer(experiment.model)
    load_safetensors_model(restored, args.output, strict=True, device="cpu")
    for name, expected in model.state_dict().items():
        if not torch.equal(expected, restored.state_dict()[name]):
            raise RuntimeError(f"SafeTensors verification failed for {name}")

    metadata_path = args.output.with_suffix(".metadata.json")
    atomic_write_json(
        metadata_path,
        {
            "schema_version": "1.0",
            "format": "safetensors model weights; non-executable",
            "model_name": experiment.identity.model_name,
            "training_step": int(checkpoint["training_step"]),
            "tokens_seen": int(checkpoint["tokens_seen"]),
            "config": experiment.to_dict(),
            "source_checkpoint": args.checkpoint.name,
            "source_checkpoint_sha256": sha256(args.checkpoint),
            "safetensors_sha256": sha256(args.output),
        },
    )
    print(f"Created {args.output} and {metadata_path}")


if __name__ == "__main__":
    main()
