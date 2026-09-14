from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from lma_phase2.train import train


def test_training_cli_core_runs_and_resumes_on_cpu(tmp_path) -> None:
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(
        """
[identity]
model_name = "tiny_smoke"
language = "bho"
language_name = "Bhojpuri"
tokenizer = "unused.model"

[model]
vocabulary_size = 50
context_length = 8
model_dimension = 16
number_of_layers = 1
number_of_heads = 4
feed_forward_dimension = 32
dropout = 0.0
attention_dropout = 0.0
embedding_dropout = 0.0
position_encoding = "learned_absolute"
normalization = "pre_layer_norm"
activation = "gelu"
tie_embeddings = true
use_bias = true

[training]
seed = 11
micro_batch_size = 2
gradient_accumulation_steps = 1
maximum_steps = 2
learning_rate = 0.001
minimum_learning_rate = 0.0001
warmup_steps = 1
weight_decay = 0.1
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
gradient_clip_norm = 1.0
validation_interval = 1
validation_batches = 1
checkpoint_interval = 1
log_interval = 1
precision = "float32"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    manifests = []
    for split in ("train", "validation"):
        token_path = tmp_path / f"{split}.tokens.bin"
        np.tile(np.arange(50, dtype="<u2"), 4).tofile(token_path)
        manifest_path = tmp_path / f"{split}.manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "language": "bho",
                    "vocabulary_size": 50,
                    "token_file": token_path.name,
                    "dtype": "<u2",
                    "token_count": 200,
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest_path)
    output_dir = tmp_path / "run"
    args = argparse.Namespace(
        config=str(config_path),
        train_manifest=str(manifests[0]),
        validation_manifest=str(manifests[1]),
        output_dir=str(output_dir),
        device="cpu",
        resume=None,
    )

    train(args)

    assert (output_dir / "checkpoint_step_0000002.pt").exists()
    assert json.loads((output_dir / "latest.json").read_text())["training_step"] == 2
    with (output_dir / "training_log.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["split"] for row in rows] == [
        "train",
        "validation",
        "train",
        "validation",
    ]

    args.resume = "latest"
    train(args)
