"""Check that auxiliary training recovers real optimizer progress and sample order."""

import gzip
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from lma_phase2.config import load_experiment_config
from lma_phase2.model import DecoderOnlyTransformer

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ablation", ROOT / "scripts/ablation_samples_vs_ppl.py"
)
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def test_subset_is_independent_of_requested_sweep_order(tmp_path):
    """Requesting one budget must reproduce that budget's historical full-sweep sample."""
    source = tmp_path / "train.gz"
    with gzip.open(source, "wt") as handle:
        for family in ablation.FAMILIES:
            for i in range(3000):
                handle.write(json.dumps({"id": f"{family}-{i}", "task_family": family}) + "\n")
    all_paths = ablation.create_stratified_subsets(source, tmp_path / "all", ablation.SWEEP_SIZES)
    one = ablation.create_stratified_subsets(source, tmp_path / "one", [5000])[5000]
    with gzip.open(one, "rt") as a, gzip.open(all_paths[5000], "rt") as b:
        assert a.read() == b.read()
    with gzip.open(one, "wt") as handle:
        handle.write("changed\n")
    with pytest.raises(ValueError, match="differs"):
        ablation.create_stratified_subsets(source, tmp_path / "one", [5000])


def test_ablation_resume_matches_uninterrupted_with_dropout(tmp_path, monkeypatch):
    """Crash after durable step 25, then compare final parameters with 40 uninterrupted steps."""
    experiment = load_experiment_config(ROOT / "hindi/configs/phase3.toml")
    experiment = replace(
        experiment,
        model=replace(
            experiment.model,
            vocabulary_size=32,
            model_dimension=8,
            number_of_heads=2,
            number_of_layers=1,
            feed_forward_dimension=16,
            context_length=8,
        ),
    )
    pretrained = tmp_path / "pretrained.pt"
    torch.save({"model_state": DecoderOnlyTransformer(experiment.model).state_dict()}, pretrained)
    data = tmp_path / "data.json"
    data.write_text("fixture")
    examples = [
        {
            "input_ids": torch.tensor([4 + i, 5 + i, 6 + i, 7 + i]),
            "labels": torch.tensor([5 + i, 6 + i, 7 + i, 8 + i]),
        }
        for i in range(16)
    ]
    monkeypatch.setattr(ablation, "load_experiment_config", lambda _: experiment)
    monkeypatch.setattr(ablation, "ReasoningFinetuneDataset", lambda **_: examples)
    monkeypatch.setattr(
        ablation,
        "collate_reasoning_batch",
        lambda batch: {
            key: torch.stack([row[key] for row in batch]) for key in ("input_ids", "labels")
        },
    )
    args = {
        "config_path": data, "pretrained_ckpt_path": pretrained, "train_data_path": data,
        "val_data_path": data, "template_config_path": data, "num_samples": 320,
        "device_name": "cpu",
    }
    reference = ablation.train_subset(output_dir=tmp_path / "continuous", **args)
    original_replace = Path.replace

    def crash_after_save(path, target):
        result = original_replace(path, target)
        if Path(target).name == "checkpoint_resume.pt":
            raise RuntimeError("simulated interruption")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", crash_after_save)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            ablation.train_subset(output_dir=tmp_path / "interrupted", **args)
    # A crash may leave log rows newer than the last durable optimizer state.
    with (tmp_path / "interrupted/training.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": 28, "loss": -99}) + "\n")
    resumed = ablation.train_subset(output_dir=tmp_path / "interrupted", **args)
    a = torch.load(reference, map_location="cpu", weights_only=False)
    b = torch.load(resumed, map_location="cpu", weights_only=False)
    assert a["training_step"] == b["training_step"] == 40
    assert a["scheduler_state"] == b["scheduler_state"]
    for name, value in a["model_state"].items():
        torch.testing.assert_close(value, b["model_state"][name], rtol=0, atol=0)
    assert a["config"]["training"]["maximum_steps"] == 40
    assert a["config"]["training"]["warmup_steps"] == 10
    with (tmp_path / "interrupted/training.jsonl").open() as handle:
        logs = [json.loads(line) for line in handle]
    assert sum(row["step"] == 28 for row in logs) == 0
    assert [row["step"] for row in logs] == [10, 20, 30, 40]
    assert all(row["loss"] >= 0 for row in logs)
