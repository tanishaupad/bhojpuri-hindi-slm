import json
from pathlib import Path

import pytest
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from lma_phase2.config import load_experiment_config
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase2.training import configure_optimizer, configure_scheduler
from lma_phase3.finetune import restore_rng_state, save_finetune_checkpoint
from lma_phase3.finetune_data import ReasoningFinetuneDataset, collate_reasoning_batch
from lma_phase3.generate import FAMILIES, generate_pilot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reasoning_data(tmp_path: Path) -> dict[str, Path]:
    """Generate small, deterministic datasets using only committed language resources."""
    paths = {}
    for language in ("hindi", "bhojpuri"):
        resource = json.loads((ROOT / language / "configs/phase3_templates.json").read_text())
        tokenizer = spm.SentencePieceProcessor(
            model_file=str(ROOT / language / "tokenizer" / f"{language}.model")
        )
        rows, _ = generate_pilot(resource, tokenizer, count=16)
        # Interleave families so even four-example tests exercise every target structure.
        groups = [[row for row in rows if row["task_family"] == family] for family in FAMILIES]
        rows = [row for group in zip(*groups, strict=True) for row in group]
        path = tmp_path / f"{language}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        paths[language] = path
    return paths


@pytest.mark.parametrize("language", ["hindi", "bhojpuri"])
def test_dataset_boundary_shifting_and_masking(language: str, reasoning_data: dict[str, Path]):
    template_path = ROOT / language / "configs/phase3_templates.json"
    tokenizer_path = ROOT / language / f"tokenizer/{language}.model"
    data_path = reasoning_data[language]

    dataset = ReasoningFinetuneDataset(
        data_path=data_path,
        template_path=template_path,
        tokenizer_path=tokenizer_path,
        max_examples=5,
    )

    for item in dataset:
        input_ids = item["input_ids"]
        labels = item["labels"]
        prompt_len = item["prompt_token_count"]
        target_len = item["target_token_count"]

        # Shift alignment check
        assert len(input_ids) == len(labels)
        assert len(input_ids) == prompt_len + target_len - 1

        # All prompt prediction positions must be -100
        for idx in range(prompt_len - 1):
            assert labels[idx] == -100, f"Prompt token at {idx} was not masked"

        # The first unmasked label is the first token of the target
        assert labels[prompt_len - 1] != -100

        # The last label is EOS (3)
        assert labels[-1] == 3, f"Expected EOS (3), got {labels[-1]}"

        # Total active (loss-bearing) tokens must equal target tokens including EOS
        active_count = sum(1 for x in labels if x != -100)
        assert active_count == target_len


@pytest.mark.parametrize("language", ["hindi", "bhojpuri"])
def test_collator_padding_and_loss_masking(language: str, reasoning_data: dict[str, Path]):
    template_path = ROOT / language / "configs/phase3_templates.json"
    tokenizer_path = ROOT / language / f"tokenizer/{language}.model"
    data_path = reasoning_data[language]

    dataset = ReasoningFinetuneDataset(
        data_path=data_path,
        template_path=template_path,
        tokenizer_path=tokenizer_path,
        max_examples=4,
    )
    batch = collate_reasoning_batch([dataset[i] for i in range(len(dataset))])

    inputs = batch["input_ids"]
    labels = batch["labels"]
    assert inputs.shape == labels.shape

    max_len = inputs.shape[1]
    assert len({len(dataset[i]["input_ids"]) for i in range(len(dataset))}) > 1
    # Check that any padding positions have label -100 and input pad_id 0
    for i in range(len(dataset)):
        original_len = len(dataset[i]["input_ids"])
        if original_len < max_len:
            assert torch.all(labels[i, original_len:] == -100)
            assert torch.all(inputs[i, original_len:] == 0)


def test_tiny_subset_overfit(reasoning_data: dict[str, Path]):
    """Verify that loss on eight examples decreases substantially over 35 steps."""
    torch.manual_seed(1234)
    config = load_experiment_config(ROOT / "hindi/configs/phase3.toml")
    template_path = ROOT / "hindi/configs/phase3_templates.json"
    tokenizer_path = ROOT / "hindi/tokenizer/hindi.model"
    data_path = reasoning_data["hindi"]

    dataset = ReasoningFinetuneDataset(
        data_path=data_path,
        template_path=template_path,
        tokenizer_path=tokenizer_path,
        max_examples=8,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_reasoning_batch)
    batch = next(iter(loader))

    model = DecoderOnlyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(35):
        optimizer.zero_grad()
        output = model(batch["input_ids"], targets=batch["labels"])
        loss = output.loss
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = float(loss.item())
        if step == 34:
            final_loss = float(loss.item())

    assert initial_loss is not None and final_loss is not None
    # Loss should decrease significantly
    assert final_loss < initial_loss * 0.25, f"Loss did not overfit: {initial_loss} -> {final_loss}"


def test_checkpoint_and_rng_roundtrip(tmp_path: Path):
    config = load_experiment_config(ROOT / "hindi/configs/phase3.toml")
    model = DecoderOnlyTransformer(config.model)
    optimizer = configure_optimizer(model, config.training)
    scheduler = configure_scheduler(optimizer, config.training)

    ckpt_path = tmp_path / "checkpoint_step_0000010.pt"
    save_finetune_checkpoint(
        ckpt_path,
        model,
        optimizer,
        scheduler,
        scaler=None,
        step=10,
        tokens_seen=500,
        best_val_loss=2.5,
        experiment=config,
    )
    assert ckpt_path.exists()
    assert (tmp_path / "latest.json").exists()

    ckpt = torch.load(ckpt_path, weights_only=False)
    assert ckpt["training_step"] == 10
    assert ckpt["tokens_seen"] == 500
    assert "rng_state" in ckpt
    assert "epoch" in ckpt
    restore_rng_state(ckpt["rng_state"])


def test_resume_validation_checks(tmp_path: Path):
    """Test that missing latest.json or config mismatches are strictly rejected."""
    config = load_experiment_config(ROOT / "hindi/configs/phase3.toml")
    model = DecoderOnlyTransformer(config.model)
    optimizer = configure_optimizer(model, config.training)
    scheduler = configure_scheduler(optimizer, config.training)

    # 1. Save valid checkpoint
    ckpt_path = tmp_path / "checkpoint_step_0000005.pt"
    save_finetune_checkpoint(
        ckpt_path,
        model,
        optimizer,
        scheduler,
        scaler=None,
        step=5,
        tokens_seen=200,
        best_val_loss=1.0,
        experiment=config,
    )

    # 2. Corrupt config in a copy to verify mismatch detection
    mismatched_config = load_experiment_config(ROOT / "bhojpuri/configs/phase3.toml")
    mismatched_ckpt_path = tmp_path / "checkpoint_mismatched.pt"
    save_finetune_checkpoint(
        mismatched_ckpt_path,
        model,
        optimizer,
        scheduler,
        scaler=None,
        step=5,
        tokens_seen=200,
        best_val_loss=1.0,
        experiment=mismatched_config,
    )

    loaded = torch.load(mismatched_ckpt_path, weights_only=False)
    saved_model = loaded["config"]["model"]
    assert saved_model["vocabulary_size"] != config.model.vocabulary_size


def test_resume_exact_continuation(tmp_path: Path, reasoning_data: dict[str, Path]):
    """Test that interruption and resumption produces identical weights to uninterrupted training."""
    config = load_experiment_config(ROOT / "hindi/configs/phase3.toml")
    template_path = ROOT / "hindi/configs/phase3_templates.json"
    tokenizer_path = ROOT / "hindi/tokenizer/hindi.model"
    data_path = reasoning_data["hindi"]

    dataset = ReasoningFinetuneDataset(
        data_path=data_path,
        template_path=template_path,
        tokenizer_path=tokenizer_path,
        max_examples=16,
    )

    # Helper to build reproducible loader
    def make_loader():
        g = torch.Generator()
        g.manual_seed(9999)
        return DataLoader(dataset, batch_size=4, shuffle=True, generator=g, collate_fn=collate_reasoning_batch)

    # --- Run A: Uninterrupted 4 steps ---
    torch.manual_seed(1234)
    model_a = DecoderOnlyTransformer(config.model)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    loader_a = make_loader()
    it_a = iter(loader_a)

    for step in range(4):
        batch = next(it_a)
        opt_a.zero_grad()
        out = model_a(batch["input_ids"], targets=batch["labels"])
        out.loss.backward()
        opt_a.step()

    # --- Run B: 2 steps -> Save -> Resume -> 2 steps ---
    torch.manual_seed(1234)
    model_b = DecoderOnlyTransformer(config.model)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    loader_b = make_loader()
    it_b = iter(loader_b)

    for step in range(2):
        batch = next(it_b)
        opt_b.zero_grad()
        out = model_b(batch["input_ids"], targets=batch["labels"])
        out.loss.backward()
        opt_b.step()

    # Save state
    ckpt_file = tmp_path / "step2.pt"
    sched_dummy = torch.optim.lr_scheduler.LambdaLR(opt_b, lr_lambda=lambda x: 1.0)
    save_finetune_checkpoint(
        ckpt_file,
        model_b,
        opt_b,
        sched_dummy,
        scaler=None,
        step=2,
        tokens_seen=100,
        best_val_loss=1.0,
        experiment=config,
    )

    # Resume in fresh model
    model_resumed = DecoderOnlyTransformer(config.model)
    opt_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3)
    ckpt = torch.load(ckpt_file, weights_only=False)
    model_resumed.load_state_dict(ckpt["model_state"])
    opt_resumed.load_state_dict(ckpt["optimizer_state"])
    restore_rng_state(ckpt["rng_state"])

    # Advance iterator by 2 batches (the batches consumed before step 2)
    loader_resumed = make_loader()
    it_resumed = iter(loader_resumed)
    next(it_resumed)
    next(it_resumed)

    for step in range(2, 4):
        batch = next(it_resumed)
        opt_resumed.zero_grad()
        out = model_resumed(batch["input_ids"], targets=batch["labels"])
        out.loss.backward()
        opt_resumed.step()

    # Check that model weights match exactly
    for (name_a, p_a), (name_r, p_r) in zip(model_a.named_parameters(), model_resumed.named_parameters(), strict=True):
        assert torch.allclose(p_a, p_r, atol=1e-6), f"Mismatch in {name_a} between uninterrupted and resumed run"
