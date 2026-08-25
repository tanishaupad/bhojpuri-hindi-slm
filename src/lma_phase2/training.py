"""Optimizer, cosine schedule, evaluation, and complete checkpoint utilities."""

from __future__ import annotations

import csv
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from lma_phase1.io import atomic_write_json
from lma_phase2.config import ExperimentConfig, TrainingConfig
from lma_phase2.data import PackedTokenDataset
from lma_phase2.model import DecoderOnlyTransformer

LOG_COLUMNS = [
    "step",
    "split",
    "loss",
    "perplexity",
    "learning_rate",
    "tokens_seen",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def precision_context(device: torch.device, precision: str) -> tuple[Any, torch.dtype | None, bool]:
    """Return an autocast factory, its dtype, and whether loss scaling is needed."""

    if precision == "auto":
        if device.type == "cuda":
            precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            precision = "float32"
    if precision == "float32":
        return nullcontext, None, False
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16

    def context():
        return torch.autocast(device_type=device.type, dtype=dtype)

    return context, dtype, precision == "float16" and device.type == "cuda"


def configure_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Apply AdamW decay to matrix weights but not biases or normalization scales."""

    decay = []
    no_decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )


def cosine_multiplier(step: int, config: TrainingConfig) -> float:
    """Linear warmup followed by cosine decay to a non-zero LR floor."""

    if step < config.warmup_steps:
        return (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.maximum_steps - config.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    minimum_ratio = config.minimum_learning_rate / config.learning_rate
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def configure_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cosine_multiplier(step, config)
    )


@torch.no_grad()
def evaluate_loss(
    model: DecoderOnlyTransformer,
    dataset: PackedTokenDataset,
    batch_size: int,
    context_length: int,
    maximum_batches: int,
    device: torch.device,
    autocast_context,
) -> dict[str, float]:
    """Compute token-weighted held-out cross-entropy and perplexity."""

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in dataset.sequential_batches(
        batch_size, context_length, device, maximum_batches
    ):
        with autocast_context():
            loss = model(inputs, targets=targets).loss
        tokens = targets.numel()
        total_loss += float(loss) * tokens
        total_tokens += tokens
    if was_training:
        model.train()
    if not total_tokens:
        raise ValueError("Validation dataset yielded no tokens")
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 50.0))}


def capture_rng_state(data_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_generator": data_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], data_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    data_generator.set_state(state["data_generator"])


def save_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    step: int,
    tokens_seen: int,
    best_validation_loss: float | None,
    experiment: ExperimentConfig,
    data_generator: torch.Generator,
) -> None:
    """Atomically save every state needed for exact training continuation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": "1.0",
            "model_name": experiment.identity.model_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "training_step": step,
            "tokens_seen": tokens_seen,
            "best_validation_loss": best_validation_loss,
            "config": experiment.to_dict(),
            "rng_state": capture_rng_state(data_generator),
        },
        temporary,
    )
    temporary.replace(path)
    atomic_write_json(
        path.parent / "latest.json",
        {"checkpoint": path.name, "training_step": step, "tokens_seen": tokens_seen},
    )


def load_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    experiment: ExperimentConfig,
    data_generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    """Restore model, optimizer, scheduler, scaler, counters, config, and RNGs."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != "1.0":
        raise ValueError("Unsupported checkpoint format")
    if checkpoint["model_name"] != experiment.identity.model_name:
        raise ValueError("Checkpoint belongs to a different model")
    if checkpoint["config"] != experiment.to_dict():
        raise ValueError("Checkpoint configuration does not match the requested experiment")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint["scaler_state"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    restore_rng_state(checkpoint["rng_state"], data_generator)
    return checkpoint


def resolve_latest_checkpoint(directory: Path) -> Path:
    with (directory / "latest.json").open(encoding="utf-8") as handle:
        value = json.load(handle)
    return directory / value["checkpoint"]


def append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
