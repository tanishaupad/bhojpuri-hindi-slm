"""Phase 3 reasoning finetuning engine with exact shifting, loss masking, and safe resume."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from lma_phase1.io import atomic_write_json
from lma_phase2.config import ExperimentConfig, load_experiment_config
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase2.training import (
    configure_optimizer,
    configure_scheduler,
    precision_context,
    select_device,
)
from lma_phase3.finetune_data import ReasoningFinetuneDataset, collate_reasoning_batch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "torch_mps": torch.mps.get_rng_state()
        if (hasattr(torch, "mps") and torch.backends.mps.is_available())
        else None,
    }
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        cuda_state = state["torch_cuda"]
        if isinstance(cuda_state, list):
            cuda_state = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_state]
        torch.cuda.set_rng_state_all(cuda_state)
    if (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and state.get("torch_mps") is not None
    ):
        torch.mps.set_rng_state(state["torch_mps"].cpu())


def save_finetune_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    step: int,
    tokens_seen: int,
    best_val_loss: float | None,
    experiment: ExperimentConfig,
    epoch: int = 0,
    keep_last: int = 3,
    validation_protocol: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format_version": "1.0",
        "phase": "phase3_finetune",
        "model_name": experiment.identity.model_name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "training_step": step,
        "tokens_seen": tokens_seen,
        "epoch": epoch,
        "best_validation_loss": best_val_loss,
        "validation_protocol": validation_protocol,
        "config": experiment.to_dict(),
        "rng_state": capture_rng_state(),
    }
    torch.save(payload, temp_path)
    temp_path.replace(path)

    atomic_write_json(
        path.parent / "latest.json",
        {"checkpoint": path.name, "training_step": step, "tokens_seen": tokens_seen, "epoch": epoch},
    )

    # Prune old checkpoints
    all_ckpts = sorted(
        path.parent.glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if len(all_ckpts) > keep_last:
        for old_ckpt in all_ckpts[:-keep_last]:
            old_ckpt.unlink(missing_ok=True)


@torch.no_grad()
def evaluate_validation(
    model: DecoderOnlyTransformer,
    val_loader: DataLoader,
    device: torch.device,
    autocast_context: Any,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Compute token-weighted loss on all validation examples unless explicitly capped."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_examples = 0

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        active_tokens = (labels != -100).sum().item()
        if active_tokens == 0:
            continue

        with autocast_context():
            output = model(input_ids, targets=labels)
            loss = output.loss

        total_loss += float(loss.item()) * active_tokens
        total_tokens += active_tokens
        total_examples += input_ids.shape[0]

    if was_training:
        model.train()

    if total_tokens == 0:
        return {"loss": float("inf"), "perplexity": float("inf"), "tokens": 0}

    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 50.0)),
        "tokens": total_tokens,
        "examples": total_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to phase3.toml")
    parser.add_argument(
        "--pretrained-checkpoint", type=Path, required=True, help="Pretrained model weights"
    )
    parser.add_argument("--train-data", type=Path, required=True, help="Path to train.jsonl.gz")
    parser.add_argument("--val-data", type=Path, required=True, help="Path to validation.jsonl.gz")
    parser.add_argument("--template-config", type=Path, required=True, help="Path to phase3_templates.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume from latest.json")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 30-step overfit verification")
    parser.add_argument("--pilot-steps", type=int, default=None, help="Stop early after N steps for pilot")
    parser.add_argument(
        "--validation-scope", choices=("full", "legacy_subset"), default="full",
        help="Full split for new runs; legacy_subset only to reproduce/resume historical v1 runs",
    )
    args = parser.parse_args()

    experiment = load_experiment_config(args.config)
    validation_protocol = "full_validation_v1" if args.validation_scope == "full" else None
    seed_all(experiment.training.seed)
    device = select_device(args.device)
    autocast_ctx, _, use_scaler = precision_context(device, experiment.training.precision)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    tokenizer_path = root / experiment.identity.tokenizer

    print(f"Loading datasets for {experiment.identity.language_name}...")
    max_train = 8 if args.smoke_test else None
    train_dataset = ReasoningFinetuneDataset(
        data_path=args.train_data,
        template_path=args.template_config,
        tokenizer_path=tokenizer_path,
        max_examples=max_train,
    )
    val_dataset = ReasoningFinetuneDataset(
        data_path=args.val_data,
        template_path=args.template_config,
        tokenizer_path=tokenizer_path,
    )

    micro_batch = min(experiment.training.micro_batch_size, len(train_dataset))

    def get_train_loader(epoch_idx: int) -> DataLoader:
        g = torch.Generator()
        g.manual_seed(experiment.training.seed + epoch_idx)
        return DataLoader(
            train_dataset,
            batch_size=micro_batch,
            shuffle=not args.smoke_test,
            generator=g if not args.smoke_test else None,
            collate_fn=collate_reasoning_batch,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=experiment.training.micro_batch_size,
        shuffle=False,
        collate_fn=collate_reasoning_batch,
    )

    print(f"Instantiating model ({experiment.identity.model_name})...")
    model = DecoderOnlyTransformer(experiment.model).to(device)

    start_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")

    latest_path = args.output_dir / "latest.json"
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"Cannot resume: {latest_path} does not exist.")
        latest_meta = json.loads(latest_path.read_text())
        ckpt_path = args.output_dir / latest_meta["checkpoint"]
        print(f"Resuming from checkpoint {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Validate config compatibility
        saved_cfg = ckpt.get("config", {})
        saved_model = saved_cfg.get("model", {})
        if saved_model:
            for key in ["vocabulary_size", "context_length", "d_model", "num_layers", "num_heads"]:
                if key in saved_model and getattr(experiment.model, key) != saved_model[key]:
                    raise ValueError(
                        f"Config mismatch on resume for {key}: saved={saved_model[key]}, current={getattr(experiment.model, key)}"
                    )

        model.load_state_dict(ckpt["model_state"])
        optimizer = configure_optimizer(model, experiment.training)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler = configure_scheduler(optimizer, experiment.training)
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler is not None and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_step = ckpt["training_step"]
        tokens_seen = ckpt["tokens_seen"]
        best_val_loss = ckpt.get("best_validation_loss", float("inf"))
        restore_rng_state(ckpt["rng_state"])
        if ckpt.get("validation_protocol") != validation_protocol and not args.smoke_test:
            raise ValueError(
                "This run selected checkpoints with a different validation protocol. "
                "Use --validation-scope legacy_subset to resume historical v1, or start a "
                "new output run for full-validation selection."
            )
    else:
        print(f"Initializing fresh weights from {args.pretrained_checkpoint}...")
        pre_ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(pre_ckpt["model_state"])
        optimizer = configure_optimizer(model, experiment.training)
        scheduler = configure_scheduler(optimizer, experiment.training)

    log_path = args.output_dir / "training.log"
    log_cols = ["step", "train_loss", "learning_rate", "tokens_seen", "val_loss", "val_perplexity", "step_time_sec"]
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(log_cols)

    total_steps = 30 if args.smoke_test else (args.pilot_steps or experiment.training.maximum_steps)
    grad_accum = 1 if args.smoke_test else experiment.training.gradient_accumulation_steps
    val_interval = 10 if args.smoke_test else experiment.training.validation_interval
    ckpt_interval = 10 if args.smoke_test else experiment.training.checkpoint_interval

    print(f"Starting training: total_steps={total_steps}, grad_accum={grad_accum}, device={device}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_tokens = 0
    step = start_step

    total_microbatches = start_step * grad_accum
    batches_per_epoch = len(train_dataset) // micro_batch
    current_epoch = total_microbatches // max(1, batches_per_epoch)
    batches_into_epoch = total_microbatches % max(1, batches_per_epoch)

    train_loader = get_train_loader(current_epoch)
    train_iter = iter(train_loader)
    for _ in range(batches_into_epoch):
        try:
            next(train_iter)
        except StopIteration:
            break

    step_start_time = time.time()

    while step < total_steps:
        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                current_epoch += 1
                train_loader = get_train_loader(current_epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            active_tokens = (labels != -100).sum().item()
            if active_tokens == 0:
                continue

            with autocast_ctx():
                output = model(input_ids, targets=labels)
                loss = output.loss / grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += float(loss.item()) * grad_accum * active_tokens
            accum_tokens += active_tokens
            tokens_seen += active_tokens

        if scaler is not None:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), experiment.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(model.parameters(), experiment.training.gradient_clip_norm)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        step_elapsed = time.time() - step_start_time
        step_start_time = time.time()

        train_loss = (accum_loss / accum_tokens) if accum_tokens > 0 else 0.0
        accum_loss = 0.0
        accum_tokens = 0

        # Logging & Validation
        val_metrics: dict[str, Any] = {}
        if step % val_interval == 0 or step == total_steps:
            # Default selection covers every family, including files grouped by task.
            # The shared legacy validation_batches config remains for historical reproducibility.
            val_batches = 10 if args.smoke_test else (
                None if args.validation_scope == "full" else experiment.training.validation_batches
            )
            val_metrics = evaluate_validation(model, val_loader, device, autocast_ctx, max_batches=val_batches)
            val_loss = val_metrics["loss"]
            if val_loss < best_val_loss and not args.smoke_test:
                best_val_loss = val_loss
                best_ckpt_path = args.output_dir / "best_checkpoint.pt"
                save_finetune_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    tokens_seen,
                    best_val_loss,
                    experiment,
                    epoch=current_epoch,
                    keep_last=experiment.training.checkpoint_keep_last,
                    validation_protocol=validation_protocol,
                )

        if step % experiment.training.log_interval == 0 or step == total_steps or args.smoke_test:
            lr = optimizer.param_groups[0]["lr"]
            val_loss_str = f"{val_metrics['loss']:.4f}" if val_metrics else "-"
            val_ppl_str = f"{val_metrics['perplexity']:.2f}" if val_metrics else "-"
            print(
                f"Step {step:05d}/{total_steps} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss_str} (PPL: {val_ppl_str}) | "
                f"LR: {lr:.2e} | "
                f"Tokens: {tokens_seen} | "
                f"Time: {step_elapsed:.3f}s"
            )
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    step,
                    f"{train_loss:.6f}",
                    f"{lr:.8e}",
                    tokens_seen,
                    val_metrics.get("loss", ""),
                    val_metrics.get("perplexity", ""),
                    f"{step_elapsed:.4f}",
                ])

        # Regular Checkpoint
        if (step % ckpt_interval == 0 or step == total_steps) and not args.smoke_test:
            ckpt_path = args.output_dir / f"checkpoint_step_{step:07d}.pt"
            save_finetune_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                tokens_seen,
                best_val_loss,
                experiment,
                epoch=current_epoch,
                keep_last=experiment.training.checkpoint_keep_last,
                validation_protocol=validation_protocol,
            )

    print(f"Training completed successfully at step {step}!")


if __name__ == "__main__":
    main()
