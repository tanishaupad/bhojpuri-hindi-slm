"""Pretrain one independent monolingual decoder-only Transformer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lma_phase1.io import atomic_write_json
from lma_phase2.config import load_experiment_config
from lma_phase2.data import PackedTokenDataset
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase2.training import (
    append_log,
    configure_optimizer,
    configure_scheduler,
    evaluate_loss,
    load_checkpoint,
    precision_context,
    prune_checkpoints,
    resolve_latest_checkpoint,
    save_checkpoint,
    seed_everything,
    select_device,
)


def validate_dataset_identity(dataset: PackedTokenDataset, language: str, vocabulary: int) -> None:
    if dataset.manifest.get("language") != language:
        raise ValueError("Packed dataset language does not match the experiment")
    if int(dataset.manifest.get("vocabulary_size", -1)) != vocabulary:
        raise ValueError("Packed dataset tokenizer vocabulary does not match the model")


def train(args: argparse.Namespace) -> None:
    experiment = load_experiment_config(Path(args.config))
    training = experiment.training
    seed_everything(training.seed)
    device = select_device(args.device)
    train_data = PackedTokenDataset(Path(args.train_manifest))
    validation_data = PackedTokenDataset(Path(args.validation_manifest))
    validate_dataset_identity(
        train_data, experiment.identity.language, experiment.model.vocabulary_size
    )
    validate_dataset_identity(
        validation_data, experiment.identity.language, experiment.model.vocabulary_size
    )
    model = DecoderOnlyTransformer(experiment.model).to(device)
    actual_parameters = model.parameter_count()
    expected_parameters = experiment.model.expected_parameter_count()
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"Parameter count mismatch: implementation={actual_parameters}, config={expected_parameters}"
        )
    optimizer = configure_optimizer(model, training)
    scheduler = configure_scheduler(optimizer, training)
    autocast_context, autocast_dtype, use_scaler = precision_context(device, training.precision)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if device.type == "cuda" else None
    data_generator = torch.Generator(device="cpu").manual_seed(training.seed + 1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "resolved_config.json"
    if resolved_config_path.exists():
        with resolved_config_path.open(encoding="utf-8") as handle:
            if json.load(handle) != experiment.to_dict():
                raise SystemExit("Output directory contains a different experiment configuration")
    else:
        atomic_write_json(resolved_config_path, experiment.to_dict())
    atomic_write_json(
        output_dir / "run_environment.json",
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "autocast_dtype": str(autocast_dtype),
            "parameter_count": actual_parameters,
            "train_manifest": str(Path(args.train_manifest)),
            "validation_manifest": str(Path(args.validation_manifest)),
        },
    )

    completed_step = 0
    tokens_seen = 0
    best_validation_loss = None
    if args.resume:
        checkpoint_path = (
            resolve_latest_checkpoint(output_dir) if args.resume == "latest" else Path(args.resume)
        )
        checkpoint = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            experiment,
            data_generator,
            device,
        )
        completed_step = int(checkpoint["training_step"])
        tokens_seen = int(checkpoint["tokens_seen"])
        best_validation_loss = checkpoint["best_validation_loss"]

    model.train()
    log_path = output_dir / "training_log.csv"
    start_time = time.monotonic()
    for step in range(completed_step + 1, training.maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _micro_step in range(training.gradient_accumulation_steps):
            inputs, targets = train_data.random_batch(
                training.micro_batch_size,
                experiment.model.context_length,
                data_generator,
                device,
            )
            with autocast_context():
                loss = model(inputs, targets=targets).loss
                scaled_loss = loss / training.gradient_accumulation_steps
            accumulated_loss += float(loss.detach())
            if scaler is not None and use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
        if scaler is not None and use_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
        if scaler is not None and use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        tokens_this_step = (
            training.micro_batch_size
            * training.gradient_accumulation_steps
            * experiment.model.context_length
        )
        tokens_seen += tokens_this_step
        train_loss = accumulated_loss / training.gradient_accumulation_steps
        learning_rate = optimizer.param_groups[0]["lr"]
        if step % training.log_interval == 0 or step == 1:
            append_log(
                log_path,
                {
                    "step": step,
                    "split": "train",
                    "loss": f"{train_loss:.8f}",
                    "perplexity": f"{torch.exp(torch.tensor(min(train_loss, 50.0))).item():.8f}",
                    "learning_rate": f"{learning_rate:.12g}",
                    "tokens_seen": tokens_seen,
                },
            )
            elapsed = time.monotonic() - start_time
            print(
                f"step={step} loss={train_loss:.5f} lr={learning_rate:.3e} "
                f"tokens={tokens_seen} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
        if step % training.validation_interval == 0 or step == training.maximum_steps:
            metrics = evaluate_loss(
                model,
                validation_data,
                training.micro_batch_size,
                experiment.model.context_length,
                training.validation_batches,
                device,
                autocast_context,
            )
            append_log(
                log_path,
                {
                    "step": step,
                    "split": "validation",
                    "loss": f"{metrics['loss']:.8f}",
                    "perplexity": f"{metrics['perplexity']:.8f}",
                    "learning_rate": f"{learning_rate:.12g}",
                    "tokens_seen": tokens_seen,
                },
            )
            best_validation_loss = (
                metrics["loss"]
                if best_validation_loss is None
                else min(best_validation_loss, metrics["loss"])
            )
            print(
                f"step={step} validation_loss={metrics['loss']:.5f} "
                f"validation_ppl={metrics['perplexity']:.3f}",
                flush=True,
            )
        if step % training.checkpoint_interval == 0 or step == training.maximum_steps:
            save_checkpoint(
                output_dir / f"checkpoint_step_{step:07d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                tokens_seen,
                best_validation_loss,
                experiment,
                data_generator,
            )
            prune_checkpoints(output_dir, training.checkpoint_keep_last)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", help="Checkpoint path or 'latest'")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
