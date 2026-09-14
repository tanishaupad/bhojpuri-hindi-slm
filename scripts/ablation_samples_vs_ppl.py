"""Ablation study: Impact of reasoning finetuning sample count on generic language perplexity (retention) and reasoning performance.

Five new subset runs plus the historical 20k run and pretrained baseline.
Accuracy is a diagnostic on the first 100 direct-comparison validation examples.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from lma_phase1.io import atomic_write_json
from lma_phase1.tokenizer.common import normalize_tokenizer_input
from lma_phase2.config import TrainingConfig, load_experiment_config
from lma_phase2.data import PackedTokenDataset
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase2.training import (
    configure_optimizer,
    configure_scheduler,
    precision_context,
    select_device,
)
from lma_phase3.cached_decode import greedy_cached
from lma_phase3.evaluate import file_hash
from lma_phase3.evaluate_retention import evaluate_stream
from lma_phase3.explicit_answer import extract_explicit_answer
from lma_phase3.finetune import capture_rng_state, evaluate_validation, restore_rng_state, seed_all
from lma_phase3.finetune_data import ReasoningFinetuneDataset, collate_reasoning_batch

SWEEP_SIZES = [1000, 3000, 5000, 8000, 12000]
FAMILIES = ("direct", "extremum", "equality", "transitivity")


def create_stratified_subsets(
    full_train_path: Path, output_dir: Path, sample_sizes: list[int], seed: int = 20260823
) -> dict[int, Path]:
    """Reproduce the canonical historical sweep, including its RNG draw order."""
    if any(size not in SWEEP_SIZES for size in sample_sizes):
        raise ValueError(f"Historical subset sizes must be drawn from {SWEEP_SIZES}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read full dataset
    family_buckets: dict[str, list[dict[str, Any]]] = {f: [] for f in FAMILIES}
    with gzip.open(full_train_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            fam = item.get("task_family")
            if fam in family_buckets:
                family_buckets[fam].append(item)
            else:
                family_buckets.setdefault("other", []).append(item)

    rng = random.Random(seed)
    subset_paths = {}

    for size in SWEEP_SIZES:
        out_path = output_dir / f"train_{size}.jsonl.gz"
        per_fam = size // len(FAMILIES)
        selected = []
        for fam in FAMILIES:
            bucket = list(family_buckets[fam])
            rng.shuffle(bucket)
            selected.extend(bucket[:per_fam])

        if len(selected) != size:
            raise ValueError(f"Insufficient examples for balanced subset {size}")

        rng.shuffle(selected)
        if size not in sample_sizes:
            continue
        serialized = "".join(json.dumps(ex, ensure_ascii=False) + "\n" for ex in selected)
        if out_path.exists():
            with gzip.open(out_path, "rt", encoding="utf-8") as f:
                if f.read() != serialized:
                    raise ValueError(
                        f"Existing subset differs from the canonical sample: {out_path}"
                    )
        else:
            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                f.write(serialized)

        subset_paths[size] = out_path
        print(f"Created subset {size} at {out_path} ({len(selected)} samples)")

    return subset_paths


def train_subset(
    config_path: Path,
    pretrained_ckpt_path: Path,
    train_data_path: Path,
    val_data_path: Path,
    template_config_path: Path,
    output_dir: Path,
    num_samples: int,
    epochs: int = 2,
    device_name: str = "auto",
) -> Path:
    """Finetune base model on subset for given epochs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = output_dir / "checkpoint_final.pt"
    if final_ckpt.exists():
        saved = torch.load(final_ckpt, map_location="cpu", weights_only=False)
        if saved.get("samples") != num_samples:
            raise ValueError("Existing checkpoint has a different sample budget")
        print(f"Found existing final checkpoint at {final_ckpt}, skipping training.")
        return final_ckpt

    experiment = load_experiment_config(config_path)
    seed_all(experiment.training.seed)
    device = select_device(device_name)
    autocast_ctx, _, use_scaler = precision_context(device, experiment.training.precision)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    root = Path(__file__).resolve().parents[1]
    tokenizer_path = root / experiment.identity.tokenizer

    train_dataset = ReasoningFinetuneDataset(
        data_path=train_data_path,
        template_path=template_config_path,
        tokenizer_path=tokenizer_path,
    )

    micro_batch = experiment.training.micro_batch_size
    grad_accum = experiment.training.gradient_accumulation_steps
    effective_batch = micro_batch * grad_accum
    total_steps = max(20, round(num_samples * epochs / effective_batch))
    warmup_steps = max(10, min(100, round(total_steps * 0.1)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch,
        shuffle=True,
        collate_fn=collate_reasoning_batch,
    )

    # Initialize from pretrained weights
    print(
        f"[{experiment.identity.language_name} N={num_samples}] Initializing from {pretrained_ckpt_path}..."
    )
    pre_ckpt = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=False)
    model = DecoderOnlyTransformer(experiment.model).to(device)
    model.load_state_dict(pre_ckpt["model_state"])

    # Setup training config with adjusted schedule
    custom_train_cfg = TrainingConfig(
        seed=experiment.training.seed,
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        maximum_steps=total_steps,
        learning_rate=experiment.training.learning_rate,
        minimum_learning_rate=experiment.training.minimum_learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=experiment.training.weight_decay,
        adam_beta1=experiment.training.adam_beta1,
        adam_beta2=experiment.training.adam_beta2,
        adam_epsilon=experiment.training.adam_epsilon,
        gradient_clip_norm=experiment.training.gradient_clip_norm,
        validation_interval=max(25, total_steps // 4),
        validation_batches=experiment.training.validation_batches,
        checkpoint_interval=total_steps,
        checkpoint_keep_last=1,
        log_interval=max(10, total_steps // 10),
        precision=experiment.training.precision,
    )

    optimizer = configure_optimizer(model, custom_train_cfg)
    scheduler = configure_scheduler(optimizer, custom_train_cfg)

    print(
        f"[{experiment.identity.language_name} N={num_samples}] Training {total_steps} steps (warmup={warmup_steps})..."
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    epoch_rng = capture_rng_state()
    train_iter = iter(train_loader)
    epoch_batches = 0
    resume_path = output_dir / "checkpoint_resume.pt"
    identities = {
        str(key): file_hash(path)
        for key, path in {
            "train": train_data_path,
            "validation": val_data_path,
            "tokenizer": tokenizer_path,
            "templates": template_config_path,
            "pretrained": pretrained_ckpt_path,
        }.items()
    }
    resolved = replace(experiment, training=custom_train_cfg)
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["config"] != resolved.to_dict() or saved["input_sha256"] != identities:
            raise ValueError("Resume configuration or input hashes differ from the saved run")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        if scaler is not None:
            scaler.load_state_dict(saved["scaler_state"])
        step = saved["training_step"]
        epoch_rng = saved["epoch_start_rng"]
        epoch_batches = saved["epoch_batches"]
        restore_rng_state(epoch_rng)
        train_iter = iter(train_loader)
        for _ in range(epoch_batches):
            next(train_iter)
        restore_rng_state(saved["rng_state"])
        log_path = output_dir / "training.jsonl"
        if log_path.exists():
            retained = [
                line
                for line in log_path.read_text().splitlines()
                if json.loads(line)["step"] <= step
            ]
            log_path.write_text("\n".join(retained) + ("\n" if retained else ""))
        print(f"Resumed optimizer step {step}")
    atomic_write_json(output_dir / "resolved_config.json", resolved.to_dict())
    atomic_write_json(output_dir / "input_hashes.json", identities)
    started = time.time()

    while step < total_steps:
        accum_loss = 0.0
        accum_tokens = 0
        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch_rng = capture_rng_state()
                train_iter = iter(train_loader)
                epoch_batches = 0
                batch = next(train_iter)
            epoch_batches += 1

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

        if scaler is not None:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), custom_train_cfg.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(model.parameters(), custom_train_cfg.gradient_clip_norm)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % custom_train_cfg.log_interval == 0 or step == total_steps:
            lr = optimizer.param_groups[0]["lr"]
            mean_loss = accum_loss / max(1, accum_tokens)
            print(f"  Step {step:04d}/{total_steps} | Loss: {mean_loss:.4f} | LR: {lr:.2e}")
            with (output_dir / "training.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"step": step, "loss": mean_loss, "learning_rate": lr}) + "\n")

        # Save only at optimizer boundaries; replay the current shuffled epoch on resume.
        if step % 25 == 0 or step == total_steps:
            saved = {
                "format_version": "ablation_resume_v1",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "config": resolved.to_dict(),
                "resolved_training_config": asdict(custom_train_cfg),
                "training_step": step,
                "samples": num_samples,
                "input_sha256": identities,
                "rng_state": capture_rng_state(),
                "epoch_start_rng": epoch_rng,
                "epoch_batches": epoch_batches,
            }
            temporary = resume_path.with_suffix(".tmp")
            torch.save(saved, temporary)
            temporary.replace(resume_path)

    elapsed = time.time() - started
    print(f"[{experiment.identity.language_name} N={num_samples}] Done in {elapsed:.1f}s.")

    # Historical final files are never rewritten; new runs preserve complete resume state.
    temporary = final_ckpt.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(final_ckpt)
    return final_ckpt


@torch.no_grad()
def evaluate_accuracy_on_val(
    model: DecoderOnlyTransformer,
    tokenizer_path: Path,
    val_data_path: Path,
    device: torch.device,
    max_eval_samples: int = 100,
) -> float:
    """Evaluate explicit answer accuracy using greedy decoding on validation split."""
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    rows = []
    with gzip.open(val_data_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_eval_samples:
                break

    correct = 0
    total = len(rows)

    for r in rows:
        prefix_ids = sp.encode(normalize_tokenizer_input(r["prompt"]))
        gen_ids = greedy_cached(model, [prefix_ids], sp.eos_id(), device, limit=16)[0]
        if gen_ids and gen_ids[-1] == sp.eos_id():
            gen_ids = gen_ids[:-1]
        text = sp.decode(gen_ids)
        pred_ans = extract_explicit_answer(text)
        gold_ans = extract_explicit_answer(r.get("answer_text", ""))
        if pred_ans and gold_ans and pred_ans == gold_ans:
            correct += 1

    return (correct / total) * 100.0 if total > 0 else 0.0


def run_ablation_for_language(
    lang: str,
    root: Path,
    device_name: str,
    sample_sizes: list[int],
    artifact_root: Path | None = None,
    dataset_root: Path | None = None,
    work_dir: Path | None = None,
    pretrained_checkpoint: Path | None = None,
    main_checkpoint: Path | None = None,
    test_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    """Run full sweep for a single language."""
    print("\n==========================================")
    print(f"Starting Ablation Study for: {lang.upper()}")
    print("==========================================")

    config_path = root / f"{lang}/configs/phase3.toml"
    template_path = root / f"{lang}/configs/phase3_templates.json"
    dataset_root = dataset_root or root / "tmp/phase3/reasoning_v1"
    work_dir = work_dir or root / "tmp/phase3/ablation_reproduced"
    full_train_path = dataset_root / lang / "train.jsonl.gz"
    val_data_path = dataset_root / lang / "validation.jsonl.gz"
    test_manifest_path = root / f"output/kaggle/lma_phase2_kaggle/data/{lang}/test.manifest.json"

    steps = {"hindi": "0020300", "bhojpuri": "0004150", "nepali": "0016720"}
    if lang == "hindi":
        pretrain_ckpt_path = (
            root / "lma_phase2_outputs/phase2_runs/hindi/checkpoint_step_0020300.pt"
        )
        v1_finetuned_ckpt_path = (
            root / "tmp/phase3/runs/hindi/finetune_v1/checkpoint_step_0002500.pt"
        )
    elif lang == "bhojpuri":
        pretrain_ckpt_path = (
            root / "lma_phase2_outputs/phase2_runs/bhojpuri/checkpoint_step_0004150.pt"
        )
        v1_finetuned_ckpt_path = (
            root / "tmp/phase3/runs/bhojpuri/finetune_v1/checkpoint_step_0002500.pt"
        )
    else:
        pretrain_ckpt_path = (
            root / "lma_phase2_outputs/phase2_runs/nepali/checkpoint_step_0016720.pt"
        )
        v1_finetuned_ckpt_path = (
            root / "tmp/phase3/runs/nepali/finetune_v1/checkpoint_step_0002500.pt"
        )

    if artifact_root is not None:
        step = steps[lang]
        pretrain_ckpt_path = (
            artifact_root / lang / f"phase2/runs/pretrain_v1/checkpoint_step_{step}.pt"
        )
        v1_finetuned_ckpt_path = (
            artifact_root / lang / "phase3/runs/finetune_v1/checkpoint_step_0002500.pt"
        )
        test_manifest_path = artifact_root / lang / "phase2/tokenized_v1/test.manifest.json"

    pretrain_ckpt_path = pretrained_checkpoint or pretrain_ckpt_path
    v1_finetuned_ckpt_path = main_checkpoint or v1_finetuned_ckpt_path
    test_manifest_path = test_manifest or test_manifest_path
    for label, path in (
        ("pretrained checkpoint", pretrain_ckpt_path),
        ("main 20k checkpoint", v1_finetuned_ckpt_path),
        ("packed test manifest", test_manifest_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    experiment = load_experiment_config(config_path)
    tokenizer_path = root / experiment.identity.tokenizer
    device = select_device(device_name)
    test_dataset = PackedTokenDataset(test_manifest_path)

    # 1. Create subsets
    subset_dir = work_dir / "data" / lang
    subset_paths = create_stratified_subsets(full_train_path, subset_dir, sample_sizes)

    # Dictionary of checkpoints for each N: 0, 1k, 3k, 5k, 8k, 12k, 20k
    checkpoints: dict[int, Path] = {
        0: pretrain_ckpt_path,
        20000: v1_finetuned_ckpt_path,
    }

    # 2. Train intermediate models
    for size in sample_sizes:
        out_dir = work_dir / "runs" / lang / f"run_{size}"
        ckpt = train_subset(
            config_path=config_path,
            pretrained_ckpt_path=pretrain_ckpt_path,
            train_data_path=subset_paths[size],
            val_data_path=val_data_path,
            template_config_path=template_path,
            output_dir=out_dir,
            num_samples=size,
            epochs=2,
            device_name=device_name,
        )
        checkpoints[size] = ckpt

    # 3. Evaluate each checkpoint
    results = []
    sorted_sizes = sorted(checkpoints.keys())

    # Load val dataset for reasoning validation loss
    val_dataset = ReasoningFinetuneDataset(
        data_path=val_data_path,
        template_path=template_path,
        tokenizer_path=tokenizer_path,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_reasoning_batch,
    )
    autocast_ctx, _, _ = precision_context(device, experiment.training.precision)

    for size in sorted_sizes:
        ckpt_path = checkpoints[size]
        print(f"\nEvaluating {lang.upper()} N={size} (ckpt: {ckpt_path.name})...")

        # Load model
        model = DecoderOnlyTransformer(experiment.model).to(device)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        model.eval()

        # A. Normal language retention
        retention_json = root / f"report/results/generated/phase3/retention_v1/{lang}.json"
        if size == 0 and retention_json.exists():
            print("  Using published pretrained retention metrics...")
            data = json.loads(retention_json.read_text())
            stream_metrics = data["pretrained"]
        elif size == 20000 and retention_json.exists():
            print("  Using published 20k finetuned retention metrics...")
            data = json.loads(retention_json.read_text())
            stream_metrics = data["finetuned"]
        else:
            print("  Scoring normal language held-out test stream...")
            stream_metrics = evaluate_stream(model, test_dataset, device, batch_size=8)

        # B. Reasoning validation loss
        print("  Scoring reasoning validation loss...")
        val_metrics = evaluate_validation(model, val_loader, device, autocast_ctx)

        # C. Explicit answer accuracy on validation
        print("  Scoring reasoning answer accuracy...")
        accuracy = evaluate_accuracy_on_val(
            model, tokenizer_path, val_data_path, device, max_eval_samples=100
        )

        record = {
            "language": lang,
            "samples": size,
            "generic_perplexity": stream_metrics["perplexity"],
            "generic_cross_entropy": stream_metrics["cross_entropy"],
            "generic_bits_per_byte": stream_metrics["bits_per_byte"],
            "reasoning_val_loss": val_metrics["loss"],
            "reasoning_val_perplexity": val_metrics["perplexity"],
            "reasoning_val_accuracy": accuracy,
        }
        print(
            f"  -> Normal PPL: {record['generic_perplexity']:.2f} | "
            f"Reasoning Loss: {record['reasoning_val_loss']:.4f} | "
            f"Reasoning Acc: {record['reasoning_val_accuracy']:.1f}%"
        )
        results.append(record)

    return results


def plot_tradeoff_figure(all_results: dict[str, list[dict]], output_path: Path):
    """Plot dual-axis publication-ready tradeoff figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_langs = len(all_results)
    if num_langs == 0:
        return

    fig, axes = plt.subplots(1, num_langs, figsize=(8.5 * num_langs, 6), squeeze=False)
    axes_flat = axes.flatten()

    for ax1, (lang, results) in zip(axes_flat, all_results.items()):
        lang_title = {
            "hindi": "Hindi (Model H)",
            "bhojpuri": "Bhojpuri (Model L)",
            "nepali": "Nepali (Model L)",
        }[lang]
        samples = [r["samples"] for r in results]
        ppls = [r["generic_perplexity"] for r in results]
        accs = [r["reasoning_val_accuracy"] for r in results]

        color_ppl = "#d9534f"  # Red
        color_acc = "#0275d8"  # Blue

        # Highlight TA recommended 5k - 8k zone
        ax1.axvspan(5000, 8000, color="#5cb85c", alpha=0.18, label="TA Range (5k-8k)")

        # Left Axis: Generic Perplexity (Log Scale)
        ax1.set_xlabel("Reasoning Training Samples (N)", fontsize=12, fontweight="bold")
        ax1.set_ylabel(
            "Generic Language Test Perplexity (Log Scale)",
            color=color_ppl,
            fontsize=12,
            fontweight="bold",
        )
        ax1.plot(
            samples,
            ppls,
            color=color_ppl,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label="Generic LM PPL",
        )
        ax1.set_yscale("log")
        ax1.tick_params(axis="y", labelcolor=color_ppl)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Baseline dashed line
        base_ppl = ppls[0]
        ax1.axhline(
            base_ppl,
            color=color_ppl,
            linestyle=":",
            alpha=0.7,
            label=f"Pretrained Base PPL ({base_ppl:.1f})",
        )

        # Right Axis: Reasoning Accuracy (%)
        ax2 = ax1.twinx()
        ax2.set_ylabel(
            "Accuracy: first 100 direct validation cases (%)", color=color_acc, fontsize=11
        )
        ax2.plot(
            samples,
            accs,
            color=color_acc,
            marker="s",
            linewidth=2.5,
            markersize=8,
            linestyle="--",
            label="Direct-only accuracy (n=100)",
        )
        ax2.set_ylim(-2, 102)
        ax2.tick_params(axis="y", labelcolor=color_acc)

        # Title
        ax1.set_title(
            f"{lang_title}: Samples vs. Perplexity & Accuracy",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"\nSaved tradeoff figure to {output_path}")


def main():
    """Reproduce runs into a separate directory, or redraw the published evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["hindi", "nepali"])
    parser.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--samples", nargs="+", type=int, choices=SWEEP_SIZES, default=SWEEP_SIZES)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--main-checkpoint", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/phase3/ablation_reproduced"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tmp/phase3/ablation_reproduced/results")
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.plot_only:
        published = root / "report/results/generated/phase3/samples_vs_ppl_ablation.json"
        plot_tradeoff_figure(
            json.loads(published.read_text()), args.output_dir / "samples_vs_ppl_tradeoff.png"
        )
        return
    res_dir = args.output_dir
    res_dir.mkdir(parents=True, exist_ok=True)
    json_path = res_dir / "samples_vs_ppl_ablation.json"
    csv_path = res_dir / "samples_vs_ppl_ablation.csv"

    all_results = {}
    if json_path.exists():
        try:
            all_results = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid existing result file: {json_path}") from exc

    for lang in args.languages:
        results = run_ablation_for_language(
            lang,
            root,
            args.device,
            args.samples,
            args.artifact_root,
            args.dataset_root,
            args.work_dir,
            args.pretrained_checkpoint,
            args.main_checkpoint,
            args.test_manifest,
        )
        all_results[lang] = results

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved full results to {json_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "language",
                "samples",
                "generic_perplexity",
                "generic_cross_entropy",
                "generic_bits_per_byte",
                "reasoning_val_loss",
                "reasoning_val_perplexity",
                "reasoning_val_accuracy",
            ]
        )
        for lang, rows in all_results.items():
            for r in rows:
                writer.writerow(
                    [
                        r["language"],
                        r["samples"],
                        f"{r['generic_perplexity']:.4f}",
                        f"{r['generic_cross_entropy']:.4f}",
                        f"{r['generic_bits_per_byte']:.4f}",
                        f"{r['reasoning_val_loss']:.4f}",
                        f"{r['reasoning_val_perplexity']:.4f}",
                        f"{r['reasoning_val_accuracy']:.2f}",
                    ]
                )
    print(f"Saved CSV summary to {csv_path}")

    # Plot
    if all_results:
        fig_path = res_dir / "samples_vs_ppl_tradeoff.png"
        plot_tradeoff_figure(all_results, fig_path)


if __name__ == "__main__":
    main()
