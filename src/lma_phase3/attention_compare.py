"""Attention comparison between pretrained and finetuned reasoning models.

Extracts multi-head attention weights on identical fixed reasoning prompts,
computes attention entropy and mean attention distance, and plots comparison heatmaps.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
from matplotlib.font_manager import FontProperties

from lma_phase2.config import load_experiment_config
from lma_phase2.evaluation import attention_summary
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase2.training import select_device


def plot_side_by_side_attention(
    pre_matrix,
    fine_matrix,
    labels: list[str],
    layer: int,
    head: int,
    language_name: str,
    output: Path,
    font_path: Path | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True, layout="constrained")
    font = FontProperties(fname=str(font_path)) if font_path is not None else None
    tick_options = {"fontproperties": font} if font is not None else {}

    step = max(1, len(labels) // 15)
    positions = list(range(0, len(labels), step))
    tick_labels = [labels[i] for i in positions]

    ax1.imshow(pre_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax1.set_title(f"Pretrained ({language_name}) - L{layer} H{head}", fontsize=12)
    ax1.set_xticks(positions, tick_labels, rotation=90, **tick_options)
    ax1.set_yticks(positions, tick_labels, **tick_options)
    ax1.set_xlabel("Key Position")
    ax1.set_ylabel("Query Position")

    im2 = ax2.imshow(fine_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax2.set_title(f"Finetuned ({language_name}) - L{layer} H{head}", fontsize=12)
    ax2.set_xticks(positions, tick_labels, rotation=90, **tick_options)
    ax2.set_xlabel("Key Position")

    fig.colorbar(im2, ax=[ax1, ax2], label="Attention Weight", fraction=0.03, pad=0.04)
    fig.savefig(output, dpi=180)
    plt.close(fig)


@torch.no_grad()
def extract_attention(
    model: DecoderOnlyTransformer,
    input_ids: list[int],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], list[dict[str, Any]]]:
    model.eval()
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    output = model(tensor, return_attentions=True)
    attentions = output.attentions
    assert attentions is not None
    summary = attention_summary(attentions)
    return attentions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to config.toml")
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True, help="Reasoning prompt to inspect")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--heads", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--font-path", type=Path, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    experiment = load_experiment_config(args.config)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    print(f"Loading pretrained model from {args.pretrained_checkpoint}...")
    pre_model = DecoderOnlyTransformer(experiment.model).to(device)
    pre_ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
    pre_model.load_state_dict(pre_ckpt["model_state"])

    print(f"Loading finetuned model from {args.finetuned_checkpoint}...")
    fine_model = DecoderOnlyTransformer(experiment.model).to(device)
    fine_ckpt = torch.load(args.finetuned_checkpoint, map_location="cpu", weights_only=False)
    fine_model.load_state_dict(fine_ckpt["model_state"])

    token_ids = sp.encode(args.prompt)
    labels = [sp.id_to_piece(tid).replace("\u2581", "_") for tid in token_ids]
    print(f"Prompt ({len(token_ids)} tokens): {args.prompt}")

    pre_attns, pre_summary = extract_attention(pre_model, token_ids, device)
    fine_attns, fine_summary = extract_attention(fine_model, token_ids, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary metrics comparison CSV
    csv_path = args.output_dir / "attention_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "head",
            "pretrained_entropy",
            "finetuned_entropy",
            "entropy_change",
            "pretrained_mean_distance",
            "finetuned_mean_distance",
            "distance_change",
        ])
        for p, fn in zip(pre_summary, fine_summary, strict=True):
            ent_p = float(p["mean_entropy"])
            ent_f = float(fn["mean_entropy"])
            dist_p = float(p["mean_attention_distance"])
            dist_f = float(fn["mean_attention_distance"])
            writer.writerow([
                p["layer"],
                p["head"],
                f"{ent_p:.4f}",
                f"{ent_f:.4f}",
                f"{ent_f - ent_p:+.4f}",
                f"{dist_p:.4f}",
                f"{dist_f:.4f}",
                f"{dist_f - dist_p:+.4f}",
            ])

    # Plot specified layers and heads
    for layer in args.layers:
        for head in args.heads:
            pre_mat = pre_attns[layer][0, head].cpu().numpy()
            fine_mat = fine_attns[layer][0, head].cpu().numpy()
            fig_path = args.output_dir / f"compare_layer_{layer}_head_{head}.png"
            plot_side_by_side_attention(
                pre_mat,
                fine_mat,
                labels,
                layer=layer,
                head=head,
                language_name=experiment.identity.language_name,
                output=fig_path,
                font_path=args.font_path,
            )
            print(f"Saved attention figure to {fig_path}")

    print(f"Saved attention comparison table to {csv_path}")


if __name__ == "__main__":
    main()
