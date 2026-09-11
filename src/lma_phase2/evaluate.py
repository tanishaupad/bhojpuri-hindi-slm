"""Run Phase 2 intrinsic, generation, and attention evaluation for one model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from pathlib import Path

import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
from matplotlib.font_manager import FontProperties

from lma_phase1.io import atomic_write_json, iter_jsonl
from lma_phase2.config import load_experiment_config
from lma_phase2.data import PackedTokenDataset
from lma_phase2.evaluation import (
    attention_summary,
    generate_tokens,
    generation_metrics,
    intrinsic_metrics,
)
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase2.training import precision_context, select_device


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    experiment = load_experiment_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config"] != experiment.to_dict():
        raise ValueError("Checkpoint and evaluation config differ")
    model = DecoderOnlyTransformer(experiment.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return experiment, model


def generation_examples(
    test_jsonl: Path,
    tokenizer: spm.SentencePieceProcessor,
    count: int,
    prefix_tokens: int,
    continuation_tokens: int,
    seed: int,
) -> list[dict[str, object]]:
    """Choose a stable corpus-wide hash sample of sufficiently long documents."""

    heap: list[tuple[int, str, dict[str, object]]] = []
    required = prefix_tokens + continuation_tokens
    for document in iter_jsonl(test_jsonl):
        token_ids = tokenizer.encode(document.text, out_type=int)
        if len(token_ids) < required:
            continue
        score = int.from_bytes(
            hashlib.sha256(f"{seed}\0{document.document_id}".encode()).digest()[:8],
            "big",
        )
        example = {
            "document_id": document.document_id,
            "source_name": document.source_name,
            "source_url": document.source_url,
            "prefix_ids": token_ids[:prefix_tokens],
            "reference_ids": token_ids[prefix_tokens:required],
        }
        item = (-score, document.document_id, example)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)
    if len(heap) < count:
        raise ValueError(f"Only found {len(heap)} sufficiently long test documents")
    return [item[2] for item in sorted(heap, key=lambda value: -value[0])]


def plot_attention(
    matrix,
    labels: list[str],
    layer: int,
    head: int,
    language_name: str,
    output: Path,
    font_path: Path | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    step = max(1, len(labels) // 20)
    positions = list(range(0, len(labels), step))
    font = FontProperties(fname=str(font_path)) if font_path is not None else None
    tick_options = {"fontproperties": font} if font is not None else {}
    axis.set_xticks(
        positions,
        [labels[index] for index in positions],
        rotation=90,
        **tick_options,
    )
    axis.set_yticks(positions, [labels[index] for index in positions], **tick_options)
    axis.set_xlabel("Key token position")
    axis.set_ylabel("Query token position")
    axis.set_title(f"{language_name} attention: layer {layer}, head {head}")
    figure.colorbar(image, ax=axis, label="Attention probability")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    device = select_device(args.device)
    experiment, model = load_model(Path(args.config), Path(args.checkpoint), device)
    autocast_context, _dtype, _use_scaler = precision_context(device, experiment.training.precision)
    test_data = PackedTokenDataset(Path(args.test_manifest))
    intrinsic = intrinsic_metrics(model, test_data, device, autocast_context)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_model)
    examples = generation_examples(
        Path(args.test_jsonl),
        tokenizer,
        args.generation_examples,
        args.prefix_tokens,
        args.continuation_tokens,
        args.seed,
    )
    modes = {"greedy": None, "temperature_0.5": 0.5, "temperature_1.0": 1.0, "temperature_1.5": 1.5}
    samples = []
    metrics = {}
    for mode_index, (mode, temperature) in enumerate(modes.items()):
        hypotheses = []
        references = []
        generated_ids = []
        generator = torch.Generator().manual_seed(args.seed + mode_index)
        for example in examples:
            continuation = generate_tokens(
                model,
                example["prefix_ids"],
                args.continuation_tokens,
                device,
                temperature,
                generator,
                tokenizer.eos_id(),
            )
            hypothesis = tokenizer.decode(continuation)
            reference = tokenizer.decode(example["reference_ids"])
            hypotheses.append(hypothesis)
            references.append(reference)
            generated_ids.append(continuation)
            samples.append(
                {
                    "mode": mode,
                    "document_id": example["document_id"],
                    "source_name": example["source_name"],
                    "source_url": example["source_url"],
                    "prefix": tokenizer.decode(example["prefix_ids"]),
                    "reference": reference,
                    "generated": hypothesis,
                }
            )
        metrics[mode] = generation_metrics(hypotheses, references, generated_ids)

    first = examples[0]
    attention_input_ids = first["prefix_ids"]
    input_tensor = torch.tensor([attention_input_ids], device=device)
    with torch.no_grad():
        attention_output = model(input_tensor, return_attentions=True)
    attention_rows = attention_summary(attention_output.attentions)
    token_labels = [
        tokenizer.id_to_piece(token_id).replace("▁", " ") for token_id in attention_input_ids
    ]
    figure_dir = Path(args.figure_dir)
    for layer in (0, experiment.model.number_of_layers - 1):
        for head in range(min(3, experiment.model.number_of_heads)):
            plot_attention(
                attention_output.attentions[layer][0, head].cpu().numpy(),
                token_labels,
                layer,
                head,
                experiment.identity.language_name,
                figure_dir / f"{experiment.identity.language}_layer{layer}_head{head}.png",
                Path(args.attention_font) if args.attention_font else None,
            )
    attention_csv = Path(args.attention_csv)
    attention_csv.parent.mkdir(parents=True, exist_ok=True)
    with attention_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=attention_rows[0])
        writer.writeheader()
        writer.writerows(attention_rows)
    sample_path = Path(args.samples_output)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    report = {
        "model_name": experiment.identity.model_name,
        "checkpoint": args.checkpoint,
        "intrinsic": intrinsic,
        "generation": metrics,
        "generation_protocol": {
            "examples": args.generation_examples,
            "prefix_tokens": args.prefix_tokens,
            "continuation_tokens": args.continuation_tokens,
            "bleu": "corpus BLEU-4, whitespace tokenization",
            "chrf": "chrF++ with character order 6 and word order 2",
            "rouge": "mean whitespace-token ROUGE-L F1",
            "sampling_seed": args.seed,
            "example_selection": "smallest SHA-256(seed, document_id) scores",
        },
        "attention_csv": args.attention_csv,
        "attention_figure_dir": args.figure_dir,
        "samples_output": args.samples_output,
    }
    atomic_write_json(Path(args.output), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-output", required=True)
    parser.add_argument("--attention-csv", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument(
        "--attention-font",
        help="Optional font file with glyphs for the tokenizer labels (for example Devanagari)",
    )
    parser.add_argument("--generation-examples", type=int, default=50)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
