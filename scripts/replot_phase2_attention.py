"""Redraw Phase 2 heatmaps from the saved prefixes, verifying existing head metrics."""

import argparse
import csv
import json
from pathlib import Path

import sentencepiece as spm
import torch

from lma_phase2.config import load_experiment_config
from lma_phase2.evaluate import plot_attention
from lma_phase2.evaluation import attention_summary
from lma_phase2.model import DecoderOnlyTransformer


def main() -> None:
    """Use the report's exact prompt and leave measured CSV summaries unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    for language, step, code in (("hindi", "0020300", "hi"), ("bhojpuri", "0004150", "bho")):
        config = load_experiment_config(root / language / "configs/phase2.toml")
        model = DecoderOnlyTransformer(config.model).eval()
        path = args.artifact_root / language / f"phase2/runs/pretrain_v1/checkpoint_step_{step}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        del payload
        tokenizer = spm.SentencePieceProcessor(model_file=str(root / config.identity.tokenizer))
        samples = root / f"report/samples/generated/phase2/{language}_samples.jsonl"
        with samples.open() as handle:
            prefix = json.loads(next(handle))["prefix"]
        ids = tokenizer.encode(prefix)
        with torch.no_grad():
            attention = model(torch.tensor([ids]), return_attentions=True).attentions
        with (
            root / f"report/tables/generated/phase2/{language}_attention_summary.csv"
        ).open() as handle:
            expected = list(csv.DictReader(handle))
        for actual, recorded in zip(attention_summary(attention), expected, strict=True):
            for key, value in actual.items():
                if abs(value - float(recorded[key])) > 1e-4:
                    raise ValueError(f"Attention summary mismatch: {language}/{key}")
        labels = [tokenizer.id_to_piece(i).replace("▁", " ") for i in ids]
        for layer in (0, 8):
            for head in (0, 1, 2):
                plot_attention(
                    attention[layer][0, head].numpy(),
                    labels,
                    layer,
                    head,
                    language.title(),
                    root
                    / "report/figures/generated/phase2/attention"
                    / language
                    / f"{code}_layer{layer}_head{head}.png",
                    args.font,
                )
        print(language, "all 54 head metrics matched; six figures redrawn", flush=True)


if __name__ == "__main__":
    main()
