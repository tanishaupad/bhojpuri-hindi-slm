"""Dataset, collation, and token-masking utilities for reasoning finetuning.

Ensures strict shift alignment, full prompt masking (-100), padding masking (-100),
and exact token-weighted loss computation across variable-length sequences.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

from lma_phase3.evaluate_proofs import explanation_prompt
from lma_phase3.proof_checker import reference_explanation


def build_training_pair(
    row: dict[str, Any],
    base: dict[str, Any],
    prompt_style: str = "explanation",
    include_reason_marker: bool = True,
) -> tuple[str, str]:
    """Construct (prompt, target_explanation) from problem row and templates.

    prompt_style: 'explanation' (with explicit proof instruction) or 'original'.
    include_reason_marker: True for '...। कारण: ...', False for '...। ...'.
    """
    if prompt_style == "explanation":
        prompt = explanation_prompt(row, base)
    elif prompt_style == "original":
        prompt = row["prompt"]
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")

    target = reference_explanation(row, base)
    if not include_reason_marker:
        target = target.replace("कारण: ", "")
    return prompt, target


class ReasoningFinetuneDataset(Dataset):
    """Memory-efficient dataset loading reasoning instances and tokenizing on the fly."""

    def __init__(
        self,
        data_path: Path,
        template_path: Path,
        tokenizer_path: Path,
        prompt_style: str = "explanation",
        include_reason_marker: bool = True,
        max_examples: int | None = None,
    ):
        self.data_path = data_path
        self.prompt_style = prompt_style
        self.include_reason_marker = include_reason_marker
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.base = json.loads(template_path.read_text(encoding="utf-8"))

        self.rows: list[dict[str, Any]] = []
        opener = gzip.open if str(data_path).endswith(".gz") else open
        with opener(str(data_path), "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
                if max_examples is not None and len(self.rows) >= max_examples:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt, target = build_training_pair(
            row,
            self.base,
            prompt_style=self.prompt_style,
            include_reason_marker=self.include_reason_marker,
        )

        prompt_ids = self.tokenizer.encode(prompt)
        full_text = prompt + " " + target
        full_ids = self.tokenizer.encode(full_text)

        # Boundary check: ensure prefix matches prompt tokens exactly
        prompt_len = len(prompt_ids)
        if full_ids[:prompt_len] != prompt_ids:
            # Fallback if boundary merges token at space
            target_ids = self.tokenizer.encode(" " + target)
            full_ids = prompt_ids + target_ids

        target_ids = full_ids[prompt_len:]
        target_ids_with_eos = target_ids + [self.tokenizer.eos_id()]
        combined_ids = prompt_ids + target_ids_with_eos

        # Causal sequence shifting:
        # inputs: combined_ids[:-1]
        # targets: combined_ids[1:], with prompt tokens masked to -100
        input_ids = combined_ids[:-1]
        labels = list(combined_ids[1:])
        # Prompt tokens in labels are indices 0 .. prompt_len - 2
        # Index prompt_len - 1 is the first token of target_ids, predicted from input_ids[prompt_len - 1]
        for i in range(prompt_len - 1):
            labels[i] = -100

        return {
            "id": row.get("id", f"sample-{index}"),
            "task_family": row.get("task_family", "unknown"),
            "input_ids": input_ids,
            "labels": labels,
            "prompt_token_count": prompt_len,
            "target_token_count": len(target_ids_with_eos),
        }


def collate_reasoning_batch(
    items: list[dict[str, Any]],
    pad_id: int = 0,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Pad variable-length input sequences and labels to the max length in batch."""
    batch_size = len(items)
    max_len = max(len(item["input_ids"]) for item in items)

    inputs = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(items):
        seq_len = len(item["input_ids"])
        inputs[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[i, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)
        attention_mask[i, :seq_len] = True

    return {
        "input_ids": inputs,
        "labels": labels,
        "attention_mask": attention_mask,
        "ids": [item["id"] for item in items],
        "active_tokens": (labels != ignore_index).sum().item(),
    }
