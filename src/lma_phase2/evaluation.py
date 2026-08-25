"""Intrinsic, generation, diversity, and attention metrics for Phase 2."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
from sacrebleu.metrics import BLEU, CHRF
from torch import Tensor

from lma_phase2.data import PackedTokenDataset
from lma_phase2.model import DecoderOnlyTransformer


@torch.no_grad()
def intrinsic_metrics(
    model: DecoderOnlyTransformer,
    dataset: PackedTokenDataset,
    device: torch.device,
    autocast_context,
) -> dict[str, float | int]:
    """Score every held-out token and calculate CE, PPL, and UTF-8 BPB."""

    model.eval()
    total_negative_log_likelihood = 0.0
    total_tokens = 0
    context = model.config.context_length
    for start in range(0, dataset.token_count - 1, context):
        stop = min(start + context + 1, dataset.token_count)
        values = np.asarray(dataset.tokens[start:stop], dtype=np.int64)
        if len(values) < 2:
            continue
        batch = torch.from_numpy(values).unsqueeze(0).to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        with autocast_context():
            loss = model(inputs, targets=targets).loss
        tokens = targets.numel()
        total_negative_log_likelihood += float(loss) * tokens
        total_tokens += tokens
    if not total_tokens:
        raise ValueError("Held-out token stream is empty")
    utf8_bytes = int(dataset.manifest.get("utf8_bytes", 0))
    if utf8_bytes <= 0:
        raise ValueError("Manifest must contain a positive utf8_bytes count for BPB")
    cross_entropy = total_negative_log_likelihood / total_tokens
    return {
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(min(cross_entropy, 50.0)),
        "bits_per_byte": total_negative_log_likelihood / (math.log(2.0) * utf8_bytes),
        "tokens_scored": total_tokens,
        "utf8_bytes": utf8_bytes,
    }


@torch.no_grad()
def generate_tokens(
    model: DecoderOnlyTransformer,
    prefix: Sequence[int],
    maximum_new_tokens: int,
    device: torch.device,
    temperature: float | None,
    generator: torch.Generator | None = None,
    eos_id: int | None = None,
) -> list[int]:
    """Generate greedily when temperature is None, otherwise sample reproducibly."""

    if not prefix:
        raise ValueError("Generation prefix cannot be empty")
    if temperature is not None and temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    generated = list(prefix)
    for _ in range(maximum_new_tokens):
        context = generated[-model.config.context_length :]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids).logits[0, -1]
        if temperature is None:
            next_token = int(torch.argmax(logits))
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = int(torch.multinomial(probabilities.cpu(), 1, generator=generator).item())
        generated.append(next_token)
        if eos_id is not None and next_token == eos_id:
            break
    return generated[len(prefix) :]


def longest_common_subsequence_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Memory-efficient dynamic programming for ROUGE-L."""

    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    hypothesis_tokens = hypothesis.split()
    reference_tokens = reference.split()
    if not hypothesis_tokens or not reference_tokens:
        return 0.0
    common = longest_common_subsequence_length(hypothesis_tokens, reference_tokens)
    precision = common / len(hypothesis_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def distinct_n(sequences: Sequence[Sequence[int]], order: int) -> float:
    ngrams = [
        tuple(sequence[index : index + order])
        for sequence in sequences
        for index in range(len(sequence) - order + 1)
    ]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def repetition_rate(sequences: Sequence[Sequence[int]], order: int = 4) -> float:
    ngrams = [
        tuple(sequence[index : index + order])
        for sequence in sequences
        for index in range(len(sequence) - order + 1)
    ]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated_occurrences = sum(count - 1 for count in counts.values() if count > 1)
    return repeated_occurrences / len(ngrams)


def generation_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str],
    generated_token_ids: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Compute corpus BLEU-4, chrF++, mean ROUGE-L, and diversity."""

    if len(hypotheses) != len(references) or len(hypotheses) != len(generated_token_ids):
        raise ValueError("Hypotheses, references, and token sequences must align")
    bleu = BLEU(max_ngram_order=4, tokenize="none", effective_order=True)
    chrf = CHRF(char_order=6, word_order=2, beta=2)
    return {
        "bleu_4": bleu.corpus_score(list(hypotheses), [list(references)]).score,
        "chrf_plus_plus": chrf.corpus_score(list(hypotheses), [list(references)]).score,
        "rouge_l_f1": sum(
            rouge_l_f1(hypothesis, reference)
            for hypothesis, reference in zip(hypotheses, references, strict=True)
        )
        / max(1, len(hypotheses)),
        "distinct_1": distinct_n(generated_token_ids, 1),
        "distinct_2": distinct_n(generated_token_ids, 2),
        "repeated_4gram_rate": repetition_rate(generated_token_ids, 4),
    }


def attention_summary(attentions: Sequence[Tensor]) -> list[dict[str, float | int]]:
    """Return mean causal attention entropy and distance for every layer/head."""

    summaries = []
    for layer_index, layer in enumerate(attentions):
        probabilities = layer.detach().float()
        sequence_length = probabilities.shape[-1]
        query_positions = torch.arange(sequence_length, device=probabilities.device)[:, None]
        key_positions = torch.arange(sequence_length, device=probabilities.device)[None, :]
        distance = (query_positions - key_positions).clamp_min(0).float()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        mean_distance = (probabilities * distance).sum(dim=-1)
        for head_index in range(probabilities.shape[1]):
            summaries.append(
                {
                    "layer": layer_index,
                    "head": head_index,
                    "mean_entropy": float(entropy[:, head_index].mean()),
                    "mean_attention_distance": float(mean_distance[:, head_index].mean()),
                }
            )
    return summaries
