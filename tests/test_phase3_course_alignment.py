"""Regression checks for retention metrics and explicit-answer evaluation."""

import json
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lma_phase2.config import ModelConfig
from lma_phase2.data import PackedTokenDataset
from lma_phase2.evaluation import intrinsic_metrics
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase3.evaluate_retention import evaluate_stream
from lma_phase3.explicit_answer import extract_explicit_answer, score_explicit_answer
from lma_phase3.finetune import evaluate_validation
from lma_phase3.finetune_data import collate_reasoning_batch
from lma_phase3.rescore_explicit_answers import rescore_rows


@pytest.mark.parametrize("text,expected", [
    ("अशोक। कारण: सुमित की उम्र 25 साल है।", "अशोक"),
    ("जवाब: रामनाथ। कारण: गणेश के लंबाई ढेर बा।", "रामनाथ"),
    ("उत्तर: हँ।", "हँ"),
    ("ना", "ना"),
    ("कारण: अशोक की उम्र सबसे कम है।", ""),
    ("अशोक और सुमित। कारण: दोनों का उल्लेख है।", "अशोक और सुमित"),
    ("अशोक। सुमित। कारण: विरोधी उत्तर।", "अशोक। सुमित"),
    ("अशोक\nउसकी उम्र कम है।", "अशोक"),
])
def test_explicit_field_boundaries(text, expected):
    assert extract_explicit_answer(text) == expected


def test_correct_name_in_premise_cannot_rescue_wrong_answer():
    result = score_explicit_answer(
        "बनवारी। कारण: गुड्डू के लंबाई 197 सेंटीमीटर बा।", "गुड्डू",
        {"explanation_fully_verified": True},
    )
    assert not result["explicit_answer_correct"]
    assert not result["explicit_answer_and_explanation_verified"]
    correct = score_explicit_answer("गुड्डू। कारण: गलत तथ्य।", "गुड्डू",
                                    {"explanation_fully_verified": False})
    assert correct["explicit_answer_correct"]
    assert not correct["explicit_answer_and_explanation_verified"]


def test_rescore_rejects_duplicate_and_missing_predictions():
    sources = [{"id": "one", "answer_text": "अशोक"}, {"id": "two", "answer_text": "ना"}]
    with pytest.raises(ValueError, match="Duplicate"):
        rescore_rows([{"id": "one"}, {"id": "one"}], sources)
    with pytest.raises(ValueError, match="coverage"):
        rescore_rows([{"id": "one"}], sources)


def test_generation_summary_separates_explicit_answer_from_presence(monkeypatch):
    from lma_phase3 import evaluate_finetuned as evaluation

    class Tokenizer:
        def encode(self, text):
            return [1]

        def eos_id(self):
            return 3

        def decode(self, ids):
            return "बनवारी। कारण: गुड्डू के लंबाई 197 सेंटीमीटर बा।"

    monkeypatch.setattr(evaluation, "greedy_cached", lambda *args: [[2]])
    monkeypatch.setattr(evaluation, "score_row", lambda *args: {"answer_present": True})
    monkeypatch.setattr(evaluation, "score_semantics", lambda *args: {
        "supported_claim_found": True, "has_unsupported_claim": False,
        "sufficient_supported_justification": True, "explanation_fully_verified": True,
        "joint_fully_verified": False,
    })
    rows = [{"id": "one", "prompt": "प्रश्न", "answer_text": "गुड्डू", "task_family": "direct"}]
    predictions, summary = evaluation.evaluate_split(
        None, Tokenizer(), rows, {}, "original", torch.device("cpu")
    )
    assert predictions[0]["extracted_answer"] == "बनवारी"
    assert summary["answer_present_rate"] == 1
    assert summary["verified_explanation_rate"] == 1
    assert summary["explicit_answer_accuracy"] == 0
    assert summary["explicit_answer_and_explanation_verified_rate"] == 0
    assert summary["task_breakdown"]["direct"]["explicit_answer_accuracy"] == 0


@pytest.mark.parametrize("tokens", [33, 38])
def test_batched_retention_matches_original_including_partial_tail(tmp_path, tokens):
    torch.manual_seed(72)
    config = ModelConfig(vocabulary_size=41, context_length=8, model_dimension=16,
                         number_of_layers=1, number_of_heads=2, feed_forward_dimension=32,
                         dropout=0, embedding_dropout=0, attention_dropout=0)
    model = DecoderOnlyTransformer(config).eval()
    np.arange(tokens, dtype="<u2").tofile(tmp_path / "test.bin")
    path = tmp_path / "test.json"
    path.write_text(json.dumps({"token_file": "test.bin", "dtype": "<u2",
                                "token_count": tokens, "utf8_bytes": 200}))
    dataset = PackedTokenDataset(path)
    original = intrinsic_metrics(model, dataset, torch.device("cpu"), nullcontext)
    batched = evaluate_stream(model, dataset, torch.device("cpu"), batch_size=3)
    assert batched["tokens_scored"] == tokens - 1
    for key in ("cross_entropy", "perplexity", "bits_per_byte"):
        assert batched[key] == pytest.approx(original[key], rel=1e-6)


def test_full_validation_covers_grouped_tasks_and_weights_active_tokens():
    from types import SimpleNamespace

    class PredictableModel(torch.nn.Module):
        def forward(self, inputs, targets):
            return SimpleNamespace(loss=targets[targets != -100].float().mean())

    # Ordered by family, as in the real validation files; different target lengths.
    rows = [
        {"input_ids": [1] * length, "labels": [label] * length,
         "id": str(label), "task_family": family}
        for family, label, length in [("direct", 1, 1), ("extremum", 2, 2),
                                       ("equality", 3, 3), ("transitivity", 4, 4)]
    ]
    loader = DataLoader(rows, batch_size=1, collate_fn=collate_reasoning_batch)
    model = PredictableModel().train()
    full = evaluate_validation(model, loader, torch.device("cpu"), nullcontext)
    assert full["examples"] == 4
    assert full["tokens"] == 10
    assert full["loss"] == pytest.approx(3.0)
    assert model.training
    subset = evaluate_validation(model, loader, torch.device("cpu"), nullcontext, max_batches=1)
    assert subset["loss"] == 1.0
