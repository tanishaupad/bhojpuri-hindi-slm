from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytest.importorskip("sacrebleu")

from lma_phase2.evaluation import (
    attention_summary,
    distinct_n,
    generation_metrics,
    repetition_rate,
    rouge_l_f1,
)


def test_text_and_diversity_metrics_have_expected_extremes() -> None:
    assert rouge_l_f1("एक दो तीन", "एक दो तीन") == 1.0
    assert rouge_l_f1("एक दो", "तीन चार") == 0.0
    assert distinct_n([[1, 2, 1, 2]], 1) == 0.5
    assert distinct_n([[1, 2, 1, 2]], 2) == 2 / 3
    assert repetition_rate([[1, 2, 3, 4, 1, 2, 3, 4]], 4) > 0


def test_generation_metrics_score_identical_text_highly() -> None:
    metrics = generation_metrics(
        ["ई एगो पूरा वाक्य हवे"],
        ["ई एगो पूरा वाक्य हवे"],
        [[1, 2, 3, 4, 5]],
    )

    assert metrics["bleu_4"] == pytest.approx(100.0)
    assert metrics["chrf_plus_plus"] == pytest.approx(100.0)
    assert metrics["rouge_l_f1"] == 1.0


def test_attention_summary_measures_causal_distance() -> None:
    probabilities = torch.zeros(1, 1, 3, 3)
    probabilities[0, 0, 0, 0] = 1
    probabilities[0, 0, 1, 0] = 1
    probabilities[0, 0, 2, 0] = 1

    summary = attention_summary([probabilities])[0]

    assert summary["mean_entropy"] == 0.0
    assert summary["mean_attention_distance"] == 1.0
