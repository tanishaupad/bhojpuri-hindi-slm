import torch

from lma_phase2.config import ModelConfig
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase3.evaluate import greedy_batch, next_token_logits, normalize_answer, score_prediction


def test_last_position_projection_matches_full_model():
    torch.manual_seed(12)
    model = DecoderOnlyTransformer(
        ModelConfig(
            vocabulary_size=23,
            context_length=32,
            model_dimension=16,
            number_of_layers=2,
            number_of_heads=4,
            feed_forward_dimension=32,
            dropout=0,
            attention_dropout=0,
            embedding_dropout=0,
        )
    ).eval()
    tokens = torch.randint(0, 23, (3, 7))
    torch.testing.assert_close(next_token_logits(model, tokens), model(tokens).logits[:, -1])
    prefixes = tokens.tolist()
    batch = greedy_batch(model, prefixes, 3, "cpu", 5)
    singles = [greedy_batch(model, [p], 3, "cpu", 5)[0] for p in prefixes]
    assert batch == singles


def test_full_continuation_exact_match_does_not_extract_answer():
    row = {
        "task_family": "direct",
        "answer_text": "राम",
        "problem": {"entities": ["राम", "श्याम"], "query": [0, 1]},
    }
    assert score_prediction(row, "  राम  ", {})["correct"]
    assert not score_prediction(row, "राम है", {})["correct"]
    assert score_prediction(row, "श्याम", {})["error_category"] == "wrong_canonical_answer"
    assert score_prediction(row, "", {})["error_category"] == "empty_output"
    assert normalize_answer("क\u093c") == normalize_answer("क़")


def test_eos_stops_each_sequence_individually(monkeypatch):
    import lma_phase3.evaluate as module

    calls = 0

    def fake_logits(model, tokens):
        nonlocal calls
        calls += 1
        result = torch.zeros((2, 4))
        result[0, 3] = 10
        result[1, 3 if calls == 2 else 2] = 10
        return result

    class Dummy:
        config = type("Config", (), {"context_length": 32})()

        def eval(self):
            return self

    monkeypatch.setattr(module, "next_token_logits", fake_logits)
    assert greedy_batch(Dummy(), [[1], [2]], 3, "cpu", 5) == [[3], [2, 3]]
    assert calls == 2
