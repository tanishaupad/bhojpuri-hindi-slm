from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lma_phase2.config import ModelConfig
from lma_phase2.model import DecoderOnlyTransformer


def tiny_config(**overrides) -> ModelConfig:
    values = {
        "vocabulary_size": 101,
        "context_length": 16,
        "model_dimension": 32,
        "number_of_layers": 2,
        "number_of_heads": 4,
        "feed_forward_dimension": 96,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "embedding_dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_forward_shapes_loss_attention_and_backward() -> None:
    model = DecoderOnlyTransformer(tiny_config())
    inputs = torch.randint(0, 101, (3, 12))
    targets = torch.randint(0, 101, (3, 12))

    output = model(inputs, targets=targets, return_attentions=True)
    output.loss.backward()

    assert output.logits.shape == (3, 12, 101)
    assert output.loss.ndim == 0
    assert len(output.attentions) == 2
    assert output.attentions[0].shape == (3, 4, 12, 12)
    assert model.token_embeddings.weight.grad is not None


def test_changing_future_token_does_not_change_earlier_logits() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(tiny_config()).eval()
    first = torch.tensor([[2, 10, 11, 12, 13, 14]])
    changed_future = first.clone()
    changed_future[0, 4:] = torch.tensor([77, 78])

    first_logits = model(first).logits
    changed_logits = model(changed_future).logits

    torch.testing.assert_close(first_logits[:, :4], changed_logits[:, :4])
    assert not torch.allclose(first_logits[:, 4:], changed_logits[:, 4:])


def test_attention_probabilities_are_zero_above_causal_diagonal() -> None:
    model = DecoderOnlyTransformer(tiny_config()).eval()
    output = model(torch.randint(0, 101, (1, 8)), return_attentions=True)

    future_probabilities = output.attentions[0][0, 0].triu(diagonal=1)
    assert torch.count_nonzero(future_probabilities) == 0


def test_input_and_output_embedding_weights_are_tied() -> None:
    model = DecoderOnlyTransformer(tiny_config(tie_embeddings=True))

    assert model.token_embeddings.weight is model.language_model_head.weight
    assert model.parameter_count() == model.config.expected_parameter_count()


def test_context_limit_is_enforced() -> None:
    model = DecoderOnlyTransformer(tiny_config(context_length=4))

    with pytest.raises(ValueError, match="learned-position limit"):
        model(torch.randint(0, 101, (1, 5)))


def test_model_does_not_use_prebuilt_transformer_modules() -> None:
    model = DecoderOnlyTransformer(tiny_config())
    forbidden = (
        torch.nn.Transformer,
        torch.nn.TransformerEncoder,
        torch.nn.TransformerDecoder,
        torch.nn.MultiheadAttention,
    )

    assert not any(isinstance(module, forbidden) for module in model.modules())
