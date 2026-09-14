from __future__ import annotations

from pathlib import Path

import pytest

from lma_phase2.config import ModelConfig, load_experiment_config


@pytest.mark.parametrize(
    ("path", "expected_count"),
    [
        ("hindi/configs/phase2.toml", 25_285_248),
        ("hindi/configs/phase2_no_position.toml", 25_186_944),
        ("nepali/configs/phase2.toml", 24_870_912),
    ],
)
def test_phase2_configs_have_expected_parameter_budget(path, expected_count) -> None:
    config = load_experiment_config(Path(path))

    assert config.model.expected_parameter_count() == expected_count
    assert 24_000_000 <= expected_count <= 26_000_000
    assert config.model.head_dimension == 64


def test_model_dimension_must_divide_evenly_into_heads() -> None:
    config = ModelConfig(
        vocabulary_size=100,
        context_length=16,
        model_dimension=30,
        number_of_layers=2,
        number_of_heads=8,
        feed_forward_dimension=64,
    )

    with pytest.raises(ValueError, match="divisible"):
        config.validate()
