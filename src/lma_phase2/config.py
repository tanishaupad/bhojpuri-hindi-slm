"""Typed Phase 2 experiment configuration loaded from per-language TOML files."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class IdentityConfig:
    model_name: str
    language: str
    language_name: str
    tokenizer: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocabulary_size: int
    context_length: int
    model_dimension: int
    number_of_layers: int
    number_of_heads: int
    feed_forward_dimension: int
    dropout: float = 0.1
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.1
    position_encoding: str = "learned_absolute"
    normalization: str = "pre_layer_norm"
    activation: str = "gelu"
    tie_embeddings: bool = True
    use_bias: bool = True

    def validate(self) -> None:
        positive = {
            "vocabulary_size": self.vocabulary_size,
            "context_length": self.context_length,
            "model_dimension": self.model_dimension,
            "number_of_layers": self.number_of_layers,
            "number_of_heads": self.number_of_heads,
            "feed_forward_dimension": self.feed_forward_dimension,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Model dimensions must be positive: {', '.join(invalid)}")
        if self.model_dimension % self.number_of_heads:
            raise ValueError("model_dimension must be divisible by number_of_heads")
        for name in ("dropout", "attention_dropout", "embedding_dropout"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.position_encoding != "learned_absolute":
            raise ValueError("Only student-implemented learned_absolute positions are supported")
        if self.normalization != "pre_layer_norm":
            raise ValueError("Only pre_layer_norm is supported")
        if self.activation != "gelu":
            raise ValueError("Only gelu is supported")

    @property
    def head_dimension(self) -> int:
        return self.model_dimension // self.number_of_heads

    def expected_parameter_count(self) -> int:
        """Return the exact count implied by the implemented architecture."""

        d_model = self.model_dimension
        d_ff = self.feed_forward_dimension
        bias = int(self.use_bias)
        embeddings = self.vocabulary_size * d_model + self.context_length * d_model
        attention = 4 * d_model * d_model + bias * 4 * d_model
        feed_forward = 2 * d_model * d_ff + bias * (d_ff + d_model)
        two_layer_norms = 4 * d_model
        block = attention + feed_forward + two_layer_norms
        final_layer_norm = 2 * d_model
        untied_output = 0 if self.tie_embeddings else self.vocabulary_size * d_model
        return embeddings + self.number_of_layers * block + final_layer_norm + untied_output


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    maximum_steps: int
    learning_rate: float
    minimum_learning_rate: float
    warmup_steps: int
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    gradient_clip_norm: float
    validation_interval: int
    validation_batches: int
    checkpoint_interval: int
    log_interval: int
    precision: str = "auto"
    checkpoint_keep_last: int = 3

    def validate(self) -> None:
        positive = {
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "maximum_steps": self.maximum_steps,
            "learning_rate": self.learning_rate,
            "adam_epsilon": self.adam_epsilon,
            "gradient_clip_norm": self.gradient_clip_norm,
            "validation_interval": self.validation_interval,
            "validation_batches": self.validation_batches,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
            "checkpoint_keep_last": self.checkpoint_keep_last,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Training values must be positive: {', '.join(invalid)}")
        if not 0 <= self.warmup_steps < self.maximum_steps:
            raise ValueError("warmup_steps must be non-negative and below maximum_steps")
        if not 0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum_learning_rate must be between zero and learning_rate")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported precision")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    identity: IdentityConfig
    model: ModelConfig
    training: TrainingConfig

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate one language's Phase 2 TOML configuration."""

    with path.open("rb") as handle:
        value = tomllib.load(handle)
    try:
        config = ExperimentConfig(
            identity=IdentityConfig(**value["identity"]),
            model=ModelConfig(**value["model"]),
            training=TrainingConfig(**value["training"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid experiment configuration at {path}: {exc}") from exc
    config.validate()
    return config
