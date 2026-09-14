"""Student-implemented decoder-only Transformer language model in basic PyTorch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lma_phase2.config import ModelConfig


@dataclass(slots=True)
class LanguageModelOutput:
    logits: Tensor
    loss: Tensor | None = None
    attentions: tuple[Tensor, ...] | None = None


class MultiHeadCausalSelfAttention(nn.Module):
    """Scaled dot-product attention with an explicit future-token mask."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.number_of_heads = config.number_of_heads
        self.head_dimension = config.head_dimension
        self.scale = self.head_dimension**-0.5
        self.query_key_value = nn.Linear(
            config.model_dimension,
            3 * config.model_dimension,
            bias=config.use_bias,
        )
        self.output_projection = nn.Linear(
            config.model_dimension,
            config.model_dimension,
            bias=config.use_bias,
        )
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        future_mask = torch.triu(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("future_mask", future_mask, persistent=False)

    def forward(
        self, hidden_states: Tensor, return_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        batch_size, sequence_length, model_dimension = hidden_states.shape
        query, key, value = self.query_key_value(hidden_states).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.number_of_heads,
                self.head_dimension,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            self.future_mask[:sequence_length, :sequence_length],
            torch.finfo(scores.dtype).min,
        )
        probabilities = F.softmax(scores, dim=-1)
        attended = torch.matmul(self.attention_dropout(probabilities), value)
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, model_dimension)
        )
        output = self.output_dropout(self.output_projection(attended))
        return output, probabilities if return_attention else None


class FeedForward(nn.Module):
    """Position-wise expansion, GELU activation, and projection."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_projection = nn.Linear(
            config.model_dimension,
            config.feed_forward_dimension,
            bias=config.use_bias,
        )
        self.output_projection = nn.Linear(
            config.feed_forward_dimension,
            config.model_dimension,
            bias=config.use_bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.dropout(self.output_projection(F.gelu(self.input_projection(hidden_states))))


class TransformerBlock(nn.Module):
    """Pre-normalized causal attention and feed-forward residual block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dimension)
        self.attention = MultiHeadCausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.model_dimension)
        self.feed_forward = FeedForward(config)

    def forward(
        self, hidden_states: Tensor, return_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        attention_output, probabilities = self.attention(
            self.attention_norm(hidden_states), return_attention=return_attention
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))
        return hidden_states, probabilities


class DecoderOnlyTransformer(nn.Module):
    """GPT-style monolingual language model with tied token/output weights."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocabulary_size, config.model_dimension)
        self.position_embeddings = nn.Embedding(config.context_length, config.model_dimension)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.number_of_layers)
        )
        self.final_norm = nn.LayerNorm(config.model_dimension)
        self.language_model_head = nn.Linear(
            config.model_dimension, config.vocabulary_size, bias=False
        )
        if config.tie_embeddings:
            self.language_model_head.weight = self.token_embeddings.weight
        self.apply(self._initialize_module)
        residual_standard_deviation = 0.02 / math.sqrt(2 * config.number_of_layers)
        for block in self.blocks:
            nn.init.normal_(
                block.attention.output_projection.weight,
                mean=0.0,
                std=residual_standard_deviation,
            )
            nn.init.normal_(
                block.feed_forward.output_projection.weight,
                mean=0.0,
                std=residual_standard_deviation,
            )

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        return_attentions: bool = False,
    ) -> LanguageModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds learned-position limit "
                f"{self.config.context_length}"
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.embedding_dropout(
            self.token_embeddings(input_ids) + self.position_embeddings(positions)[None, :, :]
        )
        attention_layers: list[Tensor] = []
        for block in self.blocks:
            hidden_states, attention = block(hidden_states, return_attention=return_attentions)
            if attention is not None:
                attention_layers.append(attention)
        logits = self.language_model_head(self.final_norm(hidden_states))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, self.config.vocabulary_size),
                targets.reshape(batch_size * sequence_length),
                ignore_index=-100,
            )
        return LanguageModelOutput(
            logits=logits,
            loss=loss,
            attentions=tuple(attention_layers) if return_attentions else None,
        )

    def parameter_count(self) -> int:
        """Return the number of unique trainable scalar parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
