"""Inference-only KV cache for the existing explicit-attention Transformer."""

from __future__ import annotations

import torch


@torch.inference_mode()
def cached_step(model, tokens, cache=None):
    """Compute last-position logits, caching each layer's projected keys and values."""

    offset = 0 if cache is None else cache[0][0].shape[2]
    length = tokens.shape[1]
    if offset + length > model.config.context_length:
        raise ValueError("Context limit exceeded")
    positions = torch.arange(offset, offset + length, device=tokens.device)
    hidden = model.embedding_dropout(
        model.token_embeddings(tokens) + model.position_embeddings(positions)[None]
    )
    updated = []
    for index, block in enumerate(model.blocks):
        attention = block.attention
        normalized = block.attention_norm(hidden)
        q, k, v = attention.query_key_value(normalized).chunk(3, dim=-1)

        def heads(x, attention=attention):
            return x.view(
                x.shape[0], length, attention.number_of_heads, attention.head_dimension
            ).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        if cache is not None:
            k = torch.cat((cache[index][0], k), dim=2)
            v = torch.cat((cache[index][1], v), dim=2)
        updated.append((k, v))
        scores = (q @ k.transpose(-2, -1)) * attention.scale
        mask = torch.arange(k.shape[2], device=tokens.device)[None, :] > positions[:, None]
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        values = attention.attention_dropout(scores.softmax(-1)) @ v
        values = values.transpose(1, 2).contiguous().view(hidden.shape)
        hidden = hidden + attention.output_dropout(attention.output_projection(values))
        hidden = hidden + block.feed_forward(block.feed_forward_norm(hidden))
    return model.language_model_head(model.final_norm(hidden[:, -1])), updated


@torch.inference_mode()
def greedy_cached(model, prefixes, eos_id, device, limit=96):
    """Decode equal-length prefixes with independent EOS stopping and no padding."""

    if not prefixes or not prefixes[0] or len({len(p) for p in prefixes}) != 1:
        raise ValueError("Expected nonempty equal-length prefixes")
    if len(prefixes[0]) + limit > model.config.context_length:
        raise ValueError("Generation exceeds context")
    model.eval()
    tokens = torch.tensor(prefixes, dtype=torch.long, device=device)
    cache = None
    active = [True] * len(prefixes)
    generated = [[] for _ in prefixes]
    for _ in range(limit):
        logits, cache = cached_step(model, tokens, cache)
        next_ids = logits.argmax(-1)
        for i, token in enumerate(next_ids.tolist()):
            if active[i]:
                generated[i].append(token)
                if token == eos_id:
                    active[i] = False
        if not any(active):
            break
        tokens = next_ids[:, None]
    return generated
