"""Memory-mapped packed-token batches for deterministic language-model training."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class PackedTokenDataset:
    """Read fixed-size causal examples from a compact token stream."""

    def __init__(self, manifest_path: Path):
        with manifest_path.open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.path = manifest_path.parent / self.manifest["token_file"]
        self.dtype = np.dtype(self.manifest["dtype"])
        self.token_count = int(self.manifest["token_count"])
        expected_bytes = self.token_count * self.dtype.itemsize
        if self.path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Token file size mismatch: expected {expected_bytes}, got {self.path.stat().st_size}"
            )
        self.tokens = np.memmap(self.path, mode="r", dtype=self.dtype)

    def validate_context(self, context_length: int) -> None:
        if self.token_count < context_length + 1:
            raise ValueError("Token stream is shorter than one input-target window")

    def random_batch(
        self,
        batch_size: int,
        context_length: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Sample reproducible random windows using the supplied RNG state."""

        self.validate_context(context_length)
        maximum_start = self.token_count - context_length - 1
        starts = torch.randint(
            0,
            maximum_start + 1,
            (batch_size,),
            generator=generator,
        ).numpy()
        offsets = starts[:, None] + np.arange(context_length + 1)[None, :]
        values = np.asarray(self.tokens[offsets], dtype=np.int64)
        batch = torch.from_numpy(values).to(device=device, non_blocking=True)
        return batch[:, :-1], batch[:, 1:]

    def sequential_batches(
        self,
        batch_size: int,
        context_length: int,
        device: torch.device,
        maximum_batches: int | None = None,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield deterministic non-overlapping held-out windows."""

        self.validate_context(context_length)
        examples = (self.token_count - 1) // context_length
        if maximum_batches is not None:
            examples = min(examples, maximum_batches * batch_size)
        for start_example in range(0, examples, batch_size):
            current_batch = min(batch_size, examples - start_example)
            starts = np.arange(start_example, start_example + current_batch) * context_length
            offsets = starts[:, None] + np.arange(context_length + 1)[None, :]
            values = np.asarray(self.tokens[offsets], dtype=np.int64)
            batch = torch.from_numpy(values).to(device=device, non_blocking=True)
            yield batch[:, :-1], batch[:, 1:]
