from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lma_phase2.data import PackedTokenDataset


def make_dataset(tmp_path, token_count: int = 101) -> PackedTokenDataset:
    token_path = tmp_path / "train.tokens.bin"
    values = np.arange(token_count, dtype="<u2")
    values.tofile(token_path)
    manifest_path = tmp_path / "train.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "token_file": token_path.name,
                "dtype": "<u2",
                "token_count": token_count,
            }
        ),
        encoding="utf-8",
    )
    return PackedTokenDataset(manifest_path)


def test_random_batch_is_reproducible_and_shifted(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    first_generator = torch.Generator().manual_seed(9)
    second_generator = torch.Generator().manual_seed(9)

    first_inputs, first_targets = dataset.random_batch(4, 8, first_generator, torch.device("cpu"))
    second_inputs, second_targets = dataset.random_batch(
        4, 8, second_generator, torch.device("cpu")
    )

    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_targets, second_targets)
    assert torch.equal(first_inputs[:, 1:], first_targets[:, :-1])


def test_sequential_batches_do_not_overlap_examples(tmp_path) -> None:
    dataset = make_dataset(tmp_path, token_count=41)

    batches = list(dataset.sequential_batches(2, 5, torch.device("cpu")))

    assert len(batches) == 4
    assert batches[0][0][0].tolist() == [0, 1, 2, 3, 4]
    assert batches[0][0][1].tolist() == [5, 6, 7, 8, 9]
    assert batches[-1][1][-1].tolist() == [36, 37, 38, 39, 40]
