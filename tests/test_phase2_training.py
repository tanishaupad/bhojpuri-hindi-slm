from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from lma_phase2.config import (
    ExperimentConfig,
    IdentityConfig,
    ModelConfig,
    TrainingConfig,
)
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase2.training import (
    configure_optimizer,
    configure_scheduler,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)


def experiment() -> ExperimentConfig:
    return ExperimentConfig(
        identity=IdentityConfig("tiny", "bho", "Bhojpuri", "tokenizer.model"),
        model=ModelConfig(
            vocabulary_size=50,
            context_length=8,
            model_dimension=16,
            number_of_layers=1,
            number_of_heads=4,
            feed_forward_dimension=32,
            dropout=0.0,
            attention_dropout=0.0,
            embedding_dropout=0.0,
        ),
        training=TrainingConfig(
            seed=3,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            maximum_steps=10,
            learning_rate=3e-4,
            minimum_learning_rate=3e-5,
            warmup_steps=2,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            gradient_clip_norm=1.0,
            validation_interval=2,
            validation_batches=1,
            checkpoint_interval=2,
            log_interval=1,
        ),
    )


def one_update(model, optimizer, scheduler) -> None:
    inputs = torch.randint(0, 50, (2, 8))
    loss = model(inputs, targets=torch.roll(inputs, shifts=-1, dims=1)).loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def test_checkpoint_round_trip_restores_all_training_state(tmp_path) -> None:
    config = experiment()
    model = DecoderOnlyTransformer(config.model)
    optimizer = configure_optimizer(model, config.training)
    scheduler = configure_scheduler(optimizer, config.training)
    generator = torch.Generator().manual_seed(99)
    one_update(model, optimizer, scheduler)
    expected_state = copy.deepcopy(model.state_dict())
    expected_next_random = torch.randint(0, 1000, (5,), generator=generator)
    generator.manual_seed(99)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler=None,
        step=1,
        tokens_seen=16,
        best_validation_loss=4.2,
        experiment=config,
        data_generator=generator,
    )
    restored_model = DecoderOnlyTransformer(config.model)
    restored_optimizer = configure_optimizer(restored_model, config.training)
    restored_scheduler = configure_scheduler(restored_optimizer, config.training)
    restored_generator = torch.Generator()
    checkpoint = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        scaler=None,
        experiment=config,
        data_generator=restored_generator,
        device=torch.device("cpu"),
    )

    assert checkpoint["training_step"] == 1
    assert checkpoint["tokens_seen"] == 16
    assert checkpoint["best_validation_loss"] == 4.2
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected_state[name])
    assert torch.equal(
        torch.randint(0, 1000, (5,), generator=restored_generator),
        expected_next_random,
    )
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_prune_checkpoints_keeps_newest_recovery_points(tmp_path) -> None:
    paths = [tmp_path / f"checkpoint_step_{step:07d}.pt" for step in range(1, 6)]
    for path in paths:
        path.write_bytes(b"checkpoint")

    removed = prune_checkpoints(tmp_path, keep_last=3)

    assert removed == paths[:2]
    assert [path.exists() for path in paths] == [False, False, True, True, True]
