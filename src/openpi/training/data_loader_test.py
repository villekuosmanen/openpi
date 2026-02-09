import dataclasses

import jax
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    # Expect a finite TorchDataLoader to yield exactly num_batches batches and each
    # batch to preserve the configured local batch size across all leaves.
    # Example: dataset=16, local_batch_size=4, num_batches=2 -> exactly 2 batches of 4.
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    # Expect an infinite TorchDataLoader (no num_batches limit) to keep producing
    # batches without StopIteration, even when the underlying dataset is small.
    # Example: dataset=4, local_batch_size=4 -> repeatedly returns one full batch.
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    # Expect TorchDataLoader with worker processes to still produce the requested
    # number of batches and keep the local batch size consistent across all leaves.
    # Example: dataset=10, local_batch_size=4, num_batches=2 -> 2 batches of 4.
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    # Expect create_data_loader to work end-to-end for the fake dataset config,
    # honoring the configured batch size and action tensor shapes.
    # Example: config.batch_size=4, action_horizon=50, action_dim=24 ->
    # actions shape (4, 50, 24).
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    # Expect create_data_loader to succeed for a real dataset config when norm
    # stats are skipped (so data doesn't need to be present) and produce batches
    # with the configured action shapes.
    # Example: batch_size=4, action_horizon=50, action_dim=24 ->
    # actions shape (4, 50, 24).
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def _expected_shuffled_indices(valid_indices: list[int], seed: int) -> list[int]:
    # Mirror the sampler's torch RNG-based shuffle so the expected ordering
    # matches the sampler's deterministic permutation for a given seed.
    # Example: valid_indices=[0,1,2,3], seed=7 -> deterministic permuted order.
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(valid_indices), generator=g).tolist()
    return [valid_indices[i] for i in perm]


def test_filtered_sampler_deterministic_per_epoch():
    valid_indices = list(range(10))
    sampler = _data_loader.FilteredSampler(valid_indices, shuffle=True, seed=123)

    # Epoch 0: ordering is deterministic from base seed.
    # Example: seed=123 -> same ordering every run.
    assert list(iter(sampler)) == _expected_shuffled_indices(valid_indices, 123)

    # Epoch 2: ordering is still deterministic but uses seed+epoch to reshuffle.
    # Example: seed=123, epoch=2 -> same ordering across runs, different from epoch 0.
    sampler.set_epoch(2)
    assert list(iter(sampler)) == _expected_shuffled_indices(valid_indices, 125)


def test_filtered_distributed_sampler_deterministic_per_epoch():
    valid_indices = list(range(10))
    sampler = _data_loader.FilteredDistributedSampler(
        valid_indices,
        num_replicas=2,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=7,
    )

    # Rank 0 should see its shard of the deterministic global ordering:
    # 1) shuffle with seed, 2) truncate/pad to total_size, 3) take every Nth index
    #    for this rank.
    # Example: valid_indices=0..9, num_replicas=2, rank=0 -> take even positions
    # from the global shuffled list after truncation/padding.
    expected = _expected_shuffled_indices(valid_indices, 7)
    expected = expected[: sampler.total_size]
    expected = expected[0 : sampler.total_size : sampler.num_replicas]
    assert list(iter(sampler)) == expected

    # Epoch shift should reshuffle deterministically, changing the global order
    # and therefore this rank's shard.
    # Example: epoch=3 uses seed+3, so rank 0 sees a different shard than epoch 0.
    sampler.set_epoch(3)
    expected = _expected_shuffled_indices(valid_indices, 10)
    expected = expected[: sampler.total_size]
    expected = expected[0 : sampler.total_size : sampler.num_replicas]
    assert list(iter(sampler)) == expected
