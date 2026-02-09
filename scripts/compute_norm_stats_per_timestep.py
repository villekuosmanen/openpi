"""Compute normalization statistics with per-timestep action stats.

This script computes global normalization stats for state/actions (same as
scripts/compute_norm_stats.py) and additionally computes per-timestep action
stats saved to a parallel file.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _stack_norm_stats(stats_by_timestep: list[normalize.NormStats]) -> normalize.NormStats:
    q01 = None if stats_by_timestep[0].q01 is None else np.stack([stats.q01 for stats in stats_by_timestep])
    q99 = None if stats_by_timestep[0].q99 is None else np.stack([stats.q99 for stats in stats_by_timestep])
    return normalize.NormStats(
        mean=np.stack([stats.mean for stats in stats_by_timestep]),
        std=np.stack([stats.std for stats in stats_by_timestep]),
        q01=q01,
        q99=q99,
    )


def _update_per_timestep_stats(
    stats_by_timestep: list[normalize.RunningStats],
    actions: np.ndarray,
    action_horizon: int,
) -> None:
    if actions.ndim < 2:
        raise ValueError("Expected actions to have at least 2 dimensions (horizon, action_dim).")
    if actions.shape[-2] != action_horizon:
        raise ValueError(
            f"Expected actions horizon {action_horizon}, got {actions.shape[-2]}. "
            "Make sure action_horizon matches the dataset."
        )
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    for t in range(action_horizon):
        stats_by_timestep[t].update(actions[:, t, :])


def main(config_name: str, max_frames: int | None = None, assets_base_dir: str | None = None):
    config = _config.get_config(config_name)
    if assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=assets_base_dir)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    per_timestep_stats = [normalize.RunningStats() for _ in range(config.model.action_horizon)]

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))
        _update_per_timestep_stats(per_timestep_stats, np.asarray(batch["actions"]), config.model.action_horizon)

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    per_timestep_action_stats = _stack_norm_stats([stats.get_statistics() for stats in per_timestep_stats])

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing global stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print(f"Writing per-timestep action stats to: {output_path}")
    normalize.save_actions_per_timestep(output_path, per_timestep_action_stats)


if __name__ == "__main__":
    tyro.cli(main)
