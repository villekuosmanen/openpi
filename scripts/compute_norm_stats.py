"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

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


def _serialize_running_stats(s: normalize.RunningStats) -> dict:
    return {
        "_count": s._count,
        "_mean": s._mean,
        "_mean_of_squares": s._mean_of_squares,
        "_min": s._min,
        "_max": s._max,
        "_histograms": s._histograms,
        "_bin_edges": s._bin_edges,
    }


def _deserialize_running_stats(saved: dict) -> normalize.RunningStats:
    s = normalize.RunningStats()
    s._count = saved["_count"]
    s._mean = saved["_mean"]
    s._mean_of_squares = saved["_mean_of_squares"]
    s._min = saved["_min"]
    s._max = saved["_max"]
    s._histograms = saved["_histograms"]
    s._bin_edges = saved["_bin_edges"]
    return s


def _save_checkpoint(
    checkpoint_path: str,
    batch_idx: int,
    stats: dict | None = None,
    per_dim_stats: dict | None = None,
    use_per_dim: bool = False,
):
    import pickle
    state = {"batch_idx": batch_idx, "use_per_dim": use_per_dim}
    if use_per_dim and per_dim_stats:
        state["per_dim_stats"] = {
            key: [_serialize_running_stats(s) for s in dim_list]
            for key, dim_list in per_dim_stats.items()
        }
    elif stats:
        state["stats"] = {
            key: _serialize_running_stats(s) for key, s in stats.items()
        }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(state, f)


def _load_checkpoint(checkpoint_path: str, keys: list[str]):
    """Load checkpoint. Returns (stats, per_dim_stats, use_per_dim, batch_idx) or None."""
    import pickle
    import os
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        use_per_dim = state.get("use_per_dim", False)
        batch_idx = state["batch_idx"]

        if use_per_dim and "per_dim_stats" in state:
            per_dim_stats = {}
            for key in keys:
                per_dim_stats[key] = [
                    _deserialize_running_stats(d) for d in state["per_dim_stats"][key]
                ]
            n_dims = len(per_dim_stats[keys[0]])
            counts = [per_dim_stats[keys[0]][d]._count for d in range(n_dims)]
            print(f"Resumed per-dim checkpoint at batch {batch_idx} "
                  f"(dim counts: {counts[0]:,}..{counts[-1]:,})")
            return None, per_dim_stats, True, batch_idx
        elif "stats" in state:
            stats = {}
            for key in keys:
                stats[key] = _deserialize_running_stats(state["stats"][key])
            print(f"Resumed checkpoint at batch {batch_idx} "
                  f"({stats[keys[0]]._count:,} samples)")
            return stats, None, False, batch_idx

        return None
    except Exception as e:
        print(f"Could not load checkpoint: {e}, starting fresh")
        return None


def main(
    config_name: str,
    max_frames: int | None = None,
    checkpoint_interval: int = 5000,
):
    config = _config.get_config(config_name)
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
    checkpoint_path = f"/tmp/norm_stats_checkpoint_{config_name}.pkl"

    # Per-dim stats: each dimension gets its own RunningStats so that
    # single-arm data (7 real dims) contributes to dims 0-6 while
    # bimanual data (14 real dims) contributes to all 14.
    # We always use per-dim mode to handle mixed configs correctly,
    # even if the first N batches happen to all be the same dim.
    use_per_dim = True
    per_dim_stats: dict[str, list[normalize.RunningStats]] = {}
    stats: dict = {}  # unused but kept for checkpoint compat
    start_batch = 0

    resumed = _load_checkpoint(checkpoint_path, keys)
    if resumed is not None:
        r_stats, r_per_dim, r_use_per_dim, start_batch = resumed
        if r_per_dim:
            per_dim_stats = r_per_dim
        # If resuming from old-format checkpoint, discard and restart
        if r_stats and not r_per_dim:
            print("Old-format checkpoint found — discarding and restarting with per-dim stats")
            start_batch = 0

    for batch_idx, batch in enumerate(
        tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats",
                  initial=start_batch)
    ):
        if batch_idx < start_batch:
            continue

        dim_mask = batch.get("action_dim_mask")

        # Initialize per-dim stats on first batch (need to know ndim)
        if not per_dim_stats:
            for key in keys:
                arr = np.asarray(batch[key])
                ndim = arr.shape[-1]
                per_dim_stats[key] = [normalize.RunningStats() for _ in range(ndim)]
                tqdm.tqdm.write(f"  Per-dim stats initialized for '{key}' ({ndim} dims)")

        for key in keys:
            arr = np.asarray(batch[key])
            flat = arr.reshape(-1, arr.shape[-1])

            if dim_mask is not None:
                mask = np.asarray(dim_mask)
                if arr.ndim == 3:
                    mask_expanded = np.repeat(mask, arr.shape[1], axis=0)
                else:
                    mask_expanded = mask
                for d in range(flat.shape[-1]):
                    real_vals = flat[mask_expanded[:, d], d:d+1]
                    if len(real_vals) > 0:
                        per_dim_stats[key][d].update(real_vals)
            else:
                # No mask = all dims real (bimanual data)
                for d in range(flat.shape[-1]):
                    per_dim_stats[key][d].update(flat[:, d:d+1])

        if (batch_idx + 1) % checkpoint_interval == 0:
            _save_checkpoint(
                checkpoint_path, batch_idx + 1,
                stats=stats, per_dim_stats=per_dim_stats,
                use_per_dim=use_per_dim,
            )
            tqdm.tqdm.write(f"  Checkpoint saved at batch {batch_idx + 1}")

    if use_per_dim:
        norm_stats = {}
        for key in keys:
            dim_stats_list = per_dim_stats[key]
            per_dim_results = [ds.get_statistics() for ds in dim_stats_list]
            norm_stats[key] = normalize.NormStats(
                mean=np.array([s.mean for s in per_dim_results]).squeeze(-1),
                std=np.array([s.std for s in per_dim_results]).squeeze(-1),
                q01=np.array([s.q01 for s in per_dim_results]).squeeze(-1),
                q99=np.array([s.q99 for s in per_dim_results]).squeeze(-1),
            )
            print(f"\n{key} per-dim stats:")
            for d, s in enumerate(per_dim_results):
                count = dim_stats_list[d]._count
                print(f"  [{d:2d}] mean={s.mean[0]:+.4f} std={s.std[0]:.4f} "
                      f"q01={s.q01[0]:+.4f} q99={s.q99[0]:+.4f} count={count:,}")
    else:
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    # Clean up checkpoint
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint cleaned up")


if __name__ == "__main__":
    tyro.cli(main)
