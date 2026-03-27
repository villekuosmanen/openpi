"""Compute valid sample indices for a config and save to a text file.

This script iterates over the dataset (episode_outcome=1, control_mode_autonomous=0)
and writes comma-separated indices to config.assets_dirs / data_config.repo_id / valid_indices.txt.
Training then loads this file instead of recomputing indices at startup.
"""

import dataclasses
import logging
import pathlib

import tqdm_loggable.auto as tqdm
import tyro

from robocandywrapper.factory import make_dataset_without_config
from robocandywrapper.plugins import EpisodeOutcomePlugin
from rewact_tools import PiStar0_6CumulativeRewardPlugin

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

# Use our patched ControlModePlugin that checks both legacy and new paths.
from openpi.training.data_loader import ControlModePlugin


def is_valid_frame(item: dict) -> bool:
    """Return True if a frame should be included in training.

    A frame is valid when it comes from a successful episode (episode_outcome == 1)
    AND was human-controlled, not autonomous (control_mode_autonomous == 0).
    """
    return item["episode_outcome"] == 1 and item["control_mode_autonomous"] == 0


def compute_valid_indices(dataset) -> list[int]:
    """Iterate over *dataset* and return indices of frames that pass the filter."""
    valid: list[int] = []
    n = len(dataset)
    for i in tqdm.tqdm(range(n), desc="Computing valid indices", total=n):
        item = dataset[i]
        if is_valid_frame(item):
            valid.append(i)
    return valid


def main(
    config_name: str, assets_base_dir: str | None = None, assets_dir: str | None = None
) -> None:
    config = _config.get_config(config_name)
    if assets_dir is None:
        raise ValueError("--assets-dir is required.")
    if pathlib.Path(assets_dir).name != "assets":
        raise ValueError(f"--assets-dir must end with /assets (got: {assets_dir})")
    if assets_base_dir is not None:
        raise ValueError("--assets-base-dir is not supported; use --assets-dir instead.")
    config = dataclasses.replace(config, assets_dir=assets_dir)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id to compute valid indices.")

    logging.info("Loading dataset for repo_id=%s", data_config.repo_id)
    dataset = make_dataset_without_config(
        data_config.repo_id,
        plugins=[
            EpisodeOutcomePlugin(),
            ControlModePlugin(),
            PiStar0_6CumulativeRewardPlugin(normalise=True),
        ],
        load_videos=False,
    )

    n = len(dataset)
    logging.info("Computing valid indices over %d items (episode_outcome=1, control_mode_autonomous=0).", n)
    valid = compute_valid_indices(dataset)
    logging.info("Computed %d valid indices (of %d total).", len(valid), n)

    output_dir = pathlib.Path(config.assets_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _data_loader.VALID_INDICES_FILENAME
    output_path.write_text(",".join(str(i) for i in valid))
    logging.info("Wrote valid indices to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
