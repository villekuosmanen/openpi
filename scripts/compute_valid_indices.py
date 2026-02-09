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
from rewact_tools import ControlModePlugin, PiStar0_6CumulativeRewardPlugin

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main(config_name: str, assets_base_dir: str | None = None) -> None:
    config = _config.get_config(config_name)
    if assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=assets_base_dir)
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
    valid: list[int] = []
    for i in tqdm.tqdm(range(n), desc="Computing valid indices", total=n):
        item = dataset[i]
        if item["episode_outcome"] == 1 and item["control_mode_autonomous"] == 0:
            valid.append(i)
    logging.info("Computed %d valid indices (of %d total).", len(valid), n)

    output_dir = pathlib.Path(config.assets_dirs) / data_config.repo_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _data_loader.VALID_INDICES_FILENAME
    output_path.write_text(",".join(str(i) for i in valid))
    logging.info("Wrote valid indices to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
