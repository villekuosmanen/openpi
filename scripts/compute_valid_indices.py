"""Compute valid sample indices for a config and save to a text file.

Instead of iterating over every sample (slow — requires decoding images),
this script reads episode-level metadata directly from the plugins:

  - EpisodeOutcomePlugin: outcomes.json  → filter to successful episodes
  - ControlModePlugin:    episode_modes.json → filter out autonomous frames

The result is a comma-separated list of global indices written to
``config.assets_dirs / valid_indices.txt``.
"""

import dataclasses
import json
import logging
import pathlib

import tqdm_loggable.auto as tqdm
import tyro

from robocandywrapper.factory import make_dataset_without_config
from robocandywrapper.plugins import EpisodeOutcomePlugin, ControlModePlugin

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def compute_valid_indices(dataset) -> list[int]:
    """Compute valid indices from plugin metadata without loading samples.

    A frame is valid when:
      1. Its episode has outcome == "success" (unlabeled episodes are included
         if the dataset has no outcomes.json at all, excluded otherwise).
      2. The frame is NOT in a "policy" control-mode segment.
    """
    valid: list[int] = []

    for ds_idx, ds in enumerate(dataset._datasets):
        # Resolve the global offset for this sub-dataset.
        global_offset = dataset._cumulative_lengths[ds_idx]
        index_map = dataset._index_maps[ds_idx]

        # ── Episode outcome metadata ────────────────────────────────
        from robocandywrapper.plugins.episode_outcome import EpisodeOutcomeInstance
        from robocandywrapper.plugins.control_mode import ControlModeInstance

        outcome_instance = None
        control_instance = None
        for pi in dataset._plugin_instances[ds_idx]:
            if isinstance(pi, EpisodeOutcomeInstance):
                outcome_instance = pi
            if isinstance(pi, ControlModeInstance):
                control_instance = pi

        outcomes = outcome_instance.outcomes if outcome_instance else {}
        has_outcomes = bool(outcomes)

        episode_modes = control_instance.episode_modes if control_instance else {}

        # ── Episode data index (frame ranges per episode) ───────────
        if hasattr(ds, 'episode_data_index'):
            ep_data_index = ds.episode_data_index
        else:
            import torch
            ep_from, ep_to = [], []
            current_ep = None
            for frame_idx, ep_i in enumerate(ds.hf_dataset["episode_index"]):
                if ep_i != current_ep:
                    ep_from.append(frame_idx)
                    if current_ep is not None:
                        ep_to.append(frame_idx)
                    current_ep = ep_i
            ep_to.append(frame_idx + 1)
            ep_data_index = {"from": torch.tensor(ep_from), "to": torch.tensor(ep_to)}

        num_episodes = len(ep_data_index["from"])
        total_frames = int(ep_data_index["to"][-1]) if num_episodes > 0 else 0
        ds_valid_count = 0
        eps_skipped_outcome = 0
        frames_removed_autonomous = 0
        eps_with_control_data = 0

        for ep_idx in range(num_episodes):
            ep_from = int(ep_data_index["from"][ep_idx])
            ep_to = int(ep_data_index["to"][ep_idx])
            ep_len = ep_to - ep_from

            # Filter 1: episode outcome
            if has_outcomes:
                outcome = outcomes.get(ep_idx)
                if outcome != "success":
                    eps_skipped_outcome += 1
                    continue

            # Filter 2: control mode — collect human-controlled frame ranges
            segments = episode_modes.get(ep_idx)
            if segments is None:
                human_frames = set(range(ep_len))
            else:
                eps_with_control_data += 1
                human_frames = set(range(ep_len))
                for seg in segments:
                    if seg.mode == "policy":
                        policy_frames = set(range(seg.start_index, seg.end_index + 1))
                        removed = human_frames & policy_frames
                        frames_removed_autonomous += len(removed)
                        human_frames -= removed

            # Map local dataset indices → global WrappedRobotDataset indices
            for frame_in_ep in sorted(human_frames):
                local_idx = ep_from + frame_in_ep
                if index_map is not None:
                    try:
                        virtual_idx = index_map.index(local_idx)
                    except ValueError:
                        continue
                    global_idx = global_offset + virtual_idx
                else:
                    global_idx = global_offset + local_idx
                valid.append(global_idx)
                ds_valid_count += 1

        print(
            f"  [{ds.repo_id}] {ds_valid_count}/{total_frames} frames valid | "
            f"{num_episodes} episodes, {eps_skipped_outcome} skipped (bad outcome) | "
            f"control-mode: {eps_with_control_data} eps with data, "
            f"{frames_removed_autonomous} autonomous frames removed"
        )

    return valid


def main(
    config_name: str, assets_base_dir: str = "./assets",
) -> None:
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, assets_base_dir=assets_base_dir)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id to compute valid indices.")

    repo_id = data_config.repo_id
    if repo_id.endswith(".json") and pathlib.Path(repo_id).exists():
        with open(repo_id) as f:
            repo_id = json.load(f)
        logging.info("Loaded %d repo_ids from JSON file %s", len(repo_id), data_config.repo_id)
    else:
        logging.info("Loading dataset for repo_id=%s", repo_id)

    dataset = make_dataset_without_config(
        repo_id,
        plugins=[
            EpisodeOutcomePlugin(),
            ControlModePlugin(),
        ],
        load_videos=False,
    )

    n = len(dataset)
    print(f"\nComputing valid indices over {n} items from {len(dataset._datasets)} datasets...\n")
    valid = compute_valid_indices(dataset)
    print(f"\n=> {len(valid)}/{n} valid indices ({100 * len(valid) / max(n, 1):.1f}%)")

    output_dir = pathlib.Path(config.assets_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _data_loader.VALID_INDICES_FILENAME
    output_path.write_text(",".join(str(i) for i in valid))
    print(f"=> Wrote to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
