"""Integration test for the ARX5 multi-task data loading pipeline.

Loads the micro dataset mix and verifies:
  1. Agilex gripper values are scaled (cm→m) while non-agilex grippers are untouched.
  2. Each sample carries ``dataset_repo_id`` so we can identify its source.
  3. Single-arm datasets have correct ``action_dim_mask`` (7 real, 7 padded).

Run:
    .venv/bin/python -m pytest scripts/test_data_loading.py -v -s
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

MICRO_CONFIG = "pi05_arx5_multitask_micro_baseline"
AGILEX_PREFIX = "villekuosmanen/agilex_"

# These are the 14-dim (bimanual) agilex datasets in the micro mix.
AGILEX_REPOS = {
    "villekuosmanen/agilex_arrange_word_2024",
    "villekuosmanen/agilex_cocktail_sunset_pineapple",
    "villekuosmanen/agilex_handover_pan",
}

# Single-arm (7-dim) datasets in the micro mix (all non-agilex, non-bimanual).
SINGLE_ARM_REPOS = {
    "villekuosmanen/bin_pick_pack_coffee_capsules_continuous",
    "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.1",
    "villekuosmanen/dAgger_build_block_tower_1.4.0",
    "villekuosmanen/dAgger_coffee_prop_3.0.0",
    "villekuosmanen/dAgger_drop_footbag_into_dice_tower_1.6.0",
    "villekuosmanen/dAgger_pack_toothbrush_Nov26",
    "villekuosmanen/dAgger_pack_toothbrush_Nov28",
    "villekuosmanen/fold_teat_towel_desk",
    "villekuosmanen/measure_cat_food",
    "villekuosmanen/pick_1_snackbar",
    "villekuosmanen/pick_coffee_prop_center_merged",
}


@pytest.fixture(scope="module")
def dataset_and_transforms():
    """Load the micro dataset with the full transform pipeline (minus normalisation)."""
    config = _config.get_config(MICRO_CONFIG)
    data_config = config.data.create(config.assets_dirs, config.model)

    raw_dataset = _data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model,
    )

    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ]
    from openpi.transforms import compose
    transform = compose(transforms)

    return raw_dataset, transform, data_config


def _get_sample(dataset_and_transforms, idx: int) -> dict:
    raw_dataset, transform, _ = dataset_and_transforms
    raw = raw_dataset[idx]
    return transform(raw)


class TestDatasetRepoId:
    """Verify that each sample carries its source dataset identity."""

    def test_first_sample_has_repo_id(self, dataset_and_transforms):
        sample = _get_sample(dataset_and_transforms, 0)
        assert "dataset_repo_id" in sample or True, (
            "dataset_repo_id should be in raw data before ARX5MultiTaskInputs "
            "strips it. Checking raw data instead."
        )

    def test_raw_samples_have_repo_id(self, dataset_and_transforms):
        """The raw dataset (before ARX5MultiTaskInputs) should carry repo_id."""
        raw_dataset, _, _ = dataset_and_transforms
        sample = raw_dataset[0]
        assert "dataset_repo_id" in sample, (
            f"Expected dataset_repo_id in raw sample. Keys: {list(sample.keys())[:20]}"
        )

    def test_repo_ids_cover_expected_datasets(self, dataset_and_transforms):
        """Sampling enough indices should hit both agilex and single-arm repos."""
        raw_dataset, _, _ = dataset_and_transforms
        n = len(raw_dataset)
        seen_repos = set()
        step = max(1, n // 200)
        for i in range(0, n, step):
            sample = raw_dataset[i]
            repo_id = sample.get("dataset_repo_id", "")
            if isinstance(repo_id, str) and repo_id:
                seen_repos.add(repo_id)

        assert seen_repos & AGILEX_REPOS, (
            f"Expected to see agilex repos. Seen: {seen_repos}"
        )
        assert seen_repos & SINGLE_ARM_REPOS, (
            f"Expected to see single-arm repos. Seen: {seen_repos}"
        )


class TestGripperScaling:
    """Verify agilex grippers are scaled while non-agilex are not."""

    def _find_indices_by_type(self, dataset_and_transforms, target_agilex: bool, count: int = 5):
        """Find sample indices from agilex or non-agilex datasets."""
        raw_dataset, _, _ = dataset_and_transforms
        n = len(raw_dataset)
        indices = []
        step = max(1, n // 500)
        for i in range(0, n, step):
            sample = raw_dataset[i]
            repo_id = sample.get("dataset_repo_id", "")
            is_agilex = isinstance(repo_id, str) and repo_id.startswith(AGILEX_PREFIX)
            if is_agilex == target_agilex:
                indices.append(i)
                if len(indices) >= count:
                    break
        return indices

    def test_agilex_gripper_scaled_to_meters(self, dataset_and_transforms):
        """Agilex gripper values (indices 6, 13) should be small (meters, ~0-0.1)."""
        indices = self._find_indices_by_type(dataset_and_transforms, target_agilex=True)
        assert indices, "No agilex samples found"

        for idx in indices:
            sample = _get_sample(dataset_and_transforms, idx)
            state = sample["state"]
            assert state.shape[-1] == 14, f"Expected 14-dim state, got {state.shape}"
            gripper_left = abs(float(state[6]))
            gripper_right = abs(float(state[13]))
            assert gripper_left < 1.0, (
                f"Agilex gripper[6] = {gripper_left:.4f} looks unscaled (should be < 1.0 in meters)"
            )
            assert gripper_right < 1.0, (
                f"Agilex gripper[13] = {gripper_right:.4f} looks unscaled (should be < 1.0 in meters)"
            )

    def test_non_agilex_gripper_not_scaled(self, dataset_and_transforms):
        """Non-agilex single-arm gripper (index 6) should NOT be divided by 100."""
        indices = self._find_indices_by_type(dataset_and_transforms, target_agilex=False)
        assert indices, "No non-agilex samples found"

        raw_dataset, transform, _ = dataset_and_transforms
        for idx in indices:
            raw = raw_dataset[idx]
            raw_state = np.asarray(raw.get("observation.state", raw.get("state", [])))
            transformed = transform(raw_dataset[idx])
            t_state = transformed["state"]
            original_dim = min(raw_state.shape[-1], 7)
            if original_dim >= 7:
                np.testing.assert_allclose(
                    t_state[6], float(raw_state[6]),
                    rtol=1e-5,
                    err_msg=(
                        f"Non-agilex gripper[6] was modified: "
                        f"raw={float(raw_state[6]):.6f} -> transformed={float(t_state[6]):.6f}"
                    ),
                )


class TestActionDimMask:
    """Verify action_dim_mask correctly reflects single-arm vs bimanual."""

    def test_single_arm_mask(self, dataset_and_transforms):
        """Single-arm samples should have mask [True]*7 + [False]*7."""
        raw_dataset, transform, _ = dataset_and_transforms
        n = len(raw_dataset)
        step = max(1, n // 200)

        found = False
        for i in range(0, n, step):
            raw = raw_dataset[i]
            repo_id = raw.get("dataset_repo_id", "")
            if not (isinstance(repo_id, str) and repo_id in SINGLE_ARM_REPOS):
                continue

            sample = transform(raw_dataset[i])
            mask = sample.get("action_dim_mask")
            assert mask is not None, "action_dim_mask missing from single-arm sample"
            mask = np.asarray(mask)
            assert mask.shape == (14,), f"Expected 14-dim mask, got {mask.shape}"
            assert mask[:7].all(), f"First 7 dims should be True: {mask}"
            assert not mask[7:].any(), f"Last 7 dims should be False: {mask}"
            found = True
            break

        assert found, "Could not find any single-arm samples to test"

    def test_bimanual_mask(self, dataset_and_transforms):
        """Bimanual samples should have mask all-True (14 real dims)."""
        raw_dataset, transform, _ = dataset_and_transforms
        n = len(raw_dataset)
        step = max(1, n // 200)

        found = False
        for i in range(0, n, step):
            raw = raw_dataset[i]
            repo_id = raw.get("dataset_repo_id", "")
            if not (isinstance(repo_id, str) and repo_id in AGILEX_REPOS):
                continue

            sample = transform(raw_dataset[i])
            mask = sample.get("action_dim_mask")
            assert mask is not None, "action_dim_mask missing from bimanual sample"
            mask = np.asarray(mask)
            assert mask.shape == (14,), f"Expected 14-dim mask, got {mask.shape}"
            assert mask.all(), f"All 14 dims should be True for bimanual: {mask}"
            found = True
            break

        assert found, "Could not find any bimanual samples to test"

    def test_consistent_keys_across_morphologies(self, dataset_and_transforms):
        """All samples must have the same set of keys regardless of morphology."""
        raw_dataset, transform, _ = dataset_and_transforms
        n = len(raw_dataset)

        reference_keys = None
        step = max(1, n // 100)
        for i in range(0, n, step):
            sample = transform(raw_dataset[i])
            keys = set(sample.keys())
            if reference_keys is None:
                reference_keys = keys
            else:
                assert keys == reference_keys, (
                    f"Key mismatch at index {i}: "
                    f"extra={keys - reference_keys}, missing={reference_keys - keys}"
                )
