"""Tests verifying the valid indices workflow behaves as expected.

These tests document how the EpisodeOutcomePlugin and ControlModePlugin
interact with the valid indices computation, and expose several issues:

1. A clean dataset with no outcomes.json produces ZERO valid indices
   (all frames silently rejected because episode_outcome defaults to False).

2. The ControlModePlugin looks for episode_modes.json at the WRONG path:
   it checks `candywrapper_plugins/dagger_data_source/episode_modes.json`
   but the actual HuggingFace repo stores it at
   `candywrapper_plugins/control_mode/episode_modes.json`.

3. Even when outcomes.json exists (all success), if control_mode metadata
   is in the wrong directory, control_mode defaults to "unknown" and
   control_mode_autonomous=False -- which PASSES the filter (== 0).
   So the control_mode filter is effectively a no-op.

Run:
    pytest scripts/valid_indices_test.py -v -s

For integration tests with real datasets:
    pytest scripts/valid_indices_test.py -v -s -k "TestRealDataset"
"""

import json
import os
import pathlib
import sys
import tempfile

import pytest
import torch

# ---------------------------------------------------------------------------
# Import the real filter logic from the compute script — no reimplementations.
# ---------------------------------------------------------------------------

# Ensure scripts/ is importable regardless of working directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from compute_valid_indices import is_valid_frame


def _compute_valid_indices_from_items(items: list[dict]) -> list[int]:
    """Apply the real is_valid_frame filter to a list of dicts."""
    return [i for i, item in enumerate(items) if is_valid_frame(item)]


# ===================================================================
# 1.  Pure filter-logic tests (no dataset or plugins required)
# ===================================================================

class TestFilterLogic:
    """Unit tests for the two-condition filter used by compute_valid_indices."""

    def test_all_success_teleop_passes(self):
        """All frames are success + teleop -> every index is valid."""
        items = [
            {"episode_outcome": True, "control_mode_autonomous": False}
            for _ in range(10)
        ]
        assert _compute_valid_indices_from_items(items) == list(range(10))

    def test_all_failure_rejected(self):
        """All frames have episode_outcome=False -> zero valid indices."""
        items = [
            {"episode_outcome": False, "control_mode_autonomous": False}
            for _ in range(10)
        ]
        assert _compute_valid_indices_from_items(items) == []

    def test_all_autonomous_rejected(self):
        """All frames have control_mode_autonomous=True -> zero valid indices."""
        items = [
            {"episode_outcome": True, "control_mode_autonomous": True}
            for _ in range(10)
        ]
        assert _compute_valid_indices_from_items(items) == []

    def test_mixed_episodes(self):
        """Only frames that are both success AND teleop pass the filter."""
        items = [
            {"episode_outcome": True, "control_mode_autonomous": False},   # 0 valid
            {"episode_outcome": True, "control_mode_autonomous": True},    # 1 rejected (autonomous)
            {"episode_outcome": False, "control_mode_autonomous": False},  # 2 rejected (failure)
            {"episode_outcome": False, "control_mode_autonomous": True},   # 3 rejected (both)
            {"episode_outcome": True, "control_mode_autonomous": False},   # 4 valid
        ]
        assert _compute_valid_indices_from_items(items) == [0, 4]

    def test_torch_bool_false_fails_equality_with_1(self):
        """torch.tensor(False, dtype=torch.bool) == 1 is False.

        The plugins return torch bool tensors.  The filter checks `== 1`.
        This test documents that False tensors FAIL the episode_outcome check,
        meaning a dataset with no outcomes.json gets ALL frames rejected.
        """
        assert (torch.tensor(False, dtype=torch.bool) == 1).item() is False
        assert (torch.tensor(True, dtype=torch.bool) == 1).item() is True

    def test_torch_bool_false_passes_equality_with_0(self):
        """torch.tensor(False, dtype=torch.bool) == 0 is True.

        control_mode_autonomous defaults to False when no metadata exists.
        The filter checks `== 0`, so the default PASSES the control_mode filter.
        This means the control_mode filter is effectively a no-op for datasets
        without episode_modes.json.
        """
        assert (torch.tensor(False, dtype=torch.bool) == 0).item() is True

    def test_default_plugin_values_reject_everything(self):
        """With default plugin values (no metadata files), ALL frames are rejected.

        episode_outcome = False (no outcomes.json) -> fails `== 1`
        control_mode_autonomous = False (no episode_modes.json) -> passes `== 0`
        Combined: rejected because episode_outcome fails.
        """
        items = [
            {"episode_outcome": torch.tensor(False, dtype=torch.bool),
             "control_mode_autonomous": torch.tensor([False], dtype=torch.bool)}
            for _ in range(100)
        ]
        assert _compute_valid_indices_from_items(items) == []

    def test_outcomes_true_no_control_mode_passes(self):
        """When outcomes.json marks all success BUT no control_mode metadata exists,
        all frames pass (control_mode_autonomous defaults to False which passes == 0).
        """
        items = [
            {"episode_outcome": torch.tensor(True, dtype=torch.bool),
             "control_mode_autonomous": torch.tensor([False], dtype=torch.bool)}
            for _ in range(100)
        ]
        assert _compute_valid_indices_from_items(items) == list(range(100))


# ===================================================================
# 2.  Plugin unit tests (test plugins directly with temp directories)
# ===================================================================

class TestEpisodeOutcomePlugin:
    """Test EpisodeOutcomePlugin behavior with real plugin code and temp dirs."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_plugins(self):
        pytest.importorskip("robocandywrapper")

    def _make_fake_dataset(self, tmpdir: str):
        """Create a minimal object that satisfies plugin.attach()."""

        class FakeEpisodes:
            def __len__(self):
                return 10

        class FakeMeta:
            episodes = FakeEpisodes()

        class FakeDataset:
            root = pathlib.Path(tmpdir)
            meta = FakeMeta()

        return FakeDataset()

    def test_no_outcomes_json_returns_false(self):
        """Without outcomes.json, every frame gets episode_outcome=False."""
        from robocandywrapper.plugins import EpisodeOutcomePlugin

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._make_fake_dataset(tmpdir)
            instance = EpisodeOutcomePlugin().attach(ds)
            data = instance.get_item_data(idx=0, episode_idx=0)

            assert data["episode_outcome"].item() is False
            assert data["episode_outcome_mask"].item() is False

    def test_all_success_outcomes(self):
        """With outcomes.json marking all episodes as success, outcome=True."""
        from robocandywrapper.plugins import EpisodeOutcomePlugin
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, EPISODE_OUTCOME_PLUGIN_NAME

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write outcomes.json
            plugin_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / EPISODE_OUTCOME_PLUGIN_NAME
            plugin_dir.mkdir(parents=True)
            outcomes = [{"episode_index": i, "outcome": "success"} for i in range(10)]
            (plugin_dir / "outcomes.json").write_text(json.dumps(outcomes))

            ds = self._make_fake_dataset(tmpdir)
            instance = EpisodeOutcomePlugin().attach(ds)

            for ep_idx in range(10):
                data = instance.get_item_data(idx=ep_idx * 100, episode_idx=ep_idx)
                assert data["episode_outcome"].item() is True, f"Episode {ep_idx} should be success"
                assert data["episode_outcome_mask"].item() is True

    def test_mixed_outcomes(self):
        """Success/failure outcomes are correctly distinguished."""
        from robocandywrapper.plugins import EpisodeOutcomePlugin
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, EPISODE_OUTCOME_PLUGIN_NAME

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / EPISODE_OUTCOME_PLUGIN_NAME
            plugin_dir.mkdir(parents=True)
            outcomes = [
                {"episode_index": 0, "outcome": "success"},
                {"episode_index": 1, "outcome": "failure"},
                {"episode_index": 2, "outcome": "unknown"},
                # episode 3: not in file -> unlabeled
            ]
            (plugin_dir / "outcomes.json").write_text(json.dumps(outcomes))

            ds = self._make_fake_dataset(tmpdir)
            instance = EpisodeOutcomePlugin().attach(ds)

            # success
            d0 = instance.get_item_data(idx=0, episode_idx=0)
            assert d0["episode_outcome"].item() is True
            assert d0["episode_outcome_mask"].item() is True

            # failure
            d1 = instance.get_item_data(idx=100, episode_idx=1)
            assert d1["episode_outcome"].item() is False
            assert d1["episode_outcome_mask"].item() is True

            # unknown -> treated as unlabeled
            d2 = instance.get_item_data(idx=200, episode_idx=2)
            assert d2["episode_outcome"].item() is False
            assert d2["episode_outcome_mask"].item() is False

            # missing -> unlabeled
            d3 = instance.get_item_data(idx=300, episode_idx=3)
            assert d3["episode_outcome"].item() is False
            assert d3["episode_outcome_mask"].item() is False


class TestControlModePlugin:
    """Test ControlModePlugin behavior with real plugin code and temp dirs."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_plugins(self):
        pytest.importorskip("rewact_tools")

    def test_plugin_looks_for_dagger_data_source_path(self):
        """Document that the plugin looks at candywrapper_plugins/dagger_data_source/,
        NOT candywrapper_plugins/control_mode/ where the files actually are on HuggingFace.

        This is the path mismatch bug.
        """
        from rewact_tools.control_mode_plugin import ControlModePlugin
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            # Put the file where HuggingFace stores it: control_mode/
            correct_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "control_mode"
            correct_dir.mkdir(parents=True)
            modes = {
                "0": {"segments": [{"start_index": 0, "end_index": 100, "mode": "human"}]}
            }
            (correct_dir / "episode_modes.json").write_text(json.dumps(modes))

            # But the plugin looks in dagger_data_source/
            wrong_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "dagger_data_source"
            assert not wrong_dir.exists(), "dagger_data_source/ should not exist"
            assert correct_dir.exists(), "control_mode/ should exist"

            # The plugin should NOT find the file -> warns and returns empty modes
            plugin = ControlModePlugin()

            class FakeDataset:
                root = pathlib.Path(tmpdir)
                episodes = None
                episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([101])}

            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                instance = plugin.attach(FakeDataset())

            # Should have warned about missing file
            warn_msgs = [str(warning.message) for warning in w]
            found_warning = any("not found" in msg.lower() or "unknown" in msg.lower() for msg in warn_msgs)
            assert found_warning, (
                f"Expected a warning about missing episode_modes.json. Got warnings: {warn_msgs}"
            )

    def test_no_modes_file_returns_unknown(self):
        """Without episode_modes.json, control_mode_autonomous=False (which passes the filter)."""
        from rewact_tools.control_mode_plugin import ControlModePlugin

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = ControlModePlugin()

            class FakeDataset:
                root = pathlib.Path(tmpdir)
                episodes = None
                episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([100])}

            import warnings
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                instance = plugin.attach(FakeDataset())

            data = instance.get_item_data(idx=0, episode_idx=0)
            assert data["control_mode"] == "unknown"
            # autonomous=False -> passes `== 0` check -> no filtering effect
            assert data["control_mode_autonomous"].item() is False

    def test_modes_at_correct_path_with_plugin_format(self):
        """When the file IS at dagger_data_source/ AND uses the flat-list format
        that the plugin's parser expects, control modes load correctly.

        The plugin parser does:
            for segment_data in segments_data:
                segment = ControlModeSegment(start_index=segment_data["start_index"], ...)

        So the JSON must map episode_id -> [list of segment dicts], NOT
        episode_id -> {"segments": [list]}.
        """
        from rewact_tools.control_mode_plugin import ControlModePlugin
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            # Put the file where the PLUGIN expects it, in the FLAT format it parses
            plugin_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "dagger_data_source"
            plugin_dir.mkdir(parents=True)
            modes = {
                "0": [
                    {"start_index": 0, "end_index": 49, "mode": "policy"},
                    {"start_index": 50, "end_index": 99, "mode": "human"},
                ]
            }
            (plugin_dir / "episode_modes.json").write_text(json.dumps(modes))

            class FakeDataset:
                root = pathlib.Path(tmpdir)
                episodes = None
                episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([100])}

            import warnings
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                instance = ControlModePlugin().attach(FakeDataset())

            # Frame 10 is in the "policy" segment -> autonomous=True
            d_policy = instance.get_item_data(idx=10, episode_idx=0)
            assert d_policy["control_mode"] == "policy"
            assert d_policy["control_mode_autonomous"].item() is True

            # Frame 75 is in the "human" segment -> autonomous=False
            d_human = instance.get_item_data(idx=75, episode_idx=0)
            assert d_human["control_mode"] == "human"
            assert d_human["control_mode_autonomous"].item() is False

    def test_huggingface_format_silently_fails(self):
        """The actual HuggingFace file uses {"segments": [...]} wrapper per episode,
        but the plugin parser expects a flat list. The wrapped format silently fails
        to parse -- the plugin catches the exception and returns empty modes.

        This is a THIRD bug: even if you fix the path (dagger_data_source -> control_mode),
        the JSON format mismatch means the control mode data still won't load.
        """
        from rewact_tools.control_mode_plugin import ControlModePlugin
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            # Put the file at the plugin's expected path, but in the HUGGINGFACE format
            plugin_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "dagger_data_source"
            plugin_dir.mkdir(parents=True)
            # This is the format on HuggingFace: {"segments": [...]} wrapper
            modes_hf_format = {
                "0": {
                    "segments": [
                        {"start_index": 0, "end_index": 49, "mode": "policy"},
                        {"start_index": 50, "end_index": 99, "mode": "human"},
                    ]
                }
            }
            (plugin_dir / "episode_modes.json").write_text(json.dumps(modes_hf_format))

            class FakeDataset:
                root = pathlib.Path(tmpdir)
                episodes = None
                episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([100])}

            import warnings
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                instance = ControlModePlugin().attach(FakeDataset())

            # Even though the file exists and has data, the parser fails silently
            # and all frames get control_mode="unknown"
            data = instance.get_item_data(idx=10, episode_idx=0)
            assert data["control_mode"] == "unknown", (
                "Expected 'unknown' because the HuggingFace JSON format "
                "({'segments': [...]}) is incompatible with the plugin's parser "
                "(which expects a flat list of segment dicts)"
            )
            assert data["control_mode_autonomous"].item() is False


# ===================================================================
# 2b. Patched ControlModePlugin (our fix in data_loader.py)
# ===================================================================

class TestPatchedControlModePlugin:
    """Tests for the patched ControlModePlugin in openpi.training.data_loader
    that fixes both the path and JSON format issues."""

    def setup_method(self):
        pytest.importorskip("rewact_tools")

    def test_finds_file_at_control_mode_path(self):
        """Our patched plugin finds episode_modes.json at control_mode/ (new path)."""
        from openpi.training.data_loader import ControlModePlugin
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            # Put file only at the NEW location (control_mode/)
            new_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "control_mode"
            new_dir.mkdir(parents=True)
            modes = {
                "0": [
                    {"start_index": 0, "end_index": 49, "mode": "policy"},
                    {"start_index": 50, "end_index": 99, "mode": "human"},
                ]
            }
            (new_dir / "episode_modes.json").write_text(json.dumps(modes))

            class FakeDataset:
                root = pathlib.Path(tmpdir)
                episodes = None
                episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([100])}

            import warnings
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                instance = ControlModePlugin().attach(FakeDataset())

            d = instance.get_item_data(idx=10, episode_idx=0)
            assert d["control_mode"] == "policy", f"Expected 'policy', got {d['control_mode']}"
            assert d["control_mode_autonomous"].item() is True

    def test_finds_file_at_legacy_path(self):
        """Our patched plugin still finds episode_modes.json at dagger_data_source/ (legacy)."""
        from openpi.training.data_loader import ControlModePlugin
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "dagger_data_source"
            legacy_dir.mkdir(parents=True)
            modes = {
                "0": [
                    {"start_index": 0, "end_index": 49, "mode": "policy"},
                    {"start_index": 50, "end_index": 99, "mode": "human"},
                ]
            }
            (legacy_dir / "episode_modes.json").write_text(json.dumps(modes))

            class FakeDataset:
                root = pathlib.Path(tmpdir)
                episodes = None
                episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([100])}

            import warnings
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                instance = ControlModePlugin().attach(FakeDataset())

            d = instance.get_item_data(idx=75, episode_idx=0)
            assert d["control_mode"] == "human", f"Expected 'human', got {d['control_mode']}"
            assert d["control_mode_autonomous"].item() is False

    def test_prefers_legacy_over_new_when_both_exist(self):
        """If both paths exist, legacy (dagger_data_source/) takes priority."""
        from openpi.training.data_loader import ControlModePlugin
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            base = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR

            legacy_dir = base / "dagger_data_source"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "episode_modes.json").write_text(json.dumps({
                "0": [{"start_index": 0, "end_index": 99, "mode": "human"}]
            }))

            new_dir = base / "control_mode"
            new_dir.mkdir(parents=True)
            (new_dir / "episode_modes.json").write_text(json.dumps({
                "0": [{"start_index": 0, "end_index": 99, "mode": "policy"}]
            }))

            class FakeDataset:
                root = pathlib.Path(tmpdir)
                episodes = None
                episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([100])}

            import warnings
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                instance = ControlModePlugin().attach(FakeDataset())

            d = instance.get_item_data(idx=10, episode_idx=0)
            assert d["control_mode"] == "human", (
                f"Legacy path should win, expected 'human', got {d['control_mode']}"
            )

    def test_handles_wrapped_json_format(self):
        """Our patched plugin handles the HuggingFace wrapped format:
        {"0": {"segments": [{...}]}} instead of {"0": [{...}]}.
        """
        from openpi.training.data_loader import ControlModePlugin
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "control_mode"
            new_dir.mkdir(parents=True)
            # HuggingFace format with {"segments": [...]} wrapper
            modes_hf = {
                "0": {
                    "segments": [
                        {"start_index": 0, "end_index": 49, "mode": "policy"},
                        {"start_index": 50, "end_index": 99, "mode": "human"},
                    ]
                }
            }
            (new_dir / "episode_modes.json").write_text(json.dumps(modes_hf))

            class FakeDataset:
                root = pathlib.Path(tmpdir)
                episodes = None
                episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([100])}

            import warnings
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                instance = ControlModePlugin().attach(FakeDataset())

            d_policy = instance.get_item_data(idx=10, episode_idx=0)
            assert d_policy["control_mode"] == "policy", (
                f"Expected 'policy' from wrapped format, got {d_policy['control_mode']}"
            )
            assert d_policy["control_mode_autonomous"].item() is True

            d_human = instance.get_item_data(idx=75, episode_idx=0)
            assert d_human["control_mode"] == "human", (
                f"Expected 'human' from wrapped format, got {d_human['control_mode']}"
            )
            assert d_human["control_mode_autonomous"].item() is False


# ===================================================================
# 3.  End-to-end filter simulation
# ===================================================================

class TestEndToEndFilterSimulation:
    """Simulate the full compute_valid_indices flow with crafted plugin outputs."""

    def test_clean_dataset_no_metadata(self):
        """A clean 200-episode dataset with NO candywrapper metadata
        produces 0 valid indices. Every frame is silently dropped.
        """
        # Simulate 200 episodes, ~240 frames each ≈ 48000 frames
        n_frames = 48000
        items = [
            {"episode_outcome": torch.tensor(False, dtype=torch.bool),  # default: no outcomes.json
             "control_mode_autonomous": torch.tensor([False], dtype=torch.bool)}  # default: no modes
            for _ in range(n_frames)
        ]
        valid = _compute_valid_indices_from_items(items)
        assert len(valid) == 0, (
            f"Clean dataset with no metadata should produce 0 valid indices, got {len(valid)}"
        )

    def test_clean_dataset_with_outcomes_json_all_success(self):
        """A clean dataset where outcomes.json marks all episodes as success
        AND no control_mode metadata exists -> all frames are valid.

        This is the correct state for bin_pick_pack_coffee_capsules after
        the user added outcomes.json.
        """
        n_frames = 48000
        items = [
            {"episode_outcome": torch.tensor(True, dtype=torch.bool),   # all success
             "control_mode_autonomous": torch.tensor([False], dtype=torch.bool)}  # default: no modes
            for _ in range(n_frames)
        ]
        valid = _compute_valid_indices_from_items(items)
        assert len(valid) == n_frames

    def test_dagger_dataset_with_path_mismatch(self):
        """A DAgger dataset where the control_mode file exists but at the WRONG path.

        The ControlModePlugin can't find it -> all frames get autonomous=False ->
        ALL success frames pass, including the policy-controlled segments.
        The control_mode filter is completely ineffective.
        """
        # 20 episodes, all success, mixed policy/human segments
        # But because of path mismatch, autonomous=False for all
        n_frames = 5000
        items = [
            {"episode_outcome": torch.tensor(True, dtype=torch.bool),
             "control_mode_autonomous": torch.tensor([False], dtype=torch.bool)}  # path mismatch -> default
            for _ in range(n_frames)
        ]
        valid = _compute_valid_indices_from_items(items)
        # ALL frames pass because the control mode filter is broken
        assert len(valid) == n_frames, (
            "With path mismatch, control_mode filter is a no-op: all success frames pass"
        )

    def test_dagger_dataset_with_correct_paths(self):
        """What SHOULD happen with a DAgger dataset if paths were correct:
        only human-controlled frames from success episodes pass.
        """
        # Simulate: 20 episodes, episodes 0-14 success, 15-19 failure
        # Each episode 100 frames, first 60 policy, last 40 human
        items = []
        for ep in range(20):
            is_success = ep < 15
            for frame in range(100):
                is_policy = frame < 60
                items.append({
                    "episode_outcome": torch.tensor(is_success, dtype=torch.bool),
                    "control_mode_autonomous": torch.tensor([is_policy], dtype=torch.bool),
                })

        valid = _compute_valid_indices_from_items(items)

        # Expected: 15 success episodes * 40 human frames each = 600
        assert len(valid) == 600, f"Expected 600, got {len(valid)}"

        # Verify they're all from success episodes (0-14), human segments (frames 60-99)
        for idx in valid:
            ep = idx // 100
            frame = idx % 100
            assert ep < 15, f"Index {idx} is from failure episode {ep}"
            assert frame >= 60, f"Index {idx} is from policy frame {frame}"

    def test_dagger_multi_segment_exact_indices(self):
        """Simulate a DAgger dataset with multi-segment episodes (like real data)
        and verify the exact set of valid indices matches what we expect.

        Mock layout (mimics real dAgger 1.5.1 structure):

        Episode 0 (success, 350 frames):
          frames   0-69:  policy   -> REJECT
          frames  70-150: human    -> VALID
          frames 151-313: policy   -> REJECT
          frames 314-349: human    -> VALID

        Episode 1 (success, 200 frames):
          frames   0-59:  human    -> VALID
          frames  60-199: policy   -> REJECT

        Episode 2 (failure, 100 frames):
          frames   0-49:  human    -> REJECT (failure episode)
          frames  50-99:  policy   -> REJECT (failure episode)
        """
        episodes = [
            # (outcome, total_frames, [(start, end, mode), ...])
            (True,  350, [(0, 69, "policy"), (70, 150, "human"), (151, 313, "policy"), (314, 349, "human")]),
            (True,  200, [(0, 59, "human"), (60, 199, "policy")]),
            (False, 100, [(0, 49, "human"), (50, 99, "policy")]),
        ]

        # Build the item list and expected valid indices
        items = []
        expected_valid = set()
        global_idx = 0

        for outcome, total_frames, segments in episodes:
            # Build a frame->mode lookup from segments
            frame_modes = {}
            for start, end, mode in segments:
                for f in range(start, end + 1):
                    frame_modes[f] = mode

            for frame in range(total_frames):
                mode = frame_modes.get(frame, "unknown")
                is_autonomous = (mode == "policy")
                items.append({
                    "episode_outcome": torch.tensor(outcome, dtype=torch.bool),
                    "control_mode_autonomous": torch.tensor([is_autonomous], dtype=torch.bool),
                })
                # Valid = success episode + human/unknown frame
                if outcome and not is_autonomous:
                    expected_valid.add(global_idx)
                global_idx += 1

        valid = _compute_valid_indices_from_items(items)
        actual_valid = set(valid)

        # Exact index-level comparison
        missing = expected_valid - actual_valid
        extra = actual_valid - expected_valid

        print(f"\n  Mock multi-segment DAgger:")
        print(f"    Total frames: {len(items)}")
        print(f"    Expected valid: {len(expected_valid)}")
        print(f"    Actual valid:   {len(actual_valid)}")
        print(f"    Missing (expected but not found): {len(missing)}")
        print(f"    Extra (found but not expected):   {len(extra)}")

        assert actual_valid == expected_valid, (
            f"Valid indices don't match. "
            f"Missing: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}, "
            f"Extra: {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}"
        )

        # Also verify the expected counts per episode:
        # Ep 0: human frames = (150-70+1) + (349-314+1) = 81 + 36 = 117
        # Ep 1: human frames = (59-0+1) = 60
        # Ep 2: 0 (failure)
        ep0_valid = [i for i in actual_valid if i < 350]
        ep1_valid = [i for i in actual_valid if 350 <= i < 550]
        ep2_valid = [i for i in actual_valid if 550 <= i < 650]
        assert len(ep0_valid) == 117, f"Ep 0: expected 117 valid, got {len(ep0_valid)}"
        assert len(ep1_valid) == 60, f"Ep 1: expected 60 valid, got {len(ep1_valid)}"
        assert len(ep2_valid) == 0, f"Ep 2: expected 0 valid, got {len(ep2_valid)}"


# ===================================================================
# 3b. Plugin-integrated index-level tests (real plugins + temp dirs)
# ===================================================================

class TestPluginIntegratedIndices:
    """Use real plugins with temp directories and verify the exact set of
    valid indices matches expectations.  Covers:

    1. No control_mode data (only outcomes)
    2. Control mode in flat-list format at dagger_data_source/ (legacy)
    3. Control mode in wrapped {"segments":[...]} format at control_mode/ (new)

    All tests iterate every frame and compare exact index sets.
    """

    def setup_method(self):
        pytest.importorskip("robocandywrapper")
        pytest.importorskip("rewact_tools")

    def _build_outcomes_json(self, episodes):
        """Build outcomes.json content from episode definitions.

        Args:
            episodes: list of (outcome_str, n_frames, segments) tuples.
                      outcome_str is "success" or "failure".
        """
        return [{"episode_index": i, "outcome": ep[0]} for i, ep in enumerate(episodes)]

    def _build_modes_flat(self, episodes):
        """Build episode_modes.json in flat-list format (legacy).

        Returns: {"0": [{start_index, end_index, mode}, ...], ...}
        """
        modes = {}
        for i, (_, _, segments) in enumerate(episodes):
            if segments is not None:
                modes[str(i)] = [
                    {"start_index": s, "end_index": e, "mode": m}
                    for s, e, m in segments
                ]
        return modes

    def _build_modes_wrapped(self, episodes):
        """Build episode_modes.json in wrapped format (new HuggingFace style).

        Returns: {"0": {"segments": [{start_index, end_index, mode}, ...]}, ...}
        """
        modes = {}
        for i, (_, _, segments) in enumerate(episodes):
            if segments is not None:
                modes[str(i)] = {
                    "segments": [
                        {"start_index": s, "end_index": e, "mode": m}
                        for s, e, m in segments
                    ]
                }
        return modes

    def _run_with_plugins(self, tmpdir, episodes):
        """Attach plugins to a FakeDataset and run is_valid_frame on every frame.

        Returns: (actual_valid_set, expected_valid_set, total_frames)
        """
        from openpi.training.data_loader import ControlModePlugin
        from robocandywrapper.plugins import EpisodeOutcomePlugin

        # Build episode_data_index for ControlModePlugin
        from_indices = []
        to_indices = []
        offset = 0
        for _, n_frames, _ in episodes:
            from_indices.append(offset)
            to_indices.append(offset + n_frames)
            offset += n_frames
        total_frames = offset

        class FakeMeta:
            class episodes:
                @staticmethod
                def __len__():
                    return len(from_indices)

        class FakeDataset:
            root = pathlib.Path(tmpdir)
            meta = FakeMeta()
            episode_data_index = {
                "from": torch.tensor(from_indices),
                "to": torch.tensor(to_indices),
            }

        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outcome_instance = EpisodeOutcomePlugin().attach(FakeDataset())
            mode_instance = ControlModePlugin().attach(FakeDataset())

        # Build expected valid indices
        expected_valid = set()
        global_idx = 0
        for ep_i, (outcome_str, n_frames, segments) in enumerate(episodes):
            is_success = (outcome_str == "success")
            # Build frame->mode map from segments
            frame_modes = {}
            if segments is not None:
                for s, e, m in segments:
                    for f in range(s, e + 1):
                        frame_modes[f] = m
            for frame in range(n_frames):
                mode = frame_modes.get(frame, "unknown")
                is_autonomous = (mode == "policy")
                if is_success and not is_autonomous:
                    expected_valid.add(global_idx)
                global_idx += 1

        # Run plugins on every frame and collect actual valid indices
        actual_valid = set()
        global_idx = 0
        for ep_i, (_, n_frames, _) in enumerate(episodes):
            for frame in range(n_frames):
                item = {}
                try:
                    data = outcome_instance.get_item_data(idx=global_idx, episode_idx=ep_i)
                    if data:
                        item.update(data)
                except Exception:
                    pass
                try:
                    data = mode_instance.get_item_data(idx=global_idx, episode_idx=ep_i, accumulated_data=item)
                    if data:
                        item.update(data)
                except Exception:
                    pass

                if is_valid_frame(item):
                    actual_valid.add(global_idx)
                global_idx += 1

        return actual_valid, expected_valid, total_frames

    def _print_and_assert(self, label, actual_valid, expected_valid, total_frames):
        missing = expected_valid - actual_valid
        extra = actual_valid - expected_valid
        print(f"\n  {label}:")
        print(f"    Total frames: {total_frames}")
        print(f"    Expected valid: {len(expected_valid)}")
        print(f"    Actual valid:   {len(actual_valid)}")
        print(f"    Missing: {len(missing)}, Extra: {len(extra)}")
        assert actual_valid == expected_valid, (
            f"{label}: indices don't match. "
            f"Missing: {sorted(missing)[:10]}, Extra: {sorted(extra)[:10]}"
        )

    # -----------------------------------------------------------------

    def test_no_control_mode_data(self):
        """Dataset with outcomes.json but NO episode_modes.json.

        All success frames should be valid (autonomous defaults to False).

        Episodes:
          Ep 0 (success, 200 frames): no segments -> all valid
          Ep 1 (failure, 150 frames): no segments -> all rejected
          Ep 2 (success, 100 frames): no segments -> all valid
        """
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, EPISODE_OUTCOME_PLUGIN_NAME

        episodes = [
            ("success", 200, None),
            ("failure", 150, None),
            ("success", 100, None),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write outcomes.json only — no episode_modes.json
            outcome_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / EPISODE_OUTCOME_PLUGIN_NAME
            outcome_dir.mkdir(parents=True)
            (outcome_dir / "outcomes.json").write_text(
                json.dumps(self._build_outcomes_json(episodes))
            )

            actual, expected, total = self._run_with_plugins(tmpdir, episodes)
            self._print_and_assert("No control_mode data", actual, expected, total)

            # Sanity: ep0 (200) + ep2 (100) = 300 valid
            assert len(actual) == 300

    def test_flat_list_format_at_legacy_path(self):
        """Control mode in flat-list format at dagger_data_source/ (legacy).

        Episodes:
          Ep 0 (success, 300 frames):
            policy(0-99), human(100-249), policy(250-299)
          Ep 1 (success, 200 frames):
            human(0-79), policy(80-199)
          Ep 2 (failure, 100 frames):
            human(0-99)  -> all rejected (failure)
        """
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, EPISODE_OUTCOME_PLUGIN_NAME

        episodes = [
            ("success", 300, [(0, 99, "policy"), (100, 249, "human"), (250, 299, "policy")]),
            ("success", 200, [(0, 79, "human"), (80, 199, "policy")]),
            ("failure", 100, [(0, 99, "human")]),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write outcomes.json
            outcome_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / EPISODE_OUTCOME_PLUGIN_NAME
            outcome_dir.mkdir(parents=True)
            (outcome_dir / "outcomes.json").write_text(
                json.dumps(self._build_outcomes_json(episodes))
            )

            # Write episode_modes.json in FLAT format at LEGACY path
            modes_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "dagger_data_source"
            modes_dir.mkdir(parents=True)
            (modes_dir / "episode_modes.json").write_text(
                json.dumps(self._build_modes_flat(episodes))
            )

            actual, expected, total = self._run_with_plugins(tmpdir, episodes)
            self._print_and_assert("Flat-list at dagger_data_source/", actual, expected, total)

            # Sanity: ep0 human=150, ep1 human=80, ep2=0 -> 230
            assert len(actual) == 230

    def test_wrapped_format_at_control_mode_path(self):
        """Control mode in wrapped {"segments":[...]} format at control_mode/ (new).

        Episodes:
          Ep 0 (success, 350 frames):
            policy(0-69), human(70-150), policy(151-313), human(314-349)
          Ep 1 (success, 200 frames):
            human(0-59), policy(60-199)
          Ep 2 (failure, 100 frames):
            human(0-49), policy(50-99)  -> all rejected
        """
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, EPISODE_OUTCOME_PLUGIN_NAME

        episodes = [
            ("success", 350, [(0, 69, "policy"), (70, 150, "human"), (151, 313, "policy"), (314, 349, "human")]),
            ("success", 200, [(0, 59, "human"), (60, 199, "policy")]),
            ("failure", 100, [(0, 49, "human"), (50, 99, "policy")]),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write outcomes.json
            outcome_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / EPISODE_OUTCOME_PLUGIN_NAME
            outcome_dir.mkdir(parents=True)
            (outcome_dir / "outcomes.json").write_text(
                json.dumps(self._build_outcomes_json(episodes))
            )

            # Write episode_modes.json in WRAPPED format at NEW path
            modes_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "control_mode"
            modes_dir.mkdir(parents=True)
            (modes_dir / "episode_modes.json").write_text(
                json.dumps(self._build_modes_wrapped(episodes))
            )

            actual, expected, total = self._run_with_plugins(tmpdir, episodes)
            self._print_and_assert("Wrapped format at control_mode/", actual, expected, total)

            # Sanity: ep0=81+36=117, ep1=60, ep2=0 -> 177
            assert len(actual) == 177

    def test_flat_list_format_at_control_mode_path(self):
        """Flat-list format at control_mode/ (new path, old format).

        Verifies our plugin handles the cross-combination:
        new directory + legacy JSON format.
        """
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, EPISODE_OUTCOME_PLUGIN_NAME

        episodes = [
            ("success", 300, [(0, 99, "policy"), (100, 249, "human"), (250, 299, "policy")]),
            ("success", 200, [(0, 79, "human"), (80, 199, "policy")]),
            ("failure", 100, [(0, 99, "human")]),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            outcome_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / EPISODE_OUTCOME_PLUGIN_NAME
            outcome_dir.mkdir(parents=True)
            (outcome_dir / "outcomes.json").write_text(
                json.dumps(self._build_outcomes_json(episodes))
            )

            # FLAT format at NEW path (control_mode/)
            modes_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "control_mode"
            modes_dir.mkdir(parents=True)
            (modes_dir / "episode_modes.json").write_text(
                json.dumps(self._build_modes_flat(episodes))
            )

            actual, expected, total = self._run_with_plugins(tmpdir, episodes)
            self._print_and_assert("Flat-list at control_mode/", actual, expected, total)
            assert len(actual) == 230

    def test_wrapped_format_at_legacy_path(self):
        """Wrapped {"segments":[...]} format at dagger_data_source/ (old path, new format).

        Verifies our plugin handles the cross-combination:
        legacy directory + new JSON format.
        """
        from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, EPISODE_OUTCOME_PLUGIN_NAME

        episodes = [
            ("success", 350, [(0, 69, "policy"), (70, 150, "human"), (151, 313, "policy"), (314, 349, "human")]),
            ("success", 200, [(0, 59, "human"), (60, 199, "policy")]),
            ("failure", 100, [(0, 49, "human"), (50, 99, "policy")]),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            outcome_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / EPISODE_OUTCOME_PLUGIN_NAME
            outcome_dir.mkdir(parents=True)
            (outcome_dir / "outcomes.json").write_text(
                json.dumps(self._build_outcomes_json(episodes))
            )

            # WRAPPED format at LEGACY path (dagger_data_source/)
            modes_dir = pathlib.Path(tmpdir) / CANDYWRAPPER_PLUGINS_DIR / "dagger_data_source"
            modes_dir.mkdir(parents=True)
            (modes_dir / "episode_modes.json").write_text(
                json.dumps(self._build_modes_wrapped(episodes))
            )

            actual, expected, total = self._run_with_plugins(tmpdir, episodes)
            self._print_and_assert("Wrapped format at dagger_data_source/", actual, expected, total)
            assert len(actual) == 177


# ===================================================================
# 4.  File I/O tests for valid_indices.txt
# ===================================================================

class TestValidIndicesFileIO:
    """Test the load/save of valid_indices.txt."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_loader(self):
        pytest.importorskip("openpi.training.data_loader")

    def test_load_empty_file(self):
        from openpi.training.data_loader import _load_valid_indices

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            result = _load_valid_indices(f.name)
        os.unlink(f.name)
        assert result == []

    def test_load_single_index(self):
        from openpi.training.data_loader import _load_valid_indices

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("42")
            f.flush()
            result = _load_valid_indices(f.name)
        os.unlink(f.name)
        assert result == [42]

    def test_load_multiple_indices(self):
        from openpi.training.data_loader import _load_valid_indices

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("0,5,12,23,45")
            f.flush()
            result = _load_valid_indices(f.name)
        os.unlink(f.name)
        assert result == [0, 5, 12, 23, 45]

    def test_roundtrip_matches_compute_script_format(self):
        """Verify the write format from compute_valid_indices matches the loader."""
        from openpi.training.data_loader import _load_valid_indices

        indices = [0, 100, 200, 47864]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # This is how compute_valid_indices.py writes:
            f.write(",".join(str(i) for i in indices))
            f.flush()
            result = _load_valid_indices(f.name)
        os.unlink(f.name)
        assert result == indices


# ===================================================================
# 5.  Integration tests with real datasets (skip if not available)
# ===================================================================

class TestRealDatasetIntegration:
    """Integration tests that load actual datasets with plugins.

    Three datasets, covering all the cases:

    1. bin_pick_pack_coffee_capsules         — clean teleop, no control_mode metadata
    2. dAgger_..._1.5.0                      — old DAgger, episode_modes.json in dagger_data_source/
    3. dAgger_..._1.5.1                      — new DAgger, episode_modes.json in control_mode/

    Each test loads the dataset with *both* the upstream and patched plugins
    so we can verify the patched version fixes the issues.

    These require datasets to be downloaded locally.
    Skipped automatically if not available.
    """

    CLEAN_REPO_ID = "villekuosmanen/bin_pick_pack_coffee_capsules"
    DAGGER_OLD_REPO_ID = "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.0"
    DAGGER_NEW_REPO_ID = "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.1"

    @pytest.fixture(autouse=True)
    def _skip_if_no_deps(self):
        pytest.importorskip("robocandywrapper")
        pytest.importorskip("rewact_tools")

    def _load_dataset(self, repo_id: str, use_patched_plugin: bool = False):
        """Load a dataset with either the upstream or patched ControlModePlugin."""
        from robocandywrapper.factory import make_dataset_without_config
        from robocandywrapper.plugins import EpisodeOutcomePlugin
        from rewact_tools import PiStar0_6CumulativeRewardPlugin

        if use_patched_plugin:
            from openpi.training.data_loader import ControlModePlugin
        else:
            from rewact_tools import ControlModePlugin

        try:
            dataset = make_dataset_without_config(
                repo_id,
                plugins=[
                    EpisodeOutcomePlugin(),
                    ControlModePlugin(),
                    PiStar0_6CumulativeRewardPlugin(normalise=True),
                ],
                load_videos=False,
            )
        except Exception as e:
            pytest.skip(f"Cannot load dataset {repo_id}: {e}")
        return dataset

    def _sample_and_diagnose(self, dataset, repo_id: str, label: str = "", sample_count: int = 10):
        """Sample frames via plugin instances directly (no video decode needed).

        Accesses the wrapper's _plugin_instances and the underlying hf_dataset
        to get episode_index for each frame, then calls each plugin's
        get_item_data() to collect outcome/control_mode fields.
        """
        n = len(dataset)
        step = max(1, n // sample_count)
        sample_indices = list(range(0, n, step))[:sample_count]

        # Access internals: the first (and only) underlying lerobot dataset
        lerobot_ds = dataset._datasets[0]
        plugin_instances = dataset._plugin_instances[0]

        results = []
        for i in sample_indices:
            # Get episode_index from the hf_dataset (no video decode)
            ep_idx = int(lerobot_ds.hf_dataset[i]["episode_index"])

            # Collect plugin outputs
            item = {}
            for pi in plugin_instances:
                try:
                    data = pi.get_item_data(idx=i, episode_idx=ep_idx, accumulated_data=item)
                    if data:
                        item.update(data)
                except Exception:
                    pass

            outcome = item.get("episode_outcome")
            autonomous = item.get("control_mode_autonomous")
            outcome_mask = item.get("episode_outcome_mask")
            control_mode = item.get("control_mode")

            passes = is_valid_frame(item)

            results.append({
                "index": i,
                "episode_index": ep_idx,
                "episode_outcome": outcome.item() if isinstance(outcome, torch.Tensor) else outcome,
                "outcome_mask": outcome_mask.item() if isinstance(outcome_mask, torch.Tensor) else outcome_mask,
                "control_mode": control_mode,
                "autonomous": autonomous.item() if isinstance(autonomous, torch.Tensor) else autonomous,
                "passes_filter": bool(passes),
            })

        header = f"Dataset: {repo_id} ({n} frames)"
        if label:
            header += f"  [{label}]"
        print(f"\n{'='*70}")
        print(header)
        print(f"{'='*70}")
        for r in results:
            status = "PASS" if r["passes_filter"] else "REJECT"
            print(f"  [{status}] idx={r['index']:>6d}  ep={r['episode_index']:>3}  "
                  f"outcome={r['episode_outcome']}  mask={r['outcome_mask']}  "
                  f"mode={r['control_mode']}  autonomous={r['autonomous']}")

        passed = sum(1 for r in results if r["passes_filter"])
        modes = set(r["control_mode"] for r in results)
        print(f"\n  -> {passed}/{len(results)} sampled frames pass the filter")
        print(f"  -> control_mode values seen: {modes}")

        return results

    # -----------------------------------------------------------------
    # 1. Clean teleop dataset (no control_mode metadata at all)
    # -----------------------------------------------------------------

    def test_clean_dataset(self):
        """bin_pick_pack_coffee_capsules: pure teleop, all episodes success.

        outcomes.json marks all 200 episodes as "success".
        No control_mode metadata -> control_mode="unknown", autonomous=False (passes == 0).

        Expected:
          - episode_outcome = True (all success)
          - control_mode = "unknown" (no episode_modes.json)
          - All frames PASS the filter (success + not autonomous)
        """
        dataset = self._load_dataset(self.CLEAN_REPO_ID, use_patched_plugin=False)
        results = self._sample_and_diagnose(dataset, self.CLEAN_REPO_ID, label="upstream plugin")

        n = len(dataset)
        assert 40000 < n < 55000, f"Expected ~47865 frames, got {n}"

        for r in results:
            assert r["episode_outcome"] is True, f"idx={r['index']}: expected outcome=True"
            assert r["control_mode"] == "unknown", f"idx={r['index']}: expected mode=unknown"
            assert r["passes_filter"] is True, f"idx={r['index']}: should pass (success + no control_mode)"

    # -----------------------------------------------------------------
    # 2. Old DAgger (episode_modes.json in dagger_data_source/)
    # -----------------------------------------------------------------

    def test_old_dagger_upstream(self):
        """dAgger 1.5.0 with UPSTREAM plugin.

        This dataset stores episode_modes.json at dagger_data_source/ (legacy path).
        The upstream plugin should find it and load control modes correctly.

        Expected:
          - episode_outcome: loaded from outcomes.json (mix of True/False)
          - control_mode: loaded correctly ("human" or "policy", NOT "unknown")
        """
        dataset = self._load_dataset(self.DAGGER_OLD_REPO_ID, use_patched_plugin=False)
        results = self._sample_and_diagnose(dataset, self.DAGGER_OLD_REPO_ID, label="upstream plugin")

        modes = set(r["control_mode"] for r in results)
        assert modes != {"unknown"}, (
            "Old DAgger dataset should have real control modes with the upstream plugin, "
            f"but all are 'unknown'. Modes seen: {modes}"
        )

    def test_old_dagger_patched(self):
        """dAgger 1.5.0 with PATCHED plugin.

        Should behave identically to upstream since the file is at the legacy path.

        Expected:
          - control_mode: loaded correctly (same as upstream)
        """
        dataset = self._load_dataset(self.DAGGER_OLD_REPO_ID, use_patched_plugin=True)
        results = self._sample_and_diagnose(dataset, self.DAGGER_OLD_REPO_ID, label="PATCHED plugin")

        modes = set(r["control_mode"] for r in results)
        assert modes != {"unknown"}, (
            "Old DAgger with patched plugin should still load control modes. "
            f"Modes seen: {modes}"
        )

    # -----------------------------------------------------------------
    # 3. New DAgger (episode_modes.json in control_mode/)
    # -----------------------------------------------------------------

    def test_new_dagger_upstream(self):
        """dAgger 1.5.1 with UPSTREAM plugin.

        This dataset stores episode_modes.json at control_mode/ (new path).
        The upstream plugin only checks dagger_data_source/ -> won't find it.

        Expected:
          - episode_outcome: loaded from outcomes.json (mix of True/False)
          - control_mode: "unknown" for ALL frames (path mismatch bug)
          - control_mode_autonomous: False (default) -> passes == 0 check
          - Filter is effectively: outcome-only (no control_mode filtering)
        """
        dataset = self._load_dataset(self.DAGGER_NEW_REPO_ID, use_patched_plugin=False)
        results = self._sample_and_diagnose(dataset, self.DAGGER_NEW_REPO_ID, label="upstream plugin — BUGGY")

        modes = set(r["control_mode"] for r in results)
        assert modes == {"unknown"}, (
            "New DAgger with upstream plugin should fail to load control modes "
            f"(path mismatch bug). Expected all 'unknown', got: {modes}"
        )

        # All autonomous values should be False (default) due to missing data
        for r in results:
            assert r["autonomous"] is False, (
                f"idx={r['index']}: autonomous should default to False, got {r['autonomous']}"
            )

    def test_new_dagger_patched(self):
        """dAgger 1.5.1 with PATCHED plugin.

        Our patched plugin checks both dagger_data_source/ and control_mode/,
        and handles the wrapped JSON format.

        Expected:
          - episode_outcome: loaded from outcomes.json (mix of True/False)
          - control_mode: real values ("human" or "policy", NOT all "unknown")
          - Filter correctly distinguishes human vs policy frames
        """
        dataset = self._load_dataset(self.DAGGER_NEW_REPO_ID, use_patched_plugin=True)
        results = self._sample_and_diagnose(dataset, self.DAGGER_NEW_REPO_ID, label="PATCHED plugin — FIXED")

        modes = set(r["control_mode"] for r in results)
        assert modes != {"unknown"}, (
            "New DAgger with patched plugin should load real control modes. "
            f"But all are 'unknown'. Modes seen: {modes}"
        )

    # -----------------------------------------------------------------
    # 4. Per-timestep verification within a single episode
    # -----------------------------------------------------------------

    def test_within_episode_frame_level_modes(self):
        """Verify control_mode changes at the correct frame boundary WITHIN an episode.

        dAgger 1.5.0 Episode 1 has segments:
          frames   0-63:  policy
          frames  64-169: human
          frames 170-269: policy

        dAgger 1.5.1 Episode 0 has segments (wrapped format):
          frames   0-69:  policy
          frames  70-150: human
          frames 151-313: policy
          frames 314-349: human

        We sample specific frames on both sides of each boundary and check
        that control_mode flips at the right timestep.
        """
        from openpi.training.data_loader import ControlModePlugin

        # --- Old DAgger (1.5.0): Episode 1 ---
        # Segments: policy(0-63), human(64-169), policy(170-269)
        old_ds = self._load_dataset(self.DAGGER_OLD_REPO_ID, use_patched_plugin=True)
        lerobot_ds = old_ds._datasets[0]
        plugins = old_ds._plugin_instances[0]

        # Find the global start index of episode 1
        hf = lerobot_ds.hf_dataset
        ep1_start = None
        for i in range(len(hf)):
            if int(hf[i]["episode_index"]) == 1:
                ep1_start = i
                break
        assert ep1_start is not None, "Could not find episode 1 in old DAgger"

        # Test frames at specific offsets within episode 1
        old_test_cases = [
            # (frame_offset, expected_mode, description)
            (0, "policy", "start of policy segment"),
            (30, "policy", "middle of policy segment"),
            (63, "policy", "last frame of policy segment"),
            (64, "human", "first frame of human segment"),
            (100, "human", "middle of human segment"),
            (169, "human", "last frame of human segment"),
            (170, "policy", "first frame of second policy segment"),
            (200, "policy", "middle of second policy segment"),
        ]

        print(f"\n{'='*70}")
        print(f"Within-episode test: {self.DAGGER_OLD_REPO_ID} Episode 1")
        print(f"  Episode 1 starts at global index {ep1_start}")
        print(f"  Segments: policy(0-63), human(64-169), policy(170-269)")
        print(f"{'='*70}")

        for offset, expected_mode, desc in old_test_cases:
            global_idx = ep1_start + offset
            item = {}
            for pi in plugins:
                try:
                    data = pi.get_item_data(idx=global_idx, episode_idx=1, accumulated_data=item)
                    if data:
                        item.update(data)
                except Exception:
                    pass
            actual_mode = item.get("control_mode", "MISSING")
            status = "OK" if actual_mode == expected_mode else "FAIL"
            print(f"  [{status}] frame {offset:>3d} (idx={global_idx}): "
                  f"mode={actual_mode:<8s}  expected={expected_mode:<8s}  ({desc})")
            assert actual_mode == expected_mode, (
                f"Episode 1, frame {offset} ({desc}): "
                f"expected '{expected_mode}', got '{actual_mode}'"
            )

        # --- New DAgger (1.5.1): Episode 0 ---
        # Segments: policy(0-69), human(70-150), policy(151-313), human(314-349)
        new_ds = self._load_dataset(self.DAGGER_NEW_REPO_ID, use_patched_plugin=True)
        new_lerobot_ds = new_ds._datasets[0]
        new_plugins = new_ds._plugin_instances[0]

        # Episode 0 starts at global index 0
        new_test_cases = [
            (0, "policy", "start of policy segment"),
            (50, "policy", "middle of first policy segment"),
            (69, "policy", "last frame of first policy segment"),
            (70, "human", "first frame of human segment"),
            (110, "human", "middle of human segment"),
            (150, "human", "last frame of human segment"),
            (151, "policy", "first frame of second policy segment"),
            (250, "policy", "middle of second policy segment"),
            (313, "policy", "last frame of second policy segment"),
            (314, "human", "first frame of second human segment"),
            (340, "human", "middle of second human segment"),
        ]

        print(f"\n{'='*70}")
        print(f"Within-episode test: {self.DAGGER_NEW_REPO_ID} Episode 0")
        print(f"  Segments: policy(0-69), human(70-150), policy(151-313), human(314-349)")
        print(f"{'='*70}")

        for offset, expected_mode, desc in new_test_cases:
            item = {}
            for pi in new_plugins:
                try:
                    data = pi.get_item_data(idx=offset, episode_idx=0, accumulated_data=item)
                    if data:
                        item.update(data)
                except Exception:
                    pass
            actual_mode = item.get("control_mode", "MISSING")
            status = "OK" if actual_mode == expected_mode else "FAIL"
            print(f"  [{status}] frame {offset:>3d} (idx={offset}): "
                  f"mode={actual_mode:<8s}  expected={expected_mode:<8s}  ({desc})")
            assert actual_mode == expected_mode, (
                f"Episode 0, frame {offset} ({desc}): "
                f"expected '{expected_mode}', got '{actual_mode}'"
            )


    # -----------------------------------------------------------------
    # 5. Full-episode valid indices check (every frame)
    # -----------------------------------------------------------------

    def test_valid_indices_every_frame_in_episode(self):
        """Check is_valid_frame for EVERY frame in a known episode.

        dAgger 1.5.1 Episode 0 (outcome=success):
          frames   0-69:  policy  -> REJECT (autonomous)
          frames  70-150: human   -> VALID  (success + human)
          frames 151-313: policy  -> REJECT (autonomous)
          frames 314-349: human   -> VALID  (success + human)

        Expected valid frame count: (150-70+1) + (349-314+1) = 81 + 36 = 117
        Expected rejected count:    (69-0+1) + (313-151+1) = 70 + 163 = 233
        Total: 350 frames

        This test iterates every frame and verifies is_valid_frame matches
        what the segment definitions predict.
        """
        new_ds = self._load_dataset(self.DAGGER_NEW_REPO_ID, use_patched_plugin=True)
        plugins = new_ds._plugin_instances[0]

        # Episode 0 segments (from episode_modes.json)
        segments = [
            (0, 69, "policy"),
            (70, 150, "human"),
            (151, 313, "policy"),
            (314, 349, "human"),
        ]
        total_frames = 350  # 0 through 349

        # Build expected: which frames should pass is_valid_frame?
        # Episode 0 outcome = success (True), so valid = human frames only
        expected_valid = set()
        expected_rejected = set()
        for start, end, mode in segments:
            for f in range(start, end + 1):
                if mode == "human":
                    expected_valid.add(f)
                else:
                    expected_rejected.add(f)

        # Iterate every frame and collect actual results
        actual_valid = set()
        actual_rejected = set()
        mismatches = []

        for offset in range(total_frames):
            item = {}
            for pi in plugins:
                try:
                    data = pi.get_item_data(idx=offset, episode_idx=0, accumulated_data=item)
                    if data:
                        item.update(data)
                except Exception:
                    pass

            if is_valid_frame(item):
                actual_valid.add(offset)
            else:
                actual_rejected.add(offset)

            # Check against expectation
            should_be_valid = offset in expected_valid
            actually_valid = offset in actual_valid
            if should_be_valid != actually_valid:
                mode = item.get("control_mode", "?")
                outcome = item.get("episode_outcome", "?")
                mismatches.append(
                    f"  frame {offset}: expected={'VALID' if should_be_valid else 'REJECT'}, "
                    f"got={'VALID' if actually_valid else 'REJECT'}, "
                    f"mode={mode}, outcome={outcome}"
                )

        print(f"\n{'='*70}")
        print(f"Full-episode valid indices: {self.DAGGER_NEW_REPO_ID} Episode 0")
        print(f"  Segments: policy(0-69), human(70-150), policy(151-313), human(314-349)")
        print(f"  Total frames: {total_frames}")
        print(f"  Expected valid: {len(expected_valid)}  (human frames)")
        print(f"  Expected rejected: {len(expected_rejected)}  (policy frames)")
        print(f"  Actual valid: {len(actual_valid)}")
        print(f"  Actual rejected: {len(actual_rejected)}")
        print(f"  Mismatches: {len(mismatches)}")
        if mismatches:
            for m in mismatches[:20]:
                print(m)
        print(f"{'='*70}")

        assert actual_valid == expected_valid, (
            f"{len(mismatches)} frame(s) don't match expected valid indices. "
            f"Expected {len(expected_valid)} valid, got {len(actual_valid)}."
        )
        assert actual_rejected == expected_rejected, (
            f"Rejected frames don't match. "
            f"Expected {len(expected_rejected)} rejected, got {len(actual_rejected)}."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
