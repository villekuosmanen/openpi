# Valid Indices Test Guide

Reference for `scripts/valid_indices_test.py`. Explains what each test does, why it matters,
and what passing/failing tells you.

**40 tests total** across 8 test classes.

---

## Background

The valid indices workflow filters training frames using two conditions from `compute_valid_indices.py`:

```python
if item["episode_outcome"] == 1 and item["control_mode_autonomous"] == 0:
    valid.append(i)
```

- **`episode_outcome`**: Was this episode a success? Comes from `EpisodeOutcomePlugin` which reads `outcomes.json`.
- **`control_mode_autonomous`**: Was this frame recorded by a policy (autonomous) or a human (teleop)? Comes from `ControlModePlugin` which reads `episode_modes.json`.

A frame is valid only if it's from a **successful episode** AND was **human-controlled** (not autonomous).

The filter logic lives in `compute_valid_indices.is_valid_frame()`. The tests import and use this
real function — there is no reimplementation in the test file.

### Default behaviors when metadata files are missing

| Plugin               | Missing file       | Default value              | Effect on filter       |
|----------------------|--------------------|----------------------------|------------------------|
| EpisodeOutcomePlugin | No `outcomes.json` | `episode_outcome = False`  | Frame **rejected** (fails `== 1`) |
| ControlModePlugin    | No `episode_modes.json` | `control_mode_autonomous = False` | Frame **passes** (passes `== 0`) |

This asymmetry is important: missing outcome data kills everything, but missing control mode data lets everything through.

### Bugs found in the upstream `ControlModePlugin`

1. **Path mismatch**: Plugin looks for `candywrapper_plugins/dagger_data_source/episode_modes.json` but newer datasets (e.g. `dAgger_1.5.1`) store it at `candywrapper_plugins/control_mode/episode_modes.json`.
2. **JSON format mismatch**: Plugin parser expects `{"0": [{start_index, end_index, mode}, ...]}` (flat list) but newer datasets use `{"0": {"segments": [{...}]}}` (wrapped).
3. Both bugs fail silently — `control_mode` defaults to `"unknown"`, `autonomous=False`, which **passes** the filter, so policy frames leak into training.

### Our fix

`ControlModePlugin` subclass in `data_loader.py` that:
- Checks both `dagger_data_source/` and `control_mode/` paths
- Handles both flat-list and wrapped `{"segments": [...]}` JSON formats
- All 4 path/format combinations work

---

## Section 1: `TestFilterLogic` (8 tests)

Pure logic tests. No plugins, no datasets, no files. Tests the two-condition filter with plain Python/torch values.

### `test_all_success_teleop_passes`

All frames are success + human-controlled.

```
Frame:  outcome=True, autonomous=False
Filter: True == 1 → yes,  False == 0 → yes  →  VALID
```

**Expected**: All 10 indices valid. **PASS** = filter logic is correct for the happy path.

---

### `test_all_failure_rejected`

All frames are failures.

```
Frame:  outcome=False, autonomous=False
Filter: False == 1 → no  →  REJECTED
```

**Expected**: 0 valid. **PASS** = failures are correctly filtered out.

---

### `test_all_autonomous_rejected`

All frames are autonomous (policy-controlled), even though successful.

```
Frame:  outcome=True, autonomous=True
Filter: True == 1 → yes,  True == 0 → no  →  REJECTED
```

**Expected**: 0 valid. **PASS** = policy frames are correctly excluded from training.

---

### `test_mixed_episodes`

5 frames with different combinations:

| Index | outcome | autonomous | Result |
|-------|---------|------------|--------|
| 0     | True    | False      | VALID  |
| 1     | True    | True       | REJECTED (autonomous) |
| 2     | False   | False      | REJECTED (failure) |
| 3     | False   | True       | REJECTED (both) |
| 4     | True    | False      | VALID  |

**Expected**: `[0, 4]`. **PASS** = both conditions must be met simultaneously.

---

### `test_torch_bool_false_fails_equality_with_1`

Documents a subtle PyTorch behavior that the filter relies on:

```python
torch.tensor(False, dtype=torch.bool) == 1  →  False
torch.tensor(True,  dtype=torch.bool) == 1  →  True
```

**Why it matters**: The plugins return `torch.bool` tensors, not Python bools. The filter checks `== 1`. This test proves that `False` tensors do fail the check, so datasets without `outcomes.json` (where outcome defaults to `False`) will have all frames rejected.

**PASS** = PyTorch behaves as expected. If this ever **FAILS**, the filter would silently break.

---

### `test_torch_bool_false_passes_equality_with_0`

The flip side:

```python
torch.tensor(False, dtype=torch.bool) == 0  →  True
```

**Why it matters**: `control_mode_autonomous` defaults to `False` when no metadata exists. The filter checks `== 0`. So the default **passes** the control mode check, meaning the control mode filter is a no-op for datasets without `episode_modes.json`.

**PASS** = confirms the asymmetric default behavior.

---

### `test_default_plugin_values_reject_everything`

Simulates what happens with a clean dataset that has NO metadata files at all:

```
outcome=False (no outcomes.json) → fails == 1  →  REJECTED
autonomous=False (no modes)      → passes == 0  →  (doesn't matter)
```

**Expected**: 0 valid out of 100 frames. **PASS** = confirms that enabling valid_indices on a dataset without `outcomes.json` silently drops ALL training data.

---

### `test_outcomes_true_no_control_mode_passes`

Simulates a dataset with `outcomes.json` (all success) but NO `episode_modes.json`:

```
outcome=True  → passes == 1
autonomous=False (default) → passes == 0
→  VALID
```

**Expected**: All 100 frames valid. **PASS** = this is the expected state for `bin_pick_pack_coffee_capsules` with `outcomes.json`.

---

## Section 2a: `TestEpisodeOutcomePlugin` (3 tests)

Tests the real `EpisodeOutcomePlugin` code using temporary directories to simulate dataset roots.

### `test_no_outcomes_json_returns_false`

Creates a temp dir with no `outcomes.json`. Attaches the plugin and queries frame 0.

**Expected**: `episode_outcome=False`, `episode_outcome_mask=False`.

**PASS** = confirms the plugin defaults to "unlabeled/failure" when metadata is absent.

---

### `test_all_success_outcomes`

Creates `outcomes.json` marking all 10 episodes as `"success"`.

**Expected**: All episodes return `episode_outcome=True`, `episode_outcome_mask=True`.

**PASS** = plugin correctly reads and applies outcome labels.

---

### `test_mixed_outcomes`

Creates `outcomes.json` with mixed labels:

| Episode | Outcome in JSON | `episode_outcome` | `episode_outcome_mask` |
|---------|-----------------|--------------------|-----------------------|
| 0       | `"success"`     | True               | True                  |
| 1       | `"failure"`     | False              | True (labeled, just failed) |
| 2       | `"unknown"`     | False              | False (unlabeled)     |
| 3       | (not in file)   | False              | False (unlabeled)     |

**PASS** = plugin correctly distinguishes success/failure/unknown/missing.

---

## Section 2b: `TestControlModePlugin` (4 tests)

Tests the **upstream** (unpatched) `ControlModePlugin` to document its bugs.

### `test_plugin_looks_for_dagger_data_source_path`

**Documents Bug #1: Path mismatch.**

File placed at `control_mode/` (HuggingFace location) but plugin looks at `dagger_data_source/`.

**Expected**: Plugin warns "not found". **PASS** = path mismatch bug exists.

---

### `test_no_modes_file_returns_unknown`

No `episode_modes.json` anywhere.

**Expected**: `control_mode="unknown"`, `control_mode_autonomous=False`.

**PASS** = missing control mode data is a no-op (passes the filter).

---

### `test_modes_at_correct_path_with_plugin_format`

File at `dagger_data_source/` in flat-list format — the one scenario where upstream works.

**Expected**: Frame 10 → `policy`, Frame 75 → `human`.

**PASS** = upstream works when both path AND format are correct.

---

### `test_huggingface_format_silently_fails`

**Documents Bug #2: JSON format mismatch.**

File at `dagger_data_source/` (correct path!) but in wrapped `{"segments": [...]}` format.

**Expected**: `control_mode="unknown"` even though the file exists with valid data.

**PASS** = confirms the format mismatch bug.

---

## Section 2c: `TestPatchedControlModePlugin` (4 tests)

Tests **our fix** (`ControlModePlugin` subclass in `data_loader.py`).

### `test_finds_file_at_control_mode_path`

File at `control_mode/` only. **PASS** = path fix works for newer datasets.

### `test_finds_file_at_legacy_path`

File at `dagger_data_source/` only. **PASS** = backward compatible with older datasets.

### `test_prefers_legacy_over_new_when_both_exist`

Both paths have files with different data. **PASS** = legacy takes priority (deterministic).

### `test_handles_wrapped_json_format`

File at `control_mode/` in wrapped `{"segments": [...]}` format.

**PASS** = our `_load_episode_modes()` correctly unwraps before parsing. This is the key fix for `dAgger_1.5.1`.

---

## Section 3: `TestEndToEndFilterSimulation` (5 tests)

Simulates the full `compute_valid_indices` flow with crafted torch tensor data. No real datasets or plugins.

### `test_clean_dataset_no_metadata`

48,000 frames with default plugin values (no JSON files).

**Expected**: 0 valid indices.

### `test_clean_dataset_with_outcomes_json_all_success`

48,000 frames, all `outcome=True`, no control mode metadata.

**Expected**: All 48,000 valid.

### `test_dagger_dataset_with_path_mismatch`

5,000 frames, all success, but control mode defaults to `autonomous=False` (path mismatch).

**Expected**: All 5,000 valid (control mode filter is a no-op). Shows policy frames leaking into training.

### `test_dagger_dataset_with_correct_paths`

20 episodes (100 frames each), episodes 0-14 success, 15-19 failure.
Each episode: frames 0-59 policy, frames 60-99 human.

**Expected**: 600 valid (15 success episodes × 40 human frames). Verifies per-frame episode/mode.

### `test_dagger_multi_segment_exact_indices`

Multi-segment episodes (like real DAgger data). Verifies **exact set of valid indices**, not just counts.

Mock layout:

| Episode | Outcome | Frames | Segments |
|---------|---------|--------|----------|
| 0       | success | 350    | policy(0-69), human(70-150), policy(151-313), human(314-349) |
| 1       | success | 200    | human(0-59), policy(60-199) |
| 2       | failure | 100    | human(0-49), policy(50-99) |

**Expected**: 177 valid (ep0: 117 + ep1: 60 + ep2: 0). Exact index-level set equality.

---

## Section 3b: `TestPluginIntegratedIndices` (5 tests)

Uses **real plugin code** with temp directories. Iterates every frame through both plugins and verifies
exact valid index sets. Tests all 4 path/format combinations plus the no-metadata case.

### `test_no_control_mode_data`

Only `outcomes.json` present, no `episode_modes.json`.

3 episodes (success, failure, success) × varying frame counts = 450 frames.

**Expected**: 300 valid (all frames from success episodes, since `autonomous=False` by default).

### `test_flat_list_format_at_legacy_path`

`episode_modes.json` in **flat-list format** at **`dagger_data_source/`** (legacy path, legacy format).

| Episode | Outcome | Segments |
|---------|---------|----------|
| 0       | success | policy(0-99), human(100-249), policy(250-299) |
| 1       | success | human(0-79), policy(80-199) |
| 2       | failure | human(0-99) |

**Expected**: 230 valid (ep0: 150 + ep1: 80 + ep2: 0). Exact index-level set equality.

### `test_wrapped_format_at_control_mode_path`

`episode_modes.json` in **wrapped `{"segments":[...]}` format** at **`control_mode/`** (new path, new format).

**Expected**: 177 valid. Exact index-level set equality.

### `test_flat_list_format_at_control_mode_path`

**Cross-combination**: flat-list format at `control_mode/` (new path, old format).

**Expected**: 230 valid. Exact index-level set equality.

### `test_wrapped_format_at_legacy_path`

**Cross-combination**: wrapped format at `dagger_data_source/` (old path, new format).

**Expected**: 177 valid. Exact index-level set equality.

All 4 path × format combinations produce identical results to each other (given the same data),
confirming the patched plugin is fully robust.

---

## Section 4: `TestValidIndicesFileIO` (4 tests)

Tests reading/writing of `valid_indices.txt`.

### `test_load_empty_file`

Empty file → empty list.

### `test_load_single_index`

`"42"` → `[42]`

### `test_load_multiple_indices`

`"0,5,12,23,45"` → `[0, 5, 12, 23, 45]`

### `test_roundtrip_matches_compute_script_format`

Writes indices the same way `compute_valid_indices.py` does, then reads them back.

**Expected**: Round-trip preserves exact values.

---

## Section 5: `TestRealDatasetIntegration` (8 tests)

Integration tests that load actual downloaded datasets. **Skipped** if datasets aren't available locally.

### `test_clean_dataset`

`bin_pick_pack_coffee_capsules` with upstream plugin. All 200 episodes marked success in `outcomes.json`.
No control mode metadata.

**Expected**: All sampled frames pass (outcome=True, autonomous=False by default).

### `test_old_dagger_upstream`

`dAgger_1.5.0` with upstream plugin. `episode_modes.json` at `dagger_data_source/` in flat-list format.

**Expected**: Real control modes loaded (80/80 episodes), mix of `human` and `policy`.

### `test_old_dagger_patched`

Same dataset, patched plugin. **Expected**: Identical results — backward compatible.

### `test_new_dagger_upstream`

`dAgger_1.5.1` with upstream plugin. `episode_modes.json` at `control_mode/` in wrapped format.

**Expected**: All `control_mode="unknown"` (0/0 episodes loaded). **Confirms the bug.**

### `test_new_dagger_patched`

Same dataset, patched plugin. **Expected**: Real control modes loaded (20/20 episodes). **Confirms the fix.**

### `test_within_episode_frame_level_modes`

Checks specific frames at every segment boundary within known episodes.

**dAgger 1.5.0 Episode 1**: frame 63 → policy, frame 64 → human, frame 169 → human, frame 170 → policy.

**dAgger 1.5.1 Episode 0**: frame 69 → policy, frame 70 → human, frame 150 → human, frame 151 → policy, frame 313 → policy, frame 314 → human.

**Expected**: Every boundary check exact. Verifies per-timestep precision.

### `test_valid_indices_every_frame_in_episode`

Iterates **every single frame** (0-349) of dAgger 1.5.1 Episode 0 and runs `is_valid_frame`.

```
Segments: policy(0-69), human(70-150), policy(151-313), human(314-349)
Expected valid: 117  (human frames only)
Expected rejected: 233  (policy frames)
```

Compares exact sets — not just counts, but the literal set of indices that are valid vs rejected.
If even one frame is in the wrong set, the test fails.

---

## Running the tests

```bash
# Everything (40 tests)
uv run pytest scripts/valid_indices_test.py -v -s

# All unit tests (no real dataset needed, 32 tests)
uv run pytest scripts/valid_indices_test.py -v -s -k "not TestRealDataset"

# Just the filter logic (8 tests)
uv run pytest scripts/valid_indices_test.py -v -s -k "TestFilterLogic"

# Upstream plugin bug tests (4 tests)
uv run pytest scripts/valid_indices_test.py -v -s -k "TestControlModePlugin"

# Patched plugin tests (4 tests)
uv run pytest scripts/valid_indices_test.py -v -s -k "TestPatchedControlModePlugin"

# End-to-end simulations with mock data (5 tests)
uv run pytest scripts/valid_indices_test.py -v -s -k "TestEndToEndFilterSimulation"

# Plugin-integrated index-level tests, all 4 path/format combos (5 tests)
uv run pytest scripts/valid_indices_test.py -v -s -k "TestPluginIntegratedIndices"

# Integration tests with real datasets (8 tests, requires datasets downloaded)
uv run pytest scripts/valid_indices_test.py -v -s -k "TestRealDataset"

# Just the frame-level boundary and every-frame tests
uv run pytest scripts/valid_indices_test.py -v -s -k "test_within_episode or test_valid_indices_every_frame"
```
