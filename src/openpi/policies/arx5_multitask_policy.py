"""Data transforms for multi-task ARX5 training (single-arm + bimanual mix).

Handles:
- Mixed camera configs (zero-fill + mask missing cameras)
- Agilex gripper scaling (centimeters → meters for bimanual datasets)
- Action dim padding and masking (7→14 for single-arm)
- Subtask-based prompting
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np

from openpi import transforms

_LOGGED = False


def _parse_image(img) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return (img * 255).clip(0, 255).astype(np.uint8)
    return img.clip(0, 255).astype(np.uint8)


def _get_key(data: dict, *keys):
    for k in keys:
        if k in data:
            return data[k]
    raise KeyError(f"None of {keys} found in data. Available: {list(data.keys())[:20]}")


# Bimanual agilex datasets store gripper values in centimeters.
_BIMANUAL_ACTION_DIM = 14
_GRIPPER_INDICES_BIMANUAL = (6, 13)
_GRIPPER_SCALE = 100.0


@dataclasses.dataclass(frozen=True)
class ARX5MultiTaskInputs(transforms.DataTransformFn):
    """Inputs transform for mixed ARX5 single-arm + bimanual datasets.

    After RoboCandyWrapper's key_rename_map, the data has unified keys:
      - observation.images.front, observation.images.left_wrist, observation.images.right_wrist
      - observation.state (7-dim or 14-dim)
      - action (7-dim or 14-dim, with action horizon)

    This transform:
      1. Maps images to model keys (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)
      2. Zero-fills missing cameras and sets image_mask=False
      3. Scales agilex gripper values (cm→m) for 14-dim state/actions
      4. Passes through action_is_pad, prompt, subtask
    """

    def __call__(self, data: dict) -> dict:
        global _LOGGED

        # ── Images ──────────────────────────────────────────────────
        flat = transforms.flatten_dict(data)

        image_map = {
            "base_0_rgb": ("observation/images/front", "observation.images.front"),
            "left_wrist_0_rgb": ("observation/images/left_wrist", "observation.images.left_wrist"),
            "right_wrist_0_rgb": ("observation/images/right_wrist", "observation.images.right_wrist"),
        }

        images = {}
        image_masks = {}
        placeholder = None

        for model_key, source_keys in image_map.items():
            found = False
            for sk in source_keys:
                if sk in flat:
                    raw = flat[sk]
                    raw_arr = np.asarray(raw)
                    # Ensure HWC format: handle CHW tensors (C, H, W) where C <= 4
                    if raw_arr.ndim == 3 and raw_arr.shape[0] <= 4 and raw_arr.shape[2] > 4:
                        raw_arr = np.transpose(raw_arr, (1, 2, 0))
                    # Ensure 3-channel
                    if raw_arr.ndim == 2:
                        raw_arr = np.stack([raw_arr] * 3, axis=-1)
                    elif raw_arr.ndim == 3 and raw_arr.shape[-1] == 1:
                        raw_arr = np.repeat(raw_arr, 3, axis=-1)
                    elif raw_arr.ndim == 3 and raw_arr.shape[-1] == 4:
                        raw_arr = raw_arr[..., :3]

                    if raw_arr.ndim != 3 or raw_arr.shape[-1] != 3 or raw_arr.shape[0] < 16:
                        logging.warning(
                            f"[arx5_multitask] Unexpected image shape for "
                            f"{sk} -> {model_key}: {raw_arr.shape} "
                            f"(original: {np.asarray(raw).shape}), skipping"
                        )
                        continue

                    images[model_key] = _parse_image(raw_arr)
                    image_masks[model_key] = np.True_
                    if placeholder is None:
                        placeholder = np.zeros_like(images[model_key])
                    found = True
                    break
            if not found:
                image_masks[model_key] = np.False_

        if placeholder is None:
            placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
        for model_key in image_map:
            if model_key not in images:
                images[model_key] = placeholder

        # ── State ───────────────────────────────────────────────────
        state = np.asarray(
            _get_key(data, "observation.state", "observation/state", "state")
        ).astype(np.float32)

        orig_action_dim = state.shape[-1]

        if orig_action_dim == _BIMANUAL_ACTION_DIM:
            state = state.copy()
            for idx in _GRIPPER_INDICES_BIMANUAL:
                state[..., idx] /= _GRIPPER_SCALE

        # ── Actions ─────────────────────────────────────────────────
        inputs: dict = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        # LeRobot stores action chunks as "action" (after key_rename_map) or
        # "actions" (legacy). Also check the flattened dict for nested keys.
        action_key = None
        for k in ("actions", "action"):
            if k in data:
                action_key = k
                break
            if k in flat:
                action_key = k
                break
        if action_key is not None:
            raw = data.get(action_key) if action_key in data else flat.get(action_key)
            actions = np.asarray(raw).astype(np.float32)
            if actions.shape[-1] == _BIMANUAL_ACTION_DIM:
                actions = actions.copy()
                for idx in _GRIPPER_INDICES_BIMANUAL:
                    actions[..., idx] /= _GRIPPER_SCALE
            inputs["actions"] = actions
        else:
            if not _LOGGED:
                logging.warning(
                    f"[arx5_multitask] No action key found in data. "
                    f"Available keys: {list(data.keys())[:20]}"
                )

        # ── Passthrough fields ──────────────────────────────────────
        all_keys = set(data.keys()) | set(flat.keys())
        for pad_key in ("action_is_pad", "action.pos_is_pad", "action/pos_is_pad"):
            if pad_key in data:
                inputs["action_is_pad"] = np.asarray(data[pad_key]).astype(bool)
                break
            elif pad_key in flat:
                inputs["action_is_pad"] = np.asarray(flat[pad_key]).astype(bool)
                break

        for mask_key in ("action_dim_mask", "action.pos_dim_mask", "action/dim_mask"):
            if mask_key in data:
                inputs["action_dim_mask"] = np.asarray(data[mask_key]).astype(bool)
                break
            elif mask_key in flat:
                inputs["action_dim_mask"] = np.asarray(flat[mask_key]).astype(bool)
                break

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if not _LOGGED:
            logging.info(
                f"[arx5_multitask] state_dim={orig_action_dim}, "
                f"cameras={[k for k,v in image_masks.items() if v]}, "
                f"prompt={str(data.get('prompt', ''))[:60]}"
            )
            _LOGGED = True

        return inputs


@dataclasses.dataclass(frozen=True)
class ARX5MultiTaskOutputs(transforms.DataTransformFn):
    """Output transform: slice actions back to the original dimension."""

    action_dim: int = _BIMANUAL_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            actions = np.asarray(data["actions"])
            data["actions"] = actions[..., : self.action_dim].astype(np.float32)
            if actions.shape[-1] == _BIMANUAL_ACTION_DIM:
                for idx in _GRIPPER_INDICES_BIMANUAL:
                    if idx < data["actions"].shape[-1]:
                        data["actions"][..., idx] *= _GRIPPER_SCALE
        return data
