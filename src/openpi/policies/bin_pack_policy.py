import dataclasses

import einops
import numpy as np

from openpi import transforms

_LOGGED_PROMPT = False


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_key(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing keys: {keys}")


def _rpy_to_rotm(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw (radians) to rotation matrix.

    Convention matches `arx5-sdk/python/communication/zmq_client.py`: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    rpy = np.asarray(rpy, dtype=np.float64)
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # ZYX (yaw-pitch-roll) rotation matrix.
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    return np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def _rotm_to_rpy(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to roll-pitch-yaw (radians).

    Convention matches `arx5-sdk/python/communication/zmq_client.py`.
    """
    R = np.asarray(R, dtype=np.float64)
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = np.arctan2(-R[..., 2, 0], np.sqrt(R[..., 2, 1] ** 2 + R[..., 2, 2] ** 2))
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def _rpy_to_rot6d_rows(rpy: np.ndarray) -> np.ndarray:
    """Encode RPY as 6D rotation using the top-2 rows of the rotation matrix."""
    R = _rpy_to_rotm(rpy)
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1).astype(np.float32)


def _rot6d_rows_to_rotm(rot6d: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Decode 6D rotation (top-2 rows) to a proper rotation matrix via Gram-Schmidt."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a = rot6d[..., 0:3]
    b = rot6d[..., 3:6]

    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    a_unit = a / np.maximum(a_norm, eps)

    b = b - np.sum(a_unit * b, axis=-1, keepdims=True) * a_unit
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    b_unit = b / np.maximum(b_norm, eps)

    c_unit = np.cross(a_unit, b_unit, axis=-1)

    # Stack rows.
    R = np.stack([a_unit, b_unit, c_unit], axis=-2)
    return R.astype(np.float32)


def _rot6d_rows_to_rpy(rot6d: np.ndarray) -> np.ndarray:
    """Decode 6D rotation (top-2 rows) back to roll-pitch-yaw (radians)."""
    return _rotm_to_rpy(_rot6d_rows_to_rotm(rot6d))


def _eef_pose_rpy_to_rot6d(eef_pose: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,roll,pitch,yaw,gripper] -> [x,y,z,rot6d(6),gripper]."""
    eef_pose = np.asarray(eef_pose)
    if eef_pose.shape[-1] == 10:
        return eef_pose.astype(np.float32)
    if eef_pose.shape[-1] != 7:
        raise ValueError(f"Expected eef_pose last dim 7 (RPY) or 10 (rot6d), got {eef_pose.shape[-1]}")
    xyz = eef_pose[..., 0:3]
    rpy = eef_pose[..., 3:6]
    grip = eef_pose[..., 6:7]
    rot6d = _rpy_to_rot6d_rows(rpy)
    return np.concatenate([xyz, rot6d, grip], axis=-1).astype(np.float32)


def _eef_pose_rot6d_to_rpy(eef_pose: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,rot6d(6),gripper] -> [x,y,z,roll,pitch,yaw,gripper]."""
    eef_pose = np.asarray(eef_pose)
    if eef_pose.shape[-1] == 7:
        return eef_pose.astype(np.float32)
    if eef_pose.shape[-1] != 10:
        raise ValueError(f"Expected eef_pose last dim 10 (rot6d) or 7 (RPY), got {eef_pose.shape[-1]}")
    xyz = eef_pose[..., 0:3]
    rot6d = eef_pose[..., 3:9]
    grip = eef_pose[..., 9:10]
    rpy = _rot6d_rows_to_rpy(rot6d)
    return np.concatenate([xyz, rpy, grip], axis=-1).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class BinPackInputs(transforms.DataTransformFn):
    """Inputs for the bin_pack_coffee_capsules dataset."""

    default_prompt: str = "pack coffee capsules into the cardboard bin container"

    # Determines which model will be used (unused in this transform).
    model_type: object | None = None

    def __call__(self, data: dict) -> dict:
        front = _parse_image(_get_key(data, "observation/images/front", "observation.images.front"))
        wrist = _parse_image(_get_key(data, "observation/images/wrist", "observation.images.wrist"))

        state_pos = np.asarray(_get_key(data, "observation/state/pos", "observation.state.pos"))
        state_eef = np.asarray(_get_key(data, "observation/state/eef_pose", "observation.state.eef_pose"))
        state_eef = _eef_pose_rpy_to_rot6d(state_eef)
        state = np.concatenate([state_pos, state_eef], axis=-1).astype(np.float32)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] == 17:
                inputs["actions"] = actions.astype(np.float32)
            elif actions.shape[-1] == 14:
                action_pos = actions[..., :7]
                action_eef = actions[..., 7:]
                action_eef = _eef_pose_rpy_to_rot6d(action_eef)
                inputs["actions"] = np.concatenate([action_pos, action_eef], axis=-1).astype(np.float32)
            else:
                raise ValueError(
                    f"Expected actions last dim 14 (pos+eef_rpy) or 17 (pos+eef_rot6d), got {actions.shape[-1]}"
                )
        else:
            action_pos = _get_key(data, "action/pos", "action.pos")
            action_eef = _get_key(data, "action/eef_pose", "action.eef_pose")
            action_pos = np.asarray(action_pos)
            action_eef = _eef_pose_rpy_to_rot6d(np.asarray(action_eef))
            inputs["actions"] = np.concatenate([action_pos, action_eef], axis=-1).astype(np.float32)

        # Extract action_is_pad mask if available (from LeRobot episode padding).
        # LeRobot may provide this under different keys depending on the dataset format.
        action_is_pad = None
        for pad_key in ("action_is_pad", "action.pos_is_pad", "action/pos_is_pad"):
            if pad_key in data:
                action_is_pad = np.asarray(data[pad_key]).astype(bool)
                break
        if action_is_pad is not None:
            inputs["action_is_pad"] = action_is_pad

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            task = str(data["task"])
            task = task.replace("pick a single coffee capsule", "pick coffee capsules")
            inputs["prompt"] = task
        else:
            inputs["prompt"] = self.default_prompt

        global _LOGGED_PROMPT
        if not _LOGGED_PROMPT:
            print(f"[bin_pack] prompt: {inputs['prompt']}")
            _LOGGED_PROMPT = True

        return inputs


@dataclasses.dataclass(frozen=True)
class BinPackOutputs(transforms.DataTransformFn):
    """Outputs for the bin_pack_coffee_capsules dataset."""

    action_dim: int | None = None
    # If true, decode rot6d eef orientation back to RPY for downstream consumers.
    output_rpy: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        action_dim = self.action_dim or actions.shape[-1]
        actions = actions[:, : action_dim]

        # If actions include (pos + eef_rot6d), optionally decode back to (pos + eef_rpy).
        if self.output_rpy and actions.shape[-1] == 17:
            action_pos = actions[..., :7]
            action_eef = actions[..., 7:]
            action_eef = _eef_pose_rot6d_to_rpy(action_eef)
            actions = np.concatenate([action_pos, action_eef], axis=-1).astype(np.float32)

        return {"actions": actions}

