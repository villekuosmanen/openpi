import numpy as np

from openpi.policies import bin_pack_policy


def test_eef_pose_rpy_rot6d_roundtrip_small_angles():
    # Keep angles away from wrap/gimbal for a strict numeric check.
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(128, 3)).astype(np.float32)
    rpy = rng.uniform(low=-0.8, high=0.8, size=(128, 3)).astype(np.float32)
    gripper = rng.uniform(low=0.0, high=1.0, size=(128, 1)).astype(np.float32)

    eef_rpy = np.concatenate([xyz, rpy, gripper], axis=-1)
    eef_rot6d = bin_pack_policy._eef_pose_rpy_to_rot6d(eef_rpy)
    assert eef_rot6d.shape == (128, 10)

    # rot6d components should lie in [-1, 1] (up to tiny numeric tolerance).
    rot = eef_rot6d[:, 3:9]
    assert np.all(rot <= 1.0 + 1e-5)
    assert np.all(rot >= -1.0 - 1e-5)

    eef_rpy2 = bin_pack_policy._eef_pose_rot6d_to_rpy(eef_rot6d)
    assert eef_rpy2.shape == (128, 7)

    # xyz + gripper should roundtrip closely
    assert np.allclose(eef_rpy2[:, 0:3], eef_rpy[:, 0:3], atol=1e-5)
    assert np.allclose(eef_rpy2[:, 6:7], eef_rpy[:, 6:7], atol=1e-5)
    # rpy should roundtrip closely in this restricted regime
    assert np.allclose(eef_rpy2[:, 3:6], eef_rpy[:, 3:6], atol=5e-4)


def test_bin_pack_inputs_outputs_shapes_and_decode():
    rng = np.random.default_rng(1)
    front = rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)

    state_pos = rng.normal(size=(7,)).astype(np.float32)
    state_eef_rpy = np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.05, 0.7], dtype=np.float32)
    action_pos = rng.normal(size=(50, 7)).astype(np.float32)
    action_eef_rpy = np.tile(state_eef_rpy, (50, 1)).astype(np.float32)

    raw = {
        "observation/images/front": front,
        "observation/images/wrist": wrist,
        "observation/state/pos": state_pos,
        "observation/state/eef_pose": state_eef_rpy,
        "action/pos": action_pos,
        "action/eef_pose": action_eef_rpy,
        "task": "pick a single coffee capsule and place it into the bin",
    }

    x = bin_pack_policy.BinPackInputs()(raw)
    assert x["state"].shape == (17,)
    assert x["actions"].shape == (50, 17)

    # Simulate model output: pad to 32 dims (like pi05 action_dim default).
    actions32 = np.pad(x["actions"], [(0, 0), (0, 32 - x["actions"].shape[-1])]).astype(np.float32)
    out = bin_pack_policy.BinPackOutputs(action_dim=17, output_rpy=True)({"actions": actions32})
    assert out["actions"].shape == (50, 14)  # pos(7) + eef_rpy(7)
