"""Convert DART rollout joints into the MDM-style results payload used by vrcai."""
#s
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_POSITION_SCALE = (0.7, 1.0, -0.7)
DEFAULT_Y_OFFSET = 0.42


# DART/SMPL-H 22-joint order:
# pelvis, left_hip, right_hip, spine1, left_knee, right_knee, spine2, left_ankle,
# right_ankle, spine3, left_foot, right_foot, neck, left_collar, right_collar,
# head, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
_DART_TO_VRCAI_JOINT_ORDER = (
    0,   # CenterHip <- pelvis
    2,   # RightHip <- right_hip
    1,   # LeftHip <- left_hip
    3,   # Stomach <- spine1
    5,   # RightKnee <- right_knee
    4,   # LeftKnee <- left_knee
    6,   # Chest <- spine2
    8,   # RightFoot <- right_ankle
    7,   # LeftFoot <- left_ankle
    9,   # AboveChest <- spine3
    11,  # RightFootFwd <- right_foot
    10,  # LeftFootFwd <- left_foot
    12,  # Neck <- neck
    14,  # RightShoulder <- right_collar
    13,  # LeftShoulder <- left_collar
    15,  # Head <- head
    17,  # RightShoulderOut <- right_shoulder
    16,  # LeftShoulderOut <- left_shoulder
    19,  # RightElbow <- right_elbow
    18,  # LeftElbow <- left_elbow
    21,  # RightHand <- right_wrist
    20,  # LeftHand <- left_wrist
)

# Use both ankle and foot joints for floor normalization so imported ankle trackers
# line up with the same standing height as the previous MDM/MMotion pipeline.
_DART_SUPPORT_JOINTS = (7, 8, 10, 11)
_DART_TO_HML3D_AXIS = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def reorder_dart_joints_to_vrcai(dart_joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(dart_joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(
            "Expected DART joints shaped (frames, 22, 3), got {shape}.".format(shape=joints.shape)
        )
    return joints[:, _DART_TO_VRCAI_JOINT_ORDER, :].astype(np.float32, copy=True)


def convert_dart_world_joints_to_processed_positions(dart_joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(dart_joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(
            "Expected DART joints shaped (frames, 22, 3), got {shape}.".format(shape=joints.shape)
        )
    if joints.shape[0] == 0:
        raise ValueError("DART joint sequences must contain at least one frame.")

    normalized = joints.copy()
    floor_height = float(normalized[0, _DART_SUPPORT_JOINTS, 2].min())
    normalized[:, :, 2] -= floor_height
    normalized = np.matmul(normalized, _DART_TO_HML3D_AXIS).astype(np.float32)
    processed = reorder_dart_joints_to_vrcai(normalized)
    # DART world joints are floor-relative, while the downstream package format
    # expects the same standing-height offset used by the legacy MDM/MMotion flow.
    processed[..., 1] += float(DEFAULT_Y_OFFSET)
    return processed.astype(np.float32)


def convert_dart_world_joints_to_raw_positions(
    dart_joints: np.ndarray,
    scale=DEFAULT_POSITION_SCALE,
    y_offset: float = DEFAULT_Y_OFFSET,
) -> np.ndarray:
    processed = convert_dart_world_joints_to_processed_positions(dart_joints)
    raw = processed.copy()
    raw[..., 0] /= float(scale[0])
    raw[..., 1] = (raw[..., 1] - float(y_offset)) / float(scale[1])
    raw[..., 2] /= float(scale[2])
    return raw.astype(np.float32)


def build_mdm_results_payload(raw_positions: np.ndarray, prompt: str, generator: str) -> dict[str, Any]:
    positions = np.asarray(raw_positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (22, 3):
        raise ValueError(
            "Expected raw positions shaped (frames, 22, 3), got {shape}.".format(shape=positions.shape)
        )
    frame_count = int(positions.shape[0])
    motion = np.transpose(positions, (1, 2, 0))[np.newaxis, ...]
    return {
        "motion": motion,
        "text": [str(prompt)],
        "lengths": np.array([frame_count], dtype=np.int32),
        "num_samples": 1,
        "num_repetitions": 1,
        "generator": str(generator),
    }


def write_mdm_style_results(output_dir: Path, raw_positions: np.ndarray, prompt: str, generator: str) -> Path:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    payload = build_mdm_results_payload(raw_positions=raw_positions, prompt=prompt, generator=generator)
    np.save(str(root / "results.npy"), payload, allow_pickle=True)
    (root / "results.txt").write_text(str(prompt) + "\n", encoding="utf-8")
    (root / "results_len.txt").write_text(str(int(raw_positions.shape[0])) + "\n", encoding="utf-8")
    (root / "results_meta.json").write_text(
        json.dumps({"generator": str(generator), "frame_count": int(raw_positions.shape[0])}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return root / "results.npy"
