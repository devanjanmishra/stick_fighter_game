"""
Synthetic keypoint data generator for testing without a camera.
Generates realistic pose sequences for idle, jab, cross, hook, uppercut, walking.
"""

import math
import numpy as np
from core.pose_estimator import Keypoint, PoseFrame


def _make_keypoint(x: float, y: float, z: float, name: str, vis: float = 0.95) -> Keypoint:
    return Keypoint(x=x, y=y, z=z, visibility=vis, name=name)


def generate_idle_pose(frame_index: int = 0, stance: str = "orthodox") -> PoseFrame:
    """
    Generate a front-facing idle fighting stance.
    Orthodox = left hand lead, right hand rear.
    Camera sees the person facing it.
    Coordinates: x=0-1 (left-right), y=0-1 (top-bottom), z=depth
    """
    # Slight idle sway
    sway = math.sin(frame_index * 0.1) * 0.005

    if stance == "orthodox":
        keypoints = {
            "nose": _make_keypoint(0.50 + sway, 0.20, 0.0, "nose"),
            "left_shoulder": _make_keypoint(0.42 + sway, 0.32, 0.0, "left_shoulder"),
            "right_shoulder": _make_keypoint(0.58 + sway, 0.32, 0.0, "right_shoulder"),
            # Lead arm (left) slightly forward
            "left_elbow": _make_keypoint(0.38, 0.42, -0.05, "left_elbow"),
            "left_wrist": _make_keypoint(0.40, 0.38, -0.10, "left_wrist"),
            # Rear arm (right) closer to body
            "right_elbow": _make_keypoint(0.62, 0.42, -0.02, "right_elbow"),
            "right_wrist": _make_keypoint(0.58, 0.35, -0.05, "right_wrist"),
            "left_hip": _make_keypoint(0.45, 0.55, 0.0, "left_hip"),
            "right_hip": _make_keypoint(0.55, 0.55, 0.0, "right_hip"),
        }
    else:  # southpaw
        keypoints = {
            "nose": _make_keypoint(0.50 + sway, 0.20, 0.0, "nose"),
            "left_shoulder": _make_keypoint(0.42 + sway, 0.32, 0.0, "left_shoulder"),
            "right_shoulder": _make_keypoint(0.58 + sway, 0.32, 0.0, "right_shoulder"),
            "left_elbow": _make_keypoint(0.38, 0.42, -0.02, "left_elbow"),
            "left_wrist": _make_keypoint(0.42, 0.35, -0.05, "left_wrist"),
            "right_elbow": _make_keypoint(0.62, 0.42, -0.05, "right_elbow"),
            "right_wrist": _make_keypoint(0.60, 0.38, -0.10, "right_wrist"),
            "left_hip": _make_keypoint(0.45, 0.55, 0.0, "left_hip"),
            "right_hip": _make_keypoint(0.55, 0.55, 0.0, "right_hip"),
        }

    return PoseFrame(keypoints=keypoints, timestamp_ms=frame_index * 33.3, frame_index=frame_index, valid=True)


def generate_jab_sequence(num_frames: int = 12, stance: str = "orthodox") -> list[PoseFrame]:
    """
    Generate a jab sequence (lead hand extends toward camera).
    In front-facing view: wrist z decreases rapidly (toward camera),
    x stays roughly same, arm extends.
    """
    frames = []
    lead_wrist_key = "left_wrist" if stance == "orthodox" else "right_wrist"
    lead_elbow_key = "left_elbow" if stance == "orthodox" else "right_elbow"

    for i in range(num_frames):
        t = i / (num_frames - 1)  # 0 to 1
        base = generate_idle_pose(i, stance)

        if t < 0.3:
            # Wind-up phase: slight pull back
            progress = t / 0.3
            z_offset = 0.03 * progress
            y_offset = 0.0
        elif t < 0.6:
            # Extension phase: rapid forward burst
            progress = (t - 0.3) / 0.3
            z_offset = 0.03 - 0.55 * progress  # strong z decrease (toward camera)
            y_offset = -0.02 * progress
        else:
            # Recovery phase: return to guard
            progress = (t - 0.6) / 0.4
            z_offset = -0.52 + 0.52 * progress
            y_offset = -0.02 * (1 - progress)

        wrist = base.keypoints[lead_wrist_key]
        elbow = base.keypoints[lead_elbow_key]

        base.keypoints[lead_wrist_key] = _make_keypoint(
            wrist.x, wrist.y + y_offset, wrist.z + z_offset, lead_wrist_key
        )
        base.keypoints[lead_elbow_key] = _make_keypoint(
            elbow.x, elbow.y + y_offset * 0.5, elbow.z + z_offset * 0.5, lead_elbow_key
        )

        frames.append(base)

    return frames


def generate_cross_sequence(num_frames: int = 15, stance: str = "orthodox") -> list[PoseFrame]:
    """
    Generate a cross (rear hand power punch toward camera).
    Similar to jab but rear hand, with shoulder rotation.
    """
    frames = []
    rear_wrist_key = "right_wrist" if stance == "orthodox" else "left_wrist"
    rear_elbow_key = "right_elbow" if stance == "orthodox" else "left_elbow"
    rear_shoulder_key = "right_shoulder" if stance == "orthodox" else "left_shoulder"
    lead_shoulder_key = "left_shoulder" if stance == "orthodox" else "right_shoulder"

    for i in range(num_frames):
        t = i / (num_frames - 1)
        base = generate_idle_pose(i, stance)

        if t < 0.25:
            progress = t / 0.25
            z_offset = 0.03 * progress
            rotation = 0.0
        elif t < 0.55:
            progress = (t - 0.25) / 0.30
            z_offset = 0.03 - 0.60 * progress  # strong z burst
            rotation = 0.04 * progress
        else:
            progress = (t - 0.55) / 0.45
            z_offset = -0.57 + 0.57 * progress
            rotation = 0.04 * (1 - progress)

        wrist = base.keypoints[rear_wrist_key]
        elbow = base.keypoints[rear_elbow_key]
        r_shoulder = base.keypoints[rear_shoulder_key]
        l_shoulder = base.keypoints[lead_shoulder_key]

        base.keypoints[rear_wrist_key] = _make_keypoint(
            wrist.x - rotation * 2, wrist.y - 0.02 * min(t / 0.5, 1.0),
            wrist.z + z_offset, rear_wrist_key
        )
        base.keypoints[rear_elbow_key] = _make_keypoint(
            elbow.x - rotation, elbow.y, elbow.z + z_offset * 0.5, rear_elbow_key
        )
        # Shoulder rotation
        base.keypoints[rear_shoulder_key] = _make_keypoint(
            r_shoulder.x - rotation, r_shoulder.y, r_shoulder.z - rotation * 2, rear_shoulder_key
        )
        base.keypoints[lead_shoulder_key] = _make_keypoint(
            l_shoulder.x + rotation * 0.5, l_shoulder.y, l_shoulder.z + rotation, lead_shoulder_key
        )

        frames.append(base)

    return frames


def generate_hook_sequence(num_frames: int = 15, stance: str = "orthodox") -> list[PoseFrame]:
    """
    Generate a hook (lead hand sweeps laterally then forward).
    In front-facing view: wrist moves in an arc — x shifts then z decreases.
    Strong lateral displacement to exceed hook thresholds.
    """
    frames = []
    lead_wrist_key = "left_wrist" if stance == "orthodox" else "right_wrist"
    lead_elbow_key = "left_elbow" if stance == "orthodox" else "right_elbow"
    direction = -1 if stance == "orthodox" else 1  # left hand hooks right, right hooks left

    for i in range(num_frames):
        t = i / (num_frames - 1)
        base = generate_idle_pose(i, stance)

        if t < 0.2:
            # Wind-up: pull arm slightly back/out
            progress = t / 0.2
            x_offset = direction * 0.06 * progress
            z_offset = 0.02 * progress
        elif t < 0.5:
            # Arc phase: sharp lateral sweep then forward
            progress = (t - 0.2) / 0.3
            x_offset = direction * 0.06 - direction * 0.25 * progress  # strong lateral sweep
            z_offset = 0.02 - 0.10 * progress
        else:
            # Recovery
            progress = (t - 0.5) / 0.5
            x_offset = direction * (-0.19) * (1 - progress)
            z_offset = -0.08 * (1 - progress)

        wrist = base.keypoints[lead_wrist_key]
        elbow = base.keypoints[lead_elbow_key]

        base.keypoints[lead_wrist_key] = _make_keypoint(
            wrist.x + x_offset, wrist.y - 0.01, wrist.z + z_offset, lead_wrist_key
        )
        base.keypoints[lead_elbow_key] = _make_keypoint(
            elbow.x + x_offset * 0.5, elbow.y, elbow.z + z_offset * 0.3, lead_elbow_key
        )

        frames.append(base)

    return frames


def generate_uppercut_sequence(num_frames: int = 14, stance: str = "orthodox") -> list[PoseFrame]:
    """
    Generate an uppercut (rear hand moves sharply upward).
    In front-facing view: wrist y decreases rapidly (up), z slightly decreases.
    Strong upward displacement to exceed uppercut thresholds.
    """
    frames = []
    rear_wrist_key = "right_wrist" if stance == "orthodox" else "left_wrist"
    rear_elbow_key = "right_elbow" if stance == "orthodox" else "left_elbow"

    for i in range(num_frames):
        t = i / (num_frames - 1)
        base = generate_idle_pose(i, stance)

        if t < 0.25:
            # Dip phase: hand drops slightly
            progress = t / 0.25
            y_offset = 0.05 * progress
            z_offset = 0.0
        elif t < 0.55:
            # Upward explosion — sharp burst
            progress = (t - 0.25) / 0.3
            y_offset = 0.05 - 0.35 * progress  # strong upward
            z_offset = -0.05 * progress  # minimal forward
        else:
            # Recovery
            progress = (t - 0.55) / 0.45
            y_offset = -0.30 * (1 - progress)
            z_offset = -0.05 * (1 - progress)

        wrist = base.keypoints[rear_wrist_key]
        elbow = base.keypoints[rear_elbow_key]

        base.keypoints[rear_wrist_key] = _make_keypoint(
            wrist.x, wrist.y + y_offset, wrist.z + z_offset, rear_wrist_key
        )
        base.keypoints[rear_elbow_key] = _make_keypoint(
            elbow.x, elbow.y + y_offset * 0.6, elbow.z + z_offset * 0.4, rear_elbow_key
        )

        frames.append(base)

    return frames


def generate_walking_sequence(num_frames: int = 30, direction: str = "right") -> list[PoseFrame]:
    """
    Generate walking: shoulders sway laterally with sustained movement.
    """
    frames = []
    dx_per_frame = 0.008 if direction == "right" else -0.008

    for i in range(num_frames):
        base = generate_idle_pose(i)
        offset = dx_per_frame * i
        sway = math.sin(i * 0.5) * 0.003  # natural body sway

        for name, kp in base.keypoints.items():
            base.keypoints[name] = _make_keypoint(
                kp.x + offset + sway, kp.y, kp.z, name, kp.visibility
            )

        frames.append(base)

    return frames


def generate_idle_sequence(num_frames: int = 30, stance: str = "orthodox") -> list[PoseFrame]:
    """Generate a sequence of idle frames with natural sway."""
    return [generate_idle_pose(i, stance) for i in range(num_frames)]
