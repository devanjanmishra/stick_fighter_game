"""
Milestone 1 Test: Verify pose estimator works with synthetic video frames.
- Creates synthetic BGR frames with a simple figure drawn on them
- Runs MediaPipe pose estimation on them
- Validates keypoint extraction produces expected structure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from core.pose_estimator import PoseEstimator, PoseFrame, VideoSource


def create_synthetic_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a blank BGR frame for testing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (200, 200, 200)  # light gray background
    # Draw a simple stick figure that MediaPipe might detect
    # Head
    cv2.circle(frame, (320, 100), 30, (100, 80, 60), -1)
    # Body
    cv2.line(frame, (320, 130), (320, 280), (100, 80, 60), 8)
    # Arms
    cv2.line(frame, (320, 170), (240, 230), (100, 80, 60), 6)
    cv2.line(frame, (320, 170), (400, 230), (100, 80, 60), 6)
    # Legs
    cv2.line(frame, (320, 280), (270, 400), (100, 80, 60), 6)
    cv2.line(frame, (320, 280), (370, 400), (100, 80, 60), 6)
    return frame


def test_pose_estimator_init():
    """Test that PoseEstimator initializes correctly."""
    estimator = PoseEstimator(upper_body_only=True)
    assert estimator is not None
    assert estimator.upper_body_only is True
    estimator.close()
    print("[PASS] PoseEstimator initializes correctly")


def test_pose_estimator_processes_frame():
    """Test that process_frame returns a PoseFrame."""
    estimator = PoseEstimator(upper_body_only=True)
    frame = create_synthetic_frame()

    pose_frame = estimator.process_frame(frame, timestamp_ms=0.0)

    assert isinstance(pose_frame, PoseFrame)
    assert pose_frame.frame_index == 0
    assert pose_frame.timestamp_ms == 0.0
    # Note: MediaPipe may not detect a pose in a crude synthetic drawing,
    # but the pipeline should not crash
    print(f"[PASS] process_frame returns PoseFrame (valid={pose_frame.valid}, "
          f"keypoints={len(pose_frame.keypoints)})")

    estimator.close()


def test_pose_frame_properties():
    """Test PoseFrame helper properties with synthetic keypoint data."""
    from tests.synthetic_data import generate_idle_pose

    pose = generate_idle_pose(0, "orthodox")

    assert pose.valid is True
    assert len(pose.keypoints) == 9  # upper body keypoints

    # Test shoulder midpoint
    mid = pose.shoulder_midpoint
    assert mid is not None
    assert 0.45 < mid[0] < 0.55  # roughly centered
    print(f"[PASS] Shoulder midpoint: ({mid[0]:.3f}, {mid[1]:.3f}, {mid[2]:.3f})")

    # Test hip midpoint
    hip = pose.hip_midpoint
    assert hip is not None
    assert 0.45 < hip[0] < 0.55
    print(f"[PASS] Hip midpoint: ({hip[0]:.3f}, {hip[1]:.3f}, {hip[2]:.3f})")

    # Test get
    nose = pose.get("nose")
    assert nose is not None
    assert nose.name == "nose"
    print(f"[PASS] Nose keypoint: ({nose.x:.3f}, {nose.y:.3f}, {nose.z:.3f})")


def test_synthetic_sequences():
    """Test that synthetic data generators produce valid sequences."""
    from tests.synthetic_data import (
        generate_idle_sequence,
        generate_jab_sequence,
        generate_cross_sequence,
        generate_hook_sequence,
        generate_uppercut_sequence,
        generate_walking_sequence,
    )

    sequences = {
        "idle": generate_idle_sequence(30),
        "jab": generate_jab_sequence(12),
        "cross": generate_cross_sequence(15),
        "hook": generate_hook_sequence(15),
        "uppercut": generate_uppercut_sequence(14),
        "walking_right": generate_walking_sequence(30, "right"),
    }

    for name, seq in sequences.items():
        assert len(seq) > 0, f"{name} sequence is empty"
        for frame in seq:
            assert frame.valid, f"{name} frame {frame.frame_index} is not valid"
            assert len(frame.keypoints) == 9, f"{name} has wrong keypoint count"
        print(f"[PASS] {name}: {len(seq)} frames, all valid")

    # Verify jab has z-depth change on lead wrist
    jab = sequences["jab"]
    start_z = jab[0].get("left_wrist").z
    mid_z = jab[len(jab) // 2 + 2].get("left_wrist").z
    assert mid_z < start_z, "Jab should decrease z (extend toward camera)"
    print(f"[PASS] Jab z-depth: start={start_z:.3f}, mid={mid_z:.3f} (extends forward)")

    # Verify walking has x-position change
    walk = sequences["walking_right"]
    start_x = walk[0].get("nose").x
    end_x = walk[-1].get("nose").x
    assert end_x > start_x, "Walking right should increase x"
    print(f"[PASS] Walking x: start={start_x:.3f}, end={end_x:.3f} (moves right)")


def test_video_source_with_file():
    """Test VideoSource with a generated video file."""
    # Create a short test video
    test_video_path = "/tmp/test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(test_video_path, fourcc, 30, (640, 480))

    for _ in range(10):
        frame = create_synthetic_frame()
        writer.write(frame)
    writer.release()

    # Test VideoSource
    source = VideoSource(test_video_path)
    assert source.is_open, "VideoSource should be open"

    frames_read = 0
    while True:
        ret, frame, ts = source.read()
        if not ret:
            break
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        frames_read += 1

    assert frames_read == 10, f"Expected 10 frames, got {frames_read}"
    source.close()

    os.remove(test_video_path)
    print(f"[PASS] VideoSource read {frames_read} frames from video file")


if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 1 TESTS: Camera Capture + Pose Estimation")
    print("=" * 60)

    test_pose_estimator_init()
    test_pose_estimator_processes_frame()
    test_pose_frame_properties()
    test_synthetic_sequences()
    test_video_source_with_file()

    print("=" * 60)
    print("ALL MILESTONE 1 TESTS PASSED")
    print("=" * 60)
