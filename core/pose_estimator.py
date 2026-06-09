"""
Pose estimation wrapper around MediaPipe PoseLandmarker (tasks API v0.10+).
Extracts upper-body keypoints from camera frames or video files.
Swappable interface — can replace MediaPipe with MoveNet later.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Model path
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "pose_landmarker_lite.task",
)

# MediaPipe landmark indices for upper body
LANDMARK_NAMES = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}

# Full body landmarks (for later leg extension)
FULL_BODY_LANDMARKS = {
    **LANDMARK_NAMES,
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

# Minimum visibility score to treat a keypoint as valid.
# Below this, MediaPipe is extrapolating (hand out of frame).
# Extrapolated positions create huge velocity spikes -> false move detections.
MIN_VISIBILITY = 0.5


@dataclass
class Keypoint:
    x: float  # normalized 0-1 (left-right in camera frame)
    y: float  # normalized 0-1 (top-bottom in camera frame)
    z: float  # depth relative to hip (negative = toward camera)
    visibility: float  # 0-1 confidence
    name: str = ""


@dataclass
class PoseFrame:
    keypoints: dict[str, Keypoint] = field(default_factory=dict)
    timestamp_ms: float = 0.0
    frame_index: int = 0
    valid: bool = False

    def get(self, name: str) -> Optional[Keypoint]:
        return self.keypoints.get(name)

    @property
    def shoulder_midpoint(self) -> Optional[tuple[float, float, float]]:
        ls = self.get("left_shoulder")
        rs = self.get("right_shoulder")
        if ls and rs:
            return (
                (ls.x + rs.x) / 2,
                (ls.y + rs.y) / 2,
                (ls.z + rs.z) / 2,
            )
        return None

    @property
    def hip_midpoint(self) -> Optional[tuple[float, float, float]]:
        lh = self.get("left_hip")
        rh = self.get("right_hip")
        if lh and rh:
            return (
                (lh.x + rh.x) / 2,
                (lh.y + rh.y) / 2,
                (lh.z + rh.z) / 2,
            )
        return None


class PoseEstimator:
    """Wraps MediaPipe PoseLandmarker (tasks API) for keypoint extraction."""

    def __init__(
        self,
        upper_body_only: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str = MODEL_PATH,
        running_mode: str = "VIDEO",
    ):
        self.upper_body_only = upper_body_only
        self.landmark_map = LANDMARK_NAMES if upper_body_only else FULL_BODY_LANDMARKS

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        mode_map = {
            "IMAGE": RunningMode.IMAGE,
            "VIDEO": RunningMode.VIDEO,
            "LIVE_STREAM": RunningMode.LIVE_STREAM,
        }

        options_kwargs = {
            "base_options": BaseOptions(model_asset_path=model_path),
            "running_mode": mode_map[running_mode],
            "min_pose_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
            "num_poses": 1,
        }

        if running_mode == "LIVE_STREAM":
            self._latest_result = None
            options_kwargs["result_callback"] = self._live_callback

        self.options = PoseLandmarkerOptions(**options_kwargs)
        self.landmarker = PoseLandmarker.create_from_options(self.options)
        self._frame_index = 0
        self._running_mode = running_mode

    def _live_callback(self, result, output_image, timestamp_ms):
        self._latest_result = result

    def process_frame(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> PoseFrame:
        """Process a BGR frame and return extracted keypoints.

        Only keypoints with visibility >= MIN_VISIBILITY are included.
        Invisible keypoints (hands out of frame) are omitted entirely so
        the move detector and smoother never see extrapolated positions.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_frame = PoseFrame(
            timestamp_ms=timestamp_ms,
            frame_index=self._frame_index,
        )
        self._frame_index += 1

        if self._running_mode == "IMAGE":
            result = self.landmarker.detect(mp_image)
        elif self._running_mode == "VIDEO":
            result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        else:
            self.landmarker.detect_async(mp_image, int(timestamp_ms))
            result = self._latest_result
            if result is None:
                return pose_frame

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            any_valid = False
            for idx, name in self.landmark_map.items():
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    visibility = 0.9
                    if hasattr(lm, "visibility") and lm.visibility is not None:
                        visibility = lm.visibility
                    elif hasattr(lm, "presence") and lm.presence is not None:
                        visibility = lm.presence

                    # Skip low-visibility keypoints entirely.
                    # Wrists/elbows below threshold mean the hand is out of frame.
                    # Including them would cause velocity spikes -> false moves.
                    # Shoulders and hips get a lower threshold (body is usually visible).
                    is_extremity = name in ("left_wrist", "right_wrist",
                                            "left_elbow", "right_elbow")
                    # 0.35 for wrists/elbows (was 0.5): left wrist often just
                    # barely visible when arm is bent in guard position.
                    # Raising to 0.5 was causing JAB to never be detected
                    # because the left wrist was always below threshold.
                    threshold = 0.35 if is_extremity else 0.25

                    if visibility >= threshold:
                        pose_frame.keypoints[name] = Keypoint(
                            x=lm.x,
                            y=lm.y,
                            z=lm.z,
                            visibility=visibility,
                            name=name,
                        )
                        any_valid = True

            # Frame is valid if at least shoulders are visible
            ls = pose_frame.keypoints.get("left_shoulder")
            rs = pose_frame.keypoints.get("right_shoulder")
            pose_frame.valid = bool(ls and rs)

        return pose_frame

    def close(self):
        self.landmarker.close()


class VideoSource:
    """Captures frames from a camera or video file."""

    def __init__(self, source: int | str = 0, target_fps: int = 30):
        """
        source: 0 for default camera, or path to video file
        """
        self.cap = cv2.VideoCapture(source)
        self.target_fps = target_fps

        if isinstance(source, str):
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            self.source_fps = target_fps

        self._frame_count = 0

    @property
    def is_open(self) -> bool:
        return self.cap.isOpened()

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        """Returns (success, frame, timestamp_ms)."""
        ret, frame = self.cap.read()
        if ret:
            timestamp_ms = (self._frame_count / self.source_fps) * 1000
            self._frame_count += 1
            return True, frame, timestamp_ms
        return False, None, 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def close(self):
        self.cap.release()
