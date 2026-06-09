"""
Step 1: Extract features from ALL labeled videos via MediaPipe.
Saves windowed dataset to ml/data/all_videos_dataset.npz
"""
import os
import sys
import math
import json
import numpy as np

sys.path.insert(0, '/home/ubuntu/stick_fighter')

from core.pose_estimator import PoseEstimator, PoseFrame, VideoSource
from core.smoothing import PoseSmoother

FEATURE_NAMES = sorted([
    "shoulder_mid_x", "shoulder_mid_y", "shoulder_width", "nose_y",
    "left_wx", "left_wy", "left_wz", "left_dx", "left_dy", "left_dz",
    "right_wx", "right_wy", "right_wz", "right_dx", "right_dy", "right_dz",
    "left_vwx", "left_vwy", "left_vwz", "left_vdx", "left_vdy", "left_vdz",
    "right_vwx", "right_vwy", "right_vwz", "right_vdx", "right_vdy", "right_vdz",
])

CLASS_NAMES = ["idle", "jab", "cross", "hook", "uppercut", "walking"]


def extract_position_features(pose: PoseFrame) -> dict:
    if not pose.valid:
        return {}
    kps = pose.keypoints
    ls = kps.get("left_shoulder")
    rs = kps.get("right_shoulder")
    nose = kps.get("nose")
    if not ls or not rs:
        return {}

    shoulder_mid_x = (ls.x + rs.x) / 2
    shoulder_mid_y = (ls.y + rs.y) / 2
    shoulder_width = math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2)
    if shoulder_width < 0.01:
        shoulder_width = 0.1

    features = {
        "shoulder_mid_x": shoulder_mid_x,
        "shoulder_mid_y": shoulder_mid_y,
        "shoulder_width": shoulder_width,
        "nose_y": nose.y if nose else shoulder_mid_y - 0.1,
    }
    for hand in ["left", "right"]:
        wrist = kps.get(f"{hand}_wrist")
        shoulder = kps.get(f"{hand}_shoulder")
        if not wrist or not shoulder:
            for suffix in ["wx", "wy", "wz", "dx", "dy", "dz"]:
                features[f"{hand}_{suffix}"] = 0.0
            continue
        features[f"{hand}_wx"] = wrist.x
        features[f"{hand}_wy"] = wrist.y
        features[f"{hand}_wz"] = wrist.z
        features[f"{hand}_dx"] = (wrist.x - shoulder.x) / shoulder_width
        features[f"{hand}_dy"] = (wrist.y - shoulder.y) / shoulder_width
        features[f"{hand}_dz"] = (wrist.z - shoulder.z) / shoulder_width
    return features


def add_velocity_features(frames: list[dict], window: int = 3) -> list[dict]:
    enriched = []
    for i, feat in enumerate(frames):
        if not feat:
            enriched.append({})
            continue
        new_feat = dict(feat)
        for hand in ["left", "right"]:
            for coord in ["wx", "wy", "wz", "dx", "dy", "dz"]:
                key = f"{hand}_{coord}"
                vel_key = f"{hand}_v{coord}"
                if i < window:
                    new_feat[vel_key] = 0.0
                else:
                    prev = frames[i - window]
                    if prev and key in prev and key in feat:
                        new_feat[vel_key] = (feat[key] - prev[key]) / window
                    else:
                        new_feat[vel_key] = 0.0
        enriched.append(new_feat)
    return enriched


def detect_active_frames(enriched_frames: list[dict]) -> list[bool]:
    is_active = []
    for feat in enriched_frames:
        if not feat:
            is_active.append(False)
            continue
        vel_mag = 0.0
        for hand in ["left", "right"]:
            for coord in ["vwx", "vwy", "vwz"]:
                v = feat.get(f"{hand}_{coord}", 0.0)
                vel_mag += v * v
        vel_mag = math.sqrt(vel_mag)
        is_active.append(vel_mag > 0.015)
    return is_active


def process_video(video_path: str, label: str) -> list[dict]:
    print(f"\n  Processing: {os.path.basename(video_path)} (label={label})")
    import cv2
    video_src = VideoSource(video_path)
    if not video_src.is_open:
        print(f"    ERROR: Cannot open {video_path}")
        return []

    fps = video_src.cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = video_src.total_frames
    print(f"    {total} frames @ {fps:.0f} fps")

    pose_estimator = PoseEstimator(running_mode="VIDEO")
    smoother = PoseSmoother()

    all_pos_features = []
    while True:
        ret, frame, ts_ms = video_src.read()
        if not ret:
            break
        pose = pose_estimator.process_frame(frame, ts_ms)
        smoothed = smoother.smooth(pose)
        features = extract_position_features(smoothed)
        all_pos_features.append(features)
    video_src.close()

    enriched = add_velocity_features(all_pos_features)

    if label in ("idle", "walking"):
        output_frames = []
        for i, feat in enumerate(enriched):
            if feat:
                output_frames.append({"index": i, "label": label, "features": feat})
        active_count = len(output_frames)
        idle_count = 0
    else:
        active_mask = detect_active_frames(enriched)
        output_frames = []
        active_count = 0
        idle_count = 0
        for i, (feat, is_active) in enumerate(zip(enriched, active_mask)):
            if not feat:
                continue
            frame_label = label if is_active else "idle"
            if is_active:
                active_count += 1
            else:
                idle_count += 1
            output_frames.append({"index": i, "label": frame_label, "features": feat})

    print(f"    Extracted: {len(output_frames)} frames "
          f"({active_count} {label}, {idle_count} idle)")
    return output_frames


def create_windowed_dataset(all_frames_by_video, window_size=16, stride=2):
    all_windows = []
    all_labels = []

    for video_frames in all_frames_by_video:
        if not video_frames:
            continue

        feat_matrix = []
        label_indices = []
        for fr in video_frames:
            row = [fr["features"].get(k, 0.0) for k in FEATURE_NAMES]
            feat_matrix.append(row)
            lbl = fr["label"]
            label_indices.append(CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else 0)

        feat_array = np.array(feat_matrix, dtype=np.float32)
        label_array = np.array(label_indices, dtype=np.int64)

        for start in range(0, len(feat_array) - window_size, stride):
            window = feat_array[start:start + window_size]
            win_labels = label_array[start:start + window_size]

            non_idle = win_labels[win_labels > 0]
            if len(non_idle) >= window_size // 4:
                counts = np.bincount(non_idle, minlength=len(CLASS_NAMES))
                label = int(counts.argmax())
            else:
                label = 0

            all_windows.append(window)
            all_labels.append(label)

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


def main():
    print("=" * 60)
    print("STEP 1: EXTRACT FEATURES FROM ALL LABELED VIDEOS")
    print("=" * 60)

    videos = [
        (os.path.expanduser("~/attachments/7c242094-6815-4982-8183-6ba478e39761/jab.mp4"), "jab"),
        (os.path.expanduser("~/attachments/f6777eed-4fc2-4feb-8c3b-9abc9d57dfd0/cross.mp4"), "cross"),
        (os.path.expanduser("~/attachments/e707b2dc-e951-4254-9392-d820e336513c/hook.mp4"), "hook"),
        (os.path.expanduser("~/attachments/624ad648-bd08-49bf-a27c-a490130fcc32/uppercut.mp4"), "uppercut"),
        (os.path.expanduser("~/attachments/da64e209-13fa-4872-b035-300051dc4292/walking_back_forth.mp4"), "walking"),
        (os.path.expanduser("~/attachments/d14f6498-0b85-481e-93da-0b2a78819631/idling_notwalking.mp4"), "idle"),
    ]

    all_video_frames = []
    for path, label in videos:
        frames = process_video(path, label)
        all_video_frames.append(frames)

    print("\n--- Creating windowed dataset ---")
    X, y = create_windowed_dataset(all_video_frames, window_size=16, stride=2)
    print(f"  Total: {X.shape[0]} windows, {X.shape[1]} timesteps, {X.shape[2]} features")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {int((y == i).sum())} windows")

    os.makedirs("ml/data", exist_ok=True)
    np.savez("ml/data/all_videos_dataset.npz", X=X, y=y,
             feature_names=FEATURE_NAMES, class_names=CLASS_NAMES)
    print(f"\n  Saved to: ml/data/all_videos_dataset.npz")

    # Also extract features from the mixed validation video
    print("\n--- Extracting mixed validation video ---")
    mixed_path = os.path.expanduser(
        "~/attachments/772c0aa0-9814-4cb5-b4f1-eace406bb88e/WIN_20260516_18_45_10_Pro.mp4")
    mixed_frames = process_video(mixed_path, "unknown")

    # Save raw frame features for validation
    mixed_features = []
    for fr in mixed_frames:
        row = [fr["features"].get(k, 0.0) for k in FEATURE_NAMES]
        mixed_features.append(row)
    mixed_X = np.array(mixed_features, dtype=np.float32)
    np.savez("ml/data/mixed_video_features.npz", X=mixed_X,
             feature_names=FEATURE_NAMES)
    print(f"  Saved mixed video features: {mixed_X.shape}")

    print("\n  STEP 1 COMPLETE")


if __name__ == "__main__":
    main()
