"""
Process labeled per-move-type videos through MediaPipe to extract
training data with known labels.

Each video contains ONLY one move type (multiple reps).
We extract per-frame features and label all non-idle frames with that move type.
"""
import os
import sys
import json
import math
import numpy as np

sys.path.insert(0, '/home/ubuntu/stick_fighter')

from core.pose_estimator import PoseEstimator, PoseFrame, VideoSource
from core.smoothing import PoseSmoother

# Canonical alphabetical feature order
FEATURE_NAMES = sorted([
    "shoulder_mid_x", "shoulder_mid_y", "shoulder_width", "nose_y",
    "left_wx", "left_wy", "left_wz", "left_dx", "left_dy", "left_dz",
    "right_wx", "right_wy", "right_wz", "right_dx", "right_dy", "right_dz",
    "left_vwx", "left_vwy", "left_vwz", "left_vdx", "left_vdy", "left_vdz",
    "right_vwx", "right_vwy", "right_vwz", "right_vdx", "right_vdy", "right_vdz",
])


def extract_position_features(pose: PoseFrame) -> dict:
    """Extract position features from a single pose frame."""
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
    """Add velocity features computed over sliding window."""
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


def detect_active_frames(enriched_frames: list[dict], move_type: str) -> list[bool]:
    """Heuristic: detect which frames have active movement vs idle.

    Uses velocity magnitude to identify active punch frames.
    Frames with high wrist velocity are labeled as the move type;
    frames with low velocity are labeled as idle.
    """
    is_active = []

    for feat in enriched_frames:
        if not feat:
            is_active.append(False)
            continue

        # Total wrist velocity (both hands)
        vel_mag = 0.0
        for hand in ["left", "right"]:
            for coord in ["vwx", "vwy", "vwz"]:
                key = f"{hand}_{coord}"
                v = feat.get(key, 0.0)
                vel_mag += v * v

        vel_mag = math.sqrt(vel_mag)

        # Threshold for "active" — calibrated from real data analysis
        # Idle typically has vel_mag < 0.01, active moves have vel_mag > 0.02
        is_active.append(vel_mag > 0.015)

    return is_active


def process_video(video_path: str, label: str) -> dict:
    """Process a labeled video and extract features with labels.

    Returns dict with:
      - frames: list of {index, label, features}
      - stats: summary statistics
    """
    print(f"\nProcessing: {video_path} (label={label})")

    video_src = VideoSource(video_path)
    if not video_src.is_open:
        print(f"  ERROR: Cannot open {video_path}")
        return {"frames": [], "stats": {}}

    import cv2
    fps = video_src.cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = video_src.total_frames
    print(f"  {total} frames @ {fps:.0f} fps")

    pose_estimator = PoseEstimator(running_mode="VIDEO")
    smoother = PoseSmoother()

    # Extract position features for all frames
    all_pos_features = []
    frame_idx = 0
    while True:
        ret, frame, ts_ms = video_src.read()
        if not ret:
            break
        pose = pose_estimator.process_frame(frame, ts_ms)
        smoothed = smoother.smooth(pose)
        features = extract_position_features(smoothed)
        all_pos_features.append(features)
        frame_idx += 1
    video_src.close()

    # Add velocity features
    enriched = add_velocity_features(all_pos_features)

    # Detect active vs idle frames
    active_mask = detect_active_frames(enriched, label)

    # Build output
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

        output_frames.append({
            "index": i,
            "label": frame_label,
            "features": feat,
        })

    stats = {
        "video": os.path.basename(video_path),
        "label": label,
        "total_frames": frame_idx,
        "active_frames": active_count,
        "idle_frames": idle_count,
        "active_ratio": active_count / max(1, frame_idx),
    }

    print(f"  Extracted: {len(output_frames)} frames "
          f"({active_count} {label}, {idle_count} idle)")

    return {"frames": output_frames, "stats": stats, "feature_names": FEATURE_NAMES}


def create_windowed_dataset(
    all_video_data: list[dict],
    window_size: int = 16,
    stride: int = 2,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create windowed dataset from all processed videos.

    Windows are labeled by the majority non-idle class.
    Uses canonical FEATURE_NAMES ordering for all features.
    """
    if class_names is None:
        class_names = ["idle", "jab", "cross", "hook", "uppercut"]

    all_windows = []
    all_labels = []

    for video_data in all_video_data:
        frames = video_data["frames"]
        if not frames:
            continue

        # Build feature matrix in canonical order
        feat_matrix = []
        label_indices = []
        for fr in frames:
            row = [fr["features"].get(k, 0.0) for k in FEATURE_NAMES]
            feat_matrix.append(row)
            lbl = fr["label"]
            label_indices.append(
                class_names.index(lbl) if lbl in class_names else 0
            )

        feat_array = np.array(feat_matrix, dtype=np.float32)
        label_array = np.array(label_indices, dtype=np.int64)

        # Extract windows
        for start in range(0, len(feat_array) - window_size, stride):
            window = feat_array[start:start + window_size]
            win_labels = label_array[start:start + window_size]

            non_idle = win_labels[win_labels > 0]
            if len(non_idle) >= window_size // 4:
                counts = np.bincount(non_idle, minlength=len(class_names))
                label = int(counts.argmax())
            else:
                label = 0  # idle

            all_windows.append(window)
            all_labels.append(label)

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


def main():
    # Video paths and labels
    videos = [
        ("~/attachments/7c242094-6815-4982-8183-6ba478e39761/jab.mp4", "jab"),
        ("~/attachments/f6777eed-4fc2-4feb-8c3b-9abc9d57dfd0/cross.mp4", "cross"),
        ("~/attachments/e707b2dc-e951-4254-9392-d820e336513c/hook.mp4", "hook"),
        ("~/attachments/624ad648-bd08-49bf-a27c-a490130fcc32/uppercut.mp4", "uppercut"),
    ]

    # Also include the original mixed video
    original_video = (
        "~/attachments/772c0aa0-9814-4cb5-b4f1-eace406bb88e/WIN_20260516_18_45_10_Pro.mp4",
        "mixed",  # label determined by rule-based detector
    )

    class_names = ["idle", "jab", "cross", "hook", "uppercut"]

    all_video_data = []
    all_stats = []

    # Process labeled videos
    for path, label in videos:
        path = os.path.expanduser(path)
        result = process_video(path, label)
        all_video_data.append(result)
        all_stats.append(result["stats"])

    # Create windowed dataset
    print("\n" + "=" * 60)
    print("CREATING WINDOWED DATASET")
    print("=" * 60)

    X, y = create_windowed_dataset(all_video_data, window_size=16, stride=2)

    print(f"\nDataset shape: {X.shape}")
    print(f"Class distribution:")
    for i, name in enumerate(class_names):
        count = int((y == i).sum())
        print(f"  {name}: {count}")

    # Save
    os.makedirs("ml/data", exist_ok=True)
    np.savez(
        "ml/data/real_video_dataset.npz",
        X=X, y=y, class_names=class_names,
    )
    print(f"\nSaved to ml/data/real_video_dataset.npz")

    # Also save raw per-video data for analysis
    for i, (result, (path, label)) in enumerate(zip(all_video_data, videos)):
        output_path = f"ml/data/{label}_features.json"
        save_data = {
            "video_path": path,
            "label": label,
            "feature_names": FEATURE_NAMES,
            "stats": result["stats"],
            "frames": result["frames"],
        }
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Saved {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in all_stats:
        print(f"  {s['video']}: {s['total_frames']} frames, "
              f"{s['active_frames']} {s['label']}, {s['idle_frames']} idle "
              f"({s['active_ratio']:.1%} active)")

    return X, y, class_names


if __name__ == "__main__":
    main()
