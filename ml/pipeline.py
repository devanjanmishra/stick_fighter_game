"""
Reusable ML training and validation pipeline.

Usage:
  # Train/finetune model with labeled videos
  python ml/pipeline.py train \\
      --videos "jab:path/to/jab.mp4:17" "cross:path/to/cross.mp4:9" \\
               "hook:path/to/hook.mp4:10" "uppercut:path/to/uppercut.mp4:9" \\
               "walking:path/to/walking.mp4:0" "idle:path/to/idle.mp4:0"
      --mode finetune          # or "fresh" to train from scratch
      --epochs 80

  # Validate model against labeled test videos
  python ml/pipeline.py validate \\
      --videos "jab:path/to/test_jab.mp4:5" "cross:path/to/test_cross.mp4:3"
      --model-version latest   # or a specific version number

  # Show report from last validation (or specific version)
  python ml/pipeline.py report
      --model-version latest

  # List all model versions and their metrics
  python ml/pipeline.py history

Video format: "move_type:path/to/video.mp4:expected_count"
  - move_type: jab, cross, hook, uppercut, walking, idle (or any new type)
  - path: path to the video file
  - expected_count: number of that move in the video (0 for idle/walking)

New move types are supported automatically -- just provide training videos
with the new type name. The pipeline will extend the model's class list.
"""
import argparse
import datetime
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")

DEFAULT_CLASS_NAMES = ["idle", "jab", "cross", "hook", "uppercut", "walking"]

FEATURE_NAMES = sorted([
    "shoulder_mid_x", "shoulder_mid_y", "shoulder_width", "nose_y",
    "left_wx", "left_wy", "left_wz", "left_dx", "left_dy", "left_dz",
    "right_wx", "right_wy", "right_wz", "right_dx", "right_dy", "right_dz",
    "left_vwx", "left_vwy", "left_vwz", "left_vdx", "left_vdy", "left_vdz",
    "right_vwx", "right_vwy", "right_vwz", "right_vdx", "right_vdy", "right_vdz",
])

VEL_INDICES = [FEATURE_NAMES.index(f"{h}_{c}")
               for h in ["left", "right"]
               for c in ["vwx", "vwy", "vwz"]]


# ---------------------------------------------------------------------------
# Model architecture (must match existing trained model)
# ---------------------------------------------------------------------------
class MoveClassifierCNN(nn.Module):
    def __init__(self, n_features: int = 28, n_classes: int = 6):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=2)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Model Registry — tracks versions, training data, metrics
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self, path: str = REGISTRY_PATH):
        self._path = path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            with open(self._path) as f:
                return json.load(f)
        return {"versions": [], "latest_version": 0}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    @property
    def latest_version(self) -> int:
        return self._data.get("latest_version", 0)

    def get_version_info(self, version: int) -> dict:
        for v in self._data["versions"]:
            if v["version"] == version:
                return v
        return {}

    def get_latest_info(self) -> dict:
        if self._data["versions"]:
            return self._data["versions"][-1]
        return {}

    def register_version(self, version: int, metadata: dict) -> None:
        metadata["version"] = version
        metadata["timestamp"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        self._data["versions"].append(metadata)
        self._data["latest_version"] = version
        self._save()

    def next_version(self) -> int:
        return self.latest_version + 1

    def list_versions(self) -> list:
        return list(self._data["versions"])


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
def extract_position_features(pose) -> dict:
    """Extract 16 position features from a PoseFrame."""
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


def add_velocity_features(frames: list, window: int = 3) -> list:
    """Add 12 velocity features to each frame dict."""
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


def detect_active_frames(enriched_frames: list) -> list:
    """Identify which frames have significant hand movement."""
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


def extract_video_features(video_path: str) -> np.ndarray:
    """
    Extract per-frame 28-element feature matrix from a video via MediaPipe.
    Returns shape (N, 28) float32 array.
    """
    from core.pose_estimator import PoseEstimator, VideoSource
    from core.smoothing import PoseSmoother
    import cv2

    video_src = VideoSource(video_path)
    if not video_src.is_open:
        print(f"    ERROR: Cannot open {video_path}")
        return np.array([])

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
    del pose_estimator, smoother
    gc.collect()

    enriched = add_velocity_features(all_pos_features)

    feat_matrix = []
    for feat in enriched:
        if feat:
            row = [feat.get(k, 0.0) for k in FEATURE_NAMES]
        else:
            row = [0.0] * len(FEATURE_NAMES)
        feat_matrix.append(row)

    return np.array(feat_matrix, dtype=np.float32)


def extract_labeled_frames(video_path: str, label: str) -> list:
    """
    Extract labeled frame dicts from a video.
    For punch videos: active frames get the move label, rest get "idle".
    For idle/walking: all frames get that label.
    """
    from core.pose_estimator import PoseEstimator, VideoSource
    from core.smoothing import PoseSmoother
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
    del pose_estimator, smoother
    gc.collect()

    enriched = add_velocity_features(all_pos_features)

    if label in ("idle", "walking"):
        output = []
        for i, feat in enumerate(enriched):
            if feat:
                output.append({"index": i, "label": label, "features": feat})
        print(f"    Extracted: {len(output)} frames (all {label})")
        return output

    active_mask = detect_active_frames(enriched)
    output = []
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
        output.append({"index": i, "label": frame_label, "features": feat})

    print(f"    Extracted: {len(output)} frames "
          f"({active_count} {label}, {idle_count} idle)")
    return output


def create_windowed_dataset(all_frames_by_video: list,
                            class_names: list,
                            window_size: int = 16,
                            stride: int = 2) -> tuple:
    """Create windowed (X, y) dataset from labeled frame lists."""
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
            label_indices.append(
                class_names.index(lbl) if lbl in class_names else 0)

        feat_array = np.array(feat_matrix, dtype=np.float32)
        label_array = np.array(label_indices, dtype=np.int64)

        for start in range(0, len(feat_array) - window_size, stride):
            window = feat_array[start:start + window_size]
            win_labels = label_array[start:start + window_size]

            non_idle = win_labels[win_labels > 0]
            if len(non_idle) >= window_size // 4:
                counts = np.bincount(non_idle, minlength=len(class_names))
                label = int(counts.argmax())
            else:
                label = 0

            all_windows.append(window)
            all_labels.append(label)

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Detection (mirrors ml_move_detector.py logic for offline validation)
# ---------------------------------------------------------------------------
def compute_vel_mag(features: np.ndarray) -> np.ndarray:
    return np.array([
        np.sqrt(sum(features[i, idx] ** 2 for idx in VEL_INDICES))
        for i in range(len(features))
    ])


def detect_peaks(features: np.ndarray, min_dist: int = 10,
                 vel_thresh: float = 0.040,
                 valley_ratio: float = 0.25) -> list:
    """Detect velocity peaks with valley-based suppression."""
    vel_mag = compute_vel_mag(features)
    raw_peaks = []
    for i in range(2, len(vel_mag) - 2):
        if vel_mag[i] < vel_thresh:
            continue
        if all(vel_mag[i] >= vel_mag[i + d] for d in [-2, -1, 1, 2]):
            raw_peaks.append({"frame": i, "vel": float(vel_mag[i])})

    dist_filtered = []
    for p in raw_peaks:
        if not dist_filtered or p["frame"] - dist_filtered[-1]["frame"] >= min_dist:
            dist_filtered.append(p)
        elif p["vel"] > dist_filtered[-1]["vel"]:
            dist_filtered[-1] = p

    if len(dist_filtered) <= 1:
        return dist_filtered

    valley_filtered = [dist_filtered[0]]
    for p in dist_filtered[1:]:
        prev = valley_filtered[-1]
        start_f = prev["frame"]
        end_f = p["frame"]
        if end_f - start_f < 3:
            if p["vel"] > prev["vel"]:
                valley_filtered[-1] = p
            continue
        min_vel = float(np.min(vel_mag[start_f + 1:end_f]))
        threshold = min(prev["vel"], p["vel"]) * valley_ratio
        if min_vel <= threshold:
            valley_filtered.append(p)
        else:
            if p["vel"] > prev["vel"]:
                valley_filtered[-1] = p

    return valley_filtered


def classify_peak(features: np.ndarray, frame: int,
                  model: MoveClassifierCNN,
                  mean: np.ndarray, std: np.ndarray,
                  class_names: list,
                  window_size: int = 16,
                  hook_z_thresh: float = 0.090,
                  uppercut_y_thresh: float = 0.035) -> tuple:
    """Classify a single velocity peak. Returns (move_name, confidence, reason)."""
    half = window_size // 2
    start = max(0, frame - half)
    end = min(len(features), start + window_size)
    start = max(0, end - window_size)
    if end - start < window_size:
        return "idle", 0.0, "skip"

    window = features[start:end]
    window_norm = (window - mean) / std
    with torch.no_grad():
        x = torch.from_numpy(window_norm).unsqueeze(0).float()
        probs = torch.softmax(model(x), dim=1).squeeze().numpy()

    ml_pred = int(probs.argmax())
    ml_conf = float(probs[ml_pred])

    hw = 5
    s = max(0, frame - hw)
    e = min(len(features), frame + hw + 1)
    win = features[s:e]

    def abs_mean(feat_name):
        idx = FEATURE_NAMES.index(feat_name)
        return float(np.mean(np.abs(win[:, idx])))

    y_vel = max(abs_mean("left_vwy"), abs_mean("right_vwy"))
    x_vel = max(abs_mean("left_vwx"), abs_mean("right_vwx"))
    z_vel = max(abs_mean("left_vwz"), abs_mean("right_vwz"))

    # Hook index
    hook_idx = class_names.index("hook") if "hook" in class_names else -1
    uppercut_idx = class_names.index("uppercut") if "uppercut" in class_names else -1
    idle_idx = class_names.index("idle") if "idle" in class_names else 0
    walking_idx = class_names.index("walking") if "walking" in class_names else -1

    # Rule 1: HOOK
    if hook_idx >= 0:
        if ml_pred == hook_idx and ml_conf > 0.7:
            return "hook", ml_conf, "ML"
        if z_vel > hook_z_thresh and z_vel > y_vel * 2.0:
            return "hook", max(0.8, float(probs[hook_idx]) if hook_idx < len(probs) else 0.8), "VEL:z-dominant"

    # Rule 2: UPPERCUT
    if uppercut_idx >= 0:
        if y_vel > uppercut_y_thresh and y_vel > z_vel * 1.3 and y_vel > x_vel * 1.5:
            return "uppercut", max(0.75, float(probs[uppercut_idx]) if uppercut_idx < len(probs) else 0.75), "VEL:y-dominant"

    # Rule 3: JAB/CROSS — trust ML
    jab_idx = class_names.index("jab") if "jab" in class_names else -1
    cross_idx = class_names.index("cross") if "cross" in class_names else -1
    if ml_pred in (jab_idx, cross_idx) and ml_conf > 0.5:
        return class_names[ml_pred], ml_conf, "ML"

    # Rule 4: Trust ML for idle/walking (no fallback override)
    if ml_pred in (idle_idx, walking_idx):
        return class_names[ml_pred], ml_conf, "ML:trust-idle"

    return class_names[ml_pred], ml_conf, "ML:default"


def run_detector(features: np.ndarray, model: MoveClassifierCNN,
                 mean: np.ndarray, std: np.ndarray,
                 class_names: list,
                 min_dist: int = 10, vel_thresh: float = 0.040,
                 valley_ratio: float = 0.25,
                 hook_z_thresh: float = 0.090,
                 uppercut_y_thresh: float = 0.035,
                 heavy_gap: int = 35, light_gap: int = 12) -> tuple:
    """
    Full detection pipeline:
    1. Peak detection with valley suppression
    2. ML + heuristic classification
    3. Post-classification deduplication with type-specific gaps
    Returns (detections_list, counts_dict).
    """
    peaks = detect_peaks(features, min_dist, vel_thresh,
                         valley_ratio=valley_ratio)

    classified = []
    for p in peaks:
        pred, conf, reason = classify_peak(
            features, p["frame"], model, mean, std, class_names,
            hook_z_thresh=hook_z_thresh,
            uppercut_y_thresh=uppercut_y_thresh,
        )
        if pred in ("idle", "walking"):
            continue
        classified.append({
            "frame": p["frame"],
            "time": round(p["frame"] / 30.0, 2),
            "pred": pred,
            "conf": round(conf, 3),
            "reason": reason,
            "vel": round(p["vel"], 4),
        })

    if len(classified) <= 1:
        counts = {}
        for r in classified:
            counts[r["pred"]] = counts.get(r["pred"], 0) + 1
        return classified, counts

    # Post-classification dedup with type-specific gaps
    deduped = [classified[0]]
    for r in classified[1:]:
        prev = deduped[-1]
        gap = r["frame"] - prev["frame"]

        if r["pred"] == prev["pred"]:
            is_heavy = r["pred"] in ("hook", "uppercut")
            required_gap = heavy_gap if is_heavy else light_gap
            if gap >= required_gap:
                deduped.append(r)
            elif r["vel"] > prev["vel"]:
                deduped[-1] = r
        else:
            deduped.append(r)

    counts = {}
    for r in deduped:
        counts[r["pred"]] = counts.get(r["pred"], 0) + 1
    return deduped, counts


def score_video(counts: dict, expected_move: str,
                expected_count: int) -> tuple:
    """Score a single video. Returns (correct, false_positives, missed)."""
    if expected_move in ("idle", "walking"):
        total = sum(counts.values())
        return 0, total, 0
    elif expected_move == "mixed":
        # For mixed videos we'd need a breakdown — handled separately
        total = sum(counts.values())
        return min(total, expected_count), max(0, total - expected_count), max(0, expected_count - total)
    else:
        correct_count = counts.get(expected_move, 0)
        correct = min(correct_count, expected_count)
        over = max(0, correct_count - expected_count)
        other = sum(v for k, v in counts.items() if k != expected_move)
        fp = over + other
        missed = max(0, expected_count - correct_count)
        return correct, fp, missed


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _run_extraction_subprocess(videos: list, class_names: list,
                               output_path: str,
                               mode: str = "labeled") -> None:
    """Run feature extraction in a subprocess to isolate MediaPipe from PyTorch."""
    import subprocess as sp
    import tempfile

    job = {
        "videos": videos,
        "class_names": class_names,
        "output_path": output_path,
        "mode": mode,
    }

    job_path = output_path.replace(".npz", "_job.json")
    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)

    worker = os.path.join(os.path.dirname(__file__), "_extract_worker.py")
    result = sp.run(
        [sys.executable, worker, job_path],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Feature extraction failed (exit code {result.returncode})")

    # Clean up job file
    if os.path.exists(job_path):
        os.remove(job_path)


def train_model(videos: list, mode: str = "finetune",
                epochs: int = 80, lr: float = 0.001) -> dict:
    """
    Train or finetune the CNN model on labeled videos.

    Args:
        videos: list of dicts with keys: move_type, path, expected_count
        mode: "fresh" (train from scratch) or "finetune" (continue from existing)
        epochs: number of training epochs
        lr: learning rate

    Returns:
        dict with training results and model version info
    """
    registry = ModelRegistry()
    version = registry.next_version()

    print("=" * 70)
    print(f"TRAINING MODEL v{version} ({mode} mode, {epochs} epochs)")
    print("=" * 70)

    # Determine class names
    # Start with existing classes, add any new move types
    class_names = list(DEFAULT_CLASS_NAMES)
    for v in videos:
        if v["move_type"] not in class_names:
            print(f"  New move type detected: {v['move_type']}")
            class_names.append(v["move_type"])

    # Extract features in subprocess (isolates MediaPipe from PyTorch)
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"training_features_v{version}.npz")

    print("\n--- EXTRACTING FEATURES (subprocess) ---")
    _run_extraction_subprocess(videos, class_names, cache_path, mode="labeled")

    # Load cached features
    cached = np.load(cache_path, allow_pickle=True)
    X = cached["X"]
    y = cached["y"]

    # Load video infos
    info_path = cache_path.replace(".npz", "_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            video_infos = json.load(f)
    else:
        video_infos = [{"move_type": v["move_type"], "path": v["path"],
                        "expected_count": v["expected_count"]}
                       for v in videos]

    print(f"\n  Loaded: {X.shape[0]} windows, {X.shape[1]} timesteps, "
          f"{X.shape[2]} features")
    for i, name in enumerate(class_names):
        count = int((y == i).sum())
        if count > 0:
            print(f"    {name}: {count} windows")

    # Normalize
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    # Stratified split
    np.random.seed(42)
    indices = np.arange(len(X_norm))
    np.random.shuffle(indices)
    X_norm = X_norm[indices]
    y = y[indices]

    val_size = max(int(len(X_norm) * 0.15), 30)
    X_val, y_val = X_norm[:val_size], y[:val_size]
    X_train, y_train = X_norm[val_size:], y[val_size:]
    print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}")

    # Save norm stats
    os.makedirs(MODELS_DIR, exist_ok=True)
    norm_path = os.path.join(MODELS_DIR, f"norm_stats_v{version}.npz")
    np.savez(norm_path,
             mean=mean.squeeze((0, 1)), std=std.squeeze((0, 1)))

    # Class weights for imbalanced data
    n_classes = len(class_names)
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * n_classes

    # Create model
    device = torch.device("cpu")
    n_features = X.shape[2]
    model = MoveClassifierCNN(n_features=n_features,
                              n_classes=n_classes).to(device)

    if mode == "finetune" and registry.latest_version > 0:
        # Load existing model weights
        prev_path = os.path.join(MODELS_DIR, "move_classifier.pt")
        if os.path.exists(prev_path):
            print(f"\n  Loading existing model for finetuning...")
            prev_state = torch.load(prev_path, map_location=device,
                                    weights_only=True)
            # Handle class count changes
            prev_n_classes = prev_state["classifier.3.weight"].shape[0]
            if prev_n_classes != n_classes:
                print(f"  Class count changed: {prev_n_classes} -> {n_classes}")
                print(f"  Expanding classifier layer, keeping shared weights")
                # Load conv layers (shared)
                compatible = {}
                for k, v in prev_state.items():
                    if not k.startswith("classifier.3"):
                        compatible[k] = v
                    elif k == "classifier.3.weight":
                        new_w = model.state_dict()[k]
                        new_w[:prev_n_classes] = v
                        compatible[k] = new_w
                    elif k == "classifier.3.bias":
                        new_b = model.state_dict()[k]
                        new_b[:prev_n_classes] = v
                        compatible[k] = new_b
                model.load_state_dict(compatible)
            else:
                model.load_state_dict(prev_state)
            print(f"  Finetuning from v{registry.latest_version}")
        else:
            print(f"  No existing model found, training from scratch")

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    weights_tensor = torch.from_numpy(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5)

    train_ds = TensorDataset(torch.from_numpy(X_train),
                             torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val),
                           torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    best_val_acc = 0.0
    best_epoch = 0
    training_history = []

    print(f"\n--- TRAINING ({epochs} epochs) ---")
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)
            train_loss_sum += loss.item() * xb.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        scheduler.step(val_loss_sum / max(1, val_total))

        training_history.append({
            "epoch": epoch + 1,
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:3d}/{epochs}: "
                  f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # Save versioned model
            model_path = os.path.join(MODELS_DIR,
                                      f"move_classifier_v{version}.pt")
            torch.save(model.state_dict(), model_path)

    print(f"\n  Best val accuracy: {best_val_acc:.3f} at epoch {best_epoch}")

    # Copy best to canonical paths for backward compatibility
    best_model_path = os.path.join(MODELS_DIR,
                                   f"move_classifier_v{version}.pt")
    canonical_model = os.path.join(MODELS_DIR, "move_classifier.pt")
    canonical_norm = os.path.join(MODELS_DIR, "norm_stats.npz")

    # Load best model
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    # Save to canonical paths
    torch.save(model.state_dict(), canonical_model)
    norm_data = np.load(norm_path)
    np.savez(canonical_norm, mean=norm_data["mean"], std=norm_data["std"])

    # Save model config
    config = {
        "n_features": int(n_features),
        "n_classes": n_classes,
        "window_size": 16,
        "class_names": class_names,
        "feature_names": list(FEATURE_NAMES),
    }
    config_path = os.path.join(MODELS_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Evaluate on validation set
    print("\n--- VALIDATION SET EVALUATION ---")
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in val_dl:
            logits = model(xb.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())
    all_preds_arr = np.array(all_preds)
    all_true_arr = np.array(all_true)

    per_class_acc = {}
    print("\n  Per-class accuracy:")
    for i, name in enumerate(class_names):
        mask = all_true_arr == i
        if mask.sum() == 0:
            continue
        correct = int((all_preds_arr[mask] == i).sum())
        total = int(mask.sum())
        acc = correct / total
        per_class_acc[name] = round(acc, 3)
        print(f"    {name:>12s}: {correct}/{total} = {acc:.1%}")

    # Register version
    metadata = {
        "mode": mode,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "n_classes": n_classes,
        "class_names": class_names,
        "training_videos": video_infos,
        "dataset_size": int(X.shape[0]),
        "per_class_accuracy": per_class_acc,
        "model_file": f"move_classifier_v{version}.pt",
        "norm_file": f"norm_stats_v{version}.npz",
    }
    registry.register_version(version, metadata)
    print(f"\n  Model v{version} registered")
    print(f"  Files: {best_model_path}")
    print(f"         {norm_path}")
    print(f"  Also saved to canonical: {canonical_model}, {canonical_norm}")

    return {
        "version": version,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "class_names": class_names,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_model(videos: list, model_version: str = "latest") -> dict:
    """
    Validate the model against labeled test videos.

    Args:
        videos: list of dicts with keys: move_type, path, expected_count
        model_version: "latest" or a specific version number

    Returns:
        dict with validation results
    """
    registry = ModelRegistry()

    if model_version == "latest":
        ver = registry.latest_version
    else:
        ver = int(model_version)

    if ver == 0:
        # No versioned model, use canonical
        print("  No versioned models found, using canonical model files")
        model_path = os.path.join(MODELS_DIR, "move_classifier.pt")
        norm_path = os.path.join(MODELS_DIR, "norm_stats.npz")
        config_path = os.path.join(MODELS_DIR, "model_config.json")
    else:
        model_path = os.path.join(MODELS_DIR, f"move_classifier_v{ver}.pt")
        norm_path = os.path.join(MODELS_DIR, f"norm_stats_v{ver}.npz")
        config_path = os.path.join(MODELS_DIR, "model_config.json")
        # Fall back to canonical if versioned files don't exist
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, "move_classifier.pt")
        if not os.path.exists(norm_path):
            norm_path = os.path.join(MODELS_DIR, "norm_stats.npz")

    # Load config
    with open(config_path) as f:
        config = json.load(f)
    class_names = config["class_names"]

    print("=" * 70)
    print(f"VALIDATING MODEL v{ver} ({len(videos)} videos)")
    print(f"  Classes: {class_names}")
    print("=" * 70)

    # Load model
    device = torch.device("cpu")
    model = MoveClassifierCNN(
        n_features=config["n_features"],
        n_classes=config["n_classes"]).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    norm = np.load(norm_path)
    mean, std = norm["mean"], norm["std"]

    # Process each video
    results = []
    total_correct = 0
    total_fp = 0
    total_missed = 0
    total_expected = 0

    for v in videos:
        move_type = v["move_type"]
        expected_count = v["expected_count"]
        total_expected += expected_count

        print(f"\n  [{move_type.upper()}] {os.path.basename(v['path'])} "
              f"(expected: {expected_count})")

        if not os.path.exists(v["path"]):
            print(f"    SKIPPED — file not found: {v['path']}")
            results.append({
                "video": os.path.basename(v["path"]),
                "move_type": move_type,
                "expected_count": expected_count,
                "status": "SKIPPED",
            })
            continue

        # Extract features in subprocess to isolate MediaPipe
        feat_cache = os.path.join(
            DATA_DIR, f"val_v{ver}_{move_type}_features.npz")
        _run_extraction_subprocess(
            [v], class_names, feat_cache, mode="raw")

        feat_data = np.load(feat_cache)
        features = feat_data["features"]
        if features.size == 0:
            print(f"    SKIPPED — no features extracted")
            results.append({
                "video": os.path.basename(v["path"]),
                "move_type": move_type,
                "expected_count": expected_count,
                "status": "ERROR",
            })
            continue

        detections, counts = run_detector(
            features, model, mean, std, class_names)
        correct, fp, missed = score_video(counts, move_type, expected_count)

        total_correct += correct
        total_fp += fp
        total_missed += missed

        total_det = sum(counts.values())

        if move_type in ("idle", "walking"):
            status = "PASS" if fp == 0 else f"FAIL ({fp} FP)"
        else:
            correct_type = counts.get(move_type, 0)
            diff = abs(correct_type - expected_count)
            misclass = {k: v for k, v in counts.items() if k != move_type}
            status = "PASS" if diff <= 2 else "FAIL"
            if misclass:
                status += f" (misclassified: {misclass})"

        print(f"    Detected: {total_det} total — {counts}")
        print(f"    Correct={correct}, FP={fp}, Missed={missed} → {status}")

        for d in detections:
            print(f"    [{d['time']:5.2f}s] {d['pred'].upper():>9s} "
                  f"(conf={d['conf']:.2f}, vel={d['vel']:.4f}) {d['reason']}")

        results.append({
            "video": os.path.basename(v["path"]),
            "move_type": move_type,
            "expected_count": expected_count,
            "detected_total": total_det,
            "breakdown": dict(counts),
            "detections": detections,
            "correct": correct,
            "fp": fp,
            "missed": missed,
            "status": status,
        })

    # Summary
    accuracy = (total_correct / total_expected * 100) if total_expected > 0 else 0

    print(f"\n\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Video':<25s} {'Expected':<12s} {'Detected':<12s} "
          f"{'Correct':<10s} {'FP':<6s} {'Missed':<8s} {'Status'}")
    print(f"  {'-' * 85}")
    for r in results:
        if r.get("status") in ("SKIPPED", "ERROR"):
            print(f"  {r['video']:<25s} {'—':<12s} {'—':<12s} "
                  f"{'—':<10s} {'—':<6s} {'—':<8s} {r['status']}")
            continue
        exp_str = f"{r['expected_count']} {r['move_type']}"
        det_str = str(r["detected_total"])
        print(f"  {r['video']:<25s} {exp_str:<12s} {det_str:<12s} "
              f"{r['correct']:<10d} {r['fp']:<6d} {r['missed']:<8d} "
              f"{r['status']}")

    print(f"\n  Overall: {total_correct}/{total_expected} correct "
          f"({accuracy:.1f}%), {total_fp} FP, {total_missed} missed")

    # Save validation results
    val_results = {
        "model_version": ver,
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "total_correct": total_correct,
        "total_expected": total_expected,
        "total_fp": total_fp,
        "total_missed": total_missed,
        "accuracy_pct": round(accuracy, 1),
        "per_video": results,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    val_path = os.path.join(DATA_DIR, f"validation_v{ver}.json")
    with open(val_path, "w") as f:
        json.dump(val_results, f, indent=2)
    print(f"\n  Results saved: {val_path}")

    return val_results


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------
def generate_report(model_version: str = "latest") -> str:
    """Generate a markdown validation report."""
    registry = ModelRegistry()
    if model_version == "latest":
        ver = registry.latest_version
    else:
        ver = int(model_version)

    # Load validation results
    val_path = os.path.join(DATA_DIR, f"validation_v{ver}.json")
    if not os.path.exists(val_path):
        # Try unversioned
        val_path = os.path.join(DATA_DIR, "validation_v0.json")
        if not os.path.exists(val_path):
            return "No validation results found. Run `validate` first."

    with open(val_path) as f:
        val = json.load(f)

    # Load model info
    info = registry.get_version_info(ver) if ver > 0 else {}

    lines = [
        f"# ML Model Validation Report — v{ver}",
        "",
        f"**Generated:** {val.get('timestamp', 'unknown')}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Correct | {val['total_correct']}/{val['total_expected']} |",
        f"| Accuracy | {val['accuracy_pct']}% |",
        f"| False Positives | {val['total_fp']} |",
        f"| Missed | {val['total_missed']} |",
    ]

    if info:
        lines += [
            "",
            "## Model Info",
            "",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Mode | {info.get('mode', '?')} |",
            f"| Epochs | {info.get('epochs', '?')} |",
            f"| Best Epoch | {info.get('best_epoch', '?')} |",
            f"| Val Accuracy | {info.get('best_val_acc', '?')} |",
            f"| Classes | {', '.join(info.get('class_names', []))} |",
            f"| Dataset Size | {info.get('dataset_size', '?')} windows |",
        ]

    lines += [
        "",
        "## Per-Video Results",
        "",
        "| Video | Expected | Detected | Correct | FP | Missed | Status |",
        "|-------|----------|----------|---------|-----|--------|--------|",
    ]

    for r in val.get("per_video", []):
        if r.get("status") in ("SKIPPED", "ERROR"):
            lines.append(
                f"| {r['video']} | — | — | — | — | — | {r['status']} |")
            continue
        exp = f"{r['expected_count']} {r['move_type']}"
        lines.append(
            f"| {r['video']} | {exp} | {r.get('detected_total', '?')} | "
            f"{r.get('correct', '?')} | {r.get('fp', '?')} | "
            f"{r.get('missed', '?')} | {r['status']} |")

    lines += [
        "",
        "## Detection Details",
        "",
    ]

    for r in val.get("per_video", []):
        if r.get("status") in ("SKIPPED", "ERROR"):
            continue
        lines.append(f"### {r['video']} ({r['move_type']})")
        lines.append(f"Expected: {r['expected_count']}, "
                     f"Detected: {r.get('detected_total', '?')}, "
                     f"Breakdown: {r.get('breakdown', {})}")
        lines.append("")
        lines.append("| Time | Move | Confidence | Velocity | Reason |")
        lines.append("|------|------|------------|----------|--------|")
        for d in r.get("detections", []):
            lines.append(
                f"| {d['time']:.2f}s | {d['pred'].upper()} | "
                f"{d['conf']:.2f} | {d['vel']:.4f} | {d['reason']} |")
        lines.append("")

    report = "\n".join(lines)

    # Save report
    report_path = os.path.join(DATA_DIR, f"report_v{ver}.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    return report


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def show_history() -> None:
    """Show all model versions and their metrics."""
    registry = ModelRegistry()
    versions = registry.list_versions()

    if not versions:
        print("No model versions registered yet.")
        print("Run `python ml/pipeline.py train --videos ...` to train.")
        return

    print("=" * 70)
    print("MODEL VERSION HISTORY")
    print("=" * 70)

    for v in versions:
        ver = v["version"]
        ts = v.get("timestamp", "?")
        mode = v.get("mode", "?")
        acc = v.get("best_val_acc", "?")
        n_classes = v.get("n_classes", "?")
        classes = ", ".join(v.get("class_names", []))
        ds_size = v.get("dataset_size", "?")
        n_videos = len(v.get("training_videos", []))

        print(f"\n  v{ver} — {ts}")
        print(f"    Mode: {mode}, Epochs: {v.get('epochs', '?')}, "
              f"Best epoch: {v.get('best_epoch', '?')}")
        print(f"    Val accuracy: {acc}")
        print(f"    Classes ({n_classes}): {classes}")
        print(f"    Training: {n_videos} videos, {ds_size} windows")

        # Check if validation exists
        val_path = os.path.join(DATA_DIR, f"validation_v{ver}.json")
        if os.path.exists(val_path):
            with open(val_path) as f:
                val = json.load(f)
            print(f"    Validation: {val['total_correct']}/{val['total_expected']} "
                  f"({val['accuracy_pct']}%), "
                  f"{val['total_fp']} FP, {val['total_missed']} missed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_video_arg(arg: str) -> dict:
    """
    Parse a video argument string: "move_type:path:count"
    Returns dict with keys: move_type, path, expected_count
    """
    parts = arg.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid video format: '{arg}'. "
            f"Expected 'move_type:path/to/video.mp4:count'")

    move_type = parts[0].strip().lower()
    # Path might contain colons (e.g., C:\path on Windows),
    # so join everything except first and last
    path = ":".join(parts[1:-1]).strip()
    count = int(parts[-1].strip())

    return {
        "move_type": move_type,
        "path": os.path.expanduser(path),
        "expected_count": count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ML Training & Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Train
    train_p = subparsers.add_parser(
        "train", help="Train/finetune model with labeled videos")
    train_p.add_argument(
        "--videos", nargs="+", required=True,
        help='Videos: "move_type:path:count" ...')
    train_p.add_argument(
        "--mode", choices=["fresh", "finetune"], default="finetune",
        help="Training mode (default: finetune)")
    train_p.add_argument(
        "--epochs", type=int, default=80,
        help="Number of epochs (default: 80)")
    train_p.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)")

    # Validate
    val_p = subparsers.add_parser(
        "validate", help="Validate model against labeled test videos")
    val_p.add_argument(
        "--videos", nargs="+", required=True,
        help='Videos: "move_type:path:count" ...')
    val_p.add_argument(
        "--model-version", default="latest",
        help="Model version to validate (default: latest)")

    # Report
    report_p = subparsers.add_parser(
        "report", help="Generate validation report")
    report_p.add_argument(
        "--model-version", default="latest",
        help="Model version (default: latest)")

    # History
    subparsers.add_parser("history", help="Show model version history")

    args = parser.parse_args()

    if args.command == "train":
        videos = [parse_video_arg(v) for v in args.videos]
        result = train_model(videos, mode=args.mode,
                             epochs=args.epochs, lr=args.lr)
        print(f"\n  Training complete! Model v{result['version']} saved.")
        print(f"\n  To validate:")
        print(f"    python ml/pipeline.py validate --videos ...")
        print(f"\n  To see report:")
        print(f"    python ml/pipeline.py report")

    elif args.command == "validate":
        videos = [parse_video_arg(v) for v in args.videos]
        result = validate_model(videos,
                                model_version=args.model_version)
        print(f"\n  To generate report:")
        print(f"    python ml/pipeline.py report")

    elif args.command == "report":
        report = generate_report(
            model_version=args.model_version)
        print(report)

    elif args.command == "history":
        show_history()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
