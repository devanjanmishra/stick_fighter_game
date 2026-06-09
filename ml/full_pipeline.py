"""
Full ML pipeline: process all labeled videos, fix feature ordering,
generate calibrated synthetic data, train model, and validate.

Videos:
  - jab.mp4, cross.mp4, hook.mp4, uppercut.mp4 (punch types)
  - walking_back_forth.mp4 (walking)
  - idling_notwalking.mp4 (idle/standing)
  - WIN_20260516_18_45_10_Pro.mp4 (mixed - for validation only)
"""
import os
import sys
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '/home/ubuntu/stick_fighter')

from core.pose_estimator import PoseEstimator, PoseFrame, VideoSource
from core.smoothing import PoseSmoother

# ============================================================
# CANONICAL FEATURE ORDER — alphabetical, used EVERYWHERE
# ============================================================
FEATURE_NAMES = sorted([
    "shoulder_mid_x", "shoulder_mid_y", "shoulder_width", "nose_y",
    "left_wx", "left_wy", "left_wz", "left_dx", "left_dy", "left_dz",
    "right_wx", "right_wy", "right_wz", "right_dx", "right_dy", "right_dz",
    "left_vwx", "left_vwy", "left_vwz", "left_vdx", "left_vdy", "left_vdz",
    "right_vwx", "right_vwy", "right_vwz", "right_vdx", "right_vdy", "right_vdz",
])

CLASS_NAMES = ["idle", "jab", "cross", "hook", "uppercut", "walking"]


# ============================================================
# STEP 1: FEATURE EXTRACTION FROM VIDEO
# ============================================================

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


def detect_active_frames(enriched_frames: list[dict]) -> list[bool]:
    """Detect frames with active movement using velocity magnitude."""
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
    """Process a video and extract per-frame features with labels."""
    print(f"\n  Processing: {os.path.basename(video_path)} (label={label})")

    video_src = VideoSource(video_path)
    if not video_src.is_open:
        print(f"    ERROR: Cannot open {video_path}")
        return []

    import cv2
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

    # For idle and walking videos, ALL frames get that label
    # For punch videos, use velocity to distinguish active vs idle
    if label in ("idle", "walking"):
        # All frames are labeled as the video type
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


def create_windowed_dataset(
    all_frames_by_video: list[list[dict]],
    window_size: int = 16,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create windowed dataset from processed video frames."""
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
                label = 0  # idle

            all_windows.append(window)
            all_labels.append(label)

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


# ============================================================
# STEP 2: MODEL DEFINITION
# ============================================================

class MoveClassifierCNN(nn.Module):
    """Lightweight 1D-CNN for move classification."""
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
        x = x.permute(0, 2, 1)  # (B, seq, feat) -> (B, feat, seq)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=2)  # global average pooling
        return self.classifier(x)


# ============================================================
# STEP 3: TRAINING
# ============================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 6,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 0.001,
    output_dir: str = "ml/models",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    n_features = X_train.shape[2]
    model = MoveClassifierCNN(n_features=n_features, n_classes=n_classes).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    weights_tensor = torch.from_numpy(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    best_val_acc = 0.0
    best_epoch = 0

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
            train_loss_sum += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}: "
                  f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(output_dir, "move_classifier.pt"))

    print(f"  Best val accuracy: {best_val_acc:.3f} at epoch {best_epoch}")

    # Load best model and compute per-class accuracy
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "move_classifier.pt"), weights_only=True))
    model.eval()

    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in val_dl:
            logits = model(xb.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    print("\n  Per-class validation:")
    for i, name in enumerate(CLASS_NAMES):
        mask = all_true == i
        if mask.sum() == 0:
            continue
        correct = (all_preds[mask] == i).sum()
        total = mask.sum()
        print(f"    {name:>10s}: {correct}/{total} = {correct/total:.1%}")

    # Confusion matrix
    print("\n  Confusion matrix (rows=true, cols=predicted):")
    header = "            " + " ".join(f"{n[:6]:>7s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = []
        for j in range(len(CLASS_NAMES)):
            count = int(((all_true == i) & (all_preds == j)).sum())
            row.append(f"{count:7d}")
        print(f"    {name:>8s} " + " ".join(row))

    return model


# ============================================================
# STEP 4: VALIDATION ON MIXED VIDEO
# ============================================================

def validate_on_mixed_video(
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    video_path: str,
    window_size: int = 16,
):
    """Run the trained model on the original mixed video to count detections."""
    print(f"\n  Validating on: {os.path.basename(video_path)}")

    video_src = VideoSource(video_path)
    if not video_src.is_open:
        print("    ERROR: Cannot open video")
        return

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
    print(f"    {len(enriched)} frames processed")

    # Build feature windows and classify
    device = torch.device("cpu")
    model.eval()

    frame_predictions = []
    frame_confidences = []

    buffer = []
    for i, feat in enumerate(enriched):
        if not feat:
            frame_predictions.append(0)
            frame_confidences.append(0.0)
            continue

        row = [feat.get(k, 0.0) for k in FEATURE_NAMES]
        buffer.append(row)

        if len(buffer) > window_size + 5:
            buffer = buffer[-(window_size + 2):]

        if len(buffer) < window_size:
            frame_predictions.append(0)
            frame_confidences.append(0.0)
            continue

        window = np.array(buffer[-window_size:], dtype=np.float32)
        window = (window - mean) / std

        with torch.no_grad():
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        pred = int(probs.argmax())
        conf = float(probs[pred])
        frame_predictions.append(pred)
        frame_confidences.append(conf)

    # Detect move transitions (with cooldown)
    moves = []
    prev_move = 0
    cooldown = 0
    COOLDOWN_FRAMES = 12

    for i, (pred, conf) in enumerate(zip(frame_predictions, frame_confidences)):
        if cooldown > 0:
            cooldown -= 1
            continue

        if pred > 0 and conf > 0.5 and pred != prev_move:
            ts = i / 30.0
            moves.append({
                "frame": i,
                "time": round(ts, 2),
                "type": CLASS_NAMES[pred],
                "confidence": round(conf, 3),
            })
            cooldown = COOLDOWN_FRAMES
        prev_move = pred

    # Print results
    counts = {}
    for m in moves:
        counts[m["type"]] = counts.get(m["type"], 0) + 1
        print(f"    [{m['time']:.2f}s] {m['type'].upper()} (conf={m['confidence']:.2f})")

    print(f"\n    Total: {len(moves)} moves")
    print(f"    Counts: {counts}")
    print(f"    Expected: 2 jab, 3 cross, 3 hook, 3 uppercut = 11 total")

    return moves


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FULL ML PIPELINE — REAL VIDEO TRAINING + VALIDATION")
    print("=" * 70)

    # --- Step 1: Process all labeled videos ---
    print("\n[STEP 1] Processing labeled videos...")

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

    # --- Step 2: Create windowed dataset ---
    print("\n[STEP 2] Creating windowed dataset...")
    X, y = create_windowed_dataset(all_video_frames, window_size=16, stride=2)
    print(f"  Total: {X.shape[0]} windows, {X.shape[1]} timesteps, {X.shape[2]} features")
    print(f"  Feature order: {FEATURE_NAMES[:4]}... (alphabetical)")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {int((y == i).sum())} windows")

    # --- Step 3: Normalize and split ---
    print("\n[STEP 3] Normalizing and splitting...")
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    # Stratified-ish split
    indices = np.random.permutation(len(X_norm))
    X_norm = X_norm[indices]
    y = y[indices]

    val_size = int(len(X_norm) * 0.15)
    X_val, y_val = X_norm[:val_size], y[:val_size]
    X_train, y_train = X_norm[val_size:], y[val_size:]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # Save norm stats
    os.makedirs("ml/models", exist_ok=True)
    np.savez("ml/models/norm_stats.npz",
             mean=mean.squeeze(), std=std.squeeze())

    # --- Step 4: Train ---
    print("\n[STEP 4] Training model...")
    model = train_model(X_train, y_train, X_val, y_val,
                        n_classes=len(CLASS_NAMES), epochs=60)

    # Save model config
    config = {
        "n_features": int(X.shape[2]),
        "n_classes": len(CLASS_NAMES),
        "window_size": 16,
        "class_names": CLASS_NAMES,
        "feature_names": FEATURE_NAMES,
    }
    with open("ml/models/model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Step 5: Validate on mixed video ---
    print("\n[STEP 5] Validating on original mixed video...")
    mixed_video = os.path.expanduser(
        "~/attachments/772c0aa0-9814-4cb5-b4f1-eace406bb88e/WIN_20260516_18_45_10_Pro.mp4"
    )
    mean_sq = mean.squeeze()
    std_sq = std.squeeze()
    validate_on_mixed_video(model, mean_sq, std_sq, mixed_video)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
