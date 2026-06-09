"""
Extract labeled keypoint feature sequences from the user's video.
Outputs a JSON file with per-frame feature vectors and move labels.
"""
import os, sys, json, time, math
import numpy as np

sys.path.insert(0, '/home/ubuntu/stick_fighter')

from core.pose_estimator import PoseEstimator, PoseFrame, Keypoint, VideoSource
from core.smoothing import PoseSmoother
from core.move_detector import MoveDetector, MoveType, MoveDetectorConfig


def extract_wrist_features(pose: PoseFrame) -> dict:
    """Extract feature vector from a single pose frame.
    
    Features per wrist (left + right):
      - x, y, z (normalized position)
      - dx, dy, dz (displacement from shoulder — relative position)
    Plus body reference points:
      - shoulder midpoint y (torso reference)
      - shoulder width (body scale normalization)
      - nose y (head reference)
    
    Total: 6 per wrist * 2 + 3 body = 15 features per frame.
    """
    features = {}
    
    if not pose.valid:
        return {}
    
    kps = pose.keypoints
    
    # Body reference points
    ls = kps.get("left_shoulder")
    rs = kps.get("right_shoulder")
    nose = kps.get("nose")
    
    if not ls or not rs:
        return {}
    
    shoulder_mid_x = (ls.x + rs.x) / 2
    shoulder_mid_y = (ls.y + rs.y) / 2
    shoulder_width = math.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2)
    
    if shoulder_width < 0.01:
        shoulder_width = 0.1  # avoid division by zero
    
    features["shoulder_mid_x"] = shoulder_mid_x
    features["shoulder_mid_y"] = shoulder_mid_y
    features["shoulder_width"] = shoulder_width
    features["nose_y"] = nose.y if nose else shoulder_mid_y - 0.1
    
    # Per-wrist features (normalized by shoulder width for body-size invariance)
    for hand in ["left", "right"]:
        wrist = kps.get(f"{hand}_wrist")
        elbow = kps.get(f"{hand}_elbow")
        shoulder = kps.get(f"{hand}_shoulder")
        
        if not wrist or not shoulder:
            features[f"{hand}_wx"] = 0.0
            features[f"{hand}_wy"] = 0.0
            features[f"{hand}_wz"] = 0.0
            features[f"{hand}_dx"] = 0.0
            features[f"{hand}_dy"] = 0.0
            features[f"{hand}_dz"] = 0.0
            continue
        
        # Absolute position (normalized 0-1)
        features[f"{hand}_wx"] = wrist.x
        features[f"{hand}_wy"] = wrist.y
        features[f"{hand}_wz"] = wrist.z
        
        # Displacement from shoulder (normalized by shoulder width)
        features[f"{hand}_dx"] = (wrist.x - shoulder.x) / shoulder_width
        features[f"{hand}_dy"] = (wrist.y - shoulder.y) / shoulder_width
        features[f"{hand}_dz"] = (wrist.z - shoulder.z) / shoulder_width
    
    return features


def compute_velocity_features(feature_sequence: list[dict], window: int = 3) -> list[dict]:
    """Add velocity features computed over a sliding window.
    
    For each frame, computes velocity (change per frame) for each wrist coordinate.
    Adds 6 velocity features per wrist * 2 = 12 velocity features.
    Total features per frame: 15 position + 12 velocity = 27.
    """
    enriched = []
    
    for i, feat in enumerate(feature_sequence):
        if not feat:
            enriched.append({})
            continue
        
        new_feat = dict(feat)
        
        # Compute velocities
        for hand in ["left", "right"]:
            for coord in ["wx", "wy", "wz", "dx", "dy", "dz"]:
                key = f"{hand}_{coord}"
                vel_key = f"{hand}_v{coord}"
                
                if i < window:
                    new_feat[vel_key] = 0.0
                else:
                    prev_feat = feature_sequence[i - window]
                    if prev_feat and key in prev_feat and key in feat:
                        new_feat[vel_key] = (feat[key] - prev_feat[key]) / window
                    else:
                        new_feat[vel_key] = 0.0
        
        enriched.append(new_feat)
    
    return enriched


def extract_from_video(video_path: str, output_path: str):
    """Process video and extract per-frame features with move labels."""
    
    print(f"Processing: {video_path}")
    video_src = VideoSource(video_path)
    if not video_src.is_open:
        print(f"ERROR: Cannot open {video_path}")
        return
    
    total = video_src.total_frames
    import cv2
    fps = video_src.cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"  {total} frames @ {fps:.0f} fps")
    
    pose_estimator = PoseEstimator(running_mode="VIDEO")
    smoother = PoseSmoother()
    detector = MoveDetector(MoveDetectorConfig(
        cooldown_frames=20,
        min_move_frames=4,
        warmup_frames=12,
    ))
    
    all_features = []
    all_labels = []
    all_poses = []
    prev_move = MoveType.IDLE
    
    frame_idx = 0
    while True:
        ret, frame, ts_ms = video_src.read()
        if not ret:
            break
        
        pose = pose_estimator.process_frame(frame, ts_ms)
        smoothed = smoother.smooth(pose)
        detected = detector.detect(smoothed)
        
        features = extract_wrist_features(smoothed)
        all_features.append(features)
        all_labels.append(detected.move_type.value)
        all_poses.append(smoothed)
        
        if detected.move_type != MoveType.IDLE and detected.move_type != prev_move:
            ts = ts_ms / 1000.0
            print(f"    [{ts:.2f}s] {detected.move_type.value.upper()} "
                  f"({detected.hand}, conf={detected.confidence:.2f})")
        prev_move = detected.move_type
        
        frame_idx += 1
    
    video_src.close()
    
    # Add velocity features
    enriched = compute_velocity_features(all_features)
    
    # Save
    output = {
        "video_path": video_path,
        "fps": fps,
        "total_frames": frame_idx,
        "feature_names": sorted(enriched[0].keys()) if enriched and enriched[0] else [],
        "frames": []
    }
    
    for i, (feat, label) in enumerate(zip(enriched, all_labels)):
        if feat:
            output["frames"].append({
                "index": i,
                "label": label,
                "features": feat,
            })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Summary
    label_counts = {}
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n  Extracted {len(output['frames'])} feature frames")
    print(f"  Feature dimension: {len(output['feature_names'])}")
    print(f"  Label distribution: {label_counts}")
    print(f"  Saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    video_path = os.path.expanduser(
        "~/attachments/772c0aa0-9814-4cb5-b4f1-eace406bb88e/WIN_20260516_18_45_10_Pro.mp4"
    )
    output_path = "/home/ubuntu/stick_fighter/ml/data/user_video_features.json"
    extract_from_video(video_path, output_path)
