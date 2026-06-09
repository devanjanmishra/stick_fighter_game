"""
Step 3: Analyze frame-by-frame predictions on mixed video and tune detection.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn

CLASS_NAMES = ["idle", "jab", "cross", "hook", "uppercut", "walking"]


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


def main():
    print("=" * 60)
    print("STEP 3: FRAME-BY-FRAME ANALYSIS + DETECTION TUNING")
    print("=" * 60)

    # Load model
    device = torch.device("cpu")
    model = MoveClassifierCNN(n_features=28, n_classes=6).to(device)
    model.load_state_dict(torch.load("ml/models/move_classifier.pt", weights_only=True))
    model.eval()

    # Load norm stats
    norm = np.load("ml/models/norm_stats.npz")
    mean = norm["mean"]
    std = norm["std"]

    # Load mixed video features
    mixed = np.load("ml/data/mixed_video_features.npz")
    mixed_X = mixed["X"]  # (462, 28)
    print(f"  Mixed video: {mixed_X.shape[0]} frames")

    # Classify every frame using sliding window
    window_size = 16
    frame_preds = []
    frame_confs = []
    frame_probs_all = []

    for i in range(len(mixed_X)):
        if i < window_size - 1:
            frame_preds.append(0)
            frame_confs.append(0.0)
            frame_probs_all.append([0.0] * 6)
            continue

        window = mixed_X[i - window_size + 1:i + 1]
        window_norm = (window - mean) / std

        with torch.no_grad():
            x = torch.from_numpy(window_norm).unsqueeze(0).float().to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        pred = int(probs.argmax())
        conf = float(probs[pred])
        frame_preds.append(pred)
        frame_confs.append(conf)
        frame_probs_all.append(probs.tolist())

    # Print frame-by-frame summary in segments
    print("\n  Frame-by-frame prediction segments:")
    current_pred = frame_preds[0]
    segment_start = 0
    segments = []

    for i in range(1, len(frame_preds)):
        if frame_preds[i] != current_pred:
            segments.append({
                "start": segment_start,
                "end": i - 1,
                "label": CLASS_NAMES[current_pred],
                "label_idx": current_pred,
                "length": i - segment_start,
                "avg_conf": np.mean(frame_confs[segment_start:i]),
            })
            current_pred = frame_preds[i]
            segment_start = i
    segments.append({
        "start": segment_start,
        "end": len(frame_preds) - 1,
        "label": CLASS_NAMES[current_pred],
        "label_idx": current_pred,
        "length": len(frame_preds) - segment_start,
        "avg_conf": np.mean(frame_confs[segment_start:len(frame_preds)]),
    })

    for seg in segments:
        ts_start = seg["start"] / 30.0
        ts_end = seg["end"] / 30.0
        marker = "***" if seg["label_idx"] > 0 else "   "
        print(f"    {marker} [{ts_start:5.1f}s - {ts_end:5.1f}s] "
              f"{seg['label']:>9s} ({seg['length']:3d} frames, conf={seg['avg_conf']:.2f})")

    # Expected moves with approximate timestamps from user description:
    # 2 jabs, 3 crosses, 3 hooks, 3 uppercuts = 11 total
    print("\n  Expected: 2 jab, 3 cross, 3 hook, 3 uppercut")

    # IMPROVED DETECTION: Group non-idle segments into moves
    # A "move" is a contiguous period of non-idle predictions
    print("\n\n  --- IMPROVED DETECTION ALGORITHM ---")

    # Strategy: find contiguous non-idle regions, take the dominant class
    in_move = False
    move_frames = []
    detected_moves = []

    for i in range(len(frame_preds)):
        pred = frame_preds[i]
        conf = frame_confs[i]

        if pred > 0 and pred != 5:  # non-idle, non-walking
            if not in_move:
                in_move = True
                move_frames = []
            move_frames.append((i, pred, conf))
        else:
            if in_move and len(move_frames) >= 3:
                # End of a move region — classify by dominant prediction
                preds_in_region = [mf[1] for mf in move_frames]
                confs_in_region = [mf[2] for mf in move_frames]
                counts = {}
                for p in preds_in_region:
                    counts[p] = counts.get(p, 0) + 1
                dominant = max(counts, key=counts.get)
                avg_conf = np.mean([c for p, c in zip(preds_in_region, confs_in_region) if p == dominant])
                start_frame = move_frames[0][0]
                end_frame = move_frames[-1][0]
                detected_moves.append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "type": CLASS_NAMES[dominant],
                    "type_idx": dominant,
                    "confidence": round(float(avg_conf), 3),
                    "duration_frames": len(move_frames),
                    "raw_counts": {CLASS_NAMES[k]: v for k, v in counts.items()},
                })
            in_move = False
            move_frames = []

    # Handle last region
    if in_move and len(move_frames) >= 3:
        preds_in_region = [mf[1] for mf in move_frames]
        confs_in_region = [mf[2] for mf in move_frames]
        counts = {}
        for p in preds_in_region:
            counts[p] = counts.get(p, 0) + 1
        dominant = max(counts, key=counts.get)
        avg_conf = np.mean([c for p, c in zip(preds_in_region, confs_in_region) if p == dominant])
        detected_moves.append({
            "start_frame": move_frames[0][0],
            "end_frame": move_frames[-1][0],
            "type": CLASS_NAMES[dominant],
            "type_idx": dominant,
            "confidence": round(float(avg_conf), 3),
            "duration_frames": len(move_frames),
            "raw_counts": {CLASS_NAMES[k]: v for k, v in counts.items()},
        })

    print(f"\n  Detected {len(detected_moves)} move regions:")
    for m in detected_moves:
        ts = m["start_frame"] / 30.0
        print(f"    [{ts:5.1f}s] {m['type'].upper():>9s} "
              f"(conf={m['confidence']:.2f}, dur={m['duration_frames']}f, "
              f"raw={m['raw_counts']})")

    counts = {}
    for m in detected_moves:
        counts[m["type"]] = counts.get(m["type"], 0) + 1
    print(f"\n    Total: {len(detected_moves)} moves")
    print(f"    Breakdown: {counts}")
    print(f"    Expected:  jab=2, cross=3, hook=3, uppercut=3")

    # Now try merging adjacent moves of same type that are very close together
    # (within 10 frames gap filled by idle)
    print("\n\n  --- WITH MERGE (gap <= 10 frames) ---")
    merged = []
    for m in detected_moves:
        if merged and m["type"] == merged[-1]["type"] and \
           (m["start_frame"] - merged[-1]["end_frame"]) <= 10:
            merged[-1]["end_frame"] = m["end_frame"]
            merged[-1]["duration_frames"] += m["duration_frames"]
        else:
            merged.append(dict(m))

    for m in merged:
        ts = m["start_frame"] / 30.0
        print(f"    [{ts:5.1f}s] {m['type'].upper():>9s} "
              f"(conf={m['confidence']:.2f}, dur={m['duration_frames']}f)")

    counts2 = {}
    for m in merged:
        counts2[m["type"]] = counts2.get(m["type"], 0) + 1
    print(f"\n    Total: {len(merged)} moves")
    print(f"    Breakdown: {counts2}")

    # Check: are there any frames where uppercut has high probability?
    print("\n\n  --- UPPERCUT PROBABILITY ANALYSIS ---")
    for i in range(len(frame_probs_all)):
        probs = frame_probs_all[i]
        if probs[4] > 0.1:  # uppercut index = 4
            ts = i / 30.0
            print(f"    Frame {i} [{ts:.1f}s]: idle={probs[0]:.2f} jab={probs[1]:.2f} "
                  f"cross={probs[2]:.2f} hook={probs[3]:.2f} "
                  f"UPPERCUT={probs[4]:.2f} walk={probs[5]:.2f}")

    # Check walking frames
    print("\n\n  --- WALKING PROBABILITY ANALYSIS ---")
    walk_frames = 0
    for i in range(len(frame_probs_all)):
        probs = frame_probs_all[i]
        if probs[5] > 0.3:
            walk_frames += 1
            if walk_frames <= 20:
                ts = i / 30.0
                print(f"    Frame {i} [{ts:.1f}s]: walk={probs[5]:.2f} "
                      f"idle={probs[0]:.2f} best_other={CLASS_NAMES[np.argmax(probs[:5])]}={max(probs[:5]):.2f}")
    print(f"    Total frames with walking > 0.3: {walk_frames}")


if __name__ == "__main__":
    main()
