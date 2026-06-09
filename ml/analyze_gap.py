"""Analyze the distribution gap between synthetic and real video features."""
import json
import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/stick_fighter')
from ml.synthetic_generator import ALL_FEATURES

# Load real features
with open("ml/data/user_video_features.json") as f:
    real_data = json.load(f)

real_frames = real_data["frames"]
real_feat_names = real_data["feature_names"]

# Load synthetic
syn_data = np.load("ml/data/synthetic_dataset.npz", allow_pickle=True)
X_syn = syn_data["X"]  # (N, 16, 28)

# Extract real features into matrix
real_matrix = []
real_labels = []
for fr in real_frames:
    row = [fr["features"].get(k, 0.0) for k in real_feat_names]
    real_matrix.append(row)
    real_labels.append(fr["label"])
real_matrix = np.array(real_matrix, dtype=np.float32)

print("Feature name mapping:")
print(f"  Real feature names ({len(real_feat_names)}): {real_feat_names[:5]}...")
print(f"  Synthetic feature names ({len(ALL_FEATURES)}): {ALL_FEATURES[:5]}...")

# Check feature name alignment
print("\nReal feature names:")
for i, n in enumerate(real_feat_names):
    print(f"  {i:2d}: {n}")

print("\nSynthetic feature names (ALL_FEATURES):")
for i, n in enumerate(ALL_FEATURES):
    print(f"  {i:2d}: {n}")

# Compare distributions of key features
print("\n" + "=" * 80)
print("FEATURE DISTRIBUTION COMPARISON")
print("=" * 80)

# For synthetic, flatten to (N*16, 28)
syn_flat = X_syn.reshape(-1, X_syn.shape[2])

# Compare each feature
print(f"\n{'Feature':<20} {'Real Mean':>10} {'Real Std':>10} {'Syn Mean':>10} {'Syn Std':>10} {'Gap':>10}")
print("-" * 72)

for i, name in enumerate(ALL_FEATURES):
    # Find matching real feature
    if name in real_feat_names:
        ri = real_feat_names.index(name)
        r_mean = real_matrix[:, ri].mean()
        r_std = real_matrix[:, ri].std()
    else:
        r_mean = 0.0
        r_std = 0.0

    if i < syn_flat.shape[1]:
        s_mean = syn_flat[:, i].mean()
        s_std = syn_flat[:, i].std()
    else:
        s_mean = 0.0
        s_std = 0.0

    gap = abs(r_mean - s_mean)
    flag = " ***" if gap > 0.1 else ""
    print(f"{name:<20} {r_mean:>10.4f} {r_std:>10.4f} {s_mean:>10.4f} {s_std:>10.4f} {gap:>10.4f}{flag}")

# Print real feature ranges during detected moves
print("\n" + "=" * 80)
print("REAL VIDEO: FEATURES DURING DETECTED MOVES")
print("=" * 80)

for label in ["jab", "cross", "hook", "uppercut"]:
    mask = [l == label for l in real_labels]
    if not any(mask):
        print(f"\n  {label}: NO FRAMES")
        continue
    
    label_matrix = real_matrix[mask]
    print(f"\n  {label} ({label_matrix.shape[0]} frames):")
    for i, name in enumerate(real_feat_names):
        if "v" in name:  # velocity features
            vals = label_matrix[:, i]
            print(f"    {name:<25} mean={vals.mean():.5f}  std={vals.std():.5f}  "
                  f"min={vals.min():.5f}  max={vals.max():.5f}")
