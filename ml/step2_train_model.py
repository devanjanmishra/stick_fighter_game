"""
Step 2: Train 1D-CNN model on extracted video features.
Reads ml/data/all_videos_dataset.npz, trains model, saves to ml/models/.
Also validates on the mixed video features from ml/data/mixed_video_features.npz.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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


def train():
    print("=" * 60)
    print("STEP 2: TRAIN MODEL ON REAL VIDEO DATA")
    print("=" * 60)

    # Load dataset
    data = np.load("ml/data/all_videos_dataset.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = list(data["feature_names"])
    print(f"  Dataset: {X.shape[0]} windows, {X.shape[1]} timesteps, {X.shape[2]} features")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {int((y == i).sum())} windows")

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
    os.makedirs("ml/models", exist_ok=True)
    np.savez("ml/models/norm_stats.npz",
             mean=mean.squeeze((0, 1)), std=std.squeeze((0, 1)))

    # Class weights for imbalanced data
    n_classes = len(CLASS_NAMES)
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    print(f"  Class weights: {dict(zip(CLASS_NAMES, [f'{w:.2f}' for w in class_weights]))}")

    device = torch.device("cpu")
    n_features = X.shape[2]
    model = MoveClassifierCNN(n_features=n_features, n_classes=n_classes).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    weights_tensor = torch.from_numpy(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    best_val_acc = 0.0
    best_epoch = 0
    epochs = 80

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
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

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "ml/models/move_classifier.pt")

    print(f"\n  Best val accuracy: {best_val_acc:.3f} at epoch {best_epoch}")

    # Load best and evaluate
    model.load_state_dict(
        torch.load("ml/models/move_classifier.pt", weights_only=True))
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

    print("\n  Per-class validation accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask = all_true == i
        if mask.sum() == 0:
            print(f"    {name:>10s}: (no samples)")
            continue
        correct = (all_preds[mask] == i).sum()
        total = mask.sum()
        print(f"    {name:>10s}: {correct}/{total} = {correct/total:.1%}")

    print("\n  Confusion matrix (rows=true, cols=predicted):")
    header = "            " + " ".join(f"{n[:6]:>7s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = []
        for j in range(n_classes):
            count = int(((all_true == i) & (all_preds == j)).sum())
            row.append(f"{count:7d}")
        print(f"    {name:>8s} " + " ".join(row))

    # Save config
    config = {
        "n_features": int(n_features),
        "n_classes": n_classes,
        "window_size": 16,
        "class_names": CLASS_NAMES,
        "feature_names": feature_names,
    }
    with open("ml/models/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Model config saved to ml/models/model_config.json")

    # --- Validate on mixed video ---
    print("\n" + "=" * 60)
    print("STEP 3: VALIDATE ON MIXED VIDEO")
    print("=" * 60)

    mixed_data = np.load("ml/data/mixed_video_features.npz")
    mixed_X = mixed_data["X"]  # (462, 28)
    print(f"  Mixed video: {mixed_X.shape[0]} frames")

    mean_sq = mean.squeeze((0, 1))
    std_sq = std.squeeze((0, 1))

    window_size = 16
    frame_predictions = []
    frame_confidences = []

    for i in range(len(mixed_X)):
        if i < window_size - 1:
            frame_predictions.append(0)
            frame_confidences.append(0.0)
            continue

        window = mixed_X[i - window_size + 1:i + 1]
        window_norm = (window - mean_sq) / std_sq

        with torch.no_grad():
            x = torch.from_numpy(window_norm).unsqueeze(0).float().to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        pred = int(probs.argmax())
        conf = float(probs[pred])
        frame_predictions.append(pred)
        frame_confidences.append(conf)

    # Detect move transitions with cooldown
    moves = []
    cooldown = 0
    COOLDOWN_FRAMES = 15
    CONFIDENCE_THRESHOLD = 0.6
    prev_move = 0
    consecutive_count = 0
    MIN_CONSECUTIVE = 3  # require N consecutive frames of same move

    for i, (pred, conf) in enumerate(zip(frame_predictions, frame_confidences)):
        if cooldown > 0:
            cooldown -= 1
            continue

        if pred > 0 and conf > CONFIDENCE_THRESHOLD:
            if pred == prev_move:
                consecutive_count += 1
            else:
                consecutive_count = 1
                prev_move = pred

            if consecutive_count == MIN_CONSECUTIVE:
                ts = i / 30.0
                moves.append({
                    "frame": i,
                    "time": round(ts, 2),
                    "type": CLASS_NAMES[pred],
                    "confidence": round(conf, 3),
                })
                cooldown = COOLDOWN_FRAMES
                consecutive_count = 0
        else:
            if pred == 0:
                consecutive_count = 0
                prev_move = 0

    counts = {}
    for m in moves:
        counts[m["type"]] = counts.get(m["type"], 0) + 1
        print(f"    [{m['time']:6.2f}s] {m['type'].upper():>9s} (conf={m['confidence']:.2f})")

    total_detected = len(moves)
    print(f"\n    Total detected: {total_detected} moves")
    print(f"    Breakdown: {counts}")
    print(f"    Expected:  2 jab, 3 cross, 3 hook, 3 uppercut = 11 total")

    # Score
    expected = {"jab": 2, "cross": 3, "hook": 3, "uppercut": 3}
    score = 0
    for move, exp_count in expected.items():
        det = counts.get(move, 0)
        score += min(det, exp_count)  # correct detections (capped)
    print(f"\n    Accuracy score: {score}/11 correct moves detected")

    print("\n  TRAINING + VALIDATION COMPLETE")


if __name__ == "__main__":
    train()
