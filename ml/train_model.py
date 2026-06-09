"""
Train a 1D-CNN move classifier on synthetic + real keypoint data.

Architecture: 3-layer 1D-CNN with batch norm, dropout, and global average pooling.
Input: (batch, window_size=16, features=28) time-series of keypoint features.
Output: 5-class softmax (idle, jab, cross, hook, uppercut).
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MoveClassifierCNN(nn.Module):
    """Lightweight 1D-CNN for move classification from keypoint time-series."""

    def __init__(self, n_features: int = 28, n_classes: int = 5):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> transpose to (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # Global average pooling over time
        x = x.mean(dim=2)
        x = self.classifier(x)
        return x


def load_synthetic_data(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], list(data["class_names"])


def load_real_data(
    json_path: str,
    window_size: int = 16,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load real video features and create labeled windows.

    Uses a simple heuristic: frames labeled as a move type get that label,
    idle frames get the idle label. Windows are extracted with stride=1
    and labeled by the majority non-idle class in the window (or idle
    if no move frames are present).
    """
    if class_names is None:
        class_names = ["idle", "jab", "cross", "hook", "uppercut"]

    with open(json_path) as f:
        data = json.load(f)

    feature_names = data["feature_names"]
    frames = data["frames"]

    if not frames:
        return np.array([]), np.array([])

    # Build feature matrix and label array
    all_feats = []
    all_labels = []
    for fr in frames:
        row = [fr["features"].get(k, 0.0) for k in feature_names]
        all_feats.append(row)
        label_str = fr["label"]
        label_idx = class_names.index(label_str) if label_str in class_names else 0
        all_labels.append(label_idx)

    feat_array = np.array(all_feats, dtype=np.float32)
    label_array = np.array(all_labels, dtype=np.int64)

    # Extract windows
    windows = []
    window_labels = []
    for start in range(len(feat_array) - window_size):
        win = feat_array[start : start + window_size]
        win_labels = label_array[start : start + window_size]

        # Label = majority non-idle class, or idle
        non_idle = win_labels[win_labels > 0]
        if len(non_idle) >= window_size // 4:
            # Use the most common non-idle label
            counts = np.bincount(non_idle, minlength=len(class_names))
            label = int(counts.argmax())
        else:
            label = 0  # idle

        windows.append(win)
        window_labels.append(label)

    X = np.array(windows, dtype=np.float32)
    y = np.array(window_labels, dtype=np.int64)
    return X, y


def train(
    synthetic_path: str = "ml/data/synthetic_dataset.npz",
    real_path: str = "ml/data/user_video_features.json",
    output_dir: str = "ml/models",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    val_split: float = 0.15,
    window_size: int = 16,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\nLoading synthetic data...")
    X_syn, y_syn, class_names = load_synthetic_data(synthetic_path)
    print(f"  Synthetic: {X_syn.shape[0]} samples")

    print("Loading real data...")
    X_real, y_real = load_real_data(real_path, window_size, class_names)
    if X_real.size > 0:
        print(f"  Real: {X_real.shape[0]} samples")
        # Pad or trim real features to match synthetic feature count
        if X_real.shape[2] != X_syn.shape[2]:
            target_feats = X_syn.shape[2]
            if X_real.shape[2] < target_feats:
                pad = np.zeros(
                    (X_real.shape[0], X_real.shape[1], target_feats - X_real.shape[2]),
                    dtype=np.float32,
                )
                X_real = np.concatenate([X_real, pad], axis=2)
            else:
                X_real = X_real[:, :, :target_feats]

        # Upsample real data (more weight)
        repeat_factor = max(1, X_syn.shape[0] // max(1, X_real.shape[0]) // 2)
        X_real_up = np.tile(X_real, (repeat_factor, 1, 1))
        y_real_up = np.tile(y_real, repeat_factor)
        print(f"  Real upsampled: {X_real_up.shape[0]} samples (x{repeat_factor})")

        X_all = np.concatenate([X_syn, X_real_up], axis=0)
        y_all = np.concatenate([y_syn, y_real_up], axis=0)
    else:
        print("  No real data found, using synthetic only")
        X_all = X_syn
        y_all = y_syn

    # Normalize features (per-feature z-score)
    mean = X_all.mean(axis=(0, 1), keepdims=True)
    std = X_all.std(axis=(0, 1), keepdims=True) + 1e-8
    X_all = (X_all - mean) / std

    # Save normalization stats
    np.savez(
        os.path.join(output_dir, "norm_stats.npz"),
        mean=mean.squeeze(),
        std=std.squeeze(),
    )

    # Shuffle and split
    indices = np.random.permutation(len(X_all))
    X_all = X_all[indices]
    y_all = y_all[indices]

    val_size = int(len(X_all) * val_split)
    X_val, y_val = X_all[:val_size], y_all[:val_size]
    X_train, y_train = X_all[val_size:], y_all[val_size:]

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")
    print(f"Features: {X_train.shape[2]}, Window: {X_train.shape[1]}")
    print(f"Classes: {class_names}")

    # Create dataloaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Model
    n_features = X_train.shape[2]
    n_classes = len(class_names)
    model = MoveClassifierCNN(n_features=n_features, n_classes=n_classes).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    weights_tensor = torch.from_numpy(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_dl:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += X_batch.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += X_batch.size(0)

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        avg_train_loss = train_loss / max(1, train_total)
        avg_val_loss = val_loss / max(1, val_total)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.3f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(output_dir, "move_classifier.pt"))

    print(f"\nBest val accuracy: {best_val_acc:.3f} at epoch {best_epoch}")

    # Per-class validation accuracy
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "move_classifier.pt"), weights_only=True)
    )
    model.eval()

    all_preds = []
    all_true = []
    with torch.no_grad():
        for X_batch, y_batch in val_dl:
            logits = model(X_batch.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_true.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    print("\nPer-class validation results:")
    for i, name in enumerate(class_names):
        mask = all_true == i
        if mask.sum() == 0:
            print(f"  {name}: no samples")
            continue
        correct = (all_preds[mask] == i).sum()
        total = mask.sum()
        print(f"  {name}: {correct}/{total} = {correct/total:.1%}")

    # Confusion matrix
    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "          " + " ".join(f"{n[:5]:>7s}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = []
        for j in range(len(class_names)):
            count = int(((all_true == i) & (all_preds == j)).sum())
            row.append(f"{count:7d}")
        print(f"  {name:>7s} " + " ".join(row))

    # Save model config
    config = {
        "n_features": n_features,
        "n_classes": n_classes,
        "window_size": window_size,
        "class_names": class_names,
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": best_epoch,
        "total_params": sum(p.numel() for p in model.parameters()),
    }
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to {output_dir}/")
    print(f"  move_classifier.pt  ({os.path.getsize(os.path.join(output_dir, 'move_classifier.pt')) / 1024:.1f} KB)")
    print(f"  model_config.json")
    print(f"  norm_stats.npz")

    return model, class_names


if __name__ == "__main__":
    train()
