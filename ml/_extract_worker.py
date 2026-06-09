"""
Worker script for feature extraction — runs in a subprocess to isolate
MediaPipe's OpenGL context from PyTorch training.

Usage:
  python ml/_extract_worker.py <job_json_path>

The job JSON contains:
  {
    "videos": [{"move_type": "jab", "path": "/path/to/jab.mp4", "expected_count": 17}, ...],
    "class_names": ["idle", "jab", "cross", ...],
    "output_path": "/path/to/output.npz",
    "mode": "labeled" | "raw"
  }

For "labeled" mode: extracts labeled frames and creates windowed dataset (X, y).
For "raw" mode: extracts per-frame feature matrix for validation.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    job_path = sys.argv[1]
    with open(job_path) as f:
        job = json.load(f)

    # Import pipeline functions (which will import MediaPipe)
    from ml.pipeline import (
        extract_labeled_frames, extract_video_features,
        create_windowed_dataset, FEATURE_NAMES,
    )
    import numpy as np

    mode = job.get("mode", "labeled")
    videos = job["videos"]
    output_path = job["output_path"]

    if mode == "labeled":
        class_names = job["class_names"]
        all_video_frames = []
        video_infos = []

        for v in videos:
            print(f"\n  [{v['move_type'].upper()}] {os.path.basename(v['path'])} "
                  f"(expected: {v['expected_count']})")
            frames = extract_labeled_frames(v["path"], v["move_type"])
            all_video_frames.append(frames)
            video_infos.append({
                "move_type": v["move_type"],
                "path": v["path"],
                "expected_count": v["expected_count"],
                "extracted_frames": len(frames),
            })

        print("\n--- CREATING WINDOWED DATASET ---")
        X, y = create_windowed_dataset(all_video_frames, class_names,
                                       window_size=16, stride=2)
        print(f"  Total: {X.shape[0]} windows, {X.shape[1]} timesteps, "
              f"{X.shape[2]} features")
        for i, name in enumerate(class_names):
            count = int((y == i).sum())
            if count > 0:
                print(f"    {name}: {count} windows")

        np.savez(output_path, X=X, y=y, class_names=class_names)
        print(f"  Saved: {output_path}")

        # Save video infos as JSON alongside
        info_path = output_path.replace(".npz", "_info.json")
        with open(info_path, "w") as f:
            json.dump(video_infos, f, indent=2)

    elif mode == "raw":
        # Extract raw features for a single video
        v = videos[0]
        print(f"  Extracting raw features: {os.path.basename(v['path'])}")
        features = extract_video_features(v["path"])
        import numpy as np
        np.savez(output_path, features=features)
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
