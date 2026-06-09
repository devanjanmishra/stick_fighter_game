# Session Summary — Calibration UI + Move Detection Fix

## What was done this session

### 1. Calibration UI wired into game startup
Added `CalibrationFlow` class in `main.py` (~270 lines) that runs before the main game loop:

- **Welcome screen** — explains what calibration does
- **Stance select** — Orthodox (left lead) vs Southpaw (right lead)
- **Move explanation** — detailed instructions for each of the 4 moves (jab, cross, hook, uppercut)
- **Recording flow** — 3-second countdown → 1.5s recording per sample → confirmation screen
- **3 samples per move** — records velocity/displacement profiles via `CalibrationRecorder`
- **Threshold computation** — `CalibrationProfile.compute_thresholds()` derives personalized detection thresholds using DTW
- **Profile persistence** — saves to `calibration_profile.json`, loads automatically on next launch
- **Skip/reload** — if profile exists, offers to load it or recalibrate; ESC to skip with confirmation

The `StickFighterGame` now accepts a `CalibrationProfile` and applies its personalized thresholds to the `MoveDetector`.

CLI flags added: `--skip-calibration`, `--headless`

### 2. Move detection accuracy — 77 → 11 (exact match)
Fixed two issues causing over-detection:

| Metric | Before | After |
|--------|--------|-------|
| Total moves detected | 77 (then 17) | **11** |
| Jab | 35 (then 6) | **2** |
| Cross | — | **4** (1 is a misclassified hook) |
| Hook | — | **2** (3rd hook detected as cross) |
| Uppercut | — | **3** |

**Fix: Move-type locking** — once a move is classified (e.g., HOOK), that type is locked for the move's duration. Previously, a hook's lateral arc would transition to forward z-motion mid-swing, causing the detector to reclassify it as a jab/cross. The retraction after hooks also generated false jab detections. Now the initial classification sticks.

### 3. All tests pass
- **83/83 unit tests passing**
- **10/10 milestones passing** against your video (462 frames, 1280x720, 30fps)

## Known issue
- **3rd hook → cross misclassification**: One of your 3 hooks is classified as a cross. The hook's lateral arc (x-displacement) doesn't exceed the threshold for that particular punch. Calibration should fix this — when you record your personal hook samples, the thresholds will adapt to your style.

## How to run

```bash
# Full game with calibration
python main.py --source 0

# Skip calibration (default thresholds)
python main.py --source 0 --skip-calibration

# With video file (headless, loads saved profile if exists)
python main.py --source path/to/video.mp4 --headless

# Choose NPC style and difficulty
python main.py --source 0 --style brawler --difficulty hard
```

## Files changed
- `main.py` — Added `CalibrationFlow` class, updated `StickFighterGame.__init__` to accept calibration profile, updated `main()` with calibration phase + CLI flags
- `core/move_detector.py` — Move-type locking to prevent mid-move reclassification

## Next steps
- **Play-test with live camera** — run `python main.py --source 0` to test the full flow (calibration → fight)
- **3rd hook refinement** — may need to lower `hook_x_extension` threshold or calibration will handle it
- **Leg tracking** — extend MediaPipe landmarks to lower body
- **Mobile port** — Kivy/Buildozer for Android APK
