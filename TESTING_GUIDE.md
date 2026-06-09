# Stick Fighter — Testing & Validation Guide

How to verify everything works, what to look for, and what data proves correctness.

---

## Quick Start: Three Levels of Testing

```bash
cd /home/ubuntu/stick_fighter

# Level 1: Unit tests (no camera needed, runs anywhere)
python -m pytest tests/ -q                    # expect: 83/83 passed

# Level 2: Video validation (uses your recorded video)
python validate_video.py                      # expect: 10/10 milestones PASS

# Level 3: Live testing (needs webcam)
python main.py --source 0 --dojo             # dojo mode: free practice
python main.py --source 0                    # full game: calibration + fight
```

---

## 1. Unit Tests (Level 1) — No Camera Needed

### What It Tests
83 tests across 10 milestone files using **synthetic keypoint data** (no real camera).

| File | Tests | What It Validates |
|------|-------|-------------------|
| `test_milestone1.py` | 8 | Pose estimation: keypoint extraction, confidence, upper-body filtering |
| `test_milestone2.py` | 12 | Coordinate transform: front-face to side-view mapping, rendering output |
| `test_milestone3.py` | 8 | Smoothing: jitter reduction, One Euro Filter, latency vs smoothness |
| `test_milestone4.py` | 10 | Move detection: jab/cross/hook/uppercut classification, cooldown, no false positives |
| `test_milestone5.py` | 8 | Walking: shoulder displacement detection, dead zone, direction |
| `test_milestone6.py` | 8 | Calibration: template recording, threshold computation, profile save/load |
| `test_milestone7.py` | 8 | NPC AI: behavior tree, attack patterns, collision detection |
| `test_milestone8.py` | 9 | Combat: HP, rounds, KO, timer, game phase transitions |
| `test_milestone9.py` | 6 | Effects: particles, screen shake, hitstop, combos, damage numbers |
| `test_milestone10.py` | 6 | NPC styles: 5 styles x 4 difficulties, stat scaling, style profiles |

### Running
```bash
# All tests
python -m pytest tests/ -q

# Single milestone
python -m pytest tests/test_milestone4.py -v

# With output images (saved to test_output/)
python -m pytest tests/ -q && ls test_output/
```

### Expected Output
```
83 passed, 9 warnings in ~3s
```

### What a Failure Means
- **test_milestone4**: Move detection thresholds changed — check `MoveDetectorConfig` defaults
- **test_milestone2**: Coordinate transform constants changed — check `z_to_x_scale`, `segment_max_length`
- **test_milestone6**: Calibration profile format changed — check `CalibrationProfile` dataclass
- **test_milestone7/8**: Combat or NPC config changed — check damage tables, HP values

---

## 2. Video Validation (Level 2) — Pre-Recorded Video

Uses `validate_video.py` to run a recorded video through the full pipeline and check all 10 milestones.

### What It Tests
End-to-end pipeline with real human movement:
1. MediaPipe detects keypoints in every frame
2. Keypoints transform to side-view correctly
3. Smoothing reduces jitter without adding lag
4. Move detection classifies punches correctly
5. Walking detection responds to shoulder movement
6. Calibration can extract templates from the video
7. NPC AI runs correctly alongside player data
8. Combat system processes hits and HP correctly
9. Effects trigger on hits
10. All 5 NPC styles x 4 difficulties work

### Running
```bash
python validate_video.py
```

### Expected Output
```
Milestone 1 (Pose Estimation): PASS    — 462/462 frames, 11ms avg
Milestone 2 (Side-View Transform): PASS — 462 frames transformed
...
Milestone 10 (NPC Styles): PASS         — 20/20 combinations
```

### Key Validation Data (Input → Expected Output)

| Input | Expected Output | What It Proves |
|-------|----------------|----------------|
| 15.4s video, 1280x720, 30fps | 462 frames processed | Video pipeline works |
| Every frame | Keypoints detected (>0.5 confidence) | MediaPipe works in varied lighting |
| Raw keypoints | Side-view coordinates in [0, 1280] x [0, 720] | Transform is calibrated |
| Smoothed keypoints | Jitter < raw jitter | Smoothing helps, doesn't hurt |
| 2 jabs thrown | 2 jabs detected | Jab detection works |
| 4 crosses thrown | 4 crosses detected (includes 1 misclassified hook) | Cross detection works |
| 2 hooks thrown | 2 hooks detected | Hook detection works |
| 3 uppercuts thrown | 3 uppercuts detected | Uppercut detection works |
| 11 total moves | 11 total detected | No over-detection, no missed moves |

### What to Record for Your Own Validation Video
1. Face the camera, stand ~1-2m away
2. Good lighting on your upper body
3. Perform moves **slowly and deliberately** with pauses between each:
   - 2-3 jabs (lead hand forward punch toward camera)
   - 2-3 crosses (rear hand forward punch toward camera)
   - 2-3 hooks (arm sweeps laterally then forward)
   - 2-3 uppercuts (fist rises sharply upward)
4. Wait ~1 second between moves (cooldown period)
5. Keep your torso relatively still (don't lean excessively)

### If Detection Counts Don't Match
- **Too many detections**: Lower `cooldown_frames` in `MoveDetectorConfig` or raise velocity thresholds
- **Too few detections**: Lower velocity thresholds or increase `punch_z_extension`
- **Wrong classification**: Hook vs cross is the hardest — calibration fixes this by adapting to YOUR style
- **Zero detections**: Check that MediaPipe is detecting keypoints (M1 should show >50% detection rate)

---

## 3. Live Testing (Level 3) — With Webcam

### 3A. Dojo Mode (Best for Validation)

The **Dojo** is a free practice environment with no NPC — just you and the detection system. It's the best way to validate everything works.

```bash
python main.py --source 0 --dojo                    # with calibration
python main.py --source 0 --dojo --skip-calibration  # skip calibration
```

#### What the Dojo Shows You
| UI Element | Location | What It Tells You |
|------------|----------|-------------------|
| **Stick figure** | Center | Your body is being tracked correctly (arms move when yours do) |
| **Big move label** | Center (flashes) | System detected a move — should match what you threw |
| **Detection stats** | Top-left panel | Running count per move type — should match your actual throws |
| **Move log** | Right panel | Timestamped history — verify each entry is correct |
| **Velocity bars** | Bottom-center | Z (forward), X (lateral), Y (vertical) wrist velocity — should spike when you punch |
| **Camera preview** | Top-right | Your camera feed with green keypoint dots — verify joints are tracked |
| **FPS counter** | Bottom-left (press D) | Should be 25-30 FPS for real-time feel |

#### Dojo Validation Checklist
- [ ] **Stick figure mirrors your movements** — raise your arms, the stick figure raises its arms
- [ ] **Jab detected when you jab** — lead hand punch forward, big "JAB" label appears
- [ ] **Cross detected when you cross** — rear hand punch forward, "CROSS" label appears
- [ ] **Hook detected when you hook** — arm sweeps laterally, "HOOK" label appears
- [ ] **Uppercut detected when you uppercut** — fist rises sharply, "UPPERCUT" label appears
- [ ] **No false positives when standing still** — stats panel stays at 0 when idle
- [ ] **No duplicate detections** — one punch = one detection (not 2-3)
- [ ] **Velocity bars spike on punches** — Z bar for jab/cross, X bar for hooks, Y bar for uppercuts
- [ ] **Camera preview shows green keypoint dots** on your joints
- [ ] **FPS is 25-30** (press D for debug overlay)

#### Dojo Output
When you quit (ESC), the Dojo prints a session summary and saves to `dojo_session_log.json`:
```json
{
  "duration_s": 45.2,
  "total_moves": 12,
  "counts": {"jab": 3, "cross": 4, "hook": 2, "uppercut": 3},
  "rate_per_sec": 0.27,
  "move_log": [
    {"time_s": 2.451, "move": "jab", "hand": "left"},
    {"time_s": 4.102, "move": "cross", "hand": "right"},
    ...
  ]
}
```

**Compare this to what you actually threw.** If `counts` matches your actual punches, detection is working.

---

### 3B. Calibration Testing

```bash
python main.py --source 0
```

#### Calibration Validation Checklist
- [ ] **Welcome screen** appears — press SPACE to continue
- [ ] **Stance selection** — LEFT/RIGHT to pick orthodox/southpaw, ENTER to confirm
- [ ] **Move explanation** — animated stick figure demo plays for the current move
- [ ] **LEFT/RIGHT navigation** — can browse between moves before recording
- [ ] **Recording countdown** — 3-2-1 with live camera feed + green keypoints
- [ ] **Recording** — progress bar fills while camera captures your punches with keypoint overlay
- [ ] **Playback** — recorded video replays with keypoints so you can verify the recording
- [ ] **3 samples per move** — records 3 of each (jab, cross, hook, uppercut = 12 total)
- [ ] **Profile saved** — `calibration_profile.json` created in project root
- [ ] **Thresholds printed** — terminal shows computed thresholds (punch_z_vel, hook_x_vel, upper_y_vel)

#### Calibration Profile Validation
After calibration, check the saved profile:
```bash
python -c "
import json
with open('calibration_profile.json') as f:
    p = json.load(f)
print(f'Stance: {p[\"stance\"]}')
print(f'Moves recorded: {list(p[\"templates\"].keys())}')
for move, samples in p['templates'].items():
    print(f'  {move}: {len(samples)} samples')
t = p.get('thresholds', {})
if t:
    print(f'Thresholds:')
    print(f'  punch_z_vel: {t[\"punch_z_velocity_threshold\"]:.4f}')
    print(f'  hook_x_vel:  {t[\"hook_x_velocity_threshold\"]:.4f}')
    print(f'  upper_y_vel: {t[\"uppercut_y_velocity_threshold\"]:.4f}')
"
```

**Expected:** 4 moves, 3 samples each, all thresholds > 0.

---

### 3C. Full Game Testing

```bash
python main.py --source 0                                        # full flow
python main.py --source 0 --skip-calibration                     # skip cal
python main.py --source 0 --skip-calibration --style tank --difficulty hard
```

#### Game Validation Checklist
- [ ] **Calibration runs first** (unless --skip-calibration)
- [ ] **3-2-1 countdown** with "FIGHT!" overlay before match starts
- [ ] **Player stick figure** moves with your body (blue figure, left side)
- [ ] **NPC stick figure** moves independently (red figure, right side)
- [ ] **NPC has same visual style** as player (keypoint dots, skeleton, fist circles, head with eye)
- [ ] **Punches deal damage** — NPC HP bar decreases when your punch lands
- [ ] **NPC fights back** — your HP bar decreases when NPC hits you
- [ ] **Combo counter** appears on consecutive hits (DOUBLE, TRIPLE, etc.)
- [ ] **Hit effects** — particles, screen shake, damage numbers on impact
- [ ] **KO or time-up** ends the round, match goes best of 3
- [ ] **Camera preview** in bottom-right corner
- [ ] **Style/difficulty indicator** in bottom-left corner
- [ ] **Controls work:** P=pause, D=debug, M=mute, 1-5=NPC style, F1-F4=difficulty, R=restart

---

## 4. Input/Output Validation Criteria

### What "Working Perfectly" Means

| Component | Input | Good Output | Bad Output |
|-----------|-------|-------------|------------|
| **Pose Estimation** | Camera frame (BGR) | 13+ keypoints with confidence > 0.5 | No keypoints, or < 7 detected |
| **Smoothing** | Raw keypoints sequence | Smooth trajectory, jitter < 50% of raw | Jerky movement, or excessive lag (> 3 frames) |
| **Transform** | Front-facing (x,y,z) | Side-view (game_x, game_y) in [0,1280]x[0,720] | Coordinates outside screen, or arms longer than torso |
| **Jab Detection** | Lead hand punch toward camera | "jab" detected within 0.5s, exactly once | No detection, or 2+ detections per punch |
| **Cross Detection** | Rear hand punch toward camera | "cross" detected within 0.5s, exactly once | Confused with jab (check hand label) |
| **Hook Detection** | Lateral arm sweep then forward | "hook" detected within 0.5s, exactly once | Classified as cross (X-velocity threshold too high) |
| **Uppercut Detection** | Sharp upward fist motion | "uppercut" detected within 0.5s, exactly once | Classified as jab (Y-velocity threshold too high) |
| **Idle** | Standing still, no punching | No moves detected for 5+ seconds | Phantom detections while idle |
| **Walking** | Shift shoulders left/right | `is_walking=True`, game_x changes | Walking while standing still |
| **Calibration** | 3 recorded samples per move | Valid templates + computed thresholds | Failed recordings (no keypoints captured) |
| **NPC AI** | Player at x=300, NPC at x=900 | NPC approaches, attacks, blocks, retreats | NPC stands still, or teleports |
| **Combat** | Player punch hits NPC body | NPC HP decreases, effects trigger | No damage, or damage on miss |
| **Rendering** | Valid GamePose | Stick figure with joints, bones, fists, head | Missing limbs, or figure off-screen |

### Quantitative Benchmarks

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **FPS** | >= 25 | Press D in game for debug overlay |
| **Pose detection rate** | >= 90% of frames | `validate_video.py` M1 output |
| **Pose latency** | < 20ms per frame | `validate_video.py` M1 avg time |
| **Move detection accuracy** | >= 80% (pre-calibration), >= 95% (post-calibration) | Compare dojo counts to actual throws |
| **False positive rate** | < 1 per 30 seconds of idle | Stand still in dojo for 30s, check stats |
| **Smoothing jitter reduction** | >= 10% less than raw | `validate_video.py` M3 output |
| **Arm proportion** | Upper arm + forearm <= 1.2x torso height | Visual check: arms shouldn't look comically long |

---

## 5. Troubleshooting

### "No keypoints detected"
- Check lighting — MediaPipe needs your upper body clearly visible
- Stand 1-2m from camera, face it directly
- Ensure nothing blocks your shoulders/elbows/wrists
- Try: `python -c "from core.pose_estimator import PoseEstimator; print('Model loaded OK')"`

### "Moves detected but wrong type"
- Run Dojo mode, watch velocity bars:
  - Jab/Cross should spike the **Z (forward)** bar
  - Hook should spike the **X (lateral)** bar
  - Uppercut should spike the **Y (vertical)** bar
- If hook shows as cross: your hook's lateral movement isn't strong enough — exaggerate the sweep or calibrate

### "Too many detections per punch"
- Check `cooldown_frames` in `MoveDetectorConfig` (default: 15 frames = 0.5s)
- Increase to 20-25 if you punch slowly

### "Stick figure arms too long / too short"
- Check `z_to_x_scale` in `CoordinateTransformer` (default: 150)
- Check `segment_max_length` in `CoordinateTransformer` (default: 40px per segment)

### "FPS too low (<20)"
- Close other applications
- Reduce camera resolution: `VideoSource(source, target_fps=30, width=640, height=480)`
- Check CPU usage — MediaPipe is the bottleneck

### "Camera not opening"
- `python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"`
- Try different camera indices: `--source 1`, `--source 2`
- On Linux: check `/dev/video*` devices

---

## 6. Data Files Reference

| File | Purpose | When Created |
|------|---------|-------------|
| `calibration_profile.json` | Saved calibration templates + thresholds | After completing calibration |
| `dojo_session_log.json` | Dojo practice session results | After exiting Dojo mode |
| `test_output/*.png` | Visual test outputs | After running pytest |
| `validation_report.md` | Full pipeline validation report | After running validate_video.py |
| `validation_summary.json` | Machine-readable validation results | After running validate_video.py |
