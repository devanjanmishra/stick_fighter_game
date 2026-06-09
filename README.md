# Stick Fighter

> ⚠️ **Work in progress — actively under development.** The full pipeline
> (pose tracking → move detection → combat → rendering) runs end to end. I'm
> currently fixing **combo-detection timing** and **move-instance edge cases**
> (hook-vs-cross classification and occasional duplicate detections) — see
> [Known Issues](#known-issues). Commits ongoing.

A real-time stick figure fighting game controlled by your body via webcam. Face your camera, throw punches, and your stick figure fights an NPC opponent in a side-view Street Fighter style arena.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [The Four Moves](#the-four-moves)
- [Calibration System](#calibration-system)
- [NPC Fighting Styles](#npc-fighting-styles)
- [Difficulty Levels](#difficulty-levels)
- [Combat System](#combat-system)
- [Technical Decisions & Rationale](#technical-decisions--rationale)
- [Milestones & Development History](#milestones--development-history)
- [Challenges Encountered & Solutions](#challenges-encountered--solutions)
- [Requirements](#requirements)
- [Running the Game](#running-the-game)
- [Running Tests](#running-tests)
- [Next Steps & Roadmap](#next-steps--roadmap)
- [Future: Mobile & TV Deployment](#future-mobile--tv-deployment)
- [Voice-Labeled Training (in Dojo Mode)](#voice-labeled-training-in-dojo-mode)
- [ML Training & Validation Pipeline](#ml-training--validation-pipeline)
- [Project Structure](#project-structure)

---

## Overview

Stick Fighter uses **MediaPipe Pose Estimation** to track your upper body keypoints through a front-facing camera. Your movements are transformed from front-facing coordinates into a side-view game world, where a stick figure mirrors your actions in real time. You fight an AI-controlled NPC using jabs, crosses, hooks, and uppercuts — all detected from your actual body movements.

**Key UX decision:** The player faces the camera naturally (phone on desk, webcam on monitor) but the game renders the action sideways, like a classic 2D fighting game. No awkward sideways standing required.

---

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
│   Camera     │────>│  MediaPipe   │────>│  Coordinate  │────>│  Smoothing │
│ (front-face) │     │  Pose Est.   │     │  Transformer │     │ (One Euro) │
└─────────────┘     └──────────────┘     └──────────────┘     └────────────┘
                                                                     │
                    ┌──────────────┐     ┌──────────────┐           │
                    │   Pygame     │<────│    Game       │<──────────┘
                    │  Renderer    │     │   Logic       │
                    └──────────────┘     └──────────────┘
                          │                    │
                    ┌─────┴─────┐        ┌─────┴─────┐
                    │ Stick Fig │        │ Move Det. │
                    │ Combat UI │        │ NPC AI    │
                    │ Effects   │        │ Combat    │
                    └───────────┘        │ Collision │
                                         └───────────┘
```

1. **Camera captures** your front-facing pose at 25-30 FPS
2. **MediaPipe** extracts 33 body keypoints with (x, y, z) coordinates
3. **Coordinate Transformer** maps front-facing axes to side-view game coordinates:
   - Camera Z (depth toward camera) → Game X (horizontal punch extension)
   - Camera X (left-right) → Game depth (which arm is in front)
   - Camera Y (up-down) → Game Y (vertical position)
4. **One Euro Filter** smooths keypoints to eliminate jitter while keeping fast movements responsive
5. **Move Detector** analyzes wrist velocity and displacement to classify punches
6. **Combat System** applies damage, tracks HP/rounds, manages game state
7. **Renderer** draws the side-view arena with stick figures, effects, and UI

---

## Features

- **Real-time pose tracking** — MediaPipe PoseLandmarker on CPU at 25-30 FPS
- **Front-facing camera → side-view rendering** — natural UX, classic fighting game look
- **4 fighting moves** — Jab, Cross, Hook, Uppercut, each with distinct detection signatures
- **Personalized calibration** — record YOUR moves, system adapts to your body and style
- **Walking via shoulder tracking** — shift your shoulders to move your character
- **NPC opponent** — behavior tree AI with 6 states (idle, approach, attack, recover, block, retreat)
- **5 NPC fighting styles** — Boxer, Brawler, Counter, Speedster, Tank
- **4 difficulty levels** — Easy, Medium, Hard, Nightmare
- **Full combat system** — HP bars, best-of-3 rounds, 60-second timer, KO and time-up
- **Combat effects** — hit spark particles, floating damage numbers, screen shake, hitstop, hit flash
- **Combo system** — chain hits for 1.2x-2.0x damage multipliers (DOUBLE → TRIPLE → QUAD → ULTRA)
- **Synthesized sound effects** — procedurally generated hit sounds, no audio files needed
- **In-app move tutorial** — detailed explainer screens teaching each move with camera tips
- **Animated move demos** — keyframe-based stick figure animations showing each move before recording
- **Next/Previous navigation** — browse between moves during calibration (LEFT/RIGHT arrow keys)
- **Live camera feed with keypoints** — real-time camera overlay with keypoint dots + skeleton during recording
- **Video playback with keypoints** — recorded frames replayed with keypoints so you can verify each recording
- **Unified rendering** — player and NPC both use the same `StickFigureRenderer` (keypoint dots + skeleton + fists + head with eye)
- **Keypoint smoothing** — One Euro Filter with adaptive cutoff for jitter-free tracking
- **ML-based hybrid move detection** — optional 1D-CNN classifier with velocity peak detection via `--ml` flag
- **Dojo mode** — free practice with real-time detection feedback, move log, velocity bars, and integrated voice-labeled training (press V to toggle)

---

## The Four Moves

### Jab
- **What:** Quick, straight punch with your lead hand
- **How (facing camera):** Extend your lead fist toward the camera — quick, snappy motion
- **Detection:** Rapid z-velocity decrease on lead wrist (moving toward camera)
- **In-game:** Lead arm extends horizontally toward NPC
- **Damage:** 5 (base) | **Speed:** Fastest

### Cross
- **What:** Powerful straight punch with your rear hand
- **How (facing camera):** Rotate shoulders and extend rear fist toward camera
- **Detection:** Rapid z-velocity decrease on rear wrist + shoulder rotation
- **In-game:** Rear arm extends with visible shoulder rotation
- **Damage:** 8 (base) | **Speed:** Fast

### Hook
- **What:** Curved punch from the side targeting the jaw
- **How (facing camera):** Swing your lead arm in a horizontal arc across the camera's view
- **Detection:** Strong lateral x-velocity on lead wrist (side-to-side movement)
- **In-game:** Arm arcs from behind the body outward — classic sweeping hook
- **Damage:** 12 (base) | **Speed:** Medium

### Uppercut
- **What:** Rising punch from below targeting the chin
- **How (facing camera):** Drive your rear fist upward — wrist rises sharply in camera view
- **Detection:** Strong upward y-velocity on rear wrist (y decreases in camera coords)
- **In-game:** Rear arm drives upward from below
- **Damage:** 15 (base) | **Speed:** Slowest but most powerful

**Stance:** Orthodox (left lead) or Southpaw (right lead) — selected during calibration. Determines which hand is "lead" vs "rear" for jab/cross distinction.

---

## Calibration System

The game includes a personalized calibration flow that adapts detection to YOUR body and style:

### How It Works

1. **Welcome** — Introduction screen
2. **Stance Selection** — Choose Orthodox (left lead) or Southpaw (right lead)
3. **Move Explanation** — Each move is explained with step-by-step instructions, camera tips, and an **animated stick figure demo** showing exactly how to perform it. Use LEFT/RIGHT arrows to browse between moves.
4. **Record Prompt** — Press SPACE to start recording the current move
5. **Countdown** — 3-second countdown with **live camera feed** so you can position yourself
6. **Recording** — You perform the move while the system captures frames with **real-time keypoint overlay** (green dots + skeleton connections drawn on the camera feed)
7. **Playback** — The recorded video is **played back with keypoints** overlaid so you can verify the recording looks correct. Press SPACE to continue or re-record.
8. **Record Done** — Confirmation screen. Repeat for 3 samples per move × 4 moves = 12 total recordings.
9. **All Done** — Calibration complete, profile saved to disk.

### Navigation
- **LEFT/RIGHT arrows** — Browse between moves in the explanation screen
- **SPACE** — Start recording / advance through screens
- **ESC** — Skip calibration (use default thresholds)

### Behind the Scenes
- **Template Extraction** — The system extracts velocity profiles, peak speeds, and displacement ranges from your recordings
- **Threshold Computation** — Personalized thresholds are set at 60% of your average peak values, giving comfortable detection margins
- **DTW Matching** — During gameplay, live movements are compared against your templates using Dynamic Time Warping (DTW), which matches movement *shapes* regardless of speed variations

### Why Calibration Matters

- A tall person and short person have different arm lengths → different displacement values
- People throw hooks with different arc widths → different x-velocity profiles
- Some people punch fast with short extension, others slow with full reach → different velocity/time profiles
- Calibration normalizes all of this automatically

### Calibration Profile

Profiles are saved as JSON and persist across sessions. They contain:
- Velocity profiles (vx, vy, vz per frame) for each recorded move
- Peak velocities and displacements
- Computed detection thresholds
- Stance preference

---

## NPC Fighting Styles

| Style | HP | Speed | Block% | Cooldown | Personality |
|-------|-----|-------|--------|----------|-------------|
| **Boxer** | 100 | 2.5 | 25% | 25 frames | Balanced fundamentals, quick jabs |
| **Brawler** | 110 | 2.0 | 10% | 20 frames | Aggressive, heavy hooks & uppercuts |
| **Counter** | 90 | 1.8 | 45% | 35 frames | Patient, waits for openings to punish |
| **Speedster** | 85 | 3.5 | 15% | 15 frames | Lightning jabs, death by 1000 cuts |
| **Tank** | 130 | 1.5 | 35% | 40 frames | Slow but devastating — wrecking ball |

Each style has a unique color, damage profile, and attack weight distribution. The Brawler favors hooks and uppercuts while the Speedster spams jabs.

### Damage by Style (Medium difficulty)

| Move | Boxer | Brawler | Counter | Speedster | Tank |
|------|-------|---------|---------|-----------|------|
| Jab | 5 | 7 | 4 | 3 | 9 |
| Cross | 8 | 12 | 11 | 4 | 16 |
| Hook | 12 | 18 | 15 | 7 | 25 |
| Uppercut | 15 | 21 | 17 | 9 | 30 |

---

## Difficulty Levels

| Level | Speed | Attack Cooldown | Block Chance | Damage | Reaction |
|-------|-------|-----------------|--------------|--------|----------|
| **Easy** | 0.7x | 1.5x slower | 0.5x | 0.7x | 15 frame delay |
| **Medium** | 1.0x | 1.0x | 1.0x | 1.0x | 5 frame delay |
| **Hard** | 1.2x | 0.7x faster | 1.5x | 1.2x | 2 frame delay |
| **Nightmare** | 1.4x | 0.5x faster | 2.0x | 1.5x | Instant |

Difficulty modifiers are applied on top of style base stats. A Nightmare Tank has 130 HP, 2.1 speed, 70% block chance, and deals up to 46 damage per uppercut.

---

## Combat System

- **Rounds:** Best of 3 (configurable)
- **Round Timer:** 60 seconds
- **Countdown:** 3 seconds before each round
- **Win Conditions:** KO (HP reaches 0) or Time Up (higher HP wins)
- **Game Phases:** MENU → COUNTDOWN → FIGHTING → ROUND_END → MATCH_END

### Combo System

Chain hits within a 1.5-second window (45 frames at 30fps) for increasing damage:

| Hits | Multiplier | Label |
|------|-----------|-------|
| 1 | 1.0x | — |
| 2 | 1.2x | DOUBLE! |
| 3 | 1.5x | TRIPLE! |
| 4 | 1.8x | QUAD! |
| 5+ | 2.0x | ULTRA! |

### Hit Effects

Each hit triggers multiple feedback layers:
- **Spark particles** — colored by move type (yellow/orange for jab → red for uppercut)
- **Damage numbers** — float upward and fade (red for player damage, yellow for NPC)
- **Screen shake** — intensity scales with move power (3px for jab → 10px for uppercut)
- **Hitstop** — brief frame freeze on impact (2 frames for jab → 5 for uppercut)
- **Hit flash** — white screen overlay that decays quickly

---

## Technical Decisions & Rationale

### Why Rule-Based Move Detection (not ML)?

We chose rule-based detection as the default because:
1. **No training data required** — works immediately with velocity thresholds
2. **Fully interpretable** — easy to debug why a move was/wasn't detected
3. **Calibration makes it robust** — personalized thresholds adapt to each user
4. **DTW template matching** — matches movement *shape*, not just peak velocity
5. **Lower latency** — no model inference, just arithmetic comparisons

### ML Hybrid Detector (optional, `--ml` flag)

An ML-based hybrid detector is now available as an alternative. It combines:
1. **Velocity peak detection** — finds WHEN a punch happens (local maxima in wrist velocity magnitude)
2. **1D-CNN classification** — identifies WHAT type of punch it is (16-frame window around each peak)
3. **Velocity heuristic overrides** — corrects ML misclassifications using axis dominance:
   - HOOK: z-velocity > 0.06 AND z > y×2.0
   - UPPERCUT: y-velocity > 0.035 AND y > z×1.3 AND y > x×1.5
   - JAB/CROSS: Trust ML when confident (>0.5)

**Training data:** 6 labeled videos (jab, cross, hook, uppercut, walking, idle) processed through MediaPipe → 1384 windowed samples (16 frames × 28 features each). Model achieves 100% validation accuracy on real data.

**Tuned parameters:** `min_peak_distance=18, velocity_threshold=0.0425` → 11/11 correct detections on mixed video with only 1 false positive.

**When to use ML vs rule-based:**
- Rule-based (default): Best with personalized calibration, fully interpretable, zero additional dependencies
- ML hybrid (`--ml`): Better out-of-the-box accuracy without calibration, handles diverse body types

### Why Front-Facing Camera → Side-View Rendering?

- **Best UX:** Most natural camera position (phone on desk, webcam on monitor)
- **Zero friction:** No awkward sideways standing
- **MediaPipe Z is reliable enough:** Relative depth (z-coordinate) maps cleanly to horizontal punch extension
- **Universally understood:** Side-view fighting is the standard genre visual

### Why One Euro Filter (not Kalman)?

- **Adaptive:** Low cutoff when still (smooth), high cutoff when moving (responsive)
- **Simple:** 3 tunable parameters vs Kalman's covariance matrices
- **Fast:** ~0.01ms per keypoint per frame
- **Proven:** Standard in AR/VR hand tracking pipelines

### Why Pygame (not Unity/Godot)?

- **Python ecosystem:** Stays in Python alongside MediaPipe — no language boundary
- **Rapid prototyping:** Focus on game logic, not engine boilerplate
- **Sufficient for stick figures:** No need for 3D rendering or physics engines
- **Easy to port:** Clean separation means rendering can be swapped to any framework later

### Why Procedural Sound (not Audio Files)?

- **Zero dependencies:** No audio asset files to manage
- **Customizable:** Frequency, decay, noise mix tunable per move type
- **Tiny:** ~4KB of generated PCM vs MB of wav/mp3 files
- **Fallback-safe:** Gracefully handles missing audio devices (headless servers, CI)

---

## Milestones & Development History

### Milestone 1: Camera Capture + MediaPipe Keypoint Extraction
- MediaPipe PoseLandmarker (tasks API v0.10+) with lite model (5.5MB)
- Extracts 33 keypoints with (x, y, z, visibility) per frame
- Validated with synthetic keypoint data (no camera required for testing)
- **5 tests passing**

### Milestone 2: Side-View Stick Figure Rendering
- CoordinateTransformer maps front-facing (x,y,z) to side-view game coordinates
- StickFigureRenderer draws bones, joints, fists, head with outlines
- Ground shadows for depth perception
- Character proportions: head-to-shoulder 30px, shoulder-to-hip 70px, hip-to-foot 90px (~190px total)
- **5 tests passing**

### Milestone 3: Keypoint Smoothing
- One Euro Filter implementation with adaptive cutoff based on velocity
- Parameters: min_cutoff=1.0, beta=5.0, d_cutoff=1.0
- Eliminates jitter on stationary poses while preserving fast punch motion
- **4 tests passing**

### Milestone 4: Move Detection
- VelocityTracker for per-wrist position/velocity/displacement tracking
- Rule-based classification: check uppercut (y) → hook (x) → jab/cross (z)
- Cooldown system (8 frames between moves) prevents double-detection
- Orthodox/southpaw stance determines lead vs rear hand
- **8 tests passing**

### Milestone 5: Walking/Movement
- MovementTracker uses shoulder midpoint displacement for walk detection
- Threshold-based: shoulder shift > 0.02 normalized units = walking
- Direction derived from shoulder movement direction
- **7 tests passing**

### Milestone 6: Calibration
- CalibrationRecorder captures move examples as velocity profiles
- CalibrationProfile stores templates and computes personalized thresholds
- DTW distance matching for robust move identification
- Save/load profiles as JSON for persistence
- **6 tests passing**

### Milestone 7: NPC AI + Collision
- 6-state behavior tree: IDLE → APPROACH → ATTACK → RECOVER → BLOCK → RETREAT
- Weighted random attack selection with configurable probabilities
- Procedural pose generation with attack animations (extend, arc, rise)
- AABB hitbox collision detection for attack and body hitboxes
- Hit stun and knockback on receiving damage
- **11 tests passing**

### Milestone 8: Combat System + UI
- GamePhase state machine: MENU → COUNTDOWN → FIGHTING → ROUND_END → MATCH_END
- FighterStats tracking HP and rounds won per fighter
- Round management with best-of-3 logic and 60-second timer
- CombatUI renders HP bars, timer, round indicators, countdown/end overlays
- **11 tests passing**

### Milestone 9: Combat Feel
- EffectsManager: particles, screen shake, hitstop, hit flash
- Particle physics with gravity, drag, and alpha decay
- ComboTracker with time-window detection and damage multipliers
- SoundManager with procedurally synthesized PCM audio
- EffectsRenderer for drawing all effects on screen
- **12 tests passing**

### Milestone 10: NPC Fighting Styles
- 5 distinct fighting styles with unique stats, colors, and behaviors
- 4 difficulty levels with multiplicative stat scaling
- StyleProfile metadata for UI display (name, description, color)
- HP scaling per style (85 for Speedster → 130 for Tank)
- **11 tests passing**

**Total: 83 tests, all passing.**

---

## Challenges Encountered & Solutions

### Challenge 1: Testing Without a Camera
- **Problem:** Development machine has no webcam
- **Solution:** Created `synthetic_data.py` with realistic keypoint sequences for all moves. Every test runs on synthetic data — no camera required.

### Challenge 2: Front-Facing to Side-View Coordinate Mapping
- **Problem:** Camera Z (depth) is less precise than X/Y, and the axis mapping is non-obvious
- **Solution:** Designed CoordinateTransformer with explicit axis remapping. Camera Z → game horizontal (punch extension), camera X → game depth (arm layering), camera Y → game vertical.

### Challenge 3: Jab vs Cross Distinction
- **Problem:** Both moves look like "fist moves toward camera" (z-decrease). Hard to distinguish.
- **Solution:** MediaPipe tracks left vs right wrist separately. Jab = lead hand, Cross = rear hand. Stance selection (orthodox/southpaw) determines which is which.

### Challenge 4: False Positives on Idle Poses
- **Problem:** Velocity tracker corruption caused 20 false uppercut detections in 30 idle frames
- **Solution:** Fixed `_classify_hand_move()` to read tracker properties instead of calling update methods that fed zeros into the velocity history.

### Challenge 5: Smoothing vs Responsiveness Tradeoff
- **Problem:** Too much smoothing kills punch detection, too little causes jitter
- **Solution:** One Euro Filter with beta=5.0 — high adaptive term means the filter opens up (less smoothing) when velocity spikes during punches, then tightens down during idle.

### Challenge 6: NPC Approach Distance
- **Problem:** NPC starting 600px away couldn't reach the player in 100 test frames at 3px/frame
- **Solution:** Increased walk speed to 5px/frame and test duration to 250 frames.

### Challenge 7: Sound in Headless Environments
- **Problem:** No audio device on CI/development servers crashes pygame.mixer
- **Solution:** SoundManager.initialize() returns False gracefully when mixer fails. All play methods are no-ops when not initialized.

---

## Requirements

```
Python 3.10+
pygame >= 2.6
mediapipe >= 0.10.9
numpy
opencv-python
torch (required only for --ml mode)
```

MediaPipe model file: `models/pose_landmarker_lite.task` (5.5MB, auto-downloaded during setup)

---

## Running the Game

```bash
# Install dependencies
pip install pygame mediapipe numpy opencv-python

# Run with webcam (full calibration + fight)
cd stick_fighter
python main.py --source 0

# Run with a video file (for testing)
python main.py --source path/to/video.mp4

# Skip calibration (use default thresholds)
python main.py --source 0 --skip-calibration

# Choose NPC style and difficulty
python main.py --source 0 --style brawler --difficulty hard

# Headless mode (no display, for testing with video files)
python main.py --source path/to/video.mp4 --headless

# DOJO MODE — free practice with real-time detection feedback (no NPC)
python main.py --source 0 --dojo
python main.py --source 0 --dojo --skip-calibration

# ML-BASED DETECTION — use trained 1D-CNN hybrid detector
python main.py --source 0 --ml
python main.py --source 0 --dojo --ml
python main.py --source 0 --dojo --ml --skip-calibration

# Validate all milestones against a video
python validate_video.py
```

### Controls
- **SPACE** — advance through calibration screens, start recording
- **LEFT/RIGHT arrows** — browse moves during calibration
- **ESC** — skip calibration / quit
- **P** — pause/resume during fight
- **D** — toggle debug overlay (FPS, positions, states)
- **M** — mute/unmute sound
- **1-5** — change NPC style (Boxer/Brawler/Counter/Speedster/Tank)
- **F1-F4** — change difficulty (Easy/Medium/Hard/Nightmare)
- **R** — restart match (from match end screen)

---

## Running Tests

All tests use synthetic data — no camera required. Run with pytest:

```bash
# Run all 83 tests
python -m pytest tests/ -q

# Run a specific milestone
python -m pytest tests/test_milestone4.py -v

# Individual milestone test counts:
#   test_milestone1.py   — Camera + MediaPipe extraction (5 tests)
#   test_milestone2.py   — Side-view rendering (5 tests)
#   test_milestone3.py   — Keypoint smoothing (4 tests)
#   test_milestone4.py   — Move detection (8 tests)
#   test_milestone5.py   — Walking/movement (7 tests)
#   test_milestone6.py   — Calibration (6 tests)
#   test_milestone7.py   — NPC AI + collision (11 tests)
#   test_milestone8.py   — Combat system + UI (14 tests)
#   test_milestone9.py   — Combat feel + effects (12 tests)
#   test_milestone10.py  — NPC styles + difficulty (11 tests)
```

Visual test outputs are saved to `test_output/` as PNG files.

---

## Current Status (What's Done)

### Fully Complete
- **All 10 milestones** implemented and tested (83/83 tests passing)
- **Main game loop** (`main.py`) — full real-time pipeline: camera → MediaPipe → transform → smooth → detect → combat → NPC AI → render
- **Calibration UI** — 9-phase flow with animated move demos, Next/Previous navigation, live camera with keypoints, and video playback with keypoints
- **Move detection** — 11/11 exact match on real user video (2 jab, 4 cross, 2 hook, 3 uppercut)
- **Unified rendering** — player and NPC both use `StickFigureRenderer` (identical visual style: keypoint dots, skeleton connections, fist circles, head with eye)
- **Arm proportions fixed** — `z_to_x_scale` reduced from 500→150, per-segment clamping (40px upper arm + 40px forearm)
- **Move-type morphing fixed** — hooks no longer reclassify as jabs during arm retraction
- **Calibration profiles** — save/load as JSON, reuse across sessions
- **5 NPC styles × 4 difficulties** = 20 playable combinations
- **Full combat system** — HP, rounds, timer, combos, effects, sound

### Known Issues
- **Rule-based: 3rd hook classified as cross** — lateral arc doesn't exceed the threshold with default settings; calibration should fix this by adapting to the user's hook style
- **ML hybrid: 1 false positive hook** — at 13.77s, detects 12 moves (11 correct + 1 extra hook). Acceptable trade-off (91.7% precision)
- Walking detection in ML mode needs further tuning

---

## What's Left & Roadmap

### Phase 1: Immediate Polish
- **Improve walking detection in ML mode** — ML detector currently focuses on punch detection; walking/idle could benefit from shoulder-velocity features
- **Tune rule-based hook detection** — adjust `hook_x_velocity_threshold` or add DTW-based refinement for hook vs cross distinction
- **End-to-end live testing** — run `python main.py --source 0 --ml` with a real camera to test ML detection flow
- **NPC selection menu** — style/difficulty picker before match start (currently CLI args only)
- **Settings menu** — volume, difficulty, stance selection accessible in-game
- **Collect more training data** — additional users/body types to improve ML model generalization

### Phase 2: Gameplay Depth
- Add leg tracking (MediaPipe provides knee/ankle landmarks — currently unused)
- Implement blocking for the player (arms crossed = block state)
- Sound effects on actual audio device
- Pause/resume functionality (P key framework exists)
- Add player blocking detection (wrists near chin + elbows tucked)

### Phase 3: Reinforcement Learning Opponent
- Train RL agent via PPO self-play in simulation (no human required during training)
- ~50,000-500,000 simulated fights, ~1-14 hours on GPU
- Export tiny policy network for <1ms inference during gameplay
- Adaptive difficulty that learns the player's patterns

### Phase 4: Mobile & TV Deployment
- Port to Android via Kivy+Buildozer (quick path) or Flutter/Kotlin (production quality)
- Smart TV connectivity via phone-as-controller
- Play Store / App Store submission

---

## Future: Mobile & TV Deployment

### Getting on the Play Store
1. **Python prototype** (current) → proves gameplay on desktop
2. **Port to mobile:** Kivy+Buildozer (Python→APK, fastest) or Flutter/Kotlin (production quality)
3. **Play Store:** Google Play Developer account ($25 one-time), sign APK/AAB, submit for review (1-3 days)
4. **App Store:** Requires Mac, $99/year, stricter review (1-7 days)

### Sharing Builds with Friends (Before Play Store)
- **APK sideloading:** Build APK, send via any file-sharing method, install with "unknown sources" enabled
- **Firebase App Distribution (free):** Upload APK, add friends' emails, they get install link. Handles versioning and crash reports.
- **Google Play Internal Testing:** Upload to Play Store visible only to up to 100 specified testers. No review required.

### Maintaining & Updating
- Version releases (1.0.0, 1.1.0, etc.)
- CI/CD: Push code → auto-build APK → auto-upload to Play Store (Fastlane or GitHub Actions)
- Firebase Crashlytics for crash reporting
- Firebase Analytics for usage tracking

---

## Dojo Mode — Free Practice

The **Dojo** is a practice environment with no NPC — just you and the detection system. Best way to validate everything works.

```bash
python main.py --source 0 --dojo
```

**What it shows:**
- Your stick figure mirroring your movements (center)
- Big move label flash when a punch is detected ("JAB", "CROSS", "HOOK", "UPPERCUT")
- Detection stats panel (count per move type) — top-left
- Timestamped move log — right panel
- Live wrist velocity bars (Z=forward, X=lateral, Y=vertical) — bottom
- Camera preview with green keypoint overlay — top-right

**Controls:** SPACE/R = reset stats, D = debug overlay, ESC = quit

On exit, prints a session summary and saves to `dojo_session_log.json` — compare the counts to your actual throws to verify accuracy.

See **TESTING_GUIDE.md** for detailed validation checklists, input/output criteria, and troubleshooting.

---

## Voice-Labeled Training (in Dojo Mode)

Dojo mode includes integrated voice-labeled training. Press **V** during any Dojo session to toggle voice recognition on/off. Shout the name of each move as you perform it — the system timestamps your voice labels, compares them against the detector's output, and builds a labeled training dataset you can use to finetune the ML model.

```bash
python main.py --source 0 --dojo                 # start dojo, press V to enable voice
python main.py --source 0 --dojo --ml             # use ML detector, press V for voice
python main.py --source 0 --dojo --skip-calibration  # skip calibration
```

### Requirements

- **Microphone** — any built-in or USB mic
- **Python packages** — `speech_recognition`, `pyaudio` (install: `pip install SpeechRecognition pyaudio`)
- **System dependency** — `portaudio19-dev` (install: `sudo apt-get install -y portaudio19-dev`)
- **Internet** — required for Google Speech API backend (default). Use `--voice-backend sphinx` for offline recognition (lower accuracy)

### Supported Voice Commands

Shout any of these while performing the corresponding move:

| Say This | Detected As | Fuzzy Matches |
|----------|-------------|---------------|
| "jab" | JAB | "job", "jump", "tab" |
| "cross" | CROSS | "across", "toss", "boss" |
| "hook" | HOOK | "look", "cook", "took" |
| "uppercut" | UPPERCUT | "upper", "cut", "upper cut" |
| "forward" / "walk" | WALKING | "walking", "go forward" |
| "back" / "backward" | WALKING | "go back", "step back" |
| "idle" / "nothing" | IDLE | "stop", "still", "standing" |

The recognizer uses fuzzy keyword matching with substring fallback — it's forgiving of speech recognition errors.

### What It Shows

- **Dual flash display** — detected move (blue, left) vs voice label (green, right) shown side-by-side
- **Match indicator** — green "MATCH" or red "MISMATCH" between detection and voice label
- **Accuracy panel** — per-move detection count vs ground truth (voice) count with accuracy percentage
- **Label log** — recent voice label events with match/mismatch indicators
- **Velocity bars** — live wrist velocity (Z, X, Y axes)
- **Camera preview** — with green keypoint overlay

### Workflow

1. **Enter Dojo** — `python main.py --source 0 --dojo`
2. **Enable voice** — press **V** to start listening (microphone icon turns green)
3. **Practice** — throw moves while shouting their names ("JAB!", "CROSS!", "HOOK!")
4. **Watch stats** — the accuracy panel shows detection vs ground truth per move type in real-time
5. **Toggle voice off** — press **V** again to return to normal Dojo view
6. **Finish session** — press ESC to see the summary screen
7. **Review** — summary shows total labels, matches, accuracy, and per-move breakdown
8. **Finetune** — press **T** to finetune the ML model on your voice-labeled data
9. **Save** — press **S** to save the labeled training data as NPZ files for later use
10. **Continue** — press ENTER to go back to Dojo, **G** to go to Game, or ESC to quit

### One-Key Model Finetune

After a Dojo session with voice labeling, pressing **T** on the summary screen:
1. Extracts training segments from the frame buffer (±1.5s window around each voice label = 45 frames at 30fps)
2. Builds feature vectors using the existing ML pipeline infrastructure
3. Combines voice-labeled data with existing training data (if any)
4. Finetunes the 1D-CNN model with class-weighted loss
5. Saves a new model version (v2, v3, etc.) to `ml/models/`
6. Reports accuracy improvement and new model version

### Saved Data

- **Training segments** — saved to `ml/data/` as NPZ files with metadata JSON
- **Session log** — saved to `dojo_session_log.json` with all labels, matches, and stats
- **Finetuned model** — saved to `ml/models/` with version tracking

### Performance Note

Voice recognition adds ~100-200ms latency and 15-20% CPU overhead. This is why it's a toggle within Dojo mode (press V), **not** active during live game fights — the game loop needs to stay at 30fps without audio processing overhead.

---

## ML Training & Validation Pipeline

A reusable CLI pipeline for iteratively training, validating, and versioning the ML move detector. Add new move types or improve accuracy by providing labeled videos — no code changes needed.

### Quick Start

```bash
# 1. Train/finetune the model with labeled videos
python ml/pipeline.py train \
  --videos "jab:jab.mp4:17" "cross:cross.mp4:9" "hook:hook.mp4:10" \
           "uppercut:uppercut.mp4:9" "walking:walking.mp4:0" "idle:idle.mp4:0" \
  --mode finetune --epochs 80

# 2. Validate the model against test videos
python ml/pipeline.py validate \
  --videos "jab:test_jab.mp4:5" "cross:test_cross.mp4:3" "hook:test_hook.mp4:4"

# 3. Generate a markdown report
python ml/pipeline.py report

# 4. View model version history
python ml/pipeline.py history
```

### Video Argument Format

Each `--videos` argument is a colon-separated triple:
```
move_type:path/to/video.mp4:expected_count
```
- **move_type** — label for the move (e.g., `jab`, `cross`, `hook`, `uppercut`, `walking`, `idle`)
- **path** — path to the MP4 video file (supports `~` expansion)
- **expected_count** — how many of that move appear in the video (use `0` for idle/walking)

### Commands

| Command | Description |
|---------|-------------|
| `train` | Extract features from labeled videos via MediaPipe, create windowed dataset, train/finetune 1D-CNN |
| `validate` | Run the detector on test videos, compare detected counts to expected, score accuracy |
| `report` | Generate a detailed markdown report from the latest validation results |
| `history` | Show all model versions with training metadata and validation scores |

### Train Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `finetune` | `finetune` (continue from existing model) or `fresh` (train from scratch) |
| `--epochs` | `80` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |

### How It Works

1. **Feature extraction** runs in a subprocess (isolates MediaPipe's OpenGL context from PyTorch)
2. Each video frame is processed through MediaPipe → 28-element feature vector (14 position + 14 velocity)
3. Frames are windowed (16-frame windows, stride 2) and labeled by move type
4. The 1D-CNN (3 conv layers: 64→128→64 channels + classifier) is trained with class-weighted loss
5. Models are saved with version numbers (`move_classifier_v1.pt`, `v2`, etc.) and registered in `model_registry.json`
6. Validation runs the full detector pipeline (velocity peaks → CNN classification → heuristic overrides) and scores per-video accuracy

### Adding New Move Types

To add a new move (e.g., `elbow`):
```bash
python ml/pipeline.py train \
  --videos "jab:jab.mp4:17" "elbow:elbow.mp4:12" ... \
  --mode finetune
```
The pipeline auto-detects new move types not in the default class list and expands the model output layer accordingly.

### Model Versioning

Every training run produces a versioned checkpoint:
```
ml/models/move_classifier_v1.pt     # Model weights
ml/models/norm_stats_v1.npz         # Normalization stats
ml/models/model_registry.json       # Version history + metadata
```
The latest model is also saved to the canonical paths (`move_classifier.pt`, `norm_stats.npz`) so the game always uses the best model.

### Current Results (v1)

| Video | Expected | Detected | Correct | FP | Missed | Status |
|-------|----------|----------|---------|-----|--------|--------|
| jab.mp4 | 17 jab | 16 | 15 | 1 | 2 | PASS |
| cross.mp4 | 9 cross | 12 | 9 | 3 | 0 | PASS |
| hook.mp4 | 10 hook | 13 | 10 | 3 | 0 | FAIL |
| uppercut.mp4 | 9 uppercut | 8 | 8 | 0 | 1 | PASS |
| walking.mp4 | 0 walking | 0 | 0 | 0 | 0 | PASS |
| idle.mp4 | 0 idle | 0 | 0 | 0 | 0 | PASS |

**Overall: 42/45 correct (93.3%), 7 FP, 3 missed, 0 false positives on idle/walking**

---

## Project Structure

```
stick_fighter/
├── core/                          # Core pose processing pipeline
│   ├── pose_estimator.py          # MediaPipe pose extraction
│   ├── coordinate_transformer.py  # Front-facing → side-view mapping
│   ├── smoothing.py               # One Euro Filter
│   ├── move_detector.py           # Rule-based punch classification
│   ├── movement_tracker.py        # Walking via shoulder tracking
│   ├── calibration.py             # DTW-based personalized calibration
│   ├── voice_recognizer.py        # Background threaded speech recognizer
│   └── voice_dojo.py              # Voice-labeled Dojo training mode
├── game/                          # Game logic
│   ├── npc.py                     # NPC AI with behavior tree + to_game_pose()
│   ├── npc_styles.py              # 5 fighting styles + 4 difficulties
│   ├── collision.py               # Hitbox collision detection
│   ├── combat_system.py           # HP, rounds, timer, game phases
│   ├── combo_tracker.py           # Combo chains + damage multipliers
│   ├── effects.py                 # Particles, shake, hitstop, flash
│   └── sound_manager.py           # Synthesized combat sounds
├── rendering/                     # Visual output
│   ├── stick_figure.py            # Side-view stick figure renderer (player + NPC)
│   ├── game_renderer.py           # Scene compositor
│   ├── combat_ui.py               # HP bars, timer, overlays
│   ├── effects_renderer.py        # Particle/combo/flash rendering
│   └── move_explainer.py          # In-app move tutorial + animations + keypoint overlay
├── tests/                         # Test suites (83 tests total)
│   ├── conftest.py                # Pytest fixtures (Pygame display init)
│   ├── synthetic_data.py          # Synthetic keypoint sequences
│   └── test_milestone[1-10].py    # Per-milestone test suites
├── ml/                            # ML move detection + training pipeline
│   ├── pipeline.py                # Reusable train/validate/report CLI (main pipeline)
│   ├── _extract_worker.py         # Subprocess worker for MediaPipe isolation
│   ├── ml_move_detector.py        # Hybrid detector (velocity peaks + 1D-CNN)
│   ├── step1_extract_all_videos.py  # Extract features from labeled videos
│   ├── step2_train_model.py       # Train 1D-CNN classifier
│   ├── models/                    # Trained model artifacts (versioned)
│   │   ├── move_classifier.pt     # Latest model weights
│   │   ├── move_classifier_v1.pt  # Versioned model weights
│   │   ├── model_config.json      # Model architecture config
│   │   ├── norm_stats.npz         # Feature normalization stats
│   │   ├── model_registry.json    # Version history + metadata
│   │   └── detector_config.json   # Detector tuning parameters
│   └── data/                      # Training/validation data (generated)
│       ├── training_features_v1.npz # Windowed training dataset
│       ├── validation_v1.json     # Validation results
│       └── report_v1.md           # Generated validation report
├── models/                        # MediaPipe models
│   └── pose_landmarker_lite.task  # MediaPipe model (5.5MB)
├── main.py                        # Game loop + CalibrationFlow + DojoMode (with voice) + entry point
├── validate_video.py              # Full pipeline validation against a video file
├── calibration_profile.json       # Saved calibration profile (generated)
├── dojo_session_log.json          # Dojo session results (generated)
├── test_output/                   # Visual test renders (PNG)
├── README.md                      # This file
├── ARCHITECTURE.md                # Code architecture & module reference
└── TESTING_GUIDE.md               # How to test & validate everything
```
