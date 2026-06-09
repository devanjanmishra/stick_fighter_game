# Stick Fighter — Architecture & Code Reference

This document describes the codebase structure, module responsibilities, data flow, key algorithms, and extension points. It is designed to be used by both human developers and AI agents to understand and resume development.

---

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Data Flow Pipeline](#data-flow-pipeline)
- [Module Reference](#module-reference)
  - [Core Modules](#core-modules)
  - [Game Modules](#game-modules)
  - [Rendering Modules](#rendering-modules)
  - [Tests](#tests)
- [Key Algorithms](#key-algorithms)
- [Key Constants & Tuning Parameters](#key-constants--tuning-parameters)
- [Main Game Loop Integration](#main-game-loop-integration)
- [How to Extend](#how-to-extend)
- [Dependencies](#dependencies)
- [Testing Strategy](#testing-strategy)
- [Agent Resumption Guide](#agent-resumption-guide)

---

## High-Level Architecture

The system is split into three layers:

```
┌───────────────────────────────────────────────────────┐
│                   RENDERING LAYER                      │
│  stick_figure.py  game_renderer.py  combat_ui.py      │
│  effects_renderer.py  move_explainer.py               │
├───────────────────────────────────────────────────────┤
│                    GAME LAYER                          │
│  combat_system.py  npc.py  collision.py  effects.py   │
│  combo_tracker.py  sound_manager.py  npc_styles.py    │
├───────────────────────────────────────────────────────┤
│                    CORE LAYER                          │
│  pose_estimator.py  coordinate_transformer.py         │
│  smoothing.py  move_detector.py  movement_tracker.py  │
│  calibration.py                                       │
├───────────────────────────────────────────────────────┤
│                    ML LAYER (optional)                  │
│  ml/ml_move_detector.py  ml/models/                    │
│  ml/step1_extract_all_videos.py                        │
│  ml/step2_train_model.py                               │
└───────────────────────────────────────────────────────┘
```

- **Core Layer** — Processes raw camera input into game-usable data (keypoints → smoothed side-view positions → detected moves)
- **Game Layer** — Implements all game logic (NPC AI, combat, combos, effects, sound)
- **Rendering Layer** — Draws everything to screen via Pygame

Each layer only depends on layers below it. Rendering depends on Game and Core. Game depends on Core. Core is self-contained.

---

## Data Flow Pipeline

### Per-Frame Pipeline (during gameplay)

```
Camera Frame (BGR, 640x480)
    │
    ▼
PoseEstimator.detect(frame)
    │  Returns: dict[str, tuple[float, float, float]]
    │  Keys: "left_shoulder", "right_wrist", etc.
    │  Values: (x, y, z) normalized [0,1] relative to image
    │
    ▼
CoordinateTransformer.transform(keypoints)
    │  Returns: dict[str, tuple[float, float]]
    │  Maps front-facing (x,y,z) → side-view (game_x, game_y)
    │  Camera Z → Game X (punch extension)
    │  Camera Y → Game Y (vertical)
    │  Applies scale (400px/unit) and base position offset
    │
    ▼
KeypointSmoother.smooth(transformed_keypoints)
    │  Returns: dict[str, tuple[float, float]]
    │  One Euro Filter per keypoint per axis
    │  Removes jitter, preserves fast movements
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
MoveDetector.update(smoothed)         MovementTracker.update(smoothed)
    │  Returns: Optional[MoveEvent]        │  Returns: (walking: bool, direction: float)
    │  Checks velocity thresholds          │  Uses shoulder midpoint displacement
    │  Priority: uppercut > hook >         │
    │           jab/cross                  │
    │                                      │
    ├──────────────────────────────────────┘
    ▼
CombatSystem.process_player_move(move_event)
    │  Applies damage to NPC
    │  Checks hitbox collision
    │  Updates HP, round state
    │
    ▼
NPC.update(player_pos, game_phase)
    │  Behavior tree tick
    │  Returns: Optional[NPCAttack]
    │
    ▼
CombatSystem.process_npc_attack(npc_attack)
    │  Applies damage to player
    │
    ▼
EffectsManager.update() + ComboTracker.update()
    │  Particle physics, shake decay, combo window
    │
    ▼
Renderer draws everything:
    GameRenderer.render(screen, player_keypoints, npc_pose)
    CombatUI.render(screen, combat_state)
    EffectsRenderer.render(screen, effects_state)
```

### Calibration Pipeline (one-time setup)

```
User performs move (e.g., 3 jabs)
    │
    ▼
PoseEstimator + CoordinateTransformer (per frame)
    │
    ▼
CalibrationRecorder.add_frame(keypoints)
    │  Stores wrist positions per frame
    │
    ▼
CalibrationRecorder.finish_recording()
    │  Computes velocity profile from stored positions
    │  Returns: VelocityProfile (vx, vy, vz per frame)
    │
    ▼
CalibrationProfile.add_template(move_name, profile)
    │  Stores as reference template
    │
    ▼
CalibrationProfile.compute_thresholds()
    │  Sets personalized velocity thresholds at 60% of average peaks
    │
    ▼
CalibrationProfile.save("profile.json")
    │  Persists for future sessions
```

---

## Module Reference

### Core Modules

#### `core/pose_estimator.py`

Wraps MediaPipe PoseLandmarker for keypoint extraction.

```python
class PoseEstimator:
    def __init__(self, model_path: str = "models/pose_landmarker_lite.task")
    def detect(self, frame: np.ndarray) -> dict[str, tuple[float, float, float]]
    def close(self)
```

- **Input:** BGR numpy array (camera frame)
- **Output:** Dictionary mapping landmark names to (x, y, z) normalized coordinates
- **Landmark names used:** `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `nose`
- **Model:** `pose_landmarker_lite.task` — 5.5MB, runs on CPU at 25-30 FPS
- **Latency:** 20-50ms per frame

#### `core/coordinate_transformer.py`

Transforms front-facing camera coordinates to side-view game coordinates.

```python
class CoordinateTransformer:
    def __init__(self, screen_width: int = 1280, screen_height: int = 720,
                 base_x: float = 350, ground_y: float = 580,
                 scale: float = 400, z_to_x_scale: float = 550)
    def transform(self, keypoints: dict[str, tuple[float, float, float]]) -> dict[str, tuple[float, float]]
    def transform_single(self, x: float, y: float, z: float) -> tuple[float, float]
```

- **Axis mapping:**
  - Camera Z (depth) → Game X: `game_x = base_x - z * z_to_x_scale` (toward camera = forward in game)
  - Camera Y (vertical) → Game Y: `game_y = ground_y - (1 - y) * scale` (inverted — camera y=0 is top)
- **Key params:** `base_x=350` (player horizontal position), `ground_y=580` (ground line), `scale=400` (px per normalized unit), `z_to_x_scale=550` (depth to horizontal mapping)

#### `core/smoothing.py`

One Euro Filter implementation for keypoint smoothing.

```python
class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 5.0, d_cutoff: float = 1.0)
    def filter(self, value: float, timestamp: float) -> float

class KeypointSmoother:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 5.0, d_cutoff: float = 1.0)
    def smooth(self, keypoints: dict[str, tuple[float, float]], timestamp: float) -> dict[str, tuple[float, float]]
    def reset(self)
```

- Creates two OneEuroFilter instances per keypoint (one for X, one for Y)
- **Parameters:** `min_cutoff=1.0` (smoothing at rest), `beta=5.0` (speed coefficient — higher = less smoothing during fast motion), `d_cutoff=1.0` (derivative smoothing)
- See [Key Algorithms](#one-euro-filter) for the math

#### `core/move_detector.py`

Rule-based punch detection using velocity analysis.

```python
class VelocityTracker:
    def __init__(self, history_size: int = 10)
    @property
    def velocity(self) -> tuple[float, float, float]
    @property
    def peak_velocity(self) -> tuple[float, float, float]
    @property
    def displacement(self) -> tuple[float, float, float]

class MoveDetector:
    def __init__(self, stance: str = "orthodox",
                 cooldown_frames: int = 8, min_history: int = 3)
    def update(self, keypoints: dict[str, tuple[float, float, float]]) -> Optional[MoveEvent]
    def set_thresholds(self, thresholds: dict)

@dataclass
class MoveEvent:
    move_type: str       # "jab", "cross", "hook", "uppercut"
    hand: str            # "left" or "right"
    velocity: float      # peak velocity magnitude
    timestamp: float
```

- **Detection priority:** uppercut (y-velocity) → hook (x-velocity) → jab/cross (z-velocity)
- **Lead/rear logic:** Orthodox = left lead, right rear. Jab = lead hand z-punch, Cross = rear hand z-punch.
- **Cooldown:** 8 frames between detections to prevent double-counting
- **Min history:** 3 frames of data required before any detection
- **Default thresholds:** `z_velocity=15.0` (jab/cross), `x_velocity=12.0` (hook), `y_velocity=10.0` (uppercut)

#### `core/movement_tracker.py`

Walking detection via shoulder midpoint tracking.

```python
class MovementTracker:
    def __init__(self, walk_threshold: float = 0.02, history_size: int = 5)
    def update(self, keypoints: dict[str, tuple[float, float, float]]) -> tuple[bool, float]
```

- **Input:** Raw (not transformed) keypoints with left/right shoulder
- **Output:** `(is_walking: bool, direction: float)` where direction is signed displacement
- **Threshold:** Shoulder midpoint shift > 0.02 normalized units = walking

#### `core/calibration.py`

Personalized move calibration using DTW template matching.

```python
class VelocityProfile:
    vx: list[float]
    vy: list[float]
    vz: list[float]

class CalibrationRecorder:
    def __init__(self, target_hand: str = "right")
    def start_recording(self)
    def add_frame(self, keypoints: dict[str, tuple[float, float, float]])
    def finish_recording(self) -> VelocityProfile

class CalibrationProfile:
    def __init__(self, stance: str = "orthodox")
    def add_template(self, move_name: str, profile: VelocityProfile)
    def compute_thresholds(self) -> dict[str, float]
    def match_move(self, live_profile: VelocityProfile) -> tuple[str, float]
    def save(self, path: str)
    @classmethod
    def load(cls, path: str) -> "CalibrationProfile"
```

- **DTW matching:** Compares live velocity profile shape against stored templates
- **Threshold computation:** 60% of average peak velocity across recorded samples
- **Profile persistence:** JSON file with velocity arrays and computed thresholds

### Game Modules

#### `game/npc.py`

NPC AI with behavior tree and procedural pose generation.

```python
class NPCState(Enum):
    IDLE, APPROACH, ATTACK, RECOVER, BLOCK, RETREAT

class NPC:
    def __init__(self, x: float = 900, y: float = 580, style: str = "boxer")
    def update(self, player_x: float, game_phase: str) -> Optional[NPCAttack]
    def take_damage(self, damage: float, move_type: str)
    def get_pose(self) -> dict[str, tuple[float, float]]
    def get_hitboxes(self) -> dict[str, tuple[float, float, float, float]]

@dataclass
class NPCAttack:
    move_type: str
    damage: float
    hitbox: tuple[float, float, float, float]
```

- **State transitions:** IDLE→APPROACH (if far), APPROACH→ATTACK (if in range + cooldown done), ATTACK→RECOVER (after attack), any→BLOCK (random chance), any→RETREAT (low HP)
- **Attack selection:** Weighted random from style's attack_weights
- **Procedural poses:** Base fighting stance modified by current state and animation frame
- **Hitboxes:** Attack hitbox (active during attack frames) + body hitbox (always present)

#### `game/npc_styles.py`

Fighting style definitions and difficulty scaling.

```python
@dataclass
class FightingStyle:
    name: str
    hp: int
    speed: float
    block_chance: float
    attack_cooldown: int
    damage: dict[str, int]        # per-move damage
    attack_weights: dict[str, float]  # move selection probabilities
    color: tuple[int, int, int]

@dataclass
class DifficultyLevel:
    name: str
    speed_mult: float
    cooldown_mult: float
    block_mult: float
    damage_mult: float
    reaction_delay: int

STYLES: dict[str, FightingStyle]     # "boxer", "brawler", "counter", "speedster", "tank"
DIFFICULTIES: dict[str, DifficultyLevel]  # "easy", "medium", "hard", "nightmare"

def get_scaled_style(style_name: str, difficulty: str) -> FightingStyle
```

#### `game/collision.py`

AABB hitbox collision detection.

```python
def check_collision(box_a: tuple[float, float, float, float],
                    box_b: tuple[float, float, float, float]) -> bool

def get_attack_hitbox(attacker_pos: tuple[float, float],
                      move_type: str, facing_right: bool) -> tuple[float, float, float, float]

def get_body_hitbox(pos: tuple[float, float],
                    width: float = 40, height: float = 120) -> tuple[float, float, float, float]
```

- **Box format:** `(x, y, width, height)` where (x,y) is top-left corner
- **Attack hitbox sizes:** Vary by move — jab is narrow+long, hook is wide+medium, uppercut is medium+tall

#### `game/combat_system.py`

Central game state management.

```python
class GamePhase(Enum):
    MENU, COUNTDOWN, FIGHTING, ROUND_END, MATCH_END

@dataclass
class FighterStats:
    hp: float
    max_hp: float
    rounds_won: int

class CombatSystem:
    def __init__(self, max_rounds: int = 3, rounds_to_win: int = 2,
                 round_time: float = 60.0, countdown_time: float = 3.0)
    def start_match(self)
    def update(self, dt: float) -> list[str]  # returns list of events
    def apply_damage(self, target: str, damage: float, move_type: str) -> dict
    def get_state(self) -> dict
```

- **Events emitted:** `"countdown_tick"`, `"round_start"`, `"round_end"`, `"match_end"`, `"ko"`, `"time_up"`
- **State dict keys:** `phase`, `player`, `npc`, `round`, `timer`, `winner`, `countdown`

#### `game/combo_tracker.py`

Combo chain tracking with damage multipliers.

```python
class ComboTracker:
    def __init__(self, combo_window: int = 45)  # frames
    def register_hit(self, move_type: str, base_damage: float) -> dict
    def update(self)  # call every frame to decay combo window
    def get_combo_state(self) -> dict
```

- **Combo window:** 45 frames (1.5s at 30fps)
- **Returns on hit:** `{"hits": int, "multiplier": float, "label": str, "damage": float}`
- **Multipliers:** 1→1.0x, 2→1.2x, 3→1.5x, 4→1.8x, 5+→2.0x

#### `game/effects.py`

Visual and gameplay effects.

```python
class EffectsManager:
    def __init__(self)
    def spawn_hit_sparks(self, pos: tuple, move_type: str, count: int = 8)
    def spawn_damage_number(self, pos: tuple, damage: float, is_player: bool)
    def trigger_screen_shake(self, intensity: float)
    def trigger_hitstop(self, frames: int)
    def trigger_hit_flash(self, alpha: int = 80)
    def update(self) -> dict  # returns current effects state
```

- **Particles:** Position, velocity, color, alpha, lifetime — updated with gravity and drag
- **Screen shake:** Decays by 0.85x per frame
- **Hitstop:** Freezes game logic for N frames (2-5 depending on move)
- **Hit flash:** White overlay that decays by 15 alpha per frame

#### `game/sound_manager.py`

Procedurally synthesized combat sounds.

```python
class SoundManager:
    def __init__(self)
    def initialize(self) -> bool  # returns False if no audio device
    def play_hit(self, move_type: str)
    def play_block(self)
    def play_round_start(self)
    def play_ko(self)
```

- **Synthesis:** Generates PCM waveforms (sine waves + noise) at runtime
- **No audio files needed** — all sounds are generated programmatically
- **Graceful fallback:** All methods are no-ops if `initialize()` returns False

### ML Modules (optional, activated via `--ml` flag)

#### `ml/ml_move_detector.py`

Hybrid move detector combining velocity peak detection with a trained 1D-CNN classifier.

```python
class MoveClassifierCNN(nn.Module):
    """Lightweight 1D-CNN for move classification."""
    def __init__(self, n_features: int = 28, n_classes: int = 6)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    # Architecture: Conv1d(28→64) → Conv1d(64→128) → Conv1d(128→64) → GAP → Linear(64→32→6)
    # BatchNorm + Dropout at each conv layer

class MLMoveDetector:
    DEFAULT_MIN_PEAK_DISTANCE = 18
    DEFAULT_VELOCITY_THRESHOLD = 0.0425
    DEFAULT_WINDOW_SIZE = 16

    def __init__(self, model_dir: str = "ml/models",
                 min_peak_distance: int = 18,
                 velocity_threshold: float = 0.0425,
                 window_size: int = 16,
                 stance: str = "orthodox")
    def detect(self, pose: PoseFrame) -> DetectedMove
    @property
    def velocities(self) -> dict[str, float]  # for dojo velocity bars
```

- **Feature extraction:** 28 features per frame (alphabetically sorted):
  - Position features: `left_dx`, `left_dy`, `left_dz`, `left_wx`, `left_wy`, `left_wz`, `nose_y`, `right_dx`, `right_dy`, `right_dz`, `right_wx`, `right_wy`, `right_wz`, `shoulder_mid_x`, `shoulder_mid_y`, `shoulder_width`
  - Velocity features: `left_vdx`, `left_vdy`, `left_vdz`, `left_vwx`, `left_vwy`, `left_vwz`, `right_vdx`, `right_vdy`, `right_vdz`, `right_vwx`, `right_vwy`, `right_vwz`
- **Peak detection algorithm:**
  1. Compute velocity magnitude from wrist velocity features
  2. Check if frame at `buf[n-3]` is a local maximum (±2 frame lookahead)
  3. Require `velocity_magnitude >= 0.0425` threshold
  4. Enforce minimum 18 frames between consecutive peaks
- **Classification pipeline:**
  1. Extract 16-frame window centered on peak
  2. Normalize using training mean/std
  3. Run through 1D-CNN → softmax probabilities
  4. Apply velocity heuristic overrides (see below)
- **Velocity heuristics (override ML when confident):**
  - HOOK: `z_vel > 0.06 AND z_vel > y_vel * 2.0`
  - UPPERCUT: `y_vel > 0.035 AND y_vel > z_vel * 1.3 AND y_vel > x_vel * 1.5`
  - JAB/CROSS: Trust ML when confidence > 0.5
  - FALLBACK (ML says idle/walking but velocity peak exists): Use axis dominance
- **Interface:** Same `DetectedMove` output as `MoveDetector` — drop-in replacement
- **Performance:** <5ms per frame (CPU inference)

#### `ml/step1_extract_all_videos.py`

Extracts labeled features from video files through MediaPipe.

- Processes labeled videos (one move type per video) through MediaPipe
- Extracts 28 features per frame (position + velocity)
- Creates 16-frame sliding windows with stride=2
- Handles class imbalance via majority-label voting per window
- Outputs: `ml/data/all_videos_dataset.npz` (X, y, class_names arrays)

#### `ml/step2_train_model.py`

Trains the 1D-CNN move classifier.

- Loads windowed dataset from `ml/data/all_videos_dataset.npz`
- 80/20 train/val split with stratification
- Weighted cross-entropy loss (handles class imbalance)
- Adam optimizer, 100 epochs, early stopping
- Outputs: `ml/models/move_classifier.pt`, `ml/models/model_config.json`, `ml/models/norm_stats.npz`

### Rendering Modules

#### `rendering/stick_figure.py`

Draws the side-view stick figure character. **Used for both player and NPC rendering** (unified visual style).

```python
class StickFigureRenderer:
    def __init__(self, color=(50,120,255), head_color=(70,140,255),
                 line_width=4, head_radius=18, joint_radius=5, fist_radius=8)
    def draw(self, surface: pygame.Surface, game_pose: GamePose)
    def draw_ground_shadow(self, surface: pygame.Surface, game_pose: GamePose, ground_y: int = 580)
```

- **Input:** `GamePose` object (from `CoordinateTransformer` for player, or `NPCPose.to_game_pose()` for NPC)
- **Bones drawn:** 13 connections (neck, torso cross, arms shoulder→elbow→wrist, legs hip→knee→ankle)
- **Joint circles** at each keypoint (10 joints), larger fist circles at wrists
- **Head:** Circle with eye (direction based on `facing_right`)
- **Outline rendering:** Each bone drawn twice — wider black outline then colored line on top
- **Player color:** Blue `(50, 120, 255)` | **NPC color:** Red `(255, 70, 70)`
- **Character proportions:** ~190px total height (head-to-shoulder=30, shoulder-to-hip=70, hip-to-foot=90)

#### `rendering/game_renderer.py`

Scene compositor that draws background, characters, and ground.

```python
class GameRenderer:
    def __init__(self, screen_width: int = 1280, screen_height: int = 720)
    def draw_background(self, surface: pygame.Surface)
    def draw_scene(self, surface: pygame.Surface,
                   player_pose: Optional[GamePose] = None,
                   npc_pose: Optional[GamePose] = None)
    def draw_debug_info(self, surface: pygame.Surface, info: dict, font: pygame.font.Font)
```

- Contains `player_renderer` and `npc_renderer` (both `StickFigureRenderer` instances)
- `draw_scene()` draws background, shadows, then both stick figures
- Arena: dark background `(25,25,35)` with ground plane at y=580 and subtle grid lines

#### `rendering/combat_ui.py`

Heads-up display for combat state.

```python
class CombatUI:
    def __init__(self, width: int = 1280, height: int = 720)
    def render(self, screen: pygame.Surface, combat_state: dict)
    def draw_hp_bar(self, screen, x, y, hp, max_hp, color, label, flip)
    def draw_timer(self, screen, time_remaining)
    def draw_round_indicator(self, screen, current_round, p_wins, n_wins)
    def draw_countdown(self, screen, count)
    def draw_round_end(self, screen, winner_text)
    def draw_match_end(self, screen, winner_text)
```

- **HP bars:** Colored bar with border, label, and HP text — gradient from green→yellow→red
- **Timer:** Centered top, shows MM:SS
- **Round indicator:** Dots showing rounds played and won

#### `rendering/effects_renderer.py`

Draws all visual effects on top of the scene.

```python
class EffectsRenderer:
    def __init__(self)
    def render(self, screen: pygame.Surface, effects_state: dict,
               camera_offset: tuple = (0, 0))
```

- Renders particles, damage numbers, combo text, hit flash overlay
- Applies camera_offset (from screen shake) to all positions

#### `rendering/move_explainer.py`

In-app tutorial, calibration guide, and move animation system.

```python
class MoveAnimator:
    """Keyframe-based stick figure animation for move demos."""
    def __init__(self)
    def get_frame(self, move_name: str, time_ms: float) -> dict[str, tuple[float, float]]
    # Returns interpolated keypoint positions for the given move at given time

class MoveExplainer:
    def __init__(self, width: int = 1280, height: int = 720)
    def draw_welcome(self, screen)
    def draw_stance_select(self, screen, selected: str = "orthodox")
    def draw_move_explanation(self, screen, move_name: str, time_ms: float)
        # Shows move info + animated stick figure demo + LEFT/RIGHT nav hints
    def draw_record_prompt(self, screen, move_name: str, sample_number: int, total: int)
    def draw_countdown(self, screen, count: int, camera_frame=None)
        # Shows countdown number + optional live camera feed with keypoints
    def draw_recording(self, screen, move_name: str, progress: float, camera_frame=None)
        # Shows recording progress + optional live camera feed with keypoints
    def draw_playback(self, screen, move_name: str, frame: np.ndarray)
        # Shows recorded video frame with keypoints for verification
    def draw_record_done(self, screen, move_name: str, sample_number: int, success: bool)
    def draw_all_done(self, screen)
    def draw_move_overview(self, screen)

def draw_keypoints_on_frame(frame: np.ndarray, keypoints: dict) -> np.ndarray
    # Draws green keypoint dots + skeleton connections on a camera frame (OpenCV)
```

- **MoveAnimator:** Keyframe-based animation with linear interpolation between poses. Each move has 3-4 keyframes (base, extended, return) with hold durations.
- **Calibration flow:** 9 phases (welcome → stance_select → move_explain → record_prompt → countdown → recording → playback → record_done → all_done)
- **LEFT/RIGHT navigation:** Browse between moves in move_explain phase
- **Camera overlay:** `draw_keypoints_on_frame()` renders MediaPipe-style keypoints on camera frames using OpenCV
- Text wrapping, colored panels, progress indicators, animation support

### Tests

#### `tests/synthetic_data.py`

Generates realistic keypoint sequences for all moves without a camera.

```python
def generate_idle_pose() -> dict[str, tuple[float, float, float]]
def generate_jab_sequence(n_frames: int = 15) -> list[dict[str, tuple]]
def generate_cross_sequence(n_frames: int = 15) -> list[dict]
def generate_hook_sequence(n_frames: int = 15) -> list[dict]
def generate_uppercut_sequence(n_frames: int = 15) -> list[dict]
def generate_walk_sequence(direction: float = 1.0, n_frames: int = 20) -> list[dict]
```

- Each generator creates a multi-frame sequence simulating the move's keypoint trajectory
- Includes realistic noise and timing

#### Test Files

| File | Milestone | Tests | What's Tested |
|------|-----------|-------|---------------|
| `test_milestone1.py` | Camera + MediaPipe | 5 | Keypoint extraction, landmark names, coordinate ranges |
| `test_milestone2.py` | Side-view rendering | 5 | Coordinate transformation, stick figure drawing, axis mapping |
| `test_milestone3.py` | Smoothing | 4 | One Euro Filter behavior, jitter reduction, fast motion preservation |
| `test_milestone4.py` | Move detection | 8 | Jab/cross/hook/uppercut classification, cooldown, false positive prevention |
| `test_milestone5.py` | Walking | 7 | Shoulder tracking, walk detection, direction sensing |
| `test_milestone6.py` | Calibration | 6 | Recording, DTW matching, threshold computation, save/load |
| `test_milestone7.py` | NPC AI | 11 | State transitions, attack generation, collision, hitboxes |
| `test_milestone8.py` | Combat system | 11 | HP tracking, rounds, timer, game phases, KO, time-up |
| `test_milestone9.py` | Effects | 12 | Particles, shake, hitstop, flash, combos, sound synthesis |
| `test_milestone10.py` | NPC styles | 11 | Style stats, difficulty scaling, color assignments |

**Total: 80 tests, all passing.**

---

## Key Algorithms

### One Euro Filter

Adaptive low-pass filter that reduces jitter on stationary keypoints while preserving fast movements.

```
alpha(t) = 1 / (1 + tau / Te)           # smoothing factor
tau = 1 / (2 * pi * fc)                 # time constant
fc = min_cutoff + beta * |dx_hat|       # adaptive cutoff

dx_hat = LowPass(dx, alpha_d)           # smoothed derivative
x_hat = LowPass(x, alpha)              # smoothed value
```

- When velocity `|dx|` is low (idle), `fc ≈ min_cutoff` → heavy smoothing
- When velocity is high (punching), `fc` increases → minimal smoothing (responsive)
- `beta` controls sensitivity: higher = filter opens up faster during motion

**Tuned values:** `min_cutoff=1.0`, `beta=5.0`, `d_cutoff=1.0`

### Dynamic Time Warping (DTW)

Used in calibration to match live movement profiles against recorded templates.

```
DTW(A, B):
    Create matrix D[0..n, 0..m] = infinity
    D[0,0] = 0
    For i = 1..n:
        For j = 1..m:
            cost = |A[i] - B[j]|
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    Return D[n,m] / (n + m)  # normalized distance
```

- Handles speed variations (fast jab vs slow jab match the same template)
- O(n*m) where n,m are sequence lengths (~15-30 frames = ~450-900 operations)
- Normalized by path length for comparable distances across different-length sequences

### Move Detection State Machine

```
IDLE ──(velocity exceeds threshold)──> DETECTING
DETECTING ──(peak found + minimum frames)──> CLASSIFYING
CLASSIFYING ──(priority check)──> DETECTED(move_type)
DETECTED ──(cooldown expires)──> IDLE

Priority: uppercut (y) > hook (x) > jab/cross (z)
Lead hand + z-velocity = jab
Rear hand + z-velocity = cross
```

### NPC Behavior Tree

```
Root (Selector)
├── If HP < 20% → RETREAT (move away from player)
├── If player attacking → BLOCK (chance-based)
├── If distance > attack_range → APPROACH (walk toward player)
├── If cooldown_done → ATTACK (weighted random move)
├── If just_attacked → RECOVER (brief pause)
└── Default → IDLE (fighting stance)
```

---

## Key Constants & Tuning Parameters

### Screen & Rendering
| Constant | Value | Location |
|----------|-------|----------|
| Screen width | 1280 | `coordinate_transformer.py` |
| Screen height | 720 | `coordinate_transformer.py` |
| Player base X | 350 | `coordinate_transformer.py` |
| Ground Y | 580 | `coordinate_transformer.py` |
| Scale (px/unit) | 400 | `coordinate_transformer.py` |
| Z-to-X scale | 550 | `coordinate_transformer.py` |
| Head-to-shoulder | 30px | `stick_figure.py` |
| Shoulder-to-hip | 70px | `stick_figure.py` |
| Hip-to-foot | 90px | `stick_figure.py` |

### Move Detection
| Constant | Value | Location |
|----------|-------|----------|
| Z velocity threshold | 15.0 | `move_detector.py` |
| X velocity threshold | 12.0 | `move_detector.py` |
| Y velocity threshold | 10.0 | `move_detector.py` |
| Cooldown frames | 8 | `move_detector.py` |
| Min history frames | 3 | `move_detector.py` |
| Walk threshold | 0.02 | `movement_tracker.py` |

### Smoothing
| Constant | Value | Location |
|----------|-------|----------|
| min_cutoff | 1.0 | `smoothing.py` |
| beta | 5.0 | `smoothing.py` |
| d_cutoff | 1.0 | `smoothing.py` |

### Combat
| Constant | Value | Location |
|----------|-------|----------|
| Max rounds | 3 | `combat_system.py` |
| Rounds to win | 2 | `combat_system.py` |
| Round time | 60s | `combat_system.py` |
| Countdown time | 3s | `combat_system.py` |
| Combo window | 45 frames | `combo_tracker.py` |

### Effects
| Move | Hitstop | Screen Shake | Particles |
|------|---------|-------------|-----------|
| Jab | 2 frames | 3px | 6 |
| Cross | 3 frames | 5px | 8 |
| Hook | 4 frames | 8px | 10 |
| Uppercut | 5 frames | 10px | 12 |

---

## Main Game Loop Integration

The main game loop is implemented in `main.py` as the `StickFighterGame` class. Below is a simplified version of the integration pattern (see `main.py` for the full implementation with calibration, NPC rendering via `to_game_pose()`, and all UI):

```python
import pygame
from core.pose_estimator import PoseEstimator
from core.coordinate_transformer import CoordinateTransformer
from core.smoothing import KeypointSmoother
from core.move_detector import MoveDetector
from core.movement_tracker import MovementTracker
from game.combat_system import CombatSystem
from game.npc import NPC
from game.collision import check_collision, get_attack_hitbox, get_body_hitbox
from game.effects import EffectsManager
from game.combo_tracker import ComboTracker
from game.sound_manager import SoundManager
from rendering.game_renderer import GameRenderer
from rendering.combat_ui import CombatUI
from rendering.effects_renderer import EffectsRenderer

# Initialize
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()

pose = PoseEstimator()
transformer = CoordinateTransformer()
smoother = KeypointSmoother()
detector = MoveDetector(stance="orthodox")
movement = MovementTracker()
combat = CombatSystem()
npc = NPC(x=900, y=580, style="boxer")
effects = EffectsManager()
combos = ComboTracker()
sounds = SoundManager()
sounds.initialize()
renderer = GameRenderer()
ui = CombatUI()
fx_renderer = EffectsRenderer()

combat.start_match()
cap = cv2.VideoCapture(0)

running = True
frame_num = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture
    ret, frame = cap.read()
    if not ret:
        continue

    # Pipeline
    raw_kp = pose.detect(frame)
    game_kp = transformer.transform(raw_kp)
    smooth_kp = smoother.smooth(game_kp, frame_num / 30.0)

    # Detect moves
    move = detector.update(raw_kp)  # uses raw (x,y,z) for velocity
    walking, direction = movement.update(raw_kp)

    # Process combat
    dt = clock.get_time() / 1000.0
    events = combat.update(dt)

    if move:
        player_hitbox = get_attack_hitbox(
            (smooth_kp["right_wrist"][0], smooth_kp["right_wrist"][1]),
            move.move_type, facing_right=True)
        npc_body = get_body_hitbox((npc.x, npc.y))
        if check_collision(player_hitbox, npc_body):
            result = combat.apply_damage("npc", 10, move.move_type)
            combo_result = combos.register_hit(move.move_type, 10)
            effects.spawn_hit_sparks((npc.x, npc.y - 60), move.move_type)
            effects.spawn_damage_number((npc.x, npc.y - 80), combo_result["damage"], False)
            sounds.play_hit(move.move_type)

    # NPC AI
    npc_attack = npc.update(smooth_kp.get("left_shoulder", (350, 400))[0], combat.phase.name)
    if npc_attack:
        player_body = get_body_hitbox((350, 500))
        if check_collision(npc_attack.hitbox, player_body):
            combat.apply_damage("player", npc_attack.damage, npc_attack.move_type)

    # Update effects
    effects_state = effects.update()
    combos.update()

    # Render
    screen.fill((20, 20, 30))
    renderer.render(screen, smooth_kp, npc.get_pose(), (100, 200, 255), npc.color)
    ui.render(screen, combat.get_state())
    fx_renderer.render(screen, effects_state)
    pygame.display.flip()

    frame_num += 1
    clock.tick(30)

cap.release()
pose.close()
pygame.quit()
```

---

## How to Extend

### Adding a New Move (e.g., "elbow strike")

1. **`core/move_detector.py`** — Add detection logic in `_classify_hand_move()`:
   ```python
   # Check for elbow strike: elbow moves forward (z decrease) while wrist stays back
   if abs(elbow_vz) > self.thresholds.get("elbow_z", 12.0) and abs(wrist_vz) < 5.0:
       return "elbow_strike"
   ```

2. **`game/collision.py`** — Add hitbox dimensions in `get_attack_hitbox()`:
   ```python
   elif move_type == "elbow_strike":
       return (x + (20 if facing_right else -50), y - 30, 30, 30)
   ```

3. **`game/npc.py`** — Add to NPC's attack_weights and damage dict

4. **`game/effects.py`** — Add hitstop/shake values for the new move

5. **`rendering/move_explainer.py`** — Add entry to `MOVE_DATABASE`

6. **`tests/synthetic_data.py`** — Add `generate_elbow_strike_sequence()`

### Adding a New NPC Style

1. **`game/npc_styles.py`** — Add to `STYLES` dict:
   ```python
   "ninja": FightingStyle(
       name="ninja", hp=80, speed=4.0, block_chance=0.05,
       attack_cooldown=10, damage={"jab": 4, "cross": 7, ...},
       attack_weights={"jab": 0.5, "cross": 0.3, ...},
       color=(100, 0, 150)
   )
   ```

### Adding Leg Tracking

1. **`core/pose_estimator.py`** — Already extracts knee/ankle landmarks (unused)
2. **`core/coordinate_transformer.py`** — No changes needed (transforms any keypoint)
3. **`core/move_detector.py`** — Add kick detection tracking ankle velocity
4. **`rendering/stick_figure.py`** — Already draws legs if keypoints are provided

### Adding Player Blocking

1. **`core/move_detector.py`** — Detect block pose: both wrists near chin, elbows tucked
2. **`game/combat_system.py`** — Add `player_blocking` state, reduce incoming damage by 70-80%
3. **`rendering/stick_figure.py`** — Visual indicator (blue tint, shield icon)

---

## Dependencies

```
# Core
Python 3.10+
numpy                  # Array operations
mediapipe >= 0.10.9    # Pose estimation (tasks API)
pygame >= 2.6          # Rendering and input

# MediaPipe model (auto-downloaded)
models/pose_landmarker_lite.task  (5.5MB)

# Test dependencies (all in stdlib)
unittest
os, sys, json, time, math
```

No external dependencies beyond numpy, mediapipe, and pygame. All sounds are synthesized at runtime. All test data is generated programmatically.

---

## Testing Strategy

- **All tests use synthetic data** — no camera required, runs anywhere
- **Visual outputs** saved to `test_output/` as PNG for visual verification
- **Each milestone has its own test file** — can be run independently
- **No test framework dependency** — uses stdlib `unittest`
- **Total: 83 tests across 10 milestones**

Run all tests:
```bash
cd stick_fighter
python -m pytest tests/ -q
```

---

## Agent Resumption Guide

If you are an AI agent picking up this project, here is what you need to know:

### Current State (as of last session)
- All 10 milestones are implemented and tested (**83/83 tests passing**)
- `main.py` exists and wires everything together: camera → MediaPipe → transform → smooth → detect → combat → NPC AI → render
- `CalibrationFlow` class in `main.py` drives the full 9-phase calibration UI with animated demos, camera feed, and video playback
- Move detection validated on real user video: **11/11 moves detected** (2 jab, 4 cross, 2 hook, 3 uppercut)
- NPC rendering unified with player — both use `StickFigureRenderer` via `NPCPose.to_game_pose()`
- Arm proportions fixed (z_to_x_scale=150, per-segment clamping)
- Move-type morphing bug fixed (hooks no longer reclassify as jabs during retraction)
- **ML hybrid detector integrated** — `--ml` flag activates `MLMoveDetector` in both `StickFighterGame` and `DojoMode`
- ML model trained on 6 real labeled videos (1384 windows, 28 features, 100% val accuracy)
- ML detector tuned: `min_peak_distance=18, velocity_threshold=0.0425` → 11/11 correct on mixed video

### What to Do Next
1. **Improve walking detection in ML mode** — ML detector focuses on punch peaks; walking/idle classification could benefit from shoulder-velocity features or a separate walking detector.
2. **Collect more training data** — Current model trained on single user's videos. Additional body types/styles will improve generalization.
3. **End-to-end live test with ML** — Run `python main.py --source 0 --dojo --ml` to test ML detection in real time.
4. **Tune rule-based hook detection** — 3rd hook in test video classified as cross. Adjust `hook_x_velocity_threshold` or add DTW refinement.
5. **NPC selection menu** — Add a Pygame menu screen before match start to pick style/difficulty (currently CLI args: `--style`, `--difficulty`).
6. **Player blocking** — Detect block pose in `move_detector.py` (both wrists near chin, elbows tucked), reduce incoming damage in `combat_system.py`.
7. **Leg tracking** — MediaPipe already provides knee/ankle landmarks. Add kick detection to `move_detector.py` tracking ankle velocity.
8. **RL opponent** — Train via PPO self-play in simulation. Define reward function in `game/combat_system.py`, export tiny policy for <1ms inference.
9. **Mobile port** — Kivy+Buildozer for quick APK, or Flutter/Kotlin for production. See README.md for deployment strategy.

### Key Files to Understand First
1. `main.py` — **Start here.** Contains `StickFighterGame` (game loop), `DojoMode`, and `CalibrationFlow`. Both game classes accept `use_ml=True` to switch detectors. ~1540 lines.
2. `core/move_detector.py` — Rule-based move detection; velocity thresholds, classification priority, cooldown logic.
3. `ml/ml_move_detector.py` — ML hybrid detector; velocity peak detection + 1D-CNN classification + velocity heuristic overrides.
4. `core/coordinate_transformer.py` — The front-face → side-view axis mapping. Contains `GamePose` and `GameKeypoint` dataclasses.
5. `game/npc.py` — NPC AI behavior tree + `NPCPose.to_game_pose()` for unified rendering.
6. `rendering/stick_figure.py` — `StickFigureRenderer` used for both player and NPC.
7. `rendering/move_explainer.py` — `MoveAnimator` (keyframe animations) + `MoveExplainer` (calibration UI screens) + `draw_keypoints_on_frame()` (camera overlay).

### Common Pitfalls
- `move_detector.update()` expects raw (x,y,z) keypoints, NOT transformed game coordinates
- `smoother.smooth()` expects transformed (x,y) game coordinates
- NPC position is in game-world pixels (x=900, y=580), not normalized
- One Euro Filter needs monotonically increasing timestamps
- Combo window is in frames (45), not seconds — must call `combos.update()` every frame
- `NPCPose.to_game_pose()` must be called each frame to convert NPC procedural pose to `GamePose` for rendering
- Camera overlay uses OpenCV (BGR); Pygame surfaces are RGB — color conversion needed when displaying camera frames
- Calibration stores 3 samples per move × 4 moves = 12 recordings. `SAMPLES_PER_MOVE=3`, `RECORDING_FRAMES=45` (~1.5s at 30fps)

### Running Everything
```bash
# Run all tests
python -m pytest tests/ -q              # expect 83/83 passing

# Validate against a video file
python validate_video.py                # expect 10/10 milestones PASS

# Run the game with webcam (rule-based detection)
python main.py --source 0               # full calibration + fight
python main.py --source 0 --skip-calibration --style tank --difficulty hard

# Run with ML hybrid detector
python main.py --source 0 --ml          # ML detection + calibration + fight
python main.py --source 0 --dojo --ml   # ML detection in dojo mode
python main.py --source 0 --dojo --ml --skip-calibration

# Run with video file (headless testing)
python main.py --source path/to/video.mp4 --headless
```

### Project Constraints
- **No GitHub pushes** — all code stays local at `/home/ubuntu/stick_fighter/`
- **Python only** — no external build tools or compilers
- **CPU-only** — MediaPipe runs on CPU, no GPU required
- **No audio files** — all sounds synthesized at runtime
- **User video for validation:** `~/attachments/772c0aa0-.../WIN_20260516_18_45_10_Pro.mp4` (15.4s, 1280x720, 30fps)
