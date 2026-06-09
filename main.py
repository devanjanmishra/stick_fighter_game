"""
Stick Fighter - Real-time fighting game driven by camera pose estimation.

Pipeline per frame:
  Camera -> PoseEstimator -> PoseSmoother -> CoordinateTransformer -> GamePose
                                          -> MoveDetector -> DetectedMove
                                          -> MovementTracker -> MovementState
  DetectedMove + NPC -> Collision -> CombatSystem -> Effects/Sound
  Everything -> GameRenderer + CombatUI + EffectsRenderer

Controls:
  ESC / Q    - Quit
  M          - Mute/unmute sound
  P          - Pause
  R          - Restart match (from match end screen)
  D          - Toggle debug overlay
  1-5        - Select NPC style (Boxer/Brawler/Counter/Speedster/Tank)
  F1-F4      - Select difficulty (Easy/Medium/Hard/Nightmare)
"""

import os
import sys
import time
import pygame

from core.pose_estimator import PoseEstimator, VideoSource, PoseFrame
from core.smoothing import PoseSmoother, SmoothingConfig
from core.coordinate_transformer import CoordinateTransformer, GamePose, GameKeypoint
from core.move_detector import MoveDetector, MoveDetectorConfig, MoveType, MovePhase
from core.movement_tracker import MovementTracker, MovementConfig
from core.calibration import CalibrationRecorder, CalibrationProfile, CalibratedThresholds

from game.combat_system import CombatSystem, CombatConfig, GamePhase
from game.npc import NPC, NPCConfig, NPCState, NPCAttackType, NPCPose, Hitbox
from game.collision import (
    check_collision, get_player_attack_hitbox, get_player_body_hitbox,
)
from game.combo_tracker import ComboTracker
from game.effects import EffectsManager
from game.sound_manager import SoundManager
from game.npc_styles import (
    FightingStyle, Difficulty, get_npc_config, get_npc_hp,
    STYLE_PROFILES, list_styles, list_difficulties,
)

from rendering.game_renderer import GameRenderer
from rendering.stick_figure import StickFigureRenderer, NPC_COLOR, HEAD_COLOR_NPC
from rendering.combat_ui import CombatUI
from rendering.effects_renderer import EffectsRenderer
from rendering.move_explainer import (
    MoveExplainer, ExplainerPhase, MOVE_ORDER, draw_keypoints_on_frame,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 30
GROUND_Y = 580
PLAYER_START_X = 300.0
NPC_START_X = 900.0

CAMERA_PREVIEW_W = 240
CAMERA_PREVIEW_H = 180

CALIBRATION_PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "calibration_profile.json"
)
SAMPLES_PER_MOVE = 3
RECORDING_FRAMES = 45  # ~1.5s at 30fps
COUNTDOWN_SECONDS = 3


# ---------------------------------------------------------------------------
# NPC stick figure drawing helper
# ---------------------------------------------------------------------------
def draw_npc_stick_figure(
    surface: pygame.Surface,
    npc_pose: NPCPose,
    color: tuple[int, int, int] = NPC_COLOR,
    head_color: tuple[int, int, int] = HEAD_COLOR_NPC,
    line_width: int = 4,
    head_radius: int = 16,
):
    """Draw the NPC as a proper stick figure from its NPCPose."""
    pts = npc_pose.as_dict()

    bones = [
        ("head", "neck"),
        ("neck", "left_shoulder"),
        ("neck", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "hip_left"),
        ("right_shoulder", "hip_right"),
        ("hip_left", "hip_right"),
        ("hip_left", "knee_left"),
        ("knee_left", "ankle_left"),
        ("hip_right", "knee_right"),
        ("knee_right", "ankle_right"),
    ]

    outline = (30, 30, 30)

    for a, b in bones:
        if a == "head":
            continue
        p1 = (int(pts[a][0]), int(pts[a][1]))
        p2 = (int(pts[b][0]), int(pts[b][1]))
        pygame.draw.line(surface, outline, p1, p2, line_width + 2)
        pygame.draw.line(surface, color, p1, p2, line_width)

    # Joints
    joint_names = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "hip_left", "hip_right",
        "knee_left", "knee_right",
        "ankle_left", "ankle_right",
    ]
    for name in joint_names:
        pos = (int(pts[name][0]), int(pts[name][1]))
        pygame.draw.circle(surface, outline, pos, 5 + 1)
        pygame.draw.circle(surface, color, pos, 5)

    # Fists
    for name in ["left_wrist", "right_wrist"]:
        pos = (int(pts[name][0]), int(pts[name][1]))
        pygame.draw.circle(surface, outline, pos, 8 + 1)
        pygame.draw.circle(surface, color, pos, 8)

    # Head
    hx, hy = int(pts["head"][0]), int(pts["head"][1])
    pygame.draw.circle(surface, outline, (hx, hy), head_radius + 2)
    pygame.draw.circle(surface, head_color, (hx, hy), head_radius)
    # Eye
    pygame.draw.circle(surface, (0, 0, 0), (hx - 5, hy - 3), 3)


# ---------------------------------------------------------------------------
# Calibration flow
# ---------------------------------------------------------------------------
class CalibrationFlow:
    """
    Runs the calibration UI before the main game starts.

    Flow: welcome -> stance_select -> (for each move: move_explain
          -> record_prompt -> countdown -> recording -> playback
          -> record_done) -> all_done

    Navigation: LEFT/RIGHT arrows in move_explain to browse moves.
    SPACE in move_explain to start recording the current move.

    During recording the live camera feed with keypoint overlay is shown.
    After recording the captured video is played back with keypoints so
    the user can verify the recording before continuing.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        video_source: "VideoSource",
        pose_estimator: "PoseEstimator",
        smoother: "PoseSmoother",
    ):
        self.screen = screen
        self.clock = clock
        self.video_source = video_source
        self.pose_estimator = pose_estimator
        self.smoother = smoother

        self.explainer = MoveExplainer(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.recorder = CalibrationRecorder()
        self.profile = CalibrationProfile()

        self.stance = "orthodox"
        self.phase = ExplainerPhase.WELCOME
        self.current_move_idx = 0
        self.current_sample = 1
        self.countdown_timer = 0
        self.recording_frames = 0
        self.last_record_success = True

        # Camera / playback state
        self._camera_surface: pygame.Surface | None = None
        self._recorded_frames: list[tuple] = []  # (bgr_frame, keypoints_dict)
        self._playback_idx = 0
        self._playback_speed = 2  # show every Nth frame for faster playback

    def run(self) -> CalibrationProfile | None:
        """
        Run the full calibration flow.
        Returns a CalibrationProfile if completed, or None if skipped/cancelled.
        """
        # Check for existing profile
        if os.path.exists(CALIBRATION_PROFILE_PATH):
            action = self._ask_load_or_recalibrate()
            if action == "load":
                try:
                    profile = CalibrationProfile.load(CALIBRATION_PROFILE_PATH)
                    if profile.is_fully_calibrated():
                        return profile
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass  # Corrupted file, proceed with calibration
            elif action == "quit":
                return None

        self.phase = ExplainerPhase.WELCOME
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        result = self._confirm_skip()
                        if result == "skip":
                            return None
                    else:
                        running = self._handle_key(event.key)

            # Process camera during countdown / recording / playback
            if self.phase == ExplainerPhase.COUNTDOWN:
                self._process_countdown_frame()
            elif self.phase == ExplainerPhase.RECORDING:
                self._process_recording_frame()
            elif self.phase == ExplainerPhase.PLAYBACK:
                self._process_playback_frame()

            self._draw()
            self.clock.tick(FPS)

        # Compute thresholds and save
        if self.profile.is_fully_calibrated():
            self.profile.compute_thresholds()
            self.profile.save(CALIBRATION_PROFILE_PATH)
            return self.profile
        return None

    def _ask_load_or_recalibrate(self) -> str:
        """Show screen asking to load existing profile or recalibrate."""
        fonts = self.explainer._get_fonts()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_l:
                        return "load"
                    elif event.key == pygame.K_r:
                        return "recalibrate"
                    elif event.key == pygame.K_ESCAPE:
                        return "load"

            self.screen.fill((20, 20, 30))
            title = fonts["title"].render("CALIBRATION PROFILE FOUND", True, (80, 255, 120))
            self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 200))

            opt1 = fonts["body"].render(
                "Press ENTER or L  —  Load saved profile and start fighting", True, (220, 220, 220)
            )
            self.screen.blit(opt1, (SCREEN_WIDTH // 2 - opt1.get_width() // 2, 320))

            opt2 = fonts["body"].render(
                "Press R  —  Recalibrate from scratch", True, (220, 220, 220)
            )
            self.screen.blit(opt2, (SCREEN_WIDTH // 2 - opt2.get_width() // 2, 360))

            pygame.display.flip()
            self.clock.tick(FPS)

    def _confirm_skip(self) -> str:
        """Show confirmation to skip calibration."""
        fonts = self.explainer._get_fonts()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "skip"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        return "skip"
                    elif event.key == pygame.K_n or event.key == pygame.K_ESCAPE:
                        return "continue"

            self.screen.fill((20, 20, 30))
            title = fonts["subtitle"].render(
                "Skip calibration? (Y/N)", True, (255, 200, 80)
            )
            self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 300))

            hint = fonts["body"].render(
                "Default thresholds will be used — detection may be less accurate.",
                True, (140, 140, 150),
            )
            self.screen.blit(hint, (SCREEN_WIDTH // 2 - hint.get_width() // 2, 360))

            pygame.display.flip()
            self.clock.tick(FPS)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        """Handle a keypress in the current phase. Returns False to exit loop."""
        if self.phase == ExplainerPhase.WELCOME:
            if key in (pygame.K_SPACE, pygame.K_RETURN):
                self.phase = ExplainerPhase.STANCE_SELECT

        elif self.phase == ExplainerPhase.STANCE_SELECT:
            if key == pygame.K_LEFT:
                self.stance = "orthodox"
            elif key == pygame.K_RIGHT:
                self.stance = "southpaw"
            elif key == pygame.K_RETURN:
                self.profile = CalibrationProfile(stance=self.stance)
                self.current_move_idx = 0
                self.current_sample = 1
                self.phase = ExplainerPhase.MOVE_EXPLAIN

        elif self.phase == ExplainerPhase.MOVE_EXPLAIN:
            if key == pygame.K_SPACE:
                # Start recording this move
                self.phase = ExplainerPhase.RECORD_PROMPT
            elif key == pygame.K_RIGHT:
                # Next move (wrap around)
                if self.current_move_idx < len(MOVE_ORDER) - 1:
                    self.current_move_idx += 1
                    self.current_sample = 1
                    self.explainer._animators[MOVE_ORDER[self.current_move_idx]].reset()
            elif key == pygame.K_LEFT:
                # Previous move
                if self.current_move_idx > 0:
                    self.current_move_idx -= 1
                    self.current_sample = 1
                    self.explainer._animators[MOVE_ORDER[self.current_move_idx]].reset()

        elif self.phase == ExplainerPhase.RECORD_PROMPT:
            if key == pygame.K_SPACE:
                # Start countdown
                self.countdown_timer = COUNTDOWN_SECONDS * FPS
                self._camera_surface = None
                self.phase = ExplainerPhase.COUNTDOWN

        elif self.phase == ExplainerPhase.PLAYBACK:
            if key == pygame.K_SPACE:
                # Skip playback, advance to record_done
                self.phase = ExplainerPhase.RECORD_DONE

        elif self.phase == ExplainerPhase.RECORD_DONE:
            if key == pygame.K_SPACE:
                if self.last_record_success:
                    self.current_sample += 1
                    move_name = MOVE_ORDER[self.current_move_idx]
                    if self.profile.has_enough_samples(move_name, SAMPLES_PER_MOVE):
                        # Move to next move type
                        self.current_move_idx += 1
                        self.current_sample = 1
                        if self.current_move_idx >= len(MOVE_ORDER):
                            self.phase = ExplainerPhase.ALL_DONE
                        else:
                            self.phase = ExplainerPhase.MOVE_EXPLAIN
                    else:
                        self.phase = ExplainerPhase.RECORD_PROMPT
                else:
                    # Retry failed recording
                    self.phase = ExplainerPhase.RECORD_PROMPT

        elif self.phase == ExplainerPhase.ALL_DONE:
            if key == pygame.K_RETURN:
                return False  # Exit calibration loop

        return True  # Keep running

    # ------------------------------------------------------------------
    # Frame helpers: BGR numpy → pygame Surface
    # ------------------------------------------------------------------

    @staticmethod
    def _bgr_to_surface(bgr_frame) -> pygame.Surface:
        """Convert a BGR numpy array to a pygame Surface."""
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

    # ------------------------------------------------------------------
    # Phase processors (called every tick)
    # ------------------------------------------------------------------

    def _process_countdown_frame(self):
        """Tick down the countdown, grab camera for live preview."""
        # Grab a camera frame for live preview during countdown
        ok, frame, _ts = self.video_source.read()
        if ok:
            self._camera_surface = self._bgr_to_surface(frame)

        self.countdown_timer -= 1
        if self.countdown_timer <= 0:
            # Start actual recording
            move_name = MOVE_ORDER[self.current_move_idx]
            hand = "left" if (
                (self.stance == "orthodox" and move_name in ("jab", "hook"))
                or (self.stance == "southpaw" and move_name in ("cross", "uppercut"))
            ) else "right"
            self.recorder.start_recording(move_name, hand)
            self.recording_frames = 0
            self._recorded_frames = []
            self.phase = ExplainerPhase.RECORDING

    def _process_recording_frame(self):
        """Capture one frame, draw keypoints, build camera surface."""
        ok, frame, ts_ms = self.video_source.read()
        if not ok:
            return

        pose = self.pose_estimator.process_frame(frame, ts_ms)
        smoothed = self.smoother.smooth(pose)
        self.recorder.add_frame(smoothed)

        # Draw keypoints onto a copy for display
        annotated = frame.copy()
        if smoothed and smoothed.keypoints:
            draw_keypoints_on_frame(annotated, smoothed.keypoints)

        # Store for playback
        self._recorded_frames.append((annotated, smoothed.keypoints if smoothed else {}))
        self._camera_surface = self._bgr_to_surface(annotated)
        self.recording_frames += 1

        if self.recording_frames >= RECORDING_FRAMES:
            template = self.recorder.finish_recording()
            if template is not None:
                self.profile.add_template(template)
                self.last_record_success = True
            else:
                self.last_record_success = False
            # Go to playback phase
            self._playback_idx = 0
            self.phase = ExplainerPhase.PLAYBACK

    def _process_playback_frame(self):
        """Step through recorded frames for playback."""
        if not self._recorded_frames:
            self.phase = ExplainerPhase.RECORD_DONE
            return

        idx = min(self._playback_idx, len(self._recorded_frames) - 1)
        bgr_frame, _kps = self._recorded_frames[idx]
        self._camera_surface = self._bgr_to_surface(bgr_frame)
        self._playback_idx += self._playback_speed

        if self._playback_idx >= len(self._recorded_frames):
            # Playback done — advance to record_done
            self.phase = ExplainerPhase.RECORD_DONE

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self):
        """Draw the current calibration screen."""
        if self.phase == ExplainerPhase.WELCOME:
            self.explainer.draw_welcome(self.screen)

        elif self.phase == ExplainerPhase.STANCE_SELECT:
            self.explainer.draw_stance_select(self.screen, self.stance)

        elif self.phase == ExplainerPhase.MOVE_EXPLAIN:
            move_name = MOVE_ORDER[self.current_move_idx]
            self.explainer.draw_move_explanation(
                self.screen, move_name,
                move_idx=self.current_move_idx,
                total_moves=len(MOVE_ORDER),
            )

        elif self.phase == ExplainerPhase.RECORD_PROMPT:
            move_name = MOVE_ORDER[self.current_move_idx]
            self.explainer.draw_record_prompt(
                self.screen, move_name, self.current_sample, SAMPLES_PER_MOVE
            )

        elif self.phase == ExplainerPhase.COUNTDOWN:
            move_name = MOVE_ORDER[self.current_move_idx]
            seconds_left = max(1, self.countdown_timer // FPS + 1)
            self.explainer.draw_countdown(
                self.screen, move_name, seconds_left,
                camera_surface=self._camera_surface,
            )

        elif self.phase == ExplainerPhase.RECORDING:
            move_name = MOVE_ORDER[self.current_move_idx]
            self.explainer.draw_recording(
                self.screen, move_name,
                self.recording_frames, RECORDING_FRAMES,
                camera_surface=self._camera_surface,
            )

        elif self.phase == ExplainerPhase.PLAYBACK:
            move_name = MOVE_ORDER[self.current_move_idx]
            total = len(self._recorded_frames) if self._recorded_frames else 1
            self.explainer.draw_playback(
                self.screen, move_name,
                sample_number=self.current_sample,
                camera_surface=self._camera_surface,
                frame_idx=min(self._playback_idx, total - 1),
                total_frames=total,
                success=self.last_record_success,
            )

        elif self.phase == ExplainerPhase.RECORD_DONE:
            move_name = MOVE_ORDER[self.current_move_idx]
            self.explainer.draw_record_done(
                self.screen, move_name,
                self.current_sample, self.last_record_success,
            )

        elif self.phase == ExplainerPhase.ALL_DONE:
            self.explainer.draw_all_done(self.screen)

        pygame.display.flip()


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------
class StickFighterGame:
    """Main game class wiring all subsystems together."""

    def __init__(
        self,
        camera_source: int | str = 0,
        calibration_profile: CalibrationProfile | None = None,
        use_ml: bool = False,
    ):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Stick Fighter")
        self.clock = pygame.time.Clock()

        # --- Core pipeline ---
        self.video_source = VideoSource(camera_source, target_fps=FPS)
        if not self.video_source.is_open:
            print(f"ERROR: Cannot open camera/video source: {camera_source}")
            sys.exit(1)

        self.pose_estimator = PoseEstimator(
            upper_body_only=True,
            running_mode="VIDEO",
        )
        self.smoother = PoseSmoother(SmoothingConfig(min_cutoff=1.0, beta=5.0))
        self.transformer = CoordinateTransformer(
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
            player_base_x=PLAYER_START_X,
            ground_y=GROUND_Y,
        )

        # Apply calibrated thresholds if available
        stance = "orthodox"
        detector_cfg = MoveDetectorConfig(stance=stance)
        if calibration_profile is not None:
            stance = calibration_profile.stance
            detector_cfg.stance = stance
            ct = calibration_profile.thresholds
            if ct is not None:
                detector_cfg.punch_z_velocity_threshold = ct.punch_z_velocity_threshold
                detector_cfg.hook_x_velocity_threshold = ct.hook_x_velocity_threshold
                detector_cfg.uppercut_y_velocity_threshold = ct.uppercut_y_velocity_threshold
                detector_cfg.punch_z_extension = ct.punch_z_extension
                detector_cfg.hook_x_extension = ct.hook_x_extension
                detector_cfg.uppercut_y_extension = ct.uppercut_y_extension

        if use_ml:
            from ml.ml_move_detector import MLMoveDetector
            self.move_detector = MLMoveDetector(
                model_dir="ml/models",
                stance=stance,
            )
        else:
            self.move_detector = MoveDetector(detector_cfg)
        self.calibration_profile = calibration_profile
        self.movement_tracker = MovementTracker(initial_x=PLAYER_START_X)

        # --- Game systems ---
        self.npc_style = FightingStyle.BOXER
        self.difficulty = Difficulty.MEDIUM
        npc_cfg = get_npc_config(self.npc_style, self.difficulty)
        self.npc = NPC(config=npc_cfg, game_x=NPC_START_X, ground_y=GROUND_Y)
        self.combat = CombatSystem(CombatConfig(fps=FPS))
        self.combo_tracker = ComboTracker(combo_window=45)
        self.effects = EffectsManager()
        self.sound = SoundManager()
        self.sound.initialize()

        # --- Renderers ---
        self.game_renderer = GameRenderer(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.combat_ui = CombatUI(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.effects_renderer = EffectsRenderer(SCREEN_WIDTH, SCREEN_HEIGHT)

        # --- State ---
        self.running = True
        self.paused = False
        self.debug_mode = False
        self.last_player_pose: GamePose | None = None
        self.last_raw_frame = None
        self.last_move = MoveType.IDLE
        self.player_hit_this_move = False
        self._prev_countdown_val = 0
        self._calibrated = False
        self._frame_count = 0

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    def run(self):
        """Main game loop."""
        while self.running:
            self._handle_events()

            if not self.paused:
                self._update()

            self._render()
            self.clock.tick(FPS)

        self._cleanup()

    # -----------------------------------------------------------------------
    # Event handling
    # -----------------------------------------------------------------------
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_m:
                    if self.sound.is_muted:
                        self.sound.unmute()
                    else:
                        self.sound.mute()
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_d:
                    self.debug_mode = not self.debug_mode
                elif event.key == pygame.K_r:
                    if self.combat.phase == GamePhase.MATCH_END:
                        self._restart_match()
                # NPC style selection (1-5)
                elif event.key == pygame.K_1:
                    self._set_npc_style(FightingStyle.BOXER)
                elif event.key == pygame.K_2:
                    self._set_npc_style(FightingStyle.BRAWLER)
                elif event.key == pygame.K_3:
                    self._set_npc_style(FightingStyle.COUNTER)
                elif event.key == pygame.K_4:
                    self._set_npc_style(FightingStyle.SPEEDSTER)
                elif event.key == pygame.K_5:
                    self._set_npc_style(FightingStyle.TANK)
                # Difficulty (F1-F4)
                elif event.key == pygame.K_F1:
                    self._set_difficulty(Difficulty.EASY)
                elif event.key == pygame.K_F2:
                    self._set_difficulty(Difficulty.MEDIUM)
                elif event.key == pygame.K_F3:
                    self._set_difficulty(Difficulty.HARD)
                elif event.key == pygame.K_F4:
                    self._set_difficulty(Difficulty.NIGHTMARE)

    # -----------------------------------------------------------------------
    # Update (one frame)
    # -----------------------------------------------------------------------
    def _update(self):
        self._frame_count += 1

        # 1. Capture camera frame
        ok, frame, ts_ms = self.video_source.read()
        if not ok:
            return
        self.last_raw_frame = frame

        # 2. Pose estimation
        pose = self.pose_estimator.process_frame(frame, ts_ms)

        # 3. Smoothing
        smoothed = self.smoother.smooth(pose)

        # 4. Auto-calibrate on first valid frame
        if not self._calibrated and smoothed.valid:
            self.transformer.calibrate(smoothed)
            self._calibrated = True

        # 5. Coordinate transformation -> player GamePose
        movement_state = self.movement_tracker.update(smoothed, facing_right=True)
        self.transformer.set_player_position(movement_state.game_x)
        player_pose = self.transformer.transform(smoothed, facing_right=True)
        self.last_player_pose = player_pose

        # 6. Move detection
        detected = self.move_detector.detect(smoothed)
        current_move = detected.move_type

        # Track whether we've already registered a hit for this move instance
        if current_move == MoveType.IDLE:
            self.player_hit_this_move = False

        # 7. Update combat system phase
        phase = self.combat.update()

        # 8. Countdown sound
        if phase == GamePhase.COUNTDOWN:
            cv = self.combat.countdown_value
            if cv != self._prev_countdown_val:
                if cv > 0:
                    self.sound.play_countdown()
                else:
                    self.sound.play_fight()
                self._prev_countdown_val = cv
        else:
            self._prev_countdown_val = 0

        # 9. NPC update
        player_is_attacking = (
            current_move != MoveType.IDLE
            and detected.phase == MovePhase.ACTIVE
        )
        self.npc.update(movement_state.game_x, player_attacking=player_is_attacking)

        # 10. Collision detection (only during FIGHTING phase)
        if phase == GamePhase.FIGHTING:
            self._process_combat(player_pose, current_move, detected)

        # 11. Combo tracker tick
        self.combo_tracker.update()

        # 12. Effects tick
        self.effects.update()

        # 13. Track move for next frame
        self.last_move = current_move

    # -----------------------------------------------------------------------
    # Combat processing
    # -----------------------------------------------------------------------
    def _process_combat(self, player_pose: GamePose, move: MoveType, detected):
        """Handle collision, damage, effects for one frame."""
        # Player attack hitbox
        player_wrist = None
        is_attacking = (
            move != MoveType.IDLE
            and detected.phase == MovePhase.ACTIVE
            and not self.player_hit_this_move
        )
        if is_attacking:
            # Use right wrist for orthodox, left for southpaw
            wrist_name = f"{detected.hand}_wrist"
            wrist_kp = player_pose.keypoints.get(wrist_name)
            if wrist_kp:
                player_wrist = (wrist_kp.game_x, wrist_kp.game_y)

        player_attack_hb = None
        if player_wrist and is_attacking:
            player_attack_hb = get_player_attack_hitbox(
                player_wrist[0], player_wrist[1], True
            )

        # Player body hitbox
        nose = player_pose.keypoints.get("nose")
        head_y = nose.game_y - 18 if nose else GROUND_Y - 220
        player_body_hb = get_player_body_hitbox(
            self.transformer.player_base_x, head_y, GROUND_Y
        )

        # NPC hitboxes
        npc_attack_hb = self.npc.get_attack_hitbox()
        npc_body_hb = self.npc.get_body_hitbox()

        # Damage from move type
        move_str = move.value if move != MoveType.IDLE else "jab"
        combat_cfg = self.combat.config
        player_dmg = combat_cfg.damage_table.get(move_str, 5)
        npc_atk = self.npc.attack_type
        npc_dmg = 0
        if npc_atk:
            npc_dmg = self.npc.config.damage.get(npc_atk.value, 5)

        collision = check_collision(
            player_attack_hitbox=player_attack_hb,
            player_body_hitbox=player_body_hb,
            npc_attack_hitbox=npc_attack_hb,
            npc_body_hitbox=npc_body_hb,
            player_damage=player_dmg,
            npc_damage=npc_dmg,
            npc_blocking=self.npc.is_blocking,
        )

        # Apply player hitting NPC
        if collision.player_hit_npc:
            actual_dmg, mult = self.combo_tracker.register_hit(
                move_str, collision.npc_damage
            )
            self.combat.apply_damage_to_npc(move_str)
            self.npc.receive_hit(actual_dmg)
            self.player_hit_this_move = True

            # Effects
            npc_pose = self.npc.get_pose()
            hit_x = (player_wrist[0] + self.npc.game_x) / 2
            hit_y = npc_pose.head[1] + 40
            self.effects.trigger_hit(hit_x, hit_y, move_str, actual_dmg)
            self.sound.play_hit(move_str)

        elif is_attacking and player_attack_hb:
            self.sound.play_whoosh()

        # Apply NPC hitting player
        if collision.npc_hit_player:
            self.combat.apply_damage_to_player(npc_atk.value if npc_atk else "jab")
            npc_pose = self.npc.get_pose()
            hit_x = self.transformer.player_base_x
            hit_y = head_y + 30
            self.effects.trigger_hit(
                hit_x, hit_y, npc_atk.value if npc_atk else "jab",
                collision.player_damage, is_player_hit=True
            )
            self.sound.play_hit(npc_atk.value if npc_atk else "jab")

        # KO sound
        if self.combat.player.is_ko or self.combat.npc.is_ko:
            self.sound.play_ko()

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------
    def _render(self):
        # Screen shake offset
        shake_x, shake_y = self.effects_renderer.apply_screen_shake(self.effects)

        # Create a buffer surface for shake
        buffer = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Background + player stick figure
        self.game_renderer.draw_background(buffer)

        # Draw ground shadow for NPC
        npc_pose = self.npc.get_pose()

        # Draw player
        if self.last_player_pose and self.last_player_pose.valid:
            self.game_renderer.player_renderer.draw_ground_shadow(
                buffer, self.last_player_pose, GROUND_Y
            )
            self.game_renderer.player_renderer.draw(buffer, self.last_player_pose)

        # Draw NPC using the same StickFigureRenderer as the player
        npc_game_pose = npc_pose.to_game_pose(facing_right=self.npc.facing_right)
        self.game_renderer.npc_renderer.draw_ground_shadow(
            buffer, npc_game_pose, GROUND_Y
        )
        self.game_renderer.npc_renderer.draw(buffer, npc_game_pose)

        # Effects (particles, damage numbers, combos)
        self.effects_renderer.draw(buffer, self.effects, self.combo_tracker)

        # Combat UI (HP bars, timer, overlays)
        self.combat_ui.draw(buffer, self.combat)

        # Camera preview (small picture-in-picture)
        if self.last_raw_frame is not None:
            self._draw_camera_preview(buffer)

        # Debug overlay
        if self.debug_mode:
            self._draw_debug(buffer)

        # Style/difficulty indicator
        self._draw_style_indicator(buffer)

        # Pause overlay
        if self.paused:
            self._draw_pause_overlay(buffer)

        # Blit buffer with shake offset
        self.screen.fill((0, 0, 0))
        self.screen.blit(buffer, (shake_x, shake_y))
        pygame.display.flip()

    def _draw_camera_preview(self, surface: pygame.Surface):
        """Draw a small camera feed in the bottom-right corner."""
        import cv2
        frame = self.last_raw_frame
        small = cv2.resize(frame, (CAMERA_PREVIEW_W, CAMERA_PREVIEW_H))
        # BGR -> RGB
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # numpy -> pygame surface
        cam_surf = pygame.surfarray.make_surface(small.swapaxes(0, 1))
        x = SCREEN_WIDTH - CAMERA_PREVIEW_W - 10
        y = SCREEN_HEIGHT - CAMERA_PREVIEW_H - 10
        # Border
        pygame.draw.rect(surface, (100, 100, 100),
                         (x - 2, y - 2, CAMERA_PREVIEW_W + 4, CAMERA_PREVIEW_H + 4), 2)
        surface.blit(cam_surf, (x, y))
        # Label
        font = pygame.font.SysFont("monospace", 12)
        label = font.render("CAMERA", True, (180, 180, 180))
        surface.blit(label, (x, y - 14))

    def _draw_debug(self, surface: pygame.Surface):
        """Draw debug information."""
        font = pygame.font.SysFont("monospace", 14)
        fps = self.clock.get_fps()
        move = self.last_move.value
        ms = self.movement_tracker.state
        npc_state = self.npc.state.value
        phase = self.combat.phase.value

        lines = [
            f"FPS: {fps:.0f}",
            f"Move: {move}",
            f"Player X: {ms.game_x:.0f}  Walk: {ms.is_walking}",
            f"NPC X: {self.npc.game_x:.0f}  State: {npc_state}",
            f"Phase: {phase}  Round: {self.combat.current_round}",
            f"P HP: {self.combat.player.current_hp}  N HP: {self.combat.npc.current_hp}",
            f"Combo: {self.combo_tracker.state.count}",
            f"Frame: {self._frame_count}",
        ]
        y = 100
        for line in lines:
            text = font.render(line, True, (0, 255, 0))
            bg = pygame.Surface((text.get_width() + 4, text.get_height() + 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            surface.blit(bg, (8, y - 1))
            surface.blit(text, (10, y))
            y += 16

    def _draw_style_indicator(self, surface: pygame.Surface):
        """Draw current NPC style and difficulty in bottom-left."""
        font = pygame.font.SysFont("monospace", 14)
        profile = STYLE_PROFILES[self.npc_style]
        text = f"{profile.name} | {self.difficulty.value.upper()}"
        text_surf = font.render(text, True, profile.color)
        surface.blit(text_surf, (10, SCREEN_HEIGHT - 24))

    def _draw_pause_overlay(self, surface: pygame.Surface):
        """Draw pause screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        surface.blit(overlay, (0, 0))

        font = pygame.font.SysFont("monospace", 72, bold=True)
        text = font.render("PAUSED", True, (255, 255, 255))
        x = SCREEN_WIDTH // 2 - text.get_width() // 2
        y = SCREEN_HEIGHT // 2 - text.get_height() // 2
        surface.blit(text, (x, y))

        hint_font = pygame.font.SysFont("monospace", 20)
        hint = hint_font.render("Press P to resume", True, (180, 180, 180))
        hx = SCREEN_WIDTH // 2 - hint.get_width() // 2
        surface.blit(hint, (hx, y + 80))

    # -----------------------------------------------------------------------
    # Style / difficulty changes
    # -----------------------------------------------------------------------
    def _set_npc_style(self, style: FightingStyle):
        if style == self.npc_style:
            return
        self.npc_style = style
        self._rebuild_npc()

    def _set_difficulty(self, diff: Difficulty):
        if diff == self.difficulty:
            return
        self.difficulty = diff
        self._rebuild_npc()

    def _rebuild_npc(self):
        cfg = get_npc_config(self.npc_style, self.difficulty)
        hp = get_npc_hp(self.npc_style)
        self.npc = NPC(config=cfg, game_x=self.npc.game_x, ground_y=GROUND_Y)
        self.combat.npc.max_hp = hp
        self.combat.npc.current_hp = hp

    def _restart_match(self):
        self.combat.reset_match()
        self.npc.reset(NPC_START_X)
        self.combo_tracker.reset()
        self.effects.clear()
        self.movement_tracker.reset(PLAYER_START_X)
        self.transformer.set_player_position(PLAYER_START_X)
        self.move_detector.reset()
        self.player_hit_this_move = False
        self._prev_countdown_val = 0
        hp = get_npc_hp(self.npc_style)
        self.combat.npc.max_hp = hp
        self.combat.npc.current_hp = hp
        self.sound.play_round_bell()

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    def _cleanup(self):
        self.pose_estimator.close()
        self.video_source.close()
        pygame.quit()


# ---------------------------------------------------------------------------
# Dojo mode — free practice with real-time detection feedback
# ---------------------------------------------------------------------------
class DojoMode:
    """
    Practice mode with optional voice-labeled training.

    Without voice:  free practice with real-time detection feedback.
    With voice (V): shout move names to label training data, see
                    detection vs ground truth stats, and finetune the
                    ML model on collected samples.

    Controls:
      V          - toggle voice labeling on/off
      ESC        - show session summary (finetune / save / go to game)
      SPACE / R  - reset stats
      D          - toggle debug overlay
    """

    # Move colors shared across drawing methods
    MOVE_COLORS = {
        "jab": (100, 200, 255), "cross": (255, 100, 100),
        "hook": (255, 200, 50), "uppercut": (100, 255, 100),
        "walking": (180, 180, 180), "idle": (120, 120, 120),
    }

    # Half-window for training segment extraction (±1.5s at 30fps)
    SEGMENT_HALF_WINDOW = 45

    def __init__(
        self,
        camera_source: int | str,
        calibration_profile: CalibrationProfile | None = None,
        use_ml: bool = False,
    ):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Stick Fighter — DOJO")
        self.clock = pygame.time.Clock()

        # Core pipeline
        self.video_source = VideoSource(camera_source, target_fps=FPS)
        if not self.video_source.is_open:
            print(f"ERROR: Cannot open camera/video source: {camera_source}")
            sys.exit(1)

        self.pose_estimator = PoseEstimator(upper_body_only=True, running_mode="VIDEO")
        # Two smoothers: one for detection (responsive), one for rendering (stable).
        # High beta (5.0) preserves fast punches for detection.
        # Low beta (0.5) keeps the stick figure stable when idle.
        self.smoother = PoseSmoother(SmoothingConfig(min_cutoff=1.0, beta=5.0))
        self.render_smoother = PoseSmoother(SmoothingConfig(min_cutoff=1.5, beta=0.5))
        self.transformer = CoordinateTransformer(
            screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
            player_base_x=SCREEN_WIDTH // 2, ground_y=GROUND_Y,
        )

        stance = "orthodox"
        detector_cfg = MoveDetectorConfig(stance=stance)
        if calibration_profile is not None:
            stance = calibration_profile.stance
            detector_cfg.stance = stance
            ct = calibration_profile.thresholds
            if ct is not None:
                detector_cfg.punch_z_velocity_threshold = ct.punch_z_velocity_threshold
                detector_cfg.hook_x_velocity_threshold = ct.hook_x_velocity_threshold
                detector_cfg.uppercut_y_velocity_threshold = ct.uppercut_y_velocity_threshold
                detector_cfg.punch_z_extension = ct.punch_z_extension
                detector_cfg.hook_x_extension = ct.hook_x_extension
                detector_cfg.uppercut_y_extension = ct.uppercut_y_extension

        self.use_ml = use_ml
        if use_ml:
            from ml.ml_move_detector import MLMoveDetector
            self.move_detector = MLMoveDetector(
                model_dir="ml/models",
                stance=stance,
            )
        else:
            self.move_detector = MoveDetector(detector_cfg)
        self.movement_tracker = MovementTracker(initial_x=SCREEN_WIDTH // 2)
        self.renderer = GameRenderer(SCREEN_WIDTH, SCREEN_HEIGHT)

        # State
        self.running = True
        self.debug_mode = False
        self._calibrated = False
        self._frame_count = 0
        self.last_player_pose: GamePose | None = None
        self.last_raw_frame = None
        self._last_smoothed_kps: dict = {}
        self._start_time = time.time()
        self._result = "quit"  # "quit" or "game" — what to do after dojo

        # Detection tracking
        self.move_counts: dict[str, int] = {
            "jab": 0, "cross": 0, "hook": 0, "uppercut": 0,
        }
        self.move_log: list[tuple[float, str, str]] = []
        self.current_move = MoveType.IDLE
        self.current_hand = "right"
        self.move_flash_timer = 0
        self.move_flash_name = ""

        # Velocity tracking for bars
        self._vel_z = 0.0
        self._vel_x = 0.0
        self._vel_y = 0.0

        # ── Combo + effects (bag-work: every clean punch chains) ─────────────
        self.combo_tracker = ComboTracker(combo_window=45)
        self.effects = EffectsManager()
        self.effects_renderer = EffectsRenderer(SCREEN_WIDTH, SCREEN_HEIGHT)

        # Global gap: after ANY counted move, suppress all new moves for N frames.
        # This is the safety net for cross-type retraction ghosts
        # (jab extend -> uppercut classified from retraction arc).
        # 18 frames = 0.6s — below real combo speed but above retraction timing.
        self._GLOBAL_GAP = 18
        self._global_gap_counter = 0   # counts down from _GLOBAL_GAP after each hit

        # ---- Voice labeling (toggled with V key) ----
        self._voice_active = False
        self._voice_available = False
        self._voice_error = ""
        self._voice = None  # VoiceRecognizer instance (lazy init)

        # Voice GT tracking
        self.voice_gt_counts: dict[str, int] = {
            m: 0 for m in ["jab", "cross", "hook", "uppercut"]
        }
        self.label_events: list[dict] = []
        self.voice_flash_timer = 0
        self.voice_flash_name = ""

        # Frame buffer for training segment extraction
        self._frame_buffer: list[dict] = []
        self._max_buffer = FPS * 120  # keep last 2 minutes

        # Summary screen state
        self._show_summary = False
        self._training_status = ""

    # ------------------------------------------------------------------
    # Voice toggle
    # ------------------------------------------------------------------
    def _toggle_voice(self):
        if self._voice_active:
            # Turn off
            self._voice_active = False
            if self._voice:
                self._voice.stop()
            print("[DOJO] Voice labeling DISABLED")
            return

        # Turn on — lazy-init recognizer
        if self._voice is None:
            try:
                from core.voice_recognizer import VoiceRecognizer
                self._voice = VoiceRecognizer()
            except ImportError:
                self._voice_error = "speech_recognition not installed (pip install SpeechRecognition pyaudio)"
                print(f"[DOJO] {self._voice_error}")
                return

        print("[DOJO] Starting voice recognizer...")
        ok = self._voice.start()
        if ok:
            self._voice_active = True
            self._voice_available = True
            self._voice_error = ""
            print("[DOJO] Voice labeling ENABLED — shout JAB, CROSS, HOOK, UPPERCUT!")
        else:
            self._voice_error = self._voice.error or "Microphone not available"
            self._voice_active = False
            print(f"[DOJO] Voice start FAILED: {self._voice_error}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> str:
        """Run dojo. Returns 'game' if user wants to proceed to game, else 'quit'."""
        while self.running:
            if self._show_summary:
                self._handle_summary_events()
                self._render_summary()
            else:
                self._handle_events()
                self._update()
                self._render()
            self.clock.tick(FPS)
        self._cleanup()
        return self._result

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._show_summary = True
                elif event.key in (pygame.K_SPACE, pygame.K_r):
                    self._reset_stats()
                elif event.key == pygame.K_d:
                    self.debug_mode = not self.debug_mode
                elif event.key == pygame.K_v:
                    self._toggle_voice()

    def _handle_summary_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_RETURN:
                    self._show_summary = False
                elif event.key == pygame.K_g:
                    self._result = "game"
                    self.running = False
                elif event.key == pygame.K_t:
                    self._finetune_model()
                elif event.key == pygame.K_s:
                    self._save_training_data()

    def _reset_stats(self):
        self.move_counts = {k: 0 for k in self.move_counts}
        self.voice_gt_counts = {k: 0 for k in self.voice_gt_counts}
        self.move_log.clear()
        self.label_events.clear()
        self._frame_buffer.clear()
        self._start_time = time.time()
        self._frame_count = 0
        self.combo_tracker.reset()
        self.effects.clear()
        self._global_gap_counter = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _update(self):
        self._frame_count += 1
        ok, frame, ts_ms = self.video_source.read()
        if not ok:
            return
        self.last_raw_frame = frame

        pose = self.pose_estimator.process_frame(frame, ts_ms)
        smoothed = self.smoother.smooth(pose)

        if not self._calibrated and smoothed.valid:
            self.transformer.calibrate(smoothed)
            self._calibrated = True

        movement_state = self.movement_tracker.update(smoothed, facing_right=True)
        # Dojo: character stays at fixed centre. Movement tracker is updated
        # so walking state is tracked, but we never call set_player_position.
        # Use a separate render smoother (lower beta) so the figure is stable at idle.
        render_smoothed = self.render_smoother.smooth(pose)
        player_pose = self.transformer.transform(render_smoothed, facing_right=True)
        self.last_player_pose = player_pose
        self._last_smoothed_kps = smoothed.keypoints if smoothed and smoothed.keypoints else {}

        # Store frame for voice-label segment extraction
        elapsed = time.time() - self._start_time
        if self._voice_active:
            self._frame_buffer.append({
                "frame_idx": self._frame_count,
                "timestamp_s": elapsed,
                "pose_frame": smoothed,
            })
            if len(self._frame_buffer) > self._max_buffer:
                self._frame_buffer = self._frame_buffer[-self._max_buffer:]

        detected = self.move_detector.detect(smoothed)
        self.current_move = detected.move_type
        self.current_hand = detected.hand

        # Track velocities for display
        if hasattr(self.move_detector, '_velocities'):
            v = self.move_detector._velocities
            self._vel_z = abs(v.get('wrist_z', 0.0))
            self._vel_x = abs(v.get('wrist_x', 0.0))
            self._vel_y = abs(v.get('wrist_y', 0.0))

        # Tick combo window + effects every frame regardless of detection
        self.combo_tracker.update()
        self.effects.update()

        # Decrement global gap (cross-type retraction ghost suppressor)
        if self._global_gap_counter > 0:
            self._global_gap_counter -= 1

        # Register new move
        if detected.move_type != MoveType.IDLE and detected.phase == MovePhase.ACTIVE:
            move_name = detected.move_type.value
            # Gate 1: same-type flash guard (prevents holding same ACTIVE from counting twice)
            new_type = (self.move_flash_name != move_name or self.move_flash_timer <= 0)
            # Gate 2: global gap guard (suppresses cross-type retraction ghosts)
            past_gap = (self._global_gap_counter == 0)

            if new_type and past_gap:
                self.move_counts[move_name] = self.move_counts.get(move_name, 0) + 1
                self.move_log.append((elapsed, move_name, detected.hand))
                self.move_flash_name = move_name
                self.move_flash_timer = 25
                self._global_gap_counter = self._GLOBAL_GAP

                # Register with combo tracker (bag-work: every clean punch chains)
                base_damage = {"jab": 5, "cross": 8, "hook": 12, "uppercut": 15}.get(move_name, 5)
                actual_dmg, mult = self.combo_tracker.register_hit(move_name, base_damage)

                # Trigger effects at fist position
                hit_x, hit_y = SCREEN_WIDTH // 2, GROUND_Y - 120
                if self.last_player_pose and self.last_player_pose.valid:
                    wrist_key = f"{detected.hand}_wrist"
                    wkp = self.last_player_pose.keypoints.get(wrist_key)
                    if wkp:
                        hit_x, hit_y = int(wkp.game_x), int(wkp.game_y)
                self.effects.trigger_hit(hit_x, hit_y, move_name, actual_dmg)

        if self.move_flash_timer > 0:
            self.move_flash_timer -= 1

        # Process voice labels
        if self._voice_active and self._voice:
            for vl in self._voice.get_labels():
                self._process_voice_label(vl)

    def _process_voice_label(self, vl):
        """Handle a newly recognized voice label."""
        move_label = vl.label
        if move_label in self.voice_gt_counts:
            self.voice_gt_counts[move_label] += 1

        self.voice_flash_name = move_label
        self.voice_flash_timer = 30

        elapsed = time.time() - self._start_time

        # Find the most recent detected move within a ±3 second window.
        # Voice recognition has 1-3s latency, so the move_flash_timer
        # (25 frames = 0.83s) is almost always expired by the time the
        # voice label arrives. Instead, look at the move_log.
        detected_label = "idle"
        window_s = 3.0  # seconds to look back
        for t, name, _hand in reversed(self.move_log):
            if elapsed - t <= window_s:
                detected_label = name
                break
            if elapsed - t > window_s:
                break

        print(f"[VOICE] Label '{move_label}' | Nearest detection: '{detected_label}' | "
              f"Match: {move_label == detected_label}")

        self.label_events.append({
            "time_s": elapsed,
            "frame_idx": self._frame_count,
            "voice_label": move_label,
            "detected_label": detected_label,
            "match": move_label == detected_label,
        })

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self):
        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.renderer.draw_background(surface)

        # Draw player
        if self.last_player_pose and self.last_player_pose.valid:
            self.renderer.player_renderer.draw_ground_shadow(
                surface, self.last_player_pose, GROUND_Y
            )
            self.renderer.player_renderer.draw(surface, self.last_player_pose)

        # Camera preview with keypoints (top-right)
        if self.last_raw_frame is not None:
            self._draw_camera_preview(surface)

        if self._voice_active:
            # Voice mode: dual flash + accuracy panel + label log
            self._draw_dual_flash(surface)
            self._draw_accuracy_panel(surface)
            self._draw_label_log(surface)
        else:
            # Basic mode: big flash + stats + move log
            self._draw_move_flash(surface)
            self._draw_stats_panel(surface)
            self._draw_move_log(surface)

        # Combo overlay + hit effects (both voice and basic modes)
        self.effects_renderer.draw(surface, self.effects, self.combo_tracker)

        # Velocity bars (bottom-center)
        self._draw_velocity_bars(surface)

        # Title bar
        self._draw_title(surface)

        # Voice status indicator
        self._draw_voice_status(surface)

        # Debug
        if self.debug_mode:
            self._draw_debug_info(surface)

        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def _draw_camera_preview(self, surface: pygame.Surface):
        import cv2
        frame = self.last_raw_frame
        small = cv2.resize(frame, (CAMERA_PREVIEW_W, CAMERA_PREVIEW_H))
        if self._last_smoothed_kps:
            draw_keypoints_on_frame(small, self._last_smoothed_kps)
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        cam_surf = pygame.surfarray.make_surface(small_rgb.swapaxes(0, 1))
        x = SCREEN_WIDTH - CAMERA_PREVIEW_W - 10
        y = 50
        pygame.draw.rect(surface, (100, 100, 100),
                         (x - 2, y - 2, CAMERA_PREVIEW_W + 4, CAMERA_PREVIEW_H + 4), 2)
        surface.blit(cam_surf, (x, y))
        font = pygame.font.SysFont("monospace", 12)
        label = font.render("CAMERA + KEYPOINTS", True, (180, 180, 180))
        surface.blit(label, (x, y - 14))

    def _draw_move_flash(self, surface: pygame.Surface):
        if self.move_flash_timer <= 0:
            return
        color = self.MOVE_COLORS.get(self.move_flash_name, (255, 255, 255))
        alpha = min(255, self.move_flash_timer * 12)

        font = pygame.font.SysFont("monospace", 80, bold=True)
        text = font.render(self.move_flash_name.upper(), True, color)
        bg = pygame.Surface((text.get_width() + 40, text.get_height() + 20), pygame.SRCALPHA)
        bg.fill((0, 0, 0, min(180, alpha)))
        cx = SCREEN_WIDTH // 2 - bg.get_width() // 2
        cy = 160
        surface.blit(bg, (cx, cy))
        surface.blit(text, (cx + 20, cy + 10))

        hand_font = pygame.font.SysFont("monospace", 24)
        hand_text = hand_font.render(f"({self.current_hand} hand)", True, (180, 180, 180))
        surface.blit(hand_text, (SCREEN_WIDTH // 2 - hand_text.get_width() // 2, cy + 100))

    def _draw_dual_flash(self, surface: pygame.Surface):
        """Show both detected move and voice label side by side."""
        cx = SCREEN_WIDTH // 2
        cy = 140

        # Detection flash (left)
        if self.move_flash_timer > 0:
            color = self.MOVE_COLORS.get(self.move_flash_name, (255, 255, 255))
            alpha = min(255, self.move_flash_timer * 12)
            font = pygame.font.SysFont("monospace", 48, bold=True)
            text = font.render(self.move_flash_name.upper(), True, color)
            bg = pygame.Surface((text.get_width() + 20, text.get_height() + 10), pygame.SRCALPHA)
            bg.fill((0, 0, 0, min(180, alpha)))
            surface.blit(bg, (cx - bg.get_width() - 10, cy))
            surface.blit(text, (cx - text.get_width() - 0, cy + 5))
            lf = pygame.font.SysFont("monospace", 16)
            surface.blit(lf.render("DETECTED", True, (180, 180, 180)),
                         (cx - 120, cy - 18))

        # Voice label flash (right)
        if self.voice_flash_timer > 0:
            self.voice_flash_timer -= 1
            color = self.MOVE_COLORS.get(self.voice_flash_name, (255, 255, 255))
            alpha = min(255, self.voice_flash_timer * 10)
            font = pygame.font.SysFont("monospace", 48, bold=True)
            text = font.render(self.voice_flash_name.upper(), True, color)
            bg = pygame.Surface((text.get_width() + 20, text.get_height() + 10), pygame.SRCALPHA)
            bg.fill((0, 40, 0, min(180, alpha)))
            surface.blit(bg, (cx + 10, cy))
            surface.blit(text, (cx + 20, cy + 5))
            lf = pygame.font.SysFont("monospace", 16)
            surface.blit(lf.render("VOICE (GT)", True, (100, 255, 100)),
                         (cx + 10, cy - 18))

        # Match indicator
        if self.move_flash_timer > 0 and self.voice_flash_timer > 0:
            match = self.move_flash_name == self.voice_flash_name
            ic = (0, 255, 0) if match else (255, 60, 60)
            it = "MATCH" if match else "MISMATCH"
            ind = pygame.font.SysFont("monospace", 20, bold=True).render(it, True, ic)
            surface.blit(ind, (cx - ind.get_width() // 2, cy + 65))

    def _draw_accuracy_panel(self, surface: pygame.Surface):
        """Show detection vs ground truth accuracy stats."""
        x, y = 15, 50
        ft = pygame.font.SysFont("monospace", 20, bold=True)
        fb = pygame.font.SysFont("monospace", 16)
        fs = pygame.font.SysFont("monospace", 13)

        panel = pygame.Surface((280, 340), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 170))
        surface.blit(panel, (x, y))

        surface.blit(ft.render("DETECTION vs VOICE (GT)", True, (255, 220, 80)), (x + 8, y + 6))

        row_y = y + 32
        surface.blit(fs.render("MOVE       DET   GT   ACC", True, (160, 160, 170)), (x + 8, row_y))
        row_y += 20

        total_correct = 0
        total_gt = 0
        for mn in ["jab", "cross", "hook", "uppercut"]:
            color = self.MOVE_COLORS.get(mn, (200, 200, 200))
            det = self.move_counts.get(mn, 0)
            gt = self.voice_gt_counts.get(mn, 0)
            move_events = [e for e in self.label_events if e["voice_label"] == mn]
            correct = sum(1 for e in move_events if e["match"])
            total_correct += correct
            total_gt += gt
            acc_str = f"{correct}/{gt}" if gt > 0 else "-"

            pygame.draw.circle(surface, color, (x + 16, row_y + 8), 5)
            surface.blit(fb.render(f"{mn:<10s} {det:>3d}   {gt:>3d}   {acc_str}", True, (220, 220, 220)),
                         (x + 28, row_y))

            if gt > 0:
                bx = x + 250
                fill = int(min(correct / gt, 1.0) * 20)
                bc = (0, 200, 0) if correct == gt else (255, 165, 0)
                pygame.draw.rect(surface, (40, 40, 50), (bx, row_y + 2, 20, 12))
                pygame.draw.rect(surface, bc, (bx, row_y + 2, fill, 12))
            row_y += 28

        row_y += 8
        pygame.draw.line(surface, (80, 80, 90), (x + 8, row_y), (x + 270, row_y))
        row_y += 8
        overall_acc = (
            f"{total_correct}/{total_gt} ({100*total_correct/total_gt:.0f}%)"
            if total_gt > 0 else "\u2014"
        )
        surface.blit(fb.render(f"ACCURACY: {overall_acc}", True, (255, 255, 255)), (x + 8, row_y))
        row_y += 24
        surface.blit(fs.render(f"Total detected: {sum(self.move_counts.values())}", True, (150, 150, 160)),
                     (x + 8, row_y))
        row_y += 18
        surface.blit(fs.render(f"Total voice GT: {total_gt}", True, (150, 150, 160)),
                     (x + 8, row_y))

    def _draw_label_log(self, surface: pygame.Surface):
        """Show recent voice label events with match/mismatch indicators."""
        x = SCREEN_WIDTH - 300
        y = 250
        ft = pygame.font.SysFont("monospace", 16, bold=True)
        fb = pygame.font.SysFont("monospace", 13)

        panel = pygame.Surface((290, 320), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        surface.blit(panel, (x, y))

        surface.blit(ft.render("VOICE LABEL LOG", True, (255, 220, 80)), (x + 8, y + 5))
        surface.blit(fb.render("TIME   VOICE     DETECT    OK?", True, (140, 140, 150)),
                     (x + 8, y + 24))

        recent = self.label_events[-16:]
        row_y = y + 40
        for ev in recent:
            mins = int(ev["time_s"]) // 60
            secs = ev["time_s"] - mins * 60
            ts = f"{mins:01d}:{secs:04.1f}"
            vc = self.MOVE_COLORS.get(ev["voice_label"], (200, 200, 200))
            dc = self.MOVE_COLORS.get(ev["detected_label"], (200, 200, 200))
            ms = "YES" if ev["match"] else "NO"
            mc = (0, 220, 0) if ev["match"] else (255, 60, 60)

            surface.blit(fb.render(ts, True, (160, 160, 170)), (x + 8, row_y))
            surface.blit(fb.render(ev["voice_label"][:9], True, vc), (x + 68, row_y))
            dl = ev["detected_label"][:9] if ev["detected_label"] else "\u2014"
            surface.blit(fb.render(dl, True, dc), (x + 152, row_y))
            surface.blit(fb.render(ms, True, mc), (x + 245, row_y))
            row_y += 17

    def _draw_stats_panel(self, surface: pygame.Surface):
        x, y = 15, 50
        font_title = pygame.font.SysFont("monospace", 22, bold=True)
        font_body = pygame.font.SysFont("monospace", 18)
        font_small = pygame.font.SysFont("monospace", 14)

        panel = pygame.Surface((220, 260), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        surface.blit(panel, (x, y))

        title = font_title.render("DETECTION STATS", True, (255, 220, 80))
        surface.blit(title, (x + 10, y + 8))

        total = sum(self.move_counts.values())
        row_y = y + 40
        for move_name in ["jab", "cross", "hook", "uppercut"]:
            count = self.move_counts[move_name]
            color = self.MOVE_COLORS[move_name]
            pygame.draw.circle(surface, color, (x + 20, row_y + 9), 6)
            label = font_body.render(f"{move_name.upper()}: {count}", True, (220, 220, 220))
            surface.blit(label, (x + 32, row_y))
            bar_w = int(min(count * 12, 100))
            pygame.draw.rect(surface, color, (x + 170, row_y + 3, bar_w, 14))
            row_y += 35

        row_y += 10
        surface.blit(font_body.render(f"TOTAL: {total}", True, (255, 255, 255)), (x + 10, row_y))
        elapsed = time.time() - self._start_time
        rate = total / elapsed if elapsed > 1 else 0
        surface.blit(font_small.render(f"{rate:.1f} moves/sec", True, (150, 150, 150)),
                     (x + 10, row_y + 25))

    def _draw_move_log(self, surface: pygame.Surface):
        x = SCREEN_WIDTH - 260
        y = 250
        font_title = pygame.font.SysFont("monospace", 18, bold=True)
        font_body = pygame.font.SysFont("monospace", 14)

        panel = pygame.Surface((250, 320), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        surface.blit(panel, (x, y))

        surface.blit(font_title.render("MOVE LOG", True, (255, 220, 80)), (x + 10, y + 5))

        recent = self.move_log[-18:]
        row_y = y + 28
        for elapsed, move_name, hand in recent:
            color = self.MOVE_COLORS.get(move_name, (200, 200, 200))
            mins = int(elapsed) // 60
            secs = elapsed - mins * 60
            ts = f"{mins:01d}:{secs:05.2f}"
            line = f"{ts}  {move_name:<10s} ({hand})"
            surface.blit(font_body.render(line, True, color), (x + 8, row_y))
            row_y += 16

    def _draw_velocity_bars(self, surface: pygame.Surface):
        bar_x = SCREEN_WIDTH // 2 - 150
        bar_y = SCREEN_HEIGHT - 80
        bar_w = 300
        bar_h = 16
        font = pygame.font.SysFont("monospace", 12)

        labels = [
            ("Z (forward)", self._vel_z, 0.15, (100, 200, 255)),
            ("X (lateral)", self._vel_x, 0.15, (255, 200, 50)),
            ("Y (vertical)", self._vel_y, 0.15, (100, 255, 100)),
        ]
        for i, (label, val, max_val, color) in enumerate(labels):
            y = bar_y + i * 22
            pygame.draw.rect(surface, (40, 40, 50), (bar_x, y, bar_w, bar_h))
            fill_w = int(min(val / max_val, 1.0) * bar_w)
            pygame.draw.rect(surface, color, (bar_x, y, fill_w, bar_h))
            pygame.draw.rect(surface, (80, 80, 90), (bar_x, y, bar_w, bar_h), 1)
            lbl = font.render(f"{label}: {val:.4f}", True, (180, 180, 180))
            surface.blit(lbl, (bar_x - lbl.get_width() - 8, y))

    def _draw_title(self, surface: pygame.Surface):
        font = pygame.font.SysFont("monospace", 28, bold=True)
        if self._voice_active:
            title_text = "DOJO — VOICE TRAINING"
            hint_text = (
                'Shout "JAB" "CROSS" "HOOK" "UPPERCUT" | '
                "V=voice off | SPACE=reset | D=debug | ESC=finish"
            )
        else:
            title_text = "DOJO — FREE PRACTICE"
            hint_text = (
                "Throw combos freely | V=voice on | SPACE=reset | D=debug | ESC=finish"
            )
        title = font.render(title_text, True, (255, 220, 80))
        surface.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 8))

        hint_font = pygame.font.SysFont("monospace", 14)
        hint = hint_font.render(hint_text, True, (120, 120, 130))
        surface.blit(hint, (SCREEN_WIDTH // 2 - hint.get_width() // 2, 38))

    def _draw_voice_status(self, surface: pygame.Surface):
        """Show microphone status indicator with live diagnostics."""
        font = pygame.font.SysFont("monospace", 14)
        x = SCREEN_WIDTH - 260
        y = SCREEN_HEIGHT - 45

        if self._voice_active and self._voice:
            # Check for errors that occurred during the session
            if self._voice.error:
                self._voice_error = self._voice.error

            # Show live stats
            pygame.draw.circle(surface, (0, 200, 0), (x, y + 6), 5)
            labels_n = len(self.label_events)
            heard = self._voice._recognize_count if self._voice else 0
            status = f"MIC ON | Labels:{labels_n} Heard:{heard} (V=off)"
            surface.blit(font.render(status, True, (0, 200, 0)), (x + 10, y))

            # Show last heard text or error on second line
            y2 = y + 16
            if self._voice._last_raw_text:
                last = self._voice._last_raw_text[:35]
                surface.blit(font.render(f'Last: "{last}"', True, (150, 255, 150)), (x + 10, y2))
            elif self._voice._unknown_count > 0:
                surface.blit(font.render(
                    f"({self._voice._unknown_count}x could not understand)",
                    True, (255, 200, 100)), (x + 10, y2))
        elif self._voice_error:
            pygame.draw.circle(surface, (255, 60, 60), (x, y + 6), 5)
            surface.blit(font.render(f"MIC ERROR: {self._voice_error[:30]}", True, (255, 60, 60)),
                         (x + 10, y))
        else:
            pygame.draw.circle(surface, (100, 100, 100), (x, y + 6), 5)
            surface.blit(font.render("V = enable voice labeling", True, (100, 100, 100)), (x + 10, y))

    def _draw_debug_info(self, surface: pygame.Surface):
        font = pygame.font.SysFont("monospace", 14)
        fps = self.clock.get_fps()
        combo = self.combo_tracker.state
        lines = [
            f"FPS: {fps:.0f}",
            f"Frame: {self._frame_count}",
            f"Move: {self.current_move.value}",
            f"Hand: {self.current_hand}",
            f"Vel Z: {self._vel_z:.4f}  X: {self._vel_x:.4f}  Y: {self._vel_y:.4f}",
            f"Combo: {combo.count} hits  x{combo.multiplier:.1f}  gap_left:{self._global_gap_counter}",
        ]
        if self._voice_active:
            lines.append(f"Voice labels: {len(self.label_events)}")
            lines.append(f"Buffer frames: {len(self._frame_buffer)}")
        y = SCREEN_HEIGHT - 180
        for line in lines:
            text = font.render(line, True, (0, 255, 0))
            bg = pygame.Surface((text.get_width() + 4, text.get_height() + 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            surface.blit(bg, (8, y - 1))
            surface.blit(text, (10, y))
            y += 16

    # ------------------------------------------------------------------
    # Summary screen
    # ------------------------------------------------------------------
    def _render_summary(self):
        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        surface.fill((20, 20, 30))

        ft = pygame.font.SysFont("monospace", 32, bold=True)
        fs = pygame.font.SysFont("monospace", 22, bold=True)
        fb = pygame.font.SysFont("monospace", 18)

        y = 30
        surface.blit(ft.render("DOJO SESSION SUMMARY", True, (255, 220, 80)),
                     (SCREEN_WIDTH // 2 - 220, y))

        y += 60
        elapsed = time.time() - self._start_time
        total_det = sum(self.move_counts.values())
        surface.blit(fb.render(
            f"Session: {elapsed:.0f}s | Moves detected: {total_det} | "
            f"Voice labels: {len(self.label_events)}",
            True, (180, 180, 190)), (80, y))

        # Per-move table
        y += 50
        has_voice = len(self.label_events) > 0
        if has_voice:
            surface.blit(fs.render(
                "MOVE       DETECTED  VOICE(GT)  MATCHED  ACC", True, (200, 200, 210)), (80, y))
        else:
            surface.blit(fs.render(
                "MOVE       DETECTED", True, (200, 200, 210)), (80, y))
        y += 30

        total_correct = 0
        total_gt = 0
        for mn in ["jab", "cross", "hook", "uppercut"]:
            color = self.MOVE_COLORS.get(mn, (200, 200, 200))
            det = self.move_counts.get(mn, 0)
            if has_voice:
                gt = self.voice_gt_counts.get(mn, 0)
                move_events = [e for e in self.label_events if e["voice_label"] == mn]
                correct = sum(1 for e in move_events if e["match"])
                total_correct += correct
                total_gt += gt
                acc_pct = f"{100*correct/gt:.0f}%" if gt > 0 else "\u2014"
                line = f"{mn:<10s}  {det:>5d}     {gt:>5d}      {correct:>3d}    {acc_pct:>4s}"
            else:
                line = f"{mn:<10s}  {det:>5d}"
            surface.blit(fb.render(line, True, color), (80, y))
            y += 28

        # Overall
        y += 10
        if has_voice:
            overall_pct = f"{100*total_correct/total_gt:.0f}%" if total_gt > 0 else "\u2014"
            surface.blit(fs.render(
                f"OVERALL: {total_correct}/{total_gt} ({overall_pct})",
                True, (255, 255, 255)), (80, y))
            y += 30
            n_seg = len([
                e for e in self.label_events
                if e["voice_label"] in ("jab", "cross", "hook", "uppercut")
            ])
            surface.blit(fb.render(
                f"Training segments: {n_seg} (from {len(self._frame_buffer)} buffered frames)",
                True, (180, 180, 190)), (80, y))
        else:
            surface.blit(fs.render(
                f"TOTAL: {total_det} moves",
                True, (255, 255, 255)), (80, y))

        # Actions
        y += 60
        actions = [
            ("ENTER", "Go back to Dojo", (255, 220, 80)),
            ("G", "Go to Game (fight NPC)", (100, 255, 100)),
        ]
        if has_voice:
            actions.insert(0, ("T", "Finetune model on collected data", (100, 255, 100)))
            actions.insert(1, ("S", "Save training data (without finetuning)", (100, 200, 255)))
        actions.append(("ESC", "Quit", (180, 180, 180)))

        for key, desc, color in actions:
            kt = fb.render(f"[{key}]", True, color)
            dt = fb.render(f" {desc}", True, (200, 200, 210))
            surface.blit(kt, (80, y))
            surface.blit(dt, (80 + kt.get_width(), y))
            y += 28

        if self._training_status:
            y += 20
            surface.blit(fb.render(self._training_status, True, (100, 255, 100)), (80, y))

        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Training data extraction, save, and finetune
    # ------------------------------------------------------------------
    def _extract_training_segments(self) -> dict:
        segments = []
        punch_events = [
            e for e in self.label_events
            if e["voice_label"] in ("jab", "cross", "hook", "uppercut")
        ]
        if not self._frame_buffer:
            return {"segments": [], "summary": {}}

        buf_start = self._frame_buffer[0]["frame_idx"]
        buf_end = self._frame_buffer[-1]["frame_idx"]

        for ev in punch_events:
            center = ev["frame_idx"]
            s = max(center - self.SEGMENT_HALF_WINDOW, buf_start)
            e = min(center + self.SEGMENT_HALF_WINDOW, buf_end)
            seg_frames = [
                fr["pose_frame"] for fr in self._frame_buffer
                if s <= fr["frame_idx"] <= e
            ]
            if seg_frames:
                segments.append({
                    "label": ev["voice_label"],
                    "frames": seg_frames,
                    "center_idx": center,
                })

        summary = {}
        for seg in segments:
            summary[seg["label"]] = summary.get(seg["label"], 0) + 1
        return {"segments": segments, "summary": summary}

    def _save_training_data(self):
        import json as _json
        import numpy as np
        self._training_status = "Extracting training segments..."
        self._render_summary()

        data = self._extract_training_segments()
        segments = data["segments"]
        if not segments:
            self._training_status = "No training segments to save (shout some moves first!)"
            return

        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from ml.pipeline import (
            extract_position_features, add_velocity_features,
            create_windowed_dataset, DEFAULT_CLASS_NAMES,
        )

        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml", "data")
        os.makedirs(save_dir, exist_ok=True)

        all_video_frames = []
        class_names = list(DEFAULT_CLASS_NAMES)

        for seg in segments:
            label = seg["label"]
            if label not in class_names:
                class_names.append(label)
            pos_features = [extract_position_features(pf) for pf in seg["frames"]]
            enriched = add_velocity_features(pos_features)
            labeled_frames = []
            hw = self.SEGMENT_HALF_WINDOW
            for i, feat in enumerate(enriched):
                if feat:
                    center_rel = abs(i - len(enriched) // 2)
                    frame_label = label if center_rel < hw // 2 else "idle"
                    labeled_frames.append({"index": i, "label": frame_label, "features": feat})
            if labeled_frames:
                all_video_frames.append(labeled_frames)

        if not all_video_frames:
            self._training_status = "No valid features extracted from segments."
            return

        X, y = create_windowed_dataset(all_video_frames, class_names)
        timestamp = int(time.time())
        save_path = os.path.join(save_dir, f"voice_dojo_{timestamp}.npz")
        np.savez(save_path, X=X, y=y, class_names=class_names)

        meta = {
            "timestamp": timestamp,
            "n_segments": len(segments),
            "n_windows": int(X.shape[0]),
            "class_names": class_names,
            "summary": data["summary"],
        }
        meta_path = save_path.replace(".npz", "_meta.json")
        with open(meta_path, "w") as f:
            _json.dump(meta, f, indent=2)

        self._training_status = (
            f"Saved {X.shape[0]} windows from {len(segments)} segments "
            f"to {os.path.basename(save_path)}"
        )

    def _finetune_model(self):
        import json as _json
        import numpy as np
        self._training_status = "Saving training data first..."
        self._render_summary()

        self._save_training_data()
        if "No" in self._training_status:
            return

        self._training_status = "Finetuning model... (this may take a minute)"
        self._render_summary()

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml", "data")
        voice_files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith("voice_dojo_") and f.endswith(".npz")
        ])
        if not voice_files:
            self._training_status = "No voice dojo training data found."
            return

        latest = os.path.join(data_dir, voice_files[-1])
        cached = np.load(latest, allow_pickle=True)
        X_new = cached["X"]
        y_new = cached["y"]
        new_class_names = list(cached["class_names"])

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from ml.pipeline import MoveClassifierCNN, ModelRegistry, MODELS_DIR, FEATURE_NAMES

        registry = ModelRegistry()
        version = registry.next_version()

        # Combine with existing data if available
        existing_files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith("training_features_") and f.endswith(".npz")
        ])
        if existing_files:
            existing = np.load(os.path.join(data_dir, existing_files[-1]), allow_pickle=True)
            X_combined = np.concatenate([existing["X"], X_new], axis=0)
            y_combined = np.concatenate([existing["y"], y_new], axis=0)
        else:
            X_combined = X_new
            y_combined = y_new

        mean = X_combined.mean(axis=(0, 1))
        std = X_combined.std(axis=(0, 1)) + 1e-8
        X_norm = (X_combined - mean) / std

        device = torch.device("cpu")
        n_classes = len(new_class_names)
        model = MoveClassifierCNN(n_features=len(FEATURE_NAMES), n_classes=n_classes)

        latest_model = os.path.join(MODELS_DIR, "move_classifier.pt")
        if os.path.exists(latest_model):
            try:
                state = torch.load(latest_model, map_location=device, weights_only=True)
                old_n = state["classifier.3.weight"].shape[0]
                if old_n != n_classes:
                    new_state = {}
                    for k, v in state.items():
                        if "classifier.3" in k:
                            if "weight" in k:
                                w = torch.zeros(n_classes, v.shape[1])
                                w[:old_n] = v
                                new_state[k] = w
                            elif "bias" in k:
                                b = torch.zeros(n_classes)
                                b[:old_n] = v
                                new_state[k] = b
                            else:
                                new_state[k] = v
                        else:
                            new_state[k] = v
                    model.load_state_dict(new_state)
                else:
                    model.load_state_dict(state)
            except Exception:
                pass

        model.train()
        class_counts = np.bincount(y_combined, minlength=n_classes).astype(float)
        class_counts[class_counts == 0] = 1.0
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * n_classes

        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        dataset = TensorDataset(torch.FloatTensor(X_norm), torch.LongTensor(y_combined))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        epochs = 40
        best_acc = 0.0
        best_state = None
        for epoch in range(epochs):
            correct = total = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                correct += (out.argmax(dim=1) == yb).sum().item()
                total += len(yb)
            acc = correct / total if total > 0 else 0
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        os.makedirs(MODELS_DIR, exist_ok=True)
        version_path = os.path.join(MODELS_DIR, f"move_classifier_v{version}.pt")
        torch.save(model.state_dict(), version_path)
        torch.save(model.state_dict(), latest_model)
        np.savez(os.path.join(MODELS_DIR, f"norm_stats_v{version}.npz"), mean=mean, std=std)
        np.savez(os.path.join(MODELS_DIR, "norm_stats.npz"), mean=mean, std=std)

        config = {
            "n_features": len(FEATURE_NAMES), "n_classes": n_classes,
            "class_names": new_class_names, "window_size": 16,
            "feature_names": FEATURE_NAMES,
        }
        with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
            _json.dump(config, f, indent=2)

        registry.register_version(version, {
            "mode": "voice_dojo_finetune", "epochs": epochs,
            "val_accuracy": best_acc, "class_names": new_class_names,
            "n_windows": int(X_combined.shape[0]),
        })

        self._training_status = (
            f"Model v{version} saved! Accuracy: {best_acc:.1%} on "
            f"{X_combined.shape[0]} windows. Restart with --ml to use it."
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _cleanup(self):
        # Stop voice if active
        if self._voice and self._voice_active:
            self._voice.stop()

        import json as _json
        total_det = sum(self.move_counts.values())
        total_gt = sum(self.voice_gt_counts.values())
        elapsed = time.time() - self._start_time
        total_correct = sum(1 for e in self.label_events if e["match"])

        print("\n--- DOJO SESSION SUMMARY ---")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Moves detected: {total_det}")
        if total_gt > 0:
            print(f"Voice labels (GT): {total_gt}")
            print(f"Accuracy: {total_correct}/{total_gt} ({100*total_correct/total_gt:.0f}%)")
        for mn in ["jab", "cross", "hook", "uppercut"]:
            det = self.move_counts.get(mn, 0)
            gt = self.voice_gt_counts.get(mn, 0)
            if gt > 0:
                print(f"  {mn}: detected={det}, GT={gt}")
            else:
                print(f"  {mn}: {det}")
        print("----------------------------\n")

        log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dojo_session_log.json"
        )
        log_data = {
            "duration_s": round(elapsed, 2),
            "total_moves": total_det,
            "counts": self.move_counts,
            "rate_per_sec": round(total_det / elapsed, 2) if elapsed > 1 else 0,
            "move_log": [
                {"time_s": round(t, 3), "move": m, "hand": h}
                for t, m, h in self.move_log
            ],
        }
        if total_gt > 0:
            log_data["voice_gt_counts"] = self.voice_gt_counts
            log_data["voice_accuracy"] = round(total_correct / total_gt, 4) if total_gt > 0 else 0
            log_data["label_events"] = self.label_events

        with open(log_path, "w") as f:
            _json.dump(log_data, f, indent=2)
        print(f"Session log saved to: {log_path}")

        self.pose_estimator.close()
        self.video_source.close()
        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Stick Fighter - Camera-driven fighting game")
    parser.add_argument(
        "--source", default="0",
        help="Camera index (0 for default) or path to video file",
    )
    parser.add_argument(
        "--style", default="boxer",
        choices=["boxer", "brawler", "counter", "speedster", "tank"],
        help="NPC fighting style",
    )
    parser.add_argument(
        "--difficulty", default="medium",
        choices=["easy", "medium", "hard", "nightmare"],
        help="Difficulty level",
    )
    parser.add_argument(
        "--skip-calibration", action="store_true",
        help="Skip calibration and use default thresholds",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without display (for testing with video files)",
    )
    parser.add_argument(
        "--dojo", action="store_true",
        help="Dojo mode: free practice with real-time detection feedback (press V for voice labeling)",
    )
    parser.add_argument(
        "--ml", action="store_true",
        help="Use ML-based hybrid move detector instead of rule-based",
    )
    args = parser.parse_args()

    # Parse camera source
    source = int(args.source) if args.source.isdigit() else args.source

    # --- Calibration phase ---
    calibration_profile = None

    if not args.skip_calibration and not args.headless:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Stick Fighter — Calibration")
        clock = pygame.time.Clock()

        video_source = VideoSource(source, target_fps=FPS)
        if not video_source.is_open:
            print(f"ERROR: Cannot open camera/video source: {source}")
            sys.exit(1)

        pose_estimator = PoseEstimator(upper_body_only=True, running_mode="VIDEO")
        smoother = PoseSmoother(SmoothingConfig(min_cutoff=1.0, beta=5.0))

        cal_flow = CalibrationFlow(screen, clock, video_source, pose_estimator, smoother)
        calibration_profile = cal_flow.run()

        # Clean up calibration resources — game will create its own
        pose_estimator.close()
        video_source.close()
        pygame.quit()

        if calibration_profile is not None:
            print(f"Calibration complete — stance: {calibration_profile.stance}")
            if calibration_profile.thresholds:
                ct = calibration_profile.thresholds
                print(f"  punch_z_vel: {ct.punch_z_velocity_threshold:.4f}")
                print(f"  hook_x_vel:  {ct.hook_x_velocity_threshold:.4f}")
                print(f"  upper_y_vel: {ct.uppercut_y_velocity_threshold:.4f}")
        else:
            print("Calibration skipped — using default thresholds")
    elif not args.skip_calibration and os.path.exists(CALIBRATION_PROFILE_PATH):
        # Headless mode but profile exists — load it
        try:
            calibration_profile = CalibrationProfile.load(CALIBRATION_PROFILE_PATH)
            if not calibration_profile.is_fully_calibrated():
                calibration_profile = None
            else:
                print(f"Loaded calibration profile from {CALIBRATION_PROFILE_PATH}")
        except (json.JSONDecodeError, KeyError, ValueError):
            calibration_profile = None

    # --- Game phase ---
    start_game = not args.dojo  # True if going straight to game

    if args.dojo:
        dojo = DojoMode(
            camera_source=source,
            calibration_profile=calibration_profile,
            use_ml=args.ml,
        )
        result = dojo.run()  # returns "game" or "quit"
        start_game = (result == "game")

    if start_game:
        game = StickFighterGame(
            camera_source=source,
            calibration_profile=calibration_profile,
            use_ml=args.ml,
        )
        game.npc_style = FightingStyle(args.style)
        game.difficulty = Difficulty(args.difficulty)
        game._rebuild_npc()
        game.run()


if __name__ == "__main__":
    main()
