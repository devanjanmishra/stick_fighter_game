"""
Voice-labeled Dojo training mode.

The user shouts move names ("jab!", "cross!", "hook!", "uppercut!") while
performing them. Speech recognition timestamps the label as ground truth.
The system simultaneously runs the move detector and compares detection vs
what the user said (GT). Live stats show accuracy in real time.

At the end of the session the user can:
  - Save collected labeled segments as training data
  - Finetune the ML model on the new data with a single key press

Usage (from main.py):
    python main.py --voice-dojo --source 0
    python main.py --voice-dojo --source 0 --ml
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pygame

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pose_estimator import PoseEstimator, VideoSource, PoseFrame
from core.smoothing import PoseSmoother, SmoothingConfig
from core.coordinate_transformer import CoordinateTransformer, GamePose
from core.move_detector import MoveDetector, MoveDetectorConfig, MoveType, MovePhase
from core.movement_tracker import MovementTracker
from core.voice_recognizer import VoiceRecognizer, VoiceLabel, VALID_LABELS
from rendering.game_renderer import GameRenderer
from rendering.move_explainer import draw_keypoints_on_frame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 30
GROUND_Y = 580

CAMERA_PREVIEW_W = 240
CAMERA_PREVIEW_H = 180

# How many frames to keep around each voice label for training segments
SEGMENT_HALF_WINDOW = 45  # ±1.5s at 30fps → 3s total per segment

MOVE_COLORS = {
    "jab": (100, 200, 255),
    "cross": (255, 100, 100),
    "hook": (255, 200, 50),
    "uppercut": (100, 255, 100),
    "walking": (180, 180, 180),
    "idle": (120, 120, 120),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class LabelEvent:
    """A voice label event with matched detection result."""
    time_s: float              # seconds since session start
    frame_idx: int             # frame index when label arrived
    voice_label: str           # what the user said (GT)
    detected_label: str = ""   # what the detector predicted
    match: bool = False        # GT == detected?


@dataclass
class FrameRecord:
    """Stored frame data for training segment extraction."""
    frame_idx: int
    timestamp_s: float
    pose_frame: object         # PoseFrame (smoothed)
    features: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VoiceDojoMode
# ---------------------------------------------------------------------------
class VoiceDojoMode:
    """
    Dojo mode with voice labeling for training data collection.

    Flow:
      1. User enters dojo, mic starts listening
      2. User shouts "jab!" and throws a jab
      3. System timestamps the voice label and records what the detector sees
      4. Live panel shows: voice GT vs detection, accuracy stats
      5. When done (ESC), user can press T to finetune model on collected data

    Controls:
      ESC        - finish session (shows summary + finetune option)
      SPACE / R  - reset stats
      D          - toggle debug overlay
      T          - finetune model on collected data (from summary screen)
      S          - save training data without finetuning
    """

    def __init__(
        self,
        camera_source: int | str,
        calibration_profile=None,
        use_ml: bool = False,
    ):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Stick Fighter — VOICE DOJO (Training)")
        self.clock = pygame.time.Clock()

        # Core pipeline
        self.video_source = VideoSource(camera_source, target_fps=FPS)
        if not self.video_source.is_open:
            print(f"ERROR: Cannot open camera/video source: {camera_source}")
            sys.exit(1)

        self.pose_estimator = PoseEstimator(
            upper_body_only=True, running_mode="VIDEO"
        )
        self.smoother = PoseSmoother(SmoothingConfig(min_cutoff=1.0, beta=5.0))
        self.transformer = CoordinateTransformer(
            screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
            player_base_x=SCREEN_WIDTH // 2, ground_y=GROUND_Y,
        )

        # Move detector
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
                model_dir="ml/models", stance=stance,
            )
        else:
            self.move_detector = MoveDetector(detector_cfg)

        self.movement_tracker = MovementTracker(initial_x=SCREEN_WIDTH // 2)
        self.renderer = GameRenderer(SCREEN_WIDTH, SCREEN_HEIGHT)

        # Voice recognizer
        self.voice = VoiceRecognizer()

        # State
        self.running = True
        self.debug_mode = False
        self._calibrated = False
        self._frame_count = 0
        self._start_time = time.time()
        self.last_player_pose: GamePose | None = None
        self.last_raw_frame = None
        self._last_smoothed_kps: dict = {}

        # Detection tracking
        self.move_counts: dict[str, int] = {
            m: 0 for m in ["jab", "cross", "hook", "uppercut"]
        }
        self.current_move = MoveType.IDLE
        self.current_hand = "right"
        self.move_flash_timer = 0
        self.move_flash_name = ""

        # Voice label tracking
        self.voice_gt_counts: dict[str, int] = {
            m: 0 for m in ["jab", "cross", "hook", "uppercut"]
        }
        self.label_events: list[LabelEvent] = []
        self.voice_flash_timer = 0
        self.voice_flash_name = ""

        # Frame buffer for segment extraction
        self._frame_buffer: list[FrameRecord] = []
        self._max_buffer = FPS * 120  # keep last 2 minutes

        # Velocity tracking for bars
        self._vel_z = 0.0
        self._vel_x = 0.0
        self._vel_y = 0.0

        # Summary screen state
        self._show_summary = False
        self._training_status = ""

        # Voice status
        self._voice_ok = False
        self._voice_error = ""

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        # Start voice recognizer
        self._voice_ok = self.voice.start()
        if not self._voice_ok:
            self._voice_error = self.voice.error or "Microphone not available"
            print(f"WARNING: Voice recognition unavailable: {self._voice_error}")
            print("  Voice Dojo will run but voice labels won't work.")
            print("  Make sure a microphone is connected and accessible.")

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

    def _handle_summary_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_t:
                    self._finetune_model()
                elif event.key == pygame.K_s:
                    self._save_training_data()
                elif event.key == pygame.K_RETURN:
                    # Go back to dojo
                    self._show_summary = False

    def _reset_stats(self):
        self.move_counts = {k: 0 for k in self.move_counts}
        self.voice_gt_counts = {k: 0 for k in self.voice_gt_counts}
        self.label_events.clear()
        self._frame_buffer.clear()
        self._start_time = time.time()
        self._frame_count = 0

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
        self.transformer.set_player_position(movement_state.game_x)
        player_pose = self.transformer.transform(smoothed, facing_right=True)
        self.last_player_pose = player_pose
        self._last_smoothed_kps = (
            smoothed.keypoints if smoothed and smoothed.keypoints else {}
        )

        # Store frame for training segment extraction
        elapsed = time.time() - self._start_time
        self._frame_buffer.append(FrameRecord(
            frame_idx=self._frame_count,
            timestamp_s=elapsed,
            pose_frame=smoothed,
        ))
        # Trim buffer if too long
        if len(self._frame_buffer) > self._max_buffer:
            self._frame_buffer = self._frame_buffer[-self._max_buffer:]

        # Run detector
        detected = self.move_detector.detect(smoothed)
        self.current_move = detected.move_type
        self.current_hand = detected.hand

        # Track velocities for display
        if hasattr(self.move_detector, '_velocities'):
            v = self.move_detector._velocities
            self._vel_z = abs(v.get('wrist_z', 0.0))
            self._vel_x = abs(v.get('wrist_x', 0.0))
            self._vel_y = abs(v.get('wrist_y', 0.0))

        # Register detected move
        if detected.move_type != MoveType.IDLE and detected.phase == MovePhase.ACTIVE:
            move_name = detected.move_type.value
            if self.move_flash_name != move_name or self.move_flash_timer <= 0:
                self.move_counts[move_name] = self.move_counts.get(move_name, 0) + 1
                self.move_flash_name = move_name
                self.move_flash_timer = 25

        if self.move_flash_timer > 0:
            self.move_flash_timer -= 1

        # Process voice labels
        if self._voice_ok:
            voice_labels = self.voice.get_labels()
            for vl in voice_labels:
                self._process_voice_label(vl)

    def _process_voice_label(self, vl: VoiceLabel):
        """Handle a newly recognized voice label."""
        move_label = vl.label

        # Only track punch-type moves for GT counting
        if move_label in self.voice_gt_counts:
            self.voice_gt_counts[move_label] += 1

        # Flash the voice label
        self.voice_flash_name = move_label
        self.voice_flash_timer = 30  # ~1s

        # Find what the detector said around this time
        # Look at recent detections (within ±0.5s)
        elapsed = time.time() - self._start_time
        detected_label = "idle"
        # Check recent move log entries
        if self.move_flash_name and self.move_flash_timer > 15:
            detected_label = self.move_flash_name

        event = LabelEvent(
            time_s=elapsed,
            frame_idx=self._frame_count,
            voice_label=move_label,
            detected_label=detected_label,
            match=(move_label == detected_label),
        )
        self.label_events.append(event)

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

        # Big move flash (center) — shows both detection and voice label
        self._draw_dual_flash(surface)

        # Detection vs GT stats panel (left side)
        self._draw_accuracy_panel(surface)

        # Voice label log (right side)
        self._draw_label_log(surface)

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
        pygame.draw.rect(
            surface, (100, 100, 100),
            (x - 2, y - 2, CAMERA_PREVIEW_W + 4, CAMERA_PREVIEW_H + 4), 2,
        )
        surface.blit(cam_surf, (x, y))
        font = pygame.font.SysFont("monospace", 12)
        label = font.render("CAMERA + KEYPOINTS", True, (180, 180, 180))
        surface.blit(label, (x, y - 14))

    def _draw_dual_flash(self, surface: pygame.Surface):
        """Show both detected move and voice label side by side."""
        cx = SCREEN_WIDTH // 2
        cy = 140

        # Detection flash (left side)
        if self.move_flash_timer > 0:
            color = MOVE_COLORS.get(self.move_flash_name, (255, 255, 255))
            alpha = min(255, self.move_flash_timer * 12)
            font = pygame.font.SysFont("monospace", 48, bold=True)
            text = font.render(self.move_flash_name.upper(), True, color)
            bg = pygame.Surface(
                (text.get_width() + 20, text.get_height() + 10), pygame.SRCALPHA
            )
            bg.fill((0, 0, 0, min(180, alpha)))
            surface.blit(bg, (cx - bg.get_width() - 10, cy))
            surface.blit(text, (cx - text.get_width() - 0, cy + 5))

            label_font = pygame.font.SysFont("monospace", 16)
            det_label = label_font.render("DETECTED", True, (180, 180, 180))
            surface.blit(det_label, (cx - det_label.get_width() - 10, cy - 18))

        # Voice label flash (right side)
        if self.voice_flash_timer > 0:
            self.voice_flash_timer -= 1
            color = MOVE_COLORS.get(self.voice_flash_name, (255, 255, 255))
            alpha = min(255, self.voice_flash_timer * 10)
            font = pygame.font.SysFont("monospace", 48, bold=True)
            text = font.render(self.voice_flash_name.upper(), True, color)
            bg = pygame.Surface(
                (text.get_width() + 20, text.get_height() + 10), pygame.SRCALPHA
            )
            bg.fill((0, 40, 0, min(180, alpha)))
            surface.blit(bg, (cx + 10, cy))
            surface.blit(text, (cx + 20, cy + 5))

            label_font = pygame.font.SysFont("monospace", 16)
            gt_label = label_font.render("VOICE (GT)", True, (100, 255, 100))
            surface.blit(gt_label, (cx + 10, cy - 18))

        # Match indicator between the two
        if self.move_flash_timer > 0 and self.voice_flash_timer > 0:
            match = self.move_flash_name == self.voice_flash_name
            indicator_color = (0, 255, 0) if match else (255, 60, 60)
            indicator_text = "MATCH" if match else "MISMATCH"
            ind_font = pygame.font.SysFont("monospace", 20, bold=True)
            ind = ind_font.render(indicator_text, True, indicator_color)
            surface.blit(ind, (cx - ind.get_width() // 2, cy + 65))

    def _draw_accuracy_panel(self, surface: pygame.Surface):
        """Show detection vs ground truth accuracy stats."""
        x, y = 15, 50
        font_title = pygame.font.SysFont("monospace", 20, bold=True)
        font_body = pygame.font.SysFont("monospace", 16)
        font_small = pygame.font.SysFont("monospace", 13)

        # Panel background
        panel = pygame.Surface((280, 340), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 170))
        surface.blit(panel, (x, y))

        title = font_title.render("DETECTION vs VOICE (GT)", True, (255, 220, 80))
        surface.blit(title, (x + 8, y + 6))

        # Header row
        row_y = y + 32
        headers = font_small.render(
            "MOVE       DET   GT   ACC", True, (160, 160, 170)
        )
        surface.blit(headers, (x + 8, row_y))
        row_y += 20

        total_correct = 0
        total_gt = 0
        for move_name in ["jab", "cross", "hook", "uppercut"]:
            color = MOVE_COLORS.get(move_name, (200, 200, 200))
            det = self.move_counts.get(move_name, 0)
            gt = self.voice_gt_counts.get(move_name, 0)

            # Calculate per-move accuracy from label events
            move_events = [
                e for e in self.label_events if e.voice_label == move_name
            ]
            correct = sum(1 for e in move_events if e.match)
            total_correct += correct
            total_gt += gt

            acc_str = f"{correct}/{gt}" if gt > 0 else "-"

            # Color dot
            pygame.draw.circle(surface, color, (x + 16, row_y + 8), 5)

            line = f"{move_name:<10s} {det:>3d}   {gt:>3d}   {acc_str}"
            text = font_body.render(line, True, (220, 220, 220))
            surface.blit(text, (x + 28, row_y))

            # Accuracy bar
            if gt > 0:
                bar_x = x + 250
                bar_w = 20
                fill = int(min(correct / gt, 1.0) * bar_w)
                bar_color = (0, 200, 0) if correct == gt else (255, 165, 0)
                pygame.draw.rect(surface, (40, 40, 50), (bar_x, row_y + 2, bar_w, 12))
                pygame.draw.rect(surface, bar_color, (bar_x, row_y + 2, fill, 12))

            row_y += 28

        # Overall accuracy
        row_y += 8
        pygame.draw.line(
            surface, (80, 80, 90), (x + 8, row_y), (x + 270, row_y)
        )
        row_y += 8

        total_det = sum(self.move_counts.values())
        overall_acc = (
            f"{total_correct}/{total_gt} ({100*total_correct/total_gt:.0f}%)"
            if total_gt > 0
            else "—"
        )
        acc_text = font_body.render(f"ACCURACY: {overall_acc}", True, (255, 255, 255))
        surface.blit(acc_text, (x + 8, row_y))

        row_y += 24
        det_text = font_small.render(
            f"Total detected: {total_det}", True, (150, 150, 160)
        )
        surface.blit(det_text, (x + 8, row_y))

        row_y += 18
        gt_text = font_small.render(
            f"Total voice GT: {total_gt}", True, (150, 150, 160)
        )
        surface.blit(gt_text, (x + 8, row_y))

        row_y += 18
        elapsed = time.time() - self._start_time
        elapsed_text = font_small.render(
            f"Session: {elapsed:.0f}s | Labels: {len(self.label_events)}",
            True, (120, 120, 130),
        )
        surface.blit(elapsed_text, (x + 8, row_y))

    def _draw_label_log(self, surface: pygame.Surface):
        """Show recent voice label events with match/mismatch indicators."""
        x = SCREEN_WIDTH - 300
        y = 250
        font_title = pygame.font.SysFont("monospace", 16, bold=True)
        font_body = pygame.font.SysFont("monospace", 13)

        panel = pygame.Surface((290, 320), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        surface.blit(panel, (x, y))

        title = font_title.render("VOICE LABEL LOG", True, (255, 220, 80))
        surface.blit(title, (x + 8, y + 5))

        # Column headers
        hdr = font_body.render("TIME   VOICE     DETECT    OK?", True, (140, 140, 150))
        surface.blit(hdr, (x + 8, y + 24))

        recent = self.label_events[-16:]
        row_y = y + 40
        for event in recent:
            mins = int(event.time_s) // 60
            secs = event.time_s - mins * 60
            ts = f"{mins:01d}:{secs:04.1f}"

            voice_color = MOVE_COLORS.get(event.voice_label, (200, 200, 200))
            det_color = MOVE_COLORS.get(event.detected_label, (200, 200, 200))
            match_str = "YES" if event.match else "NO"
            match_color = (0, 220, 0) if event.match else (255, 60, 60)

            # Timestamp
            ts_text = font_body.render(ts, True, (160, 160, 170))
            surface.blit(ts_text, (x + 8, row_y))

            # Voice label
            vl_text = font_body.render(event.voice_label[:9], True, voice_color)
            surface.blit(vl_text, (x + 68, row_y))

            # Detected label
            dl_text = font_body.render(
                event.detected_label[:9] if event.detected_label else "—",
                True, det_color,
            )
            surface.blit(dl_text, (x + 152, row_y))

            # Match indicator
            match_text = font_body.render(match_str, True, match_color)
            surface.blit(match_text, (x + 245, row_y))

            row_y += 17

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
        font = pygame.font.SysFont("monospace", 26, bold=True)
        title = font.render(
            "VOICE DOJO — SHOUT MOVES TO TRAIN", True, (255, 220, 80)
        )
        surface.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 6))

        hint_font = pygame.font.SysFont("monospace", 13)
        hint = hint_font.render(
            'Shout "JAB" "CROSS" "HOOK" "UPPERCUT" while throwing | '
            "SPACE=reset | D=debug | ESC=finish",
            True, (120, 120, 130),
        )
        surface.blit(hint, (SCREEN_WIDTH // 2 - hint.get_width() // 2, 34))

    def _draw_voice_status(self, surface: pygame.Surface):
        """Show microphone status indicator."""
        font = pygame.font.SysFont("monospace", 14)
        x = SCREEN_WIDTH - 200
        y = SCREEN_HEIGHT - 25

        if self._voice_ok:
            # Green mic indicator
            pygame.draw.circle(surface, (0, 200, 0), (x, y + 6), 5)
            text = font.render("MIC ACTIVE", True, (0, 200, 0))
            surface.blit(text, (x + 10, y))

            # Show if voice error occurred
            if self.voice.error:
                err = font.render(
                    f"Warn: {self.voice.error[:30]}", True, (255, 165, 0)
                )
                surface.blit(err, (x - 200, y))
        else:
            # Red mic indicator
            pygame.draw.circle(surface, (255, 60, 60), (x, y + 6), 5)
            text = font.render("MIC OFF", True, (255, 60, 60))
            surface.blit(text, (x + 10, y))

    def _draw_debug_info(self, surface: pygame.Surface):
        font = pygame.font.SysFont("monospace", 14)
        fps = self.clock.get_fps()
        lines = [
            f"FPS: {fps:.0f}",
            f"Frame: {self._frame_count}",
            f"Move: {self.current_move.value}",
            f"Hand: {self.current_hand}",
            f"Vel Z: {self._vel_z:.4f}  X: {self._vel_x:.4f}  Y: {self._vel_y:.4f}",
            f"Voice labels: {len(self.label_events)}",
            f"Buffer frames: {len(self._frame_buffer)}",
        ]
        y = SCREEN_HEIGHT - 180
        for line in lines:
            text = font.render(line, True, (0, 255, 0))
            bg = pygame.Surface(
                (text.get_width() + 4, text.get_height() + 2), pygame.SRCALPHA
            )
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

        font_title = pygame.font.SysFont("monospace", 32, bold=True)
        font_sub = pygame.font.SysFont("monospace", 22, bold=True)
        font_body = pygame.font.SysFont("monospace", 18)
        font_small = pygame.font.SysFont("monospace", 14)

        y = 30
        title = font_title.render("VOICE DOJO SESSION SUMMARY", True, (255, 220, 80))
        surface.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, y))

        y += 60
        elapsed = time.time() - self._start_time
        info = font_body.render(
            f"Session: {elapsed:.0f}s | Total voice labels: {len(self.label_events)}",
            True, (180, 180, 190),
        )
        surface.blit(info, (80, y))

        # Per-move stats table
        y += 50
        header = font_sub.render("MOVE       DETECTED  VOICE(GT)  MATCHED  ACC", True, (200, 200, 210))
        surface.blit(header, (80, y))
        y += 30

        total_correct = 0
        total_gt = 0
        for move_name in ["jab", "cross", "hook", "uppercut"]:
            color = MOVE_COLORS.get(move_name, (200, 200, 200))
            det = self.move_counts.get(move_name, 0)
            gt = self.voice_gt_counts.get(move_name, 0)
            move_events = [e for e in self.label_events if e.voice_label == move_name]
            correct = sum(1 for e in move_events if e.match)
            total_correct += correct
            total_gt += gt
            acc_pct = f"{100*correct/gt:.0f}%" if gt > 0 else "—"

            line = f"{move_name:<10s}  {det:>5d}     {gt:>5d}      {correct:>3d}    {acc_pct:>4s}"
            text = font_body.render(line, True, color)
            surface.blit(text, (80, y))
            y += 28

        # Overall
        y += 10
        overall_pct = f"{100*total_correct/total_gt:.0f}%" if total_gt > 0 else "—"
        overall = font_sub.render(
            f"OVERALL: {total_correct}/{total_gt} ({overall_pct})",
            True, (255, 255, 255),
        )
        surface.blit(overall, (80, y))

        # Training data info
        y += 50
        n_segments = len([
            e for e in self.label_events
            if e.voice_label in ("jab", "cross", "hook", "uppercut")
        ])
        seg_info = font_body.render(
            f"Training segments available: {n_segments} "
            f"(from {len(self._frame_buffer)} buffered frames)",
            True, (180, 180, 190),
        )
        surface.blit(seg_info, (80, y))

        # Action buttons
        y += 60
        actions = [
            ("T", "Finetune model on collected data", (100, 255, 100)),
            ("S", "Save training data (without finetuning)", (100, 200, 255)),
            ("ENTER", "Go back to Voice Dojo", (255, 220, 80)),
            ("ESC", "Quit", (180, 180, 180)),
        ]
        for key, desc, color in actions:
            key_text = font_body.render(f"[{key}]", True, color)
            desc_text = font_body.render(f" {desc}", True, (200, 200, 210))
            surface.blit(key_text, (80, y))
            surface.blit(desc_text, (80 + key_text.get_width(), y))
            y += 28

        # Training status
        if self._training_status:
            y += 20
            status = font_body.render(self._training_status, True, (100, 255, 100))
            surface.blit(status, (80, y))

        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Training data save and finetune
    # ------------------------------------------------------------------
    def _extract_training_segments(self) -> dict:
        """
        Extract labeled training segments from frame buffer.

        Returns dict with:
          - segments: list of {label, frames: [PoseFrame, ...], start_idx, end_idx}
          - summary: per-label counts
        """
        segments = []
        punch_events = [
            e for e in self.label_events
            if e.voice_label in ("jab", "cross", "hook", "uppercut")
        ]

        if not self._frame_buffer:
            return {"segments": [], "summary": {}}

        # Build frame index lookup
        buffer_start_idx = self._frame_buffer[0].frame_idx
        buffer_end_idx = self._frame_buffer[-1].frame_idx

        for event in punch_events:
            center_idx = event.frame_idx
            seg_start = max(center_idx - SEGMENT_HALF_WINDOW, buffer_start_idx)
            seg_end = min(center_idx + SEGMENT_HALF_WINDOW, buffer_end_idx)

            # Find frames in range
            seg_frames = []
            for fr in self._frame_buffer:
                if seg_start <= fr.frame_idx <= seg_end:
                    seg_frames.append(fr.pose_frame)

            if seg_frames:
                segments.append({
                    "label": event.voice_label,
                    "frames": seg_frames,
                    "start_idx": seg_start,
                    "end_idx": seg_end,
                    "center_idx": center_idx,
                })

        summary = {}
        for seg in segments:
            summary[seg["label"]] = summary.get(seg["label"], 0) + 1

        return {"segments": segments, "summary": summary}

    def _save_training_data(self):
        """Save extracted training segments to disk as NPZ files."""
        self._training_status = "Extracting training segments..."
        self._render_summary()

        data = self._extract_training_segments()
        segments = data["segments"]

        if not segments:
            self._training_status = "No training segments to save (shout some moves first!)"
            return

        # Import feature extraction from pipeline
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from ml.pipeline import (
            extract_position_features, add_velocity_features,
            create_windowed_dataset, FEATURE_NAMES, DEFAULT_CLASS_NAMES,
        )

        save_dir = os.path.join(
            os.path.dirname(__file__), "..", "ml", "data"
        )
        os.makedirs(save_dir, exist_ok=True)

        # Convert segments to labeled frame lists (same format as pipeline)
        all_video_frames = []
        class_names = list(DEFAULT_CLASS_NAMES)

        for seg in segments:
            label = seg["label"]
            if label not in class_names:
                class_names.append(label)

            # Extract features from PoseFrame objects
            pos_features = []
            for pose_frame in seg["frames"]:
                feat = extract_position_features(pose_frame)
                pos_features.append(feat)

            enriched = add_velocity_features(pos_features)

            labeled_frames = []
            for i, feat in enumerate(enriched):
                if feat:
                    # Frames near center are labeled with the move type
                    # Frames at edges are labeled idle
                    center_rel = abs(i - len(enriched) // 2)
                    frame_label = label if center_rel < SEGMENT_HALF_WINDOW // 2 else "idle"
                    labeled_frames.append({
                        "index": i,
                        "label": frame_label,
                        "features": feat,
                    })

            if labeled_frames:
                all_video_frames.append(labeled_frames)

        if not all_video_frames:
            self._training_status = "No valid features extracted from segments."
            return

        # Create windowed dataset
        X, y = create_windowed_dataset(all_video_frames, class_names)

        # Save
        timestamp = int(time.time())
        save_path = os.path.join(save_dir, f"voice_dojo_{timestamp}.npz")
        np.savez(save_path, X=X, y=y, class_names=class_names)

        # Also save metadata
        meta = {
            "timestamp": timestamp,
            "n_segments": len(segments),
            "n_windows": int(X.shape[0]),
            "class_names": class_names,
            "summary": data["summary"],
            "label_events": [
                {
                    "time_s": round(e.time_s, 3),
                    "voice_label": e.voice_label,
                    "detected_label": e.detected_label,
                    "match": e.match,
                }
                for e in self.label_events
            ],
        }
        meta_path = save_path.replace(".npz", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._training_status = (
            f"Saved {X.shape[0]} windows from {len(segments)} segments "
            f"to {os.path.basename(save_path)}"
        )
        print(f"\n  Training data saved: {save_path}")
        print(f"  Metadata: {meta_path}")

    def _finetune_model(self):
        """Finetune the ML model on collected voice-labeled data."""
        self._training_status = "Saving training data first..."
        self._render_summary()

        # First save the data
        self._save_training_data()

        if "No" in self._training_status or "No valid" in self._training_status:
            return

        self._training_status = "Finetuning model... (this may take a minute)"
        self._render_summary()

        # Find the most recently saved voice dojo dataset
        data_dir = os.path.join(os.path.dirname(__file__), "..", "ml", "data")
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

        # Import training machinery from pipeline
        from ml.pipeline import (
            MoveClassifierCNN, ModelRegistry, MODELS_DIR, FEATURE_NAMES,
        )

        registry = ModelRegistry()
        version = registry.next_version()

        print(f"\n{'='*60}")
        print(f"FINETUNING MODEL v{version} on {X_new.shape[0]} voice-dojo windows")
        print(f"{'='*60}")

        # Try to load existing training data and combine
        existing_data_files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith("training_features_") and f.endswith(".npz")
        ])

        if existing_data_files:
            existing = np.load(
                os.path.join(data_dir, existing_data_files[-1]),
                allow_pickle=True,
            )
            X_old = existing["X"]
            y_old = existing["y"]
            X_combined = np.concatenate([X_old, X_new], axis=0)
            y_combined = np.concatenate([y_old, y_new], axis=0)
            print(f"  Combined: {X_old.shape[0]} existing + {X_new.shape[0]} new "
                  f"= {X_combined.shape[0]} total windows")
        else:
            X_combined = X_new
            y_combined = y_new
            print(f"  Training on {X_combined.shape[0]} voice-dojo windows (no prior data)")

        # Normalize
        mean = X_combined.mean(axis=(0, 1))
        std = X_combined.std(axis=(0, 1)) + 1e-8

        X_norm = (X_combined - mean) / std

        # Train
        device = torch.device("cpu")
        n_classes = len(new_class_names)
        model = MoveClassifierCNN(n_features=len(FEATURE_NAMES), n_classes=n_classes)

        # Load existing weights if available
        latest_model = os.path.join(MODELS_DIR, "move_classifier.pt")
        if os.path.exists(latest_model):
            try:
                state = torch.load(latest_model, map_location=device, weights_only=True)
                # Handle class count mismatch
                old_n = state["classifier.3.weight"].shape[0]
                if old_n != n_classes:
                    # Expand classifier layer
                    new_state = {}
                    for k, v in state.items():
                        if "classifier.3" in k:
                            if "weight" in k:
                                new_w = torch.zeros(n_classes, v.shape[1])
                                new_w[:old_n] = v
                                new_state[k] = new_w
                            elif "bias" in k:
                                new_b = torch.zeros(n_classes)
                                new_b[:old_n] = v
                                new_state[k] = new_b
                            else:
                                new_state[k] = v
                        else:
                            new_state[k] = v
                    model.load_state_dict(new_state)
                else:
                    model.load_state_dict(state)
                print("  Loaded existing model weights for finetuning")
            except Exception as exc:
                print(f"  Warning: Could not load existing model: {exc}")
                print("  Training from scratch")

        model.train()

        # Class weights for imbalanced data
        class_counts = np.bincount(y_combined, minlength=n_classes).astype(float)
        class_counts[class_counts == 0] = 1.0
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * n_classes
        class_weights = torch.FloatTensor(weights)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        X_tensor = torch.FloatTensor(X_norm)
        y_tensor = torch.LongTensor(y_combined)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        epochs = 40  # Shorter for finetune
        best_acc = 0.0
        best_state = None

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)

            acc = correct / total if total > 0 else 0
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"loss={total_loss/len(loader):.4f}, acc={acc:.3f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        # Save model
        os.makedirs(MODELS_DIR, exist_ok=True)
        version_path = os.path.join(MODELS_DIR, f"move_classifier_v{version}.pt")
        torch.save(model.state_dict(), version_path)
        torch.save(model.state_dict(), latest_model)

        # Save norm stats
        norm_path = os.path.join(MODELS_DIR, f"norm_stats_v{version}.npz")
        np.savez(norm_path, mean=mean, std=std)
        np.savez(os.path.join(MODELS_DIR, "norm_stats.npz"), mean=mean, std=std)

        # Save config
        config = {
            "n_features": len(FEATURE_NAMES),
            "n_classes": n_classes,
            "class_names": new_class_names,
            "window_size": 16,
            "feature_names": FEATURE_NAMES,
        }
        config_path = os.path.join(MODELS_DIR, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Register version
        registry.register_version(version, {
            "mode": "voice_dojo_finetune",
            "epochs": epochs,
            "best_epoch": epochs,
            "val_accuracy": best_acc,
            "class_names": new_class_names,
            "n_windows": int(X_combined.shape[0]),
            "n_voice_segments": int(X_new.shape[0]),
        })

        self._training_status = (
            f"Model v{version} saved! Accuracy: {best_acc:.1%} on "
            f"{X_combined.shape[0]} windows. Restart with --ml to use it."
        )
        print(f"\n  Model v{version} saved: {version_path}")
        print(f"  Best accuracy: {best_acc:.1%}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _cleanup(self):
        # Stop voice recognizer
        self.voice.stop()

        # Print summary
        total_det = sum(self.move_counts.values())
        total_gt = sum(self.voice_gt_counts.values())
        elapsed = time.time() - self._start_time

        total_correct = sum(
            1 for e in self.label_events if e.match
        )

        print("\n--- VOICE DOJO SESSION SUMMARY ---")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Detected moves: {total_det}")
        print(f"Voice labels (GT): {total_gt}")
        if total_gt > 0:
            print(f"Accuracy: {total_correct}/{total_gt} ({100*total_correct/total_gt:.0f}%)")
        print(f"Label events: {len(self.label_events)}")
        for move_name in ["jab", "cross", "hook", "uppercut"]:
            det = self.move_counts.get(move_name, 0)
            gt = self.voice_gt_counts.get(move_name, 0)
            move_events = [e for e in self.label_events if e.voice_label == move_name]
            correct = sum(1 for e in move_events if e.match)
            print(f"  {move_name}: detected={det}, GT={gt}, matched={correct}")
        print("----------------------------------\n")

        # Save session log
        log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..",
            "voice_dojo_session_log.json",
        )
        log_data = {
            "duration_s": round(elapsed, 2),
            "total_detected": total_det,
            "total_voice_gt": total_gt,
            "total_correct": total_correct,
            "accuracy": round(total_correct / total_gt, 4) if total_gt > 0 else 0,
            "detection_counts": self.move_counts,
            "voice_gt_counts": self.voice_gt_counts,
            "label_events": [
                {
                    "time_s": round(e.time_s, 3),
                    "voice_label": e.voice_label,
                    "detected_label": e.detected_label,
                    "match": e.match,
                }
                for e in self.label_events
            ],
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Session log saved to: {log_path}")

        self.pose_estimator.close()
        self.video_source.close()
        pygame.quit()
