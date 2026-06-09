"""
In-app move explainer and calibration tutorial.

Renders instructional screens that teach the user each fighting move,
show the expected motion, and guide them through the calibration process.
Used both as a standalone tutorial and during the calibration flow.
"""

import math
import pygame
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExplainerPhase(Enum):
    WELCOME = "welcome"
    STANCE_SELECT = "stance_select"
    MOVE_EXPLAIN = "move_explain"
    MOVE_DEMO = "move_demo"
    RECORD_PROMPT = "record_prompt"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    PLAYBACK = "playback"
    RECORD_DONE = "record_done"
    ALL_DONE = "all_done"


@dataclass
class MoveInfo:
    """Description and visual data for a single move."""
    name: str
    display_name: str
    description: str
    how_to: list[str]
    camera_tip: str
    game_rendering: str
    key_body_parts: list[str]
    detection_signature: str
    color: tuple[int, int, int]


# All four moves with detailed explanations
MOVE_DATABASE: dict[str, MoveInfo] = {
    "jab": MoveInfo(
        name="jab",
        display_name="JAB",
        description="A quick, straight punch with your lead hand. "
                    "The fastest strike in boxing — used to keep distance and set up combos.",
        how_to=[
            "Stand in your fighting stance facing the camera.",
            "Extend your LEAD hand (left if orthodox, right if southpaw) straight forward.",
            "Punch toward the camera quickly — your fist moves TOWARD the lens.",
            "Snap it back to guard position immediately.",
            "Keep your rear hand up protecting your chin.",
        ],
        camera_tip="The camera sees your lead wrist moving toward it (z-depth decreases). "
                   "Make it a quick, snappy motion — speed matters more than reach.",
        game_rendering="Your stick figure's lead arm extends horizontally toward the NPC, "
                       "like a classic Street Fighter jab.",
        key_body_parts=["Lead wrist", "Lead elbow", "Lead shoulder"],
        detection_signature="Rapid z-velocity decrease on lead wrist (toward camera). "
                           "Arm straightens (wrist-shoulder distance increases).",
        color=(100, 200, 255),
    ),
    "cross": MoveInfo(
        name="cross",
        display_name="CROSS",
        description="A powerful straight punch with your rear hand. "
                    "Generates more power than a jab because your whole body rotates into it.",
        how_to=[
            "Stand in your fighting stance facing the camera.",
            "Rotate your hips and shoulders as you extend your REAR hand forward.",
            "Your rear shoulder should drive forward — the camera will see it rotate.",
            "Punch straight toward the camera with your rear fist.",
            "Your lead hand stays up protecting your chin.",
        ],
        camera_tip="Similar to a jab but with your REAR hand. The key difference the system "
                   "detects is WHICH hand moves forward. Your rear shoulder also rotates visibly.",
        game_rendering="Your stick figure's rear arm extends with shoulder rotation, "
                       "a more powerful-looking straight punch than the jab.",
        key_body_parts=["Rear wrist", "Rear elbow", "Rear shoulder"],
        detection_signature="Rapid z-velocity decrease on rear wrist (toward camera). "
                           "Rear shoulder moves forward (x-displacement).",
        color=(255, 180, 50),
    ),
    "hook": MoveInfo(
        name="hook",
        display_name="HOOK",
        description="A powerful curved punch that comes from the side. "
                    "Hooks target the jaw or body from an angle the opponent doesn't expect.",
        how_to=[
            "Stand in your fighting stance facing the camera.",
            "Swing your lead arm in a horizontal ARC — not straight forward.",
            "Your fist moves LEFT-TO-RIGHT (or right-to-left) across the camera's view.",
            "Keep your elbow bent at roughly 90 degrees throughout.",
            "The motion is like swinging a door open, then snapping it forward.",
        ],
        camera_tip="The camera sees strong LATERAL (x-axis) movement of your wrist. "
                   "This is the key difference from jab/cross which are mostly z-axis. "
                   "Exaggerate the side-to-side motion for better detection.",
        game_rendering="Your stick figure's arm arcs from behind the body outward, "
                       "a sweeping horizontal strike — the classic fighting game hook.",
        key_body_parts=["Lead wrist", "Lead elbow", "Hips (rotation)"],
        detection_signature="Strong x-velocity (lateral) on lead wrist, combined with "
                           "forward z-movement. The lateral displacement exceeds z-displacement.",
        color=(255, 100, 50),
    ),
    "uppercut": MoveInfo(
        name="uppercut",
        display_name="UPPERCUT",
        description="A rising punch from below that targets the chin. "
                    "The most powerful punch when it lands — a real knockout shot.",
        how_to=[
            "Stand in your fighting stance facing the camera.",
            "Dip your rear shoulder slightly, loading the punch from below.",
            "Drive your rear fist UPWARD — your wrist rises sharply in the camera's view.",
            "Your fist should travel from waist/hip level up toward chin level.",
            "The motion is vertical — think 'scooping up' rather than punching forward.",
        ],
        camera_tip="The camera sees strong UPWARD (y-axis) movement of your wrist. "
                   "This is the most distinct move — the only one with major vertical motion. "
                   "Really exaggerate the upward scoop for clean detection.",
        game_rendering="Your stick figure's rear arm drives upward from below, "
                       "the classic rising uppercut animation.",
        key_body_parts=["Rear wrist", "Rear elbow", "Rear shoulder (dip)"],
        detection_signature="Strong negative y-velocity (upward in camera coords) on rear wrist. "
                           "Vertical displacement dominates lateral and forward displacement.",
        color=(255, 50, 50),
    ),
}

MOVE_ORDER = ["jab", "cross", "hook", "uppercut"]

# Colors
BG_COLOR = (20, 20, 30)
TEXT_COLOR = (220, 220, 220)
DIM_TEXT = (140, 140, 150)
ACCENT = (80, 180, 255)
SUCCESS = (80, 255, 120)
WARNING = (255, 200, 80)
PANEL_BG = (30, 30, 45)
PANEL_BORDER = (60, 60, 80)

# Keypoint drawing colours for camera overlay
KP_COLOR = (0, 255, 0)
KP_LINE_COLOR = (0, 200, 0)
KP_RADIUS = 4
KP_LINE_WIDTH = 2

# Connections to draw on the camera overlay
KP_CONNECTIONS = [
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]


# ---------------------------------------------------------------------------
# Stick-figure move animation keyframes
# ---------------------------------------------------------------------------
# Each keyframe is a dict of keypoint-name -> (x, y) in a local 200x300
# coordinate space.  The animator interpolates between them.

_BASE_POSE = {
    "nose": (100, 30),
    "left_shoulder": (80, 70), "right_shoulder": (120, 70),
    "left_elbow": (65, 110), "right_elbow": (135, 110),
    "left_wrist": (60, 145), "right_wrist": (140, 145),
    "left_hip": (85, 170), "right_hip": (115, 170),
    "left_knee": (80, 220), "right_knee": (120, 220),
    "left_ankle": (75, 270), "right_ankle": (125, 270),
}

_JAB_EXTENDED = {
    **_BASE_POSE,
    "left_elbow": (40, 80), "left_wrist": (5, 70),
}

_CROSS_EXTENDED = {
    **_BASE_POSE,
    "right_shoulder": (110, 68),
    "right_elbow": (155, 78), "right_wrist": (195, 70),
}

_HOOK_MID = {
    **_BASE_POSE,
    "left_elbow": (40, 85), "left_wrist": (25, 75),
}
_HOOK_EXTENDED = {
    **_BASE_POSE,
    "left_elbow": (30, 75), "left_wrist": (10, 60),
}

_UPPERCUT_LOAD = {
    **_BASE_POSE,
    "right_elbow": (130, 130), "right_wrist": (135, 165),
}
_UPPERCUT_EXTENDED = {
    **_BASE_POSE,
    "right_elbow": (130, 85), "right_wrist": (135, 45),
}

# frame sequences: list of (keyframe_dict, hold_frames)
MOVE_ANIMATIONS: dict[str, list[tuple[dict, int]]] = {
    "jab": [
        (_BASE_POSE, 20),
        (_JAB_EXTENDED, 10),
        (_BASE_POSE, 15),
    ],
    "cross": [
        (_BASE_POSE, 20),
        (_CROSS_EXTENDED, 10),
        (_BASE_POSE, 15),
    ],
    "hook": [
        (_BASE_POSE, 15),
        (_HOOK_MID, 8),
        (_HOOK_EXTENDED, 10),
        (_BASE_POSE, 12),
    ],
    "uppercut": [
        (_BASE_POSE, 15),
        (_UPPERCUT_LOAD, 8),
        (_UPPERCUT_EXTENDED, 10),
        (_BASE_POSE, 12),
    ],
}

_ANIM_BONES = [
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def _lerp_poses(a: dict, b: dict, t: float) -> dict:
    """Linearly interpolate between two keyframe dicts."""
    result = {}
    for k in a:
        ax, ay = a[k]
        bx, by = b.get(k, a[k])
        result[k] = (ax + (bx - ax) * t, ay + (by - ay) * t)
    return result


class MoveAnimator:
    """Plays back a keyframed stick-figure animation for one move."""

    def __init__(self, move_name: str):
        self._frames = MOVE_ANIMATIONS.get(move_name, MOVE_ANIMATIONS["jab"])
        self._total = sum(f[1] for f in self._frames)
        self._tick = 0

    def reset(self):
        self._tick = 0

    def step(self) -> dict:
        """Advance one tick and return interpolated pose dict."""
        self._tick = self._tick % self._total
        elapsed = 0
        for i, (kf, hold) in enumerate(self._frames):
            if elapsed + hold > self._tick:
                local_t = (self._tick - elapsed) / max(hold, 1)
                next_kf = self._frames[(i + 1) % len(self._frames)][0]
                self._tick += 1
                return _lerp_poses(kf, next_kf, local_t)
            elapsed += hold
        self._tick += 1
        return self._frames[0][0]

    def draw(self, surface: pygame.Surface, offset_x: int, offset_y: int,
             color: tuple[int, int, int] = ACCENT, scale: float = 1.0):
        """Draw the current animation frame on *surface*."""
        pose = self.step()
        pts: dict[str, tuple[int, int]] = {}
        for k, (x, y) in pose.items():
            pts[k] = (int(offset_x + x * scale), int(offset_y + y * scale))

        # Bones
        for a, b in _ANIM_BONES:
            if a in pts and b in pts:
                pygame.draw.line(surface, (30, 30, 30), pts[a], pts[b], 5)
                pygame.draw.line(surface, color, pts[a], pts[b], 3)

        # Joints
        for name, pos in pts.items():
            r = 10 if name == "nose" else 5
            pygame.draw.circle(surface, (30, 30, 30), pos, r + 1)
            pygame.draw.circle(surface, color, pos, r)


def draw_keypoints_on_frame(
    frame: np.ndarray,
    keypoints: dict,
    connections: list[tuple[str, str]] = KP_CONNECTIONS,
) -> np.ndarray:
    """Draw keypoint dots and connections onto a BGR frame (in-place)."""
    import cv2
    h, w = frame.shape[:2]
    pts: dict[str, tuple[int, int]] = {}
    for name, kp in keypoints.items():
        px = int(kp.x * w) if hasattr(kp, "x") else 0
        py = int(kp.y * h) if hasattr(kp, "y") else 0
        pts[name] = (px, py)
        cv2.circle(frame, (px, py), KP_RADIUS, KP_COLOR, -1)

    for a, b in connections:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], KP_LINE_COLOR, KP_LINE_WIDTH)
    return frame


class MoveExplainer:
    """
    Renders instructional screens for the calibration tutorial.

    Usage:
        explainer = MoveExplainer(screen_width, screen_height)
        # Draw the welcome screen
        explainer.draw_welcome(screen)
        # Draw explanation for a specific move
        explainer.draw_move_explanation(screen, "jab")
        # Draw the recording prompt
        explainer.draw_record_prompt(screen, "jab", sample_number=1, total=3)
        # Draw recording in progress
        explainer.draw_recording(screen, "jab", frame_count=15, max_frames=30)
        # Draw completion
        explainer.draw_all_done(screen)
    """

    def __init__(self, screen_width: int = 1280, screen_height: int = 720):
        self.w = screen_width
        self.h = screen_height
        self._anim_frame = 0
        self._animators: dict[str, MoveAnimator] = {}
        for mn in MOVE_ORDER:
            self._animators[mn] = MoveAnimator(mn)

    def _get_fonts(self) -> dict:
        return {
            "title": pygame.font.SysFont("monospace", 36, bold=True),
            "subtitle": pygame.font.SysFont("monospace", 24, bold=True),
            "body": pygame.font.SysFont("monospace", 18),
            "small": pygame.font.SysFont("monospace", 14),
            "large": pygame.font.SysFont("monospace", 48, bold=True),
            "hint": pygame.font.SysFont("monospace", 16, italic=True),
        }

    def _draw_panel(self, screen: pygame.Surface, x: int, y: int, w: int, h: int,
                    border_color: tuple = PANEL_BORDER):
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((*PANEL_BG, 220))
        screen.blit(panel, (x, y))
        pygame.draw.rect(screen, border_color, (x, y, w, h), 2, border_radius=8)

    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        words = text.split(" ")
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            if font.size(test)[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def draw_welcome(self, screen: pygame.Surface):
        """Draw the welcome/intro screen for calibration."""
        screen.fill(BG_COLOR)
        fonts = self._get_fonts()

        # Title
        title = fonts["title"].render("STICK FIGHTER — CALIBRATION", True, ACCENT)
        screen.blit(title, (self.w // 2 - title.get_width() // 2, 60))

        # Subtitle
        sub = fonts["subtitle"].render("Personalize your move detection", True, TEXT_COLOR)
        screen.blit(sub, (self.w // 2 - sub.get_width() // 2, 110))

        # Panel with instructions
        panel_x, panel_y = 100, 160
        panel_w, panel_h = self.w - 200, 420
        self._draw_panel(screen, panel_x, panel_y, panel_w, panel_h, ACCENT)

        instructions = [
            "Welcome! This calibration will teach you the four fighting moves",
            "and record YOUR personal style so the game responds perfectly to you.",
            "",
            "What you'll do:",
            "",
            "  1. Choose your stance (Orthodox or Southpaw)",
            "  2. Learn each move with detailed instructions",
            "  3. Record 3 examples of each move",
            "  4. The game calibrates detection thresholds to YOUR body",
            "",
            "What you need:",
            "",
            "  - A webcam or phone camera facing you",
            "  - Enough room to throw punches (arm's length in all directions)",
            "  - Good lighting so the camera can see you clearly",
            "",
            "The calibration takes about 2-3 minutes. You'll learn:",
            "  JAB  |  CROSS  |  HOOK  |  UPPERCUT",
        ]

        y = panel_y + 20
        for line in instructions:
            if line == "":
                y += 10
                continue
            color = TEXT_COLOR
            if line.startswith("  -"):
                color = DIM_TEXT
            if "JAB" in line and "CROSS" in line:
                color = WARNING
            surf = fonts["body"].render(line, True, color)
            screen.blit(surf, (panel_x + 30, y))
            y += 24

        # Bottom hint
        hint = fonts["hint"].render("Press SPACE or ENTER to begin", True, DIM_TEXT)
        screen.blit(hint, (self.w // 2 - hint.get_width() // 2, self.h - 50))

    def draw_stance_select(self, screen: pygame.Surface, selected: str = "orthodox"):
        """Draw stance selection screen."""
        screen.fill(BG_COLOR)
        fonts = self._get_fonts()

        title = fonts["title"].render("SELECT YOUR STANCE", True, ACCENT)
        screen.blit(title, (self.w // 2 - title.get_width() // 2, 50))

        desc = fonts["body"].render(
            "Which hand do you naturally keep in front when you fight?", True, TEXT_COLOR
        )
        screen.blit(desc, (self.w // 2 - desc.get_width() // 2, 110))

        # Orthodox panel
        orth_x = self.w // 4 - 150
        orth_color = SUCCESS if selected == "orthodox" else PANEL_BORDER
        self._draw_panel(screen, orth_x, 170, 300, 350, orth_color)

        orth_title = fonts["subtitle"].render("ORTHODOX", True,
                                               SUCCESS if selected == "orthodox" else TEXT_COLOR)
        screen.blit(orth_title, (orth_x + 150 - orth_title.get_width() // 2, 185))

        orth_lines = [
            "Left hand in front",
            "Right hand in back",
            "",
            "Lead hand: LEFT",
            "Power hand: RIGHT",
            "",
            "Most common stance",
            "(~90% of fighters)",
            "",
            "Jab = left hand",
            "Cross = right hand",
        ]
        y = 220
        for line in orth_lines:
            if not line:
                y += 8
                continue
            color = DIM_TEXT if "Most" in line or "90%" in line else TEXT_COLOR
            surf = fonts["small"].render(line, True, color)
            screen.blit(surf, (orth_x + 30, y))
            y += 22

        # Southpaw panel
        south_x = 3 * self.w // 4 - 150
        south_color = SUCCESS if selected == "southpaw" else PANEL_BORDER
        self._draw_panel(screen, south_x, 170, 300, 350, south_color)

        south_title = fonts["subtitle"].render("SOUTHPAW", True,
                                                SUCCESS if selected == "southpaw" else TEXT_COLOR)
        screen.blit(south_title, (south_x + 150 - south_title.get_width() // 2, 185))

        south_lines = [
            "Right hand in front",
            "Left hand in back",
            "",
            "Lead hand: RIGHT",
            "Power hand: LEFT",
            "",
            "Less common stance",
            "(~10% of fighters)",
            "",
            "Jab = right hand",
            "Cross = left hand",
        ]
        y = 220
        for line in south_lines:
            if not line:
                y += 8
                continue
            color = DIM_TEXT if "Less" in line or "10%" in line else TEXT_COLOR
            surf = fonts["small"].render(line, True, color)
            screen.blit(surf, (south_x + 30, y))
            y += 22

        hint = fonts["hint"].render(
            "Press LEFT/RIGHT to select, ENTER to confirm", True, DIM_TEXT
        )
        screen.blit(hint, (self.w // 2 - hint.get_width() // 2, self.h - 50))

    def draw_move_explanation(self, screen: pygame.Surface, move_name: str,
                              move_idx: int = 0, total_moves: int = 4):
        """Draw detailed explanation with animated stick-figure demo and nav."""
        screen.fill(BG_COLOR)
        fonts = self._get_fonts()
        move = MOVE_DATABASE[move_name]

        # Progress indicator
        progress = fonts["small"].render(
            f"Move {move_idx + 1} of {total_moves}", True, DIM_TEXT
        )
        screen.blit(progress, (20, 15))

        # Move name
        title = fonts["title"].render(move.display_name, True, move.color)
        screen.blit(title, (self.w // 2 - title.get_width() // 2, 30))

        # Description (compact)
        desc_lines = self._wrap_text(move.description, fonts["body"], self.w - 300)
        y = 70
        for line in desc_lines:
            surf = fonts["body"].render(line, True, TEXT_COLOR)
            screen.blit(surf, (80, y))
            y += 20
        y += 8

        # --- Left column: stick-figure animation demo ---
        anim_panel_x = 40
        anim_panel_w = 260
        anim_panel_h = 350
        self._draw_panel(screen, anim_panel_x, y, anim_panel_w, anim_panel_h, move.color)
        label = fonts["subtitle"].render("DEMO", True, move.color)
        screen.blit(label, (anim_panel_x + anim_panel_w // 2 - label.get_width() // 2, y + 8))

        animator = self._animators.get(move_name)
        if animator:
            animator.draw(surface=screen,
                          offset_x=anim_panel_x + 30,
                          offset_y=y + 35,
                          color=move.color,
                          scale=1.0)

        # --- Middle column: How to perform ---
        mid_x = anim_panel_x + anim_panel_w + 20
        mid_w = (self.w - mid_x - 40) // 2
        self._draw_panel(screen, mid_x, y, mid_w, anim_panel_h, move.color)

        how_title = fonts["subtitle"].render("HOW TO PERFORM", True, move.color)
        screen.blit(how_title, (mid_x + 12, y + 8))

        step_y = y + 38
        for i, step in enumerate(move.how_to):
            step_lines = self._wrap_text(f"{i + 1}. {step}", fonts["small"], mid_w - 30)
            for sl in step_lines:
                surf = fonts["small"].render(sl, True, TEXT_COLOR)
                screen.blit(surf, (mid_x + 12, step_y))
                step_y += 16
            step_y += 3

        # --- Right column: Camera tips ---
        right_x = mid_x + mid_w + 20
        right_w = self.w - right_x - 40
        self._draw_panel(screen, right_x, y, right_w, anim_panel_h, move.color)

        cam_title = fonts["subtitle"].render("CAMERA TIPS", True, move.color)
        screen.blit(cam_title, (right_x + 12, y + 8))

        tip_lines = self._wrap_text(move.camera_tip, fonts["small"], right_w - 30)
        tip_y = y + 38
        for tl in tip_lines:
            surf = fonts["small"].render(tl, True, TEXT_COLOR)
            screen.blit(surf, (right_x + 12, tip_y))
            tip_y += 16

        tip_y += 10
        parts_title = fonts["small"].render("Key body parts:", True, WARNING)
        screen.blit(parts_title, (right_x + 12, tip_y))
        tip_y += 18
        parts_text = ", ".join(move.key_body_parts)
        surf = fonts["small"].render(parts_text, True, DIM_TEXT)
        screen.blit(surf, (right_x + 12, tip_y))

        # --- Navigation hint bar ---
        nav_parts = []
        if move_idx > 0:
            nav_parts.append("LEFT = Previous move")
        nav_parts.append("SPACE = Record this move")
        if move_idx < total_moves - 1:
            nav_parts.append("RIGHT = Next move")
        nav_text = "   |   ".join(nav_parts)
        hint = fonts["hint"].render(nav_text, True, DIM_TEXT)
        screen.blit(hint, (self.w // 2 - hint.get_width() // 2, self.h - 50))

    def draw_record_prompt(self, screen: pygame.Surface, move_name: str,
                           sample_number: int, total: int = 3):
        """Draw the 'get ready to record' screen."""
        screen.fill(BG_COLOR)
        fonts = self._get_fonts()
        move = MOVE_DATABASE[move_name]

        title = fonts["title"].render(f"RECORD {move.display_name}", True, move.color)
        screen.blit(title, (self.w // 2 - title.get_width() // 2, 100))

        sample_text = fonts["subtitle"].render(
            f"Sample {sample_number} of {total}", True, TEXT_COLOR
        )
        screen.blit(sample_text, (self.w // 2 - sample_text.get_width() // 2, 160))

        # Checklist of completed samples
        for i in range(1, total + 1):
            if i < sample_number:
                indicator = fonts["body"].render(f"  Sample {i}: DONE", True, SUCCESS)
            elif i == sample_number:
                indicator = fonts["body"].render(f"  Sample {i}: READY", True, WARNING)
            else:
                indicator = fonts["body"].render(f"  Sample {i}: ---", True, DIM_TEXT)
            screen.blit(indicator, (self.w // 2 - 100, 200 + (i - 1) * 30))

        # Instructions
        self._draw_panel(screen, 200, 320, self.w - 400, 200, move.color)

        instructions = [
            "1. Get into your fighting stance",
            f"2. When the countdown finishes, throw ONE {move.display_name}",
            "3. Return to guard position",
            "4. Wait for confirmation",
            "",
            "Tip: Perform the move naturally — don't exaggerate or go too slow.",
            "     The system adapts to YOUR personal style.",
        ]
        y = 340
        for line in instructions:
            if not line:
                y += 5
                continue
            color = DIM_TEXT if line.startswith("Tip") or line.startswith("     ") else TEXT_COLOR
            surf = fonts["small"].render(line, True, color)
            screen.blit(surf, (220, y))
            y += 22

        hint = fonts["hint"].render(
            "Press SPACE when ready — 3-second countdown will begin", True, DIM_TEXT
        )
        screen.blit(hint, (self.w // 2 - hint.get_width() // 2, self.h - 50))

    def draw_countdown(self, screen: pygame.Surface, move_name: str,
                       seconds_left: int,
                       camera_surface: Optional[pygame.Surface] = None):
        """Draw countdown screen with optional live camera feed."""
        screen.fill(BG_COLOR)
        fonts = self._get_fonts()
        move = MOVE_DATABASE[move_name]

        if camera_surface:
            # Show camera feed in centre
            cam_w, cam_h = camera_surface.get_size()
            cam_x = self.w // 2 - cam_w // 2
            cam_y = 80
            screen.blit(camera_surface, (cam_x, cam_y))
            pygame.draw.rect(screen, move.color, (cam_x - 2, cam_y - 2, cam_w + 4, cam_h + 4), 2)

        # Large countdown number
        count_text = fonts["large"].render(str(seconds_left), True, WARNING)
        cx = self.w // 2 - count_text.get_width() // 2
        cy = self.h // 2 + 120 if camera_surface else self.h // 2 - 40
        screen.blit(count_text, (cx, cy))
        get_ready = fonts["subtitle"].render("GET READY...", True, TEXT_COLOR)
        screen.blit(get_ready, (self.w // 2 - get_ready.get_width() // 2, cy + 60))

    def draw_recording(self, screen: pygame.Surface, move_name: str,
                       frame_count: int, max_frames: int,
                       camera_surface: Optional[pygame.Surface] = None,
                       countdown: int = 0):
        """Draw recording-in-progress with live camera + keypoint overlay."""
        screen.fill(BG_COLOR)
        fonts = self._get_fonts()
        move = MOVE_DATABASE[move_name]

        if countdown > 0:
            # Backwards compat — delegate to countdown screen
            self.draw_countdown(screen, move_name, countdown, camera_surface)
            return

        # --- Camera feed (large, centre) ---
        cam_x, cam_y = 40, 90
        cam_w = self.w - 80
        cam_h = self.h - 230
        if camera_surface:
            scaled = pygame.transform.scale(camera_surface, (cam_w, cam_h))
            screen.blit(scaled, (cam_x, cam_y))
            pygame.draw.rect(screen, move.color,
                             (cam_x - 2, cam_y - 2, cam_w + 4, cam_h + 4), 2)

        # Pulsing REC indicator
        self._anim_frame += 1
        pulse = abs(math.sin(self._anim_frame * 0.15))
        rec_color = (int(255 * pulse), 30, 30)
        rec_circle = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(rec_circle, rec_color, (15, 15), 12)
        screen.blit(rec_circle, (cam_x + 10, cam_y + 10))
        rec_text = fonts["subtitle"].render("REC", True, (255, 80, 80))
        screen.blit(rec_text, (cam_x + 40, cam_y + 10))

        # Title bar
        title_text = fonts["title"].render(
            f"Throw your {move.display_name} NOW!", True, move.color
        )
        screen.blit(title_text, (self.w // 2 - title_text.get_width() // 2, 30))

        # Progress bar below camera
        bar_w = cam_w
        bar_h = 16
        bar_x = cam_x
        bar_y = cam_y + cam_h + 15
        progress = min(1.0, frame_count / max(max_frames, 1))

        pygame.draw.rect(screen, PANEL_BORDER, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            pygame.draw.rect(screen, move.color,
                             (bar_x, bar_y, fill_w, bar_h), border_radius=4)

        pct = fonts["small"].render(f"{int(progress * 100)}%", True, TEXT_COLOR)
        screen.blit(pct, (self.w // 2 - pct.get_width() // 2, bar_y + 20))

    def draw_playback(self, screen: pygame.Surface, move_name: str,
                      sample_number: int,
                      camera_surface: Optional[pygame.Surface] = None,
                      frame_idx: int = 0, total_frames: int = 1,
                      success: bool = True):
        """Draw recorded video playback with keypoints so user can verify."""
        screen.fill(BG_COLOR)
        fonts = self._get_fonts()
        move = MOVE_DATABASE[move_name]

        # Title
        status_color = SUCCESS if success else (255, 80, 80)
        status_text = "RECORDED!" if success else "DETECTION UNCLEAR"
        title = fonts["title"].render(status_text, True, status_color)
        screen.blit(title, (self.w // 2 - title.get_width() // 2, 15))

        sub = fonts["subtitle"].render(
            f"{move.display_name} — Sample {sample_number}", True, move.color
        )
        screen.blit(sub, (self.w // 2 - sub.get_width() // 2, 55))

        # Camera playback (large, centre)
        cam_x, cam_y = 40, 95
        cam_w = self.w - 80
        cam_h = self.h - 220
        if camera_surface:
            scaled = pygame.transform.scale(camera_surface, (cam_w, cam_h))
            screen.blit(scaled, (cam_x, cam_y))
            pygame.draw.rect(screen, move.color,
                             (cam_x - 2, cam_y - 2, cam_w + 4, cam_h + 4), 2)

        # Playback progress bar
        bar_w = cam_w
        bar_h = 10
        bar_x = cam_x
        bar_y = cam_y + cam_h + 8
        progress = min(1.0, frame_idx / max(total_frames - 1, 1))
        pygame.draw.rect(screen, PANEL_BORDER, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            pygame.draw.rect(screen, move.color,
                             (bar_x, bar_y, fill_w, bar_h), border_radius=3)

        # Hint
        if success:
            hint_text = "Reviewing recording...  Press SPACE to continue"
        else:
            hint_text = "Move unclear — Press SPACE to retry"
        hint = fonts["hint"].render(hint_text, True, DIM_TEXT)
        screen.blit(hint, (self.w // 2 - hint.get_width() // 2, self.h - 40))

    def draw_record_done(self, screen: pygame.Surface, move_name: str,
                         sample_number: int, success: bool = True):
        """Draw confirmation after recording a sample."""
        screen.fill(BG_COLOR)
        fonts = self._get_fonts()
        move = MOVE_DATABASE[move_name]

        if success:
            status = fonts["title"].render("RECORDED!", True, SUCCESS)
            detail = fonts["body"].render(
                f"{move.display_name} sample {sample_number} captured successfully.", True, TEXT_COLOR
            )
        else:
            status = fonts["title"].render("TRY AGAIN", True, (255, 80, 80))
            detail = fonts["body"].render(
                "Move was too short or unclear. Please try again.", True, TEXT_COLOR
            )

        screen.blit(status, (self.w // 2 - status.get_width() // 2, self.h // 2 - 60))
        screen.blit(detail, (self.w // 2 - detail.get_width() // 2, self.h // 2))

        hint = fonts["hint"].render("Press SPACE to continue", True, DIM_TEXT)
        screen.blit(hint, (self.w // 2 - hint.get_width() // 2, self.h - 50))

    def draw_all_done(self, screen: pygame.Surface):
        """Draw calibration complete screen."""
        screen.fill(BG_COLOR)
        fonts = self._get_fonts()

        title = fonts["title"].render("CALIBRATION COMPLETE!", True, SUCCESS)
        screen.blit(title, (self.w // 2 - title.get_width() // 2, 100))

        self._draw_panel(screen, 150, 170, self.w - 300, 380, SUCCESS)

        lines = [
            "All four moves have been calibrated to your personal style.",
            "",
            "Your personalized thresholds are now active:",
            "",
            "  JAB       — Lead hand quick punch (z-velocity calibrated)",
            "  CROSS     — Rear hand power punch (z-velocity + rotation)",
            "  HOOK      — Lateral arc punch (x-velocity calibrated)",
            "  UPPERCUT  — Rising punch (y-velocity calibrated)",
            "",
            "The game uses Dynamic Time Warping (DTW) to match your",
            "live movements against the templates you just recorded.",
            "",
            "This means detection is tuned to YOUR speed, YOUR reach,",
            "and YOUR style — not generic thresholds.",
            "",
            "You can re-calibrate anytime from the settings menu.",
            "",
            "Your calibration profile has been saved and will persist",
            "across game sessions.",
        ]

        y = 190
        for line in lines:
            if not line:
                y += 8
                continue
            color = TEXT_COLOR
            if line.startswith("  JAB"):
                color = MOVE_DATABASE["jab"].color
            elif line.startswith("  CROSS"):
                color = MOVE_DATABASE["cross"].color
            elif line.startswith("  HOOK"):
                color = MOVE_DATABASE["hook"].color
            elif line.startswith("  UPPERCUT"):
                color = MOVE_DATABASE["uppercut"].color
            elif "YOUR" in line:
                color = WARNING
            surf = fonts["small"].render(line, True, color)
            screen.blit(surf, (170, y))
            y += 20

        hint = fonts["hint"].render("Press ENTER to start fighting!", True, DIM_TEXT)
        screen.blit(hint, (self.w // 2 - hint.get_width() // 2, self.h - 50))

    def draw_move_overview(self, screen: pygame.Surface):
        """Draw a quick-reference card showing all four moves at once."""
        screen.fill(BG_COLOR)
        fonts = self._get_fonts()

        title = fonts["title"].render("MOVE REFERENCE", True, ACCENT)
        screen.blit(title, (self.w // 2 - title.get_width() // 2, 20))

        card_w = (self.w - 120) // 2
        card_h = 280
        positions = [
            (40, 70),
            (40 + card_w + 40, 70),
            (40, 70 + card_h + 20),
            (40 + card_w + 40, 70 + card_h + 20),
        ]

        for i, move_name in enumerate(MOVE_ORDER):
            move = MOVE_DATABASE[move_name]
            cx, cy = positions[i]
            self._draw_panel(screen, cx, cy, card_w, card_h, move.color)

            # Move name
            name_surf = fonts["subtitle"].render(move.display_name, True, move.color)
            screen.blit(name_surf, (cx + 15, cy + 10))

            # Description (wrapped)
            desc_lines = self._wrap_text(move.description, fonts["small"], card_w - 30)
            dy = cy + 40
            for dl in desc_lines[:3]:
                surf = fonts["small"].render(dl, True, TEXT_COLOR)
                screen.blit(surf, (cx + 15, dy))
                dy += 16

            # How to (first 3 steps)
            dy += 8
            how_label = fonts["small"].render("Quick guide:", True, WARNING)
            screen.blit(how_label, (cx + 15, dy))
            dy += 18
            for step in move.how_to[:3]:
                step_lines = self._wrap_text(step, fonts["small"], card_w - 40)
                for sl in step_lines[:2]:
                    surf = fonts["small"].render(sl, True, DIM_TEXT)
                    screen.blit(surf, (cx + 20, dy))
                    dy += 16

            # Key body parts
            dy += 5
            parts = ", ".join(move.key_body_parts)
            parts_surf = fonts["small"].render(f"Key: {parts}", True, move.color)
            screen.blit(parts_surf, (cx + 15, min(dy, cy + card_h - 25)))

        hint = fonts["hint"].render(
            "Press ESC to close  |  Press 1-4 for detailed move info", True, DIM_TEXT
        )
        screen.blit(hint, (self.w // 2 - hint.get_width() // 2, self.h - 30))
