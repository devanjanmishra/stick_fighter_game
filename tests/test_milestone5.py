"""
Milestone 5 Test: Walking/movement via shoulder tracking.
- Tests calibration phase
- Tests forward/backward walking detection
- Tests dead zone (no movement on small shifts)
- Tests boundary clamping
- Renders walking sequence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from core.pose_estimator import PoseFrame, Keypoint
from core.movement_tracker import MovementTracker, MovementConfig
from core.coordinate_transformer import CoordinateTransformer
from rendering.stick_figure import StickFigureRenderer
from rendering.game_renderer import GameRenderer
from tests.synthetic_data import generate_idle_pose, generate_walking_sequence

SCREEN_W = 1280
SCREEN_H = 720
GROUND_Y = 580
OUTPUT_DIR = "/home/ubuntu/stick_fighter/test_output"


def _make_pose_with_shoulder_x(shoulder_x: float, frame_idx: int = 0) -> PoseFrame:
    """Create a pose with shoulders at a specific x position."""
    base = generate_idle_pose(frame_idx)
    kps = dict(base.keypoints)
    # Shift both shoulders to set midpoint at shoulder_x
    ls = kps["left_shoulder"]
    rs = kps["right_shoulder"]
    mid_x = (ls.x + rs.x) / 2.0
    shift = shoulder_x - mid_x

    kps["left_shoulder"] = Keypoint(
        x=ls.x + shift, y=ls.y, z=ls.z, visibility=ls.visibility, name=ls.name,
    )
    kps["right_shoulder"] = Keypoint(
        x=rs.x + shift, y=rs.y, z=rs.z, visibility=rs.visibility, name=rs.name,
    )
    return PoseFrame(keypoints=kps, timestamp_ms=frame_idx * 33.3, frame_index=frame_idx, valid=True)


def test_calibration():
    """Tracker should calibrate baseline from first N frames."""
    tracker = MovementTracker(MovementConfig(calibration_frames=10))

    # Feed 10 idle frames at shoulder_x=0.5
    for i in range(10):
        pose = _make_pose_with_shoulder_x(0.5, i)
        state = tracker.update(pose)

    assert tracker.is_calibrated, "Should be calibrated after 10 frames"
    assert not state.is_walking, "Should not be walking at baseline"
    print(f"[PASS] Calibration: baseline set after {10} frames, game_x={state.game_x:.1f}")


def test_dead_zone():
    """Small shoulder shifts within dead zone should not cause walking."""
    tracker = MovementTracker(MovementConfig(calibration_frames=5, dead_zone=0.015))

    # Calibrate at 0.5
    for i in range(5):
        tracker.update(_make_pose_with_shoulder_x(0.5, i))

    # Small shift within dead zone
    for i in range(5, 15):
        state = tracker.update(_make_pose_with_shoulder_x(0.51, i))

    assert not state.is_walking, f"Should be in dead zone, but velocity={state.velocity:.3f}"
    print(f"[PASS] Dead zone: no walking on small shift (0.01), velocity={state.velocity:.3f}")


def test_forward_walking():
    """Leaning right (positive x shift) when facing right = walk forward."""
    tracker = MovementTracker(MovementConfig(calibration_frames=5, dead_zone=0.01))

    # Calibrate at 0.5
    for i in range(5):
        tracker.update(_make_pose_with_shoulder_x(0.5, i))

    initial_x = tracker.state.game_x

    # Shift shoulder right (forward when facing right)
    for i in range(5, 25):
        state = tracker.update(_make_pose_with_shoulder_x(0.56, i), facing_right=True)

    assert state.game_x > initial_x, f"Should move forward: {state.game_x:.1f} > {initial_x:.1f}"
    assert state.is_walking
    assert state.walk_direction == 1
    print(f"[PASS] Forward walk: {initial_x:.1f} -> {state.game_x:.1f} (moved {state.game_x - initial_x:.1f}px)")


def test_backward_walking():
    """Leaning left (negative x shift) when facing right = walk backward."""
    tracker = MovementTracker(MovementConfig(calibration_frames=5, dead_zone=0.01))

    # Calibrate at 0.5
    for i in range(5):
        tracker.update(_make_pose_with_shoulder_x(0.5, i))

    initial_x = tracker.state.game_x

    # Shift shoulder left (backward when facing right)
    for i in range(5, 25):
        state = tracker.update(_make_pose_with_shoulder_x(0.44, i), facing_right=True)

    assert state.game_x < initial_x, f"Should move backward: {state.game_x:.1f} < {initial_x:.1f}"
    assert state.is_walking
    assert state.walk_direction == -1
    print(f"[PASS] Backward walk: {initial_x:.1f} -> {state.game_x:.1f} (moved {state.game_x - initial_x:.1f}px)")


def test_boundary_clamping():
    """Player should not walk past game boundaries."""
    tracker = MovementTracker(MovementConfig(
        calibration_frames=5, dead_zone=0.01,
        min_x=50.0, max_x=1230.0,
    ), initial_x=100.0)

    # Calibrate
    for i in range(5):
        tracker.update(_make_pose_with_shoulder_x(0.5, i))

    # Walk very far backward
    for i in range(5, 200):
        state = tracker.update(_make_pose_with_shoulder_x(0.35, i), facing_right=True)

    assert state.game_x >= 50.0, f"Should be clamped at min: {state.game_x:.1f}"
    print(f"[PASS] Boundary clamping: position clamped at {state.game_x:.1f} (min=50)")


def test_stop_after_walking():
    """Returning shoulders to baseline should stop walking."""
    tracker = MovementTracker(MovementConfig(calibration_frames=5, dead_zone=0.01))

    # Calibrate
    for i in range(5):
        tracker.update(_make_pose_with_shoulder_x(0.5, i))

    # Walk forward
    for i in range(5, 20):
        tracker.update(_make_pose_with_shoulder_x(0.56, i), facing_right=True)
    assert tracker.state.is_walking

    # Return to baseline
    for i in range(20, 40):
        state = tracker.update(_make_pose_with_shoulder_x(0.5, i), facing_right=True)

    assert not state.is_walking, f"Should stop walking, velocity={state.velocity:.3f}"
    print(f"[PASS] Stop after walking: velocity={state.velocity:.3f}, is_walking={state.is_walking}")


def test_render_walking_sequence(screen: pygame.Surface):
    """Render a walking sequence showing the stick figure at different positions."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    renderer = GameRenderer(SCREEN_W, SCREEN_H)
    renderer.draw_background(screen)

    font = pygame.font.SysFont("monospace", 18)
    tracker = MovementTracker(MovementConfig(calibration_frames=5, dead_zone=0.01))

    # Calibrate
    for i in range(5):
        tracker.update(_make_pose_with_shoulder_x(0.5, i))

    # Generate positions at different shoulder offsets
    positions = []
    shoulder_offsets = [0.5, 0.5, 0.53, 0.56, 0.58, 0.56, 0.53, 0.5, 0.47, 0.44]
    for i, sx in enumerate(shoulder_offsets):
        pose = _make_pose_with_shoulder_x(sx, 5 + i)
        state = tracker.update(pose, facing_right=True)
        positions.append((state.game_x, pose, i))

    # Render key frames
    colors = [
        (50, 100, 255),
        (70, 120, 255),
        (90, 160, 255),
        (110, 200, 255),
        (130, 220, 255),
        (110, 200, 255),
        (90, 160, 255),
        (70, 120, 255),
        (255, 160, 90),
        (255, 100, 50),
    ]

    for game_x, pose, idx in positions:
        t = CoordinateTransformer(
            screen_width=SCREEN_W, screen_height=SCREEN_H,
            player_base_x=game_x, ground_y=GROUND_Y,
        )
        game_pose = t.transform(pose, facing_right=True)
        fig = StickFigureRenderer(
            color=colors[idx % len(colors)],
            head_color=colors[idx % len(colors)],
            line_width=3,
            head_radius=14,
        )
        fig.draw(screen, game_pose)

    # Labels
    title = font.render("WALKING SEQUENCE (shoulder-based movement)", True, (200, 200, 200))
    screen.blit(title, (SCREEN_W // 2 - 250, 20))

    fwd = font.render("Forward ->", True, (100, 180, 255))
    bwd = font.render("<- Backward", True, (255, 130, 70))
    screen.blit(fwd, (500, 680))
    screen.blit(bwd, (100, 680))

    path = os.path.join(OUTPUT_DIR, "m5_walking_sequence.png")
    pygame.image.save(screen, path)
    print(f"[PASS] Walking sequence rendered -> {path}")
    return path


if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 5 TESTS: Walking/Movement via Shoulder Tracking")
    print("=" * 60)

    test_calibration()
    test_dead_zone()
    test_forward_walking()
    test_backward_walking()
    test_boundary_clamping()
    test_stop_after_walking()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    test_render_walking_sequence(screen)
    pygame.quit()

    print("=" * 60)
    print("ALL MILESTONE 5 TESTS PASSED")
    print("=" * 60)
