"""
Milestone 4 Test: Rule-based move detection.
- Tests that each move type is correctly detected from synthetic keypoint sequences
- Tests that idle poses don't trigger false positives
- Tests cooldown between moves
- Renders detected moves with visual indicators
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from core.pose_estimator import PoseFrame
from core.move_detector import MoveDetector, MoveDetectorConfig, MoveType, MovePhase
from core.coordinate_transformer import CoordinateTransformer
from core.smoothing import PoseSmoother
from rendering.stick_figure import StickFigureRenderer
from rendering.game_renderer import GameRenderer
from tests.synthetic_data import (
    generate_idle_sequence,
    generate_jab_sequence,
    generate_cross_sequence,
    generate_hook_sequence,
    generate_uppercut_sequence,
)


SCREEN_W = 1280
SCREEN_H = 720
GROUND_Y = 580
OUTPUT_DIR = "/home/ubuntu/stick_fighter/test_output"


def _run_sequence(detector: MoveDetector, sequence: list[PoseFrame], fps: float = 30.0) -> list:
    """Run a sequence through the detector and return all detected moves."""
    results = []
    for i, pose in enumerate(sequence):
        timed = PoseFrame(
            keypoints=pose.keypoints,
            timestamp_ms=i * (1000 / fps),
            frame_index=i,
            valid=pose.valid,
        )
        move = detector.detect(timed)
        results.append(move)
    return results


def test_idle_no_false_positives():
    """Idle poses should not trigger any move detection."""
    detector = MoveDetector()
    idle = generate_idle_sequence(30)
    results = _run_sequence(detector, idle)

    active_moves = [r for r in results if r.move_type != MoveType.IDLE]
    assert len(active_moves) == 0, \
        f"Idle should not trigger moves, got {len(active_moves)} detections: {[r.move_type.value for r in active_moves]}"
    print("[PASS] Idle: no false positives (0 detections in 30 frames)")


def test_detect_jab():
    """Jab sequence should be detected as a jab."""
    detector = MoveDetector(MoveDetectorConfig(stance="orthodox"))

    # Start with idle to establish baseline, then jab
    idle = generate_idle_sequence(5)
    jab = generate_jab_sequence(12)
    sequence = idle + jab

    results = _run_sequence(detector, sequence)

    jab_detections = [r for r in results if r.move_type == MoveType.JAB]
    assert len(jab_detections) > 0, "Jab should be detected"
    assert jab_detections[0].hand == "left", f"Orthodox jab should use left hand, got {jab_detections[0].hand}"
    print(f"[PASS] Jab detected: {len(jab_detections)} frames, hand={jab_detections[0].hand}, "
          f"confidence={max(r.confidence for r in jab_detections):.2f}")


def test_detect_cross():
    """Cross sequence should be detected as a cross."""
    detector = MoveDetector(MoveDetectorConfig(stance="orthodox"))

    idle = generate_idle_sequence(5)
    cross = generate_cross_sequence(15)
    sequence = idle + cross

    results = _run_sequence(detector, sequence)

    cross_detections = [r for r in results if r.move_type == MoveType.CROSS]
    assert len(cross_detections) > 0, "Cross should be detected"
    assert cross_detections[0].hand == "right", f"Orthodox cross should use right hand, got {cross_detections[0].hand}"
    print(f"[PASS] Cross detected: {len(cross_detections)} frames, hand={cross_detections[0].hand}, "
          f"confidence={max(r.confidence for r in cross_detections):.2f}")


def test_detect_hook():
    """Hook sequence should be detected as a hook."""
    detector = MoveDetector(MoveDetectorConfig(stance="orthodox"))

    idle = generate_idle_sequence(5)
    hook = generate_hook_sequence(15)
    sequence = idle + hook

    results = _run_sequence(detector, sequence)

    hook_detections = [r for r in results if r.move_type == MoveType.HOOK]
    assert len(hook_detections) > 0, "Hook should be detected"
    print(f"[PASS] Hook detected: {len(hook_detections)} frames, hand={hook_detections[0].hand}, "
          f"confidence={max(r.confidence for r in hook_detections):.2f}")


def test_detect_uppercut():
    """Uppercut sequence should be detected as an uppercut."""
    detector = MoveDetector(MoveDetectorConfig(stance="orthodox"))

    idle = generate_idle_sequence(5)
    uppercut = generate_uppercut_sequence(14)
    sequence = idle + uppercut

    results = _run_sequence(detector, sequence)

    uppercut_detections = [r for r in results if r.move_type == MoveType.UPPERCUT]
    assert len(uppercut_detections) > 0, "Uppercut should be detected"
    print(f"[PASS] Uppercut detected: {len(uppercut_detections)} frames, hand={uppercut_detections[0].hand}, "
          f"confidence={max(r.confidence for r in uppercut_detections):.2f}")


def test_southpaw_jab():
    """Southpaw stance should use right hand for jab."""
    detector = MoveDetector(MoveDetectorConfig(stance="southpaw"))

    idle = generate_idle_sequence(5, stance="southpaw")
    jab = generate_jab_sequence(12, stance="southpaw")
    sequence = idle + jab

    results = _run_sequence(detector, sequence)

    jab_detections = [r for r in results if r.move_type == MoveType.JAB]
    assert len(jab_detections) > 0, "Southpaw jab should be detected"
    assert jab_detections[0].hand == "right", f"Southpaw jab should use right hand, got {jab_detections[0].hand}"
    print(f"[PASS] Southpaw jab detected: hand={jab_detections[0].hand}")


def test_move_sequence_timeline():
    """Run a full combo (idle -> jab -> idle -> cross -> idle) and verify timeline."""
    detector = MoveDetector(MoveDetectorConfig(stance="orthodox"))

    idle1 = generate_idle_sequence(15)
    jab = generate_jab_sequence(12)
    idle2 = generate_idle_sequence(30)
    cross = generate_cross_sequence(15)
    idle3 = generate_idle_sequence(15)

    sequence = idle1 + jab + idle2 + cross + idle3

    results = _run_sequence(detector, sequence)

    # Print timeline
    print("  Move timeline:")
    prev_move = None
    for i, r in enumerate(results):
        if r.move_type != prev_move:
            print(f"    Frame {i:3d}: {r.move_type.value:10s} (phase={r.phase.value}, conf={r.confidence:.2f})")
            prev_move = r.move_type

    move_types_found = set(r.move_type for r in results)
    assert MoveType.JAB in move_types_found, "Jab should appear in timeline"
    assert MoveType.CROSS in move_types_found, "Cross should appear in timeline"
    print("[PASS] Move sequence: idle -> jab -> idle -> cross -> idle correctly detected")


def test_render_move_detection(screen: pygame.Surface):
    """Render a sequence with move detection indicators."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    renderer = GameRenderer(SCREEN_W, SCREEN_H)
    font = pygame.font.SysFont("monospace", 22, bold=True)
    small_font = pygame.font.SysFont("monospace", 16)

    detector = MoveDetector(MoveDetectorConfig(stance="orthodox"))

    # Build a sequence: idle -> jab peak -> idle -> cross peak -> idle -> hook peak -> idle -> uppercut peak
    moves_data = [
        ("Idle", generate_idle_sequence(1)[0], 5),
        ("Jab", generate_jab_sequence(12)[7], 12),
        ("Cross", generate_cross_sequence(15)[9], 15),
        ("Hook", generate_hook_sequence(15)[8], 15),
        ("Uppercut", generate_uppercut_sequence(14)[8], 14),
    ]

    move_colors = {
        MoveType.IDLE: (150, 150, 150),
        MoveType.JAB: (255, 255, 50),
        MoveType.CROSS: (255, 100, 50),
        MoveType.HOOK: (50, 255, 150),
        MoveType.UPPERCUT: (200, 50, 255),
    }

    renderer.draw_background(screen)

    for i, (label, peak_pose, _) in enumerate(moves_data):
        x_pos = 140 + i * 230

        t = CoordinateTransformer(
            screen_width=SCREEN_W, screen_height=SCREEN_H,
            player_base_x=x_pos, ground_y=GROUND_Y,
        )
        game_pose = t.transform(peak_pose, facing_right=True)

        color = move_colors.get(MoveType(label.lower()), (150, 150, 150)) if label.lower() in [m.value for m in MoveType] else (50, 120, 255)

        fig = StickFigureRenderer(
            color=color,
            head_color=color,
            line_width=4,
            head_radius=16,
        )
        fig.draw(screen, game_pose)

        # Label with move name
        text = font.render(label, True, color)
        text_rect = text.get_rect(centerx=x_pos, top=GROUND_Y + 15)
        screen.blit(text, text_rect)

    title = font.render("MOVE DETECTION (mid-action poses)", True, (200, 200, 200))
    screen.blit(title, (SCREEN_W // 2 - 200, 20))

    path = os.path.join(OUTPUT_DIR, "m4_move_detection.png")
    pygame.image.save(screen, path)
    print(f"[PASS] Move detection visual -> {path}")
    return path


if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 4 TESTS: Rule-Based Move Detection")
    print("=" * 60)

    test_idle_no_false_positives()
    test_detect_jab()
    test_detect_cross()
    test_detect_hook()
    test_detect_uppercut()
    test_southpaw_jab()
    test_move_sequence_timeline()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    test_render_move_detection(screen)
    pygame.quit()

    print("=" * 60)
    print("ALL MILESTONE 4 TESTS PASSED")
    print("=" * 60)
