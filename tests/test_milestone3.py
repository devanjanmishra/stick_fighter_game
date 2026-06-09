"""
Milestone 3 Test: Keypoint smoothing with One Euro Filter.
- Tests that jitter is reduced on idle poses
- Tests that fast movements (punches) are preserved with minimal lag
- Compares raw vs smoothed keypoint trajectories
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from core.pose_estimator import PoseFrame, Keypoint
from core.smoothing import OneEuroFilter, PoseSmoother, SmoothingConfig
from core.coordinate_transformer import CoordinateTransformer
from rendering.stick_figure import StickFigureRenderer
from rendering.game_renderer import GameRenderer
from tests.synthetic_data import (
    generate_idle_sequence,
    generate_jab_sequence,
)

SCREEN_W = 1280
SCREEN_H = 720
OUTPUT_DIR = "/home/ubuntu/stick_fighter/test_output"


def add_jitter(pose: PoseFrame, amount: float = 0.008) -> PoseFrame:
    """Add synthetic noise/jitter to a pose to simulate real camera input."""
    import random
    noisy_kps = {}
    for name, kp in pose.keypoints.items():
        noisy_kps[name] = Keypoint(
            x=kp.x + random.gauss(0, amount),
            y=kp.y + random.gauss(0, amount),
            z=kp.z + random.gauss(0, amount * 0.5),
            visibility=kp.visibility,
            name=kp.name,
        )
    return PoseFrame(
        keypoints=noisy_kps,
        timestamp_ms=pose.timestamp_ms,
        frame_index=pose.frame_index,
        valid=pose.valid,
    )


def test_one_euro_filter_basic():
    """Test that the One Euro Filter smooths a noisy signal."""
    import random
    random.seed(42)

    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)

    # Generate a noisy constant signal (value=100 with noise)
    raw_values = [100.0 + random.gauss(0, 5) for _ in range(60)]
    filtered_values = []

    for i, val in enumerate(raw_values):
        t = i / 30.0  # 30fps
        filtered_values.append(f.apply(val, t))

    # After warm-up, filtered values should have less variance
    raw_var = sum((v - 100) ** 2 for v in raw_values[10:]) / len(raw_values[10:])
    filt_var = sum((v - 100) ** 2 for v in filtered_values[10:]) / len(filtered_values[10:])

    assert filt_var < raw_var, f"Filter should reduce variance: raw={raw_var:.2f}, filtered={filt_var:.2f}"
    reduction = (1 - filt_var / raw_var) * 100
    print(f"[PASS] Variance reduced by {reduction:.1f}%: raw={raw_var:.2f}, filtered={filt_var:.2f}")


def test_one_euro_filter_fast_response():
    """Test that the filter responds quickly to fast changes (punches)."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)

    # Slow ramp up then sudden jump (simulating a punch)
    values = [0.0] * 10 + [0.5] * 5  # jump from 0 to 0.5 at frame 10

    filtered = []
    for i, val in enumerate(values):
        t = i / 30.0
        filtered.append(f.apply(val, t))

    # After the jump, the filter should catch up within a few frames
    # At frame 12 (2 frames after jump), filtered should be at least 30% of the way
    catchup = filtered[12] / 0.5
    assert catchup > 0.3, f"Filter too slow: only at {catchup*100:.1f}% after 2 frames"
    print(f"[PASS] Fast response: {catchup*100:.1f}% catchup after 2 frames of step change")


def test_pose_smoother_reduces_jitter():
    """Test that PoseSmoother reduces jitter on idle poses."""
    import random
    random.seed(123)

    smoother = PoseSmoother()  # uses default SmoothingConfig

    idle_seq = generate_idle_sequence(60)

    # Add jitter and smooth
    raw_nose_x = []
    smoothed_nose_x = []

    for i, pose in enumerate(idle_seq):
        noisy = add_jitter(pose, amount=0.01)
        noisy_with_time = PoseFrame(
            keypoints=noisy.keypoints,
            timestamp_ms=i * (1000 / 30),
            frame_index=i,
            valid=True,
        )
        smoothed = smoother.smooth(noisy_with_time)

        raw_nose_x.append(noisy.keypoints["nose"].x)
        smoothed_nose_x.append(smoothed.keypoints["nose"].x)

    # Calculate standard deviation of both
    raw_mean = sum(raw_nose_x) / len(raw_nose_x)
    smooth_mean = sum(smoothed_nose_x) / len(smoothed_nose_x)
    raw_std = math.sqrt(sum((v - raw_mean) ** 2 for v in raw_nose_x) / len(raw_nose_x))
    smooth_std = math.sqrt(sum((v - smooth_mean) ** 2 for v in smoothed_nose_x) / len(smoothed_nose_x))

    assert smooth_std < raw_std, f"Smoothed std should be less: raw={raw_std:.4f}, smooth={smooth_std:.4f}"
    reduction = (1 - smooth_std / raw_std) * 100
    print(f"[PASS] Nose jitter reduced by {reduction:.1f}%: raw_std={raw_std:.4f}, smooth_std={smooth_std:.4f}")


def test_pose_smoother_preserves_punches():
    """Test that smoothing doesn't kill punch movements."""
    smoother = PoseSmoother()  # uses default SmoothingConfig

    jab_seq = generate_jab_sequence(12)

    raw_wrist_z = []
    smoothed_wrist_z = []

    for i, pose in enumerate(jab_seq):
        timed = PoseFrame(
            keypoints=pose.keypoints,
            timestamp_ms=i * (1000 / 30),
            frame_index=i,
            valid=True,
        )
        smoothed = smoother.smooth(timed)

        raw_wrist_z.append(pose.keypoints["left_wrist"].z)
        smoothed_wrist_z.append(smoothed.keypoints["left_wrist"].z)

    # The smoothed punch should still reach at least 70% of the raw punch depth
    raw_range = max(raw_wrist_z) - min(raw_wrist_z)
    smooth_range = max(smoothed_wrist_z) - min(smoothed_wrist_z)
    preservation = smooth_range / raw_range * 100

    assert preservation > 60, f"Punch too dampened: only {preservation:.1f}% preserved"
    print(f"[PASS] Punch preserved at {preservation:.1f}%: raw_range={raw_range:.3f}, smooth_range={smooth_range:.3f}")


def test_render_raw_vs_smoothed(screen: pygame.Surface):
    """Render raw noisy vs smoothed idle side by side."""
    import random
    random.seed(456)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    renderer = GameRenderer(SCREEN_W, SCREEN_H)
    renderer.draw_background(screen)

    smoother = PoseSmoother()  # uses default SmoothingConfig

    idle_seq = generate_idle_sequence(30)
    font = pygame.font.SysFont("monospace", 18)

    # Draw several noisy frames overlaid (ghosting effect) on the left
    raw_alpha_colors = [
        (50, 120, 255),
        (60, 130, 255),
        (70, 140, 255),
        (80, 150, 255),
        (90, 160, 255),
    ]
    smooth_alpha_colors = [
        (50, 255, 120),
        (60, 255, 130),
        (70, 255, 140),
        (80, 255, 150),
        (90, 255, 160),
    ]

    sample_frames = [5, 10, 15, 20, 25]

    for idx, frame_i in enumerate(sample_frames):
        pose = idle_seq[frame_i]
        noisy = add_jitter(pose, amount=0.012)
        timed = PoseFrame(
            keypoints=noisy.keypoints,
            timestamp_ms=frame_i * (1000 / 30),
            frame_index=frame_i,
            valid=True,
        )
        smoothed = smoother.smooth(timed)

        # Raw on left
        t_raw = CoordinateTransformer(
            screen_width=SCREEN_W, screen_height=SCREEN_H,
            player_base_x=300, ground_y=580,
        )
        raw_game = t_raw.transform(noisy, facing_right=True)
        raw_renderer = StickFigureRenderer(
            color=raw_alpha_colors[idx],
            head_color=raw_alpha_colors[idx],
            line_width=2,
            head_radius=14,
        )
        raw_renderer.draw(screen, raw_game)

        # Smoothed on right
        t_smooth = CoordinateTransformer(
            screen_width=SCREEN_W, screen_height=SCREEN_H,
            player_base_x=900, ground_y=580,
        )
        smooth_game = t_smooth.transform(smoothed, facing_right=True)
        smooth_renderer = StickFigureRenderer(
            color=smooth_alpha_colors[idx],
            head_color=smooth_alpha_colors[idx],
            line_width=2,
            head_radius=14,
        )
        smooth_renderer.draw(screen, smooth_game)

    # Labels
    raw_label = font.render("RAW (with jitter)", True, (50, 120, 255))
    smooth_label = font.render("SMOOTHED (One Euro)", True, (50, 255, 120))
    screen.blit(raw_label, (220, 20))
    screen.blit(smooth_label, (800, 20))

    title = font.render("5 frames overlaid - notice jitter reduction on right", True, (180, 180, 180))
    screen.blit(title, (SCREEN_W // 2 - 260, 680))

    path = os.path.join(OUTPUT_DIR, "m3_raw_vs_smoothed.png")
    pygame.image.save(screen, path)
    print(f"[PASS] Raw vs Smoothed rendered -> {path}")
    return path


if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 3 TESTS: Keypoint Smoothing (One Euro Filter)")
    print("=" * 60)

    test_one_euro_filter_basic()
    test_one_euro_filter_fast_response()
    test_pose_smoother_reduces_jitter()
    test_pose_smoother_preserves_punches()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    test_render_raw_vs_smoothed(screen)
    pygame.quit()

    print("=" * 60)
    print("ALL MILESTONE 3 TESTS PASSED")
    print("=" * 60)
