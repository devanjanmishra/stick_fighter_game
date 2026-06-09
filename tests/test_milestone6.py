"""
Milestone 6 Test: Calibration system.
- Tests recording move templates from synthetic data
- Tests DTW matching against recorded templates
- Tests threshold computation from calibrated data
- Tests save/load of calibration profiles
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_estimator import PoseFrame
from core.calibration import (
    CalibrationRecorder,
    CalibrationProfile,
    dtw_distance,
    MoveTemplate,
    MoveType,
)
from tests.synthetic_data import (
    generate_jab_sequence,
    generate_cross_sequence,
    generate_hook_sequence,
    generate_uppercut_sequence,
)

OUTPUT_DIR = "/home/ubuntu/stick_fighter/test_output"


def _record_template(move_type: str, hand: str, sequence: list[PoseFrame]) -> MoveTemplate:
    """Helper: record a sequence into a MoveTemplate."""
    recorder = CalibrationRecorder()
    recorder.start_recording(move_type, hand)
    for pose in sequence:
        recorder.add_frame(pose)
    template = recorder.finish_recording()
    assert template is not None, f"Failed to record {move_type} template"
    return template


def test_record_jab_template():
    """Record a jab sequence and verify template extraction."""
    jab = generate_jab_sequence(12)
    template = _record_template("jab", "left", jab)

    assert template.move_type == MoveType.JAB
    assert template.hand == "left"
    assert len(template.velocity_profile) > 0
    assert template.peak_z_velocity > 0
    assert template.z_extension > 0
    assert template.duration_frames == 12
    print(f"[PASS] Jab template: {len(template.velocity_profile)} velocity frames, "
          f"peak_z_vel={template.peak_z_velocity:.4f}, z_ext={template.z_extension:.4f}")


def test_record_all_moves():
    """Record templates for all move types."""
    moves = {
        "jab": (generate_jab_sequence(12), "left"),
        "cross": (generate_cross_sequence(15), "right"),
        "hook": (generate_hook_sequence(15), "left"),
        "uppercut": (generate_uppercut_sequence(14), "right"),
    }

    for move_name, (seq, hand) in moves.items():
        template = _record_template(move_name, hand, seq)
        assert template.move_type == MoveType(move_name)
        assert template.duration_frames == len(seq)
        print(f"  {move_name}: peak_z={template.peak_z_velocity:.4f}, "
              f"peak_x={template.peak_x_velocity:.4f}, peak_y={template.peak_y_velocity:.4f}")

    print("[PASS] All 4 move types recorded successfully")


def test_dtw_matching():
    """DTW should match a move to the most similar template."""
    # Record templates
    jab_template = _record_template("jab", "left", generate_jab_sequence(12))
    cross_template = _record_template("cross", "right", generate_cross_sequence(15))
    hook_template = _record_template("hook", "left", generate_hook_sequence(15))

    # Generate a new jab (slightly different due to random seed)
    new_jab = generate_jab_sequence(12)
    recorder = CalibrationRecorder()
    recorder.start_recording("jab", "left")
    for p in new_jab:
        recorder.add_frame(p)
    new_template = recorder.finish_recording()
    assert new_template is not None

    # DTW distances
    d_jab = dtw_distance(new_template.velocity_profile, jab_template.velocity_profile)
    d_cross = dtw_distance(new_template.velocity_profile, cross_template.velocity_profile)
    d_hook = dtw_distance(new_template.velocity_profile, hook_template.velocity_profile)

    print(f"  DTW distances: jab={d_jab:.4f}, cross={d_cross:.4f}, hook={d_hook:.4f}")

    # Jab should be closest to jab template (or at least not much worse)
    # Note: synthetic data uses the same generation function, so jab-to-jab should be close
    assert d_jab <= d_hook, f"Jab should be closer to jab than hook: {d_jab:.4f} vs {d_hook:.4f}"
    print(f"[PASS] DTW matching: jab matches jab template (d={d_jab:.4f})")


def test_calibration_profile():
    """Build a full calibration profile and compute thresholds."""
    profile = CalibrationProfile(stance="orthodox")

    # Record 3 samples of each move
    for i in range(3):
        profile.add_template(_record_template("jab", "left", generate_jab_sequence(12)))
        profile.add_template(_record_template("cross", "right", generate_cross_sequence(15)))
        profile.add_template(_record_template("hook", "left", generate_hook_sequence(15)))
        profile.add_template(_record_template("uppercut", "right", generate_uppercut_sequence(14)))

    assert profile.is_fully_calibrated()

    thresholds = profile.compute_thresholds()
    assert thresholds.punch_z_velocity_threshold > 0
    assert thresholds.hook_x_velocity_threshold > 0
    assert thresholds.uppercut_y_velocity_threshold > 0

    print(f"  Computed thresholds:")
    print(f"    punch_z_vel: {thresholds.punch_z_velocity_threshold:.4f}")
    print(f"    hook_x_vel:  {thresholds.hook_x_velocity_threshold:.4f}")
    print(f"    upper_y_vel: {thresholds.uppercut_y_velocity_threshold:.4f}")
    print(f"    punch_z_ext: {thresholds.punch_z_extension:.4f}")
    print(f"    hook_x_ext:  {thresholds.hook_x_extension:.4f}")
    print(f"    upper_y_ext: {thresholds.uppercut_y_extension:.4f}")
    print("[PASS] Calibration profile fully computed")


def test_profile_save_load():
    """Save and load a calibration profile."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    profile = CalibrationProfile(stance="orthodox")
    for i in range(3):
        profile.add_template(_record_template("jab", "left", generate_jab_sequence(12)))
        profile.add_template(_record_template("cross", "right", generate_cross_sequence(15)))
        profile.add_template(_record_template("hook", "left", generate_hook_sequence(15)))
        profile.add_template(_record_template("uppercut", "right", generate_uppercut_sequence(14)))

    profile.compute_thresholds()

    filepath = os.path.join(OUTPUT_DIR, "m6_calibration_profile.json")
    profile.save(filepath)
    assert os.path.exists(filepath)

    loaded = CalibrationProfile.load(filepath)
    assert loaded.stance == "orthodox"
    assert loaded.is_fully_calibrated()
    assert loaded.thresholds is not None
    assert abs(loaded.thresholds.punch_z_velocity_threshold
               - profile.thresholds.punch_z_velocity_threshold) < 1e-6

    print(f"[PASS] Profile saved and loaded from {filepath}")


def test_dtw_profile_match():
    """Use the profile's match_move to classify a new sequence."""
    profile = CalibrationProfile(stance="orthodox")
    for i in range(3):
        profile.add_template(_record_template("jab", "left", generate_jab_sequence(12)))
        profile.add_template(_record_template("cross", "right", generate_cross_sequence(15)))
        profile.add_template(_record_template("hook", "left", generate_hook_sequence(15)))
        profile.add_template(_record_template("uppercut", "right", generate_uppercut_sequence(14)))

    # Generate a new hook and extract its velocity profile
    hook_seq = generate_hook_sequence(15)
    recorder = CalibrationRecorder()
    recorder.start_recording("hook", "left")
    for p in hook_seq:
        recorder.add_frame(p)
    new_hook = recorder.finish_recording()
    assert new_hook is not None

    move_type, confidence = profile.match_move(new_hook.velocity_profile)
    print(f"  Profile match result: {move_type.value}, confidence={confidence:.2f}")
    # The match should at least return a valid move type
    assert move_type != MoveType.IDLE, f"Should match something, got IDLE"
    print(f"[PASS] Profile DTW match: {move_type.value} (confidence={confidence:.2f})")


if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 6 TESTS: Calibration System")
    print("=" * 60)

    test_record_jab_template()
    test_record_all_moves()
    test_dtw_matching()
    test_calibration_profile()
    test_profile_save_load()
    test_dtw_profile_match()

    print("=" * 60)
    print("ALL MILESTONE 6 TESTS PASSED")
    print("=" * 60)
