"""
Mirror coordinate transformer — normalised body scale.

The stick figure mirrors the player exactly as seen in a camera:
  - Raise your right arm  → right arm on screen goes up
  - Swing left arm left   → left arm on screen goes left
  - Move closer/further   → character stays the same size

Scale is computed from detected body proportions each frame so distance
from the camera doesn't matter. Character is always the same pixel height.

Rendering rules:
  REQUIRED:  both shoulders visible (to anchor position and compute scale)
  FORBIDDEN: fewer than 2 keypoints → return invalid GamePose (nothing drawn)
  Hips: synthesised if absent. Arms: omitted if absent. Legs: always synthesised.
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from core.pose_estimator import PoseFrame, Keypoint


@dataclass
class GameKeypoint:
    game_x: float
    game_y: float
    depth:  float = 0.0
    name:   str   = ""


@dataclass
class GamePose:
    keypoints: dict[str, GameKeypoint] = field(default_factory=dict)
    facing_right: bool = True
    valid: bool = False


class CoordinateTransformer:
    """Normalised-scale true-mirror transformer."""

    # Target torso height in pixels — character always this tall regardless of distance
    TARGET_SHOULDER_TO_HIP = 80    # px
    TARGET_HEAD_RADIUS     = 18    # px

    # Arm segment max lengths (clamped so arms don't stretch unrealistically)
    ARM_UPPER_MAX = 55
    ARM_LOWER_MAX = 55

    # Fixed vertical position: shoulders always this many px above ground
    SHOULDER_ABOVE_GROUND = 180   # px

    # Minimum total keypoints required to draw anything
    MIN_KEYPOINTS = 2

    # Fixed fallback scale used when hips are absent and we can't measure torso
    FALLBACK_SCALE = 280.0        # px per normalised unit — reasonable for ~1m distance

    # Z-depth to screen-x contribution for arm extension.
    # When punching toward camera (wrist_z goes negative relative to shoulder),
    # the fist extends FORWARD in the game world. We visualise this as lateral
    # extension toward the NPC side. Scale of 80px per unit gives clear visible
    # extension without making the arm unrealistically long.
    Z_FORWARD_SCALE = 80.0        # px per normalised z unit

    # Fixed half-width of synthesised hips in pixels.
    # Using a fixed value prevents hips going very wide when person is close to camera.
    SYNTH_HIP_HALF_WIDTH = 30     # px from centre to each hip joint

    def __init__(
        self,
        screen_width:  int   = 1280,
        screen_height: int   = 720,
        player_base_x: float = 300.0,
        ground_y:      float = 560.0,
        # kept for API compat
        z_to_x_scale:  float = 0.0,
        mirror_x_scale: float = 0.0,
        mirror_y_scale: float = 1.0,
    ):
        self.screen_width   = screen_width
        self.screen_height  = screen_height
        self.player_base_x  = player_base_x
        self.ground_y       = ground_y

    def calibrate(self, pose: PoseFrame):
        """No-op — scale computed dynamically each frame."""
        pass

    def transform(self, pose: PoseFrame, facing_right: bool = True) -> GamePose:
        """
        Mirror camera keypoints to a normalised-scale GamePose.

        Returns invalid GamePose when both shoulders absent or total
        keypoints < MIN_KEYPOINTS (prevents fly-off-screen with face-only detection).
        """
        INVALID = GamePose(keypoints={}, facing_right=facing_right, valid=False)

        if not pose.valid:
            return INVALID

        ls_kp = pose.get("left_shoulder")
        rs_kp = pose.get("right_shoulder")
        if not ls_kp or not rs_kp:
            return INVALID

        if len(pose.keypoints) < self.MIN_KEYPOINTS:
            return INVALID

        # ── Anchor: shoulder midpoint ──────────────────────────────────────
        cam_ax = (ls_kp.x + rs_kp.x) / 2   # camera x anchor
        cam_ay = (ls_kp.y + rs_kp.y) / 2   # camera y anchor

        # Screen anchor: shoulders always SHOULDER_ABOVE_GROUND px above ground
        scr_ax = self.player_base_x
        scr_ay = self.ground_y - self.SHOULDER_ABOVE_GROUND

        # ── Dynamic scale from shoulder-to-hip distance ────────────────────
        lh_kp = pose.get("left_hip")
        rh_kp = pose.get("right_hip")
        scale = self.FALLBACK_SCALE  # default if hips absent

        if lh_kp and rh_kp:
            cam_hip_y = (lh_kp.y + rh_kp.y) / 2
            cam_torso = abs(cam_hip_y - cam_ay)
            if cam_torso > 0.01:
                scale = self.TARGET_SHOULDER_TO_HIP / cam_torso

        # Clamp to sane range (very close or partial detection)
        scale = max(120.0, min(scale, 550.0))

        # ── Camera → screen mapping ────────────────────────────────────────
        # TRUE MIRROR: right in camera = right on screen.
        # cam_x > cam_ax  means arm is to the RIGHT of the body → positive dx
        # We add that to scr_ax so it appears on the RIGHT of the stick figure.
        # (The earlier version subtracted dx, which flipped left/right.)
        def to_screen(cam_x: float, cam_y: float) -> tuple[float, float]:
            dx = (cam_x - cam_ax) * scale
            dy = (cam_y - cam_ay) * scale
            return scr_ax + dx, scr_ay + dy   # +dx = true mirror direction

        def clamp_seg(ax, ay, tx, ty, maxlen):
            ddx, ddy = tx - ax, ty - ay
            d = math.hypot(ddx, ddy)
            if d > maxlen and d > 0:
                f = maxlen / d
                return ax + ddx * f, ay + ddy * f
            return tx, ty

        kps: dict[str, GameKeypoint] = {}

        # ── Head ───────────────────────────────────────────────────────────
        nose_kp = pose.get("nose")
        if nose_kp:
            nx, ny = to_screen(nose_kp.x, nose_kp.y)
            kps["nose"] = GameKeypoint(nx, ny, 0.0, "nose")
        else:
            # Synthesise head above shoulders
            kps["nose"] = GameKeypoint(
                scr_ax, scr_ay - self.TARGET_HEAD_RADIUS * 2.5, 0.0, "nose")

        # ── Shoulders ──────────────────────────────────────────────────────
        for name, kp in (("left_shoulder", ls_kp), ("right_shoulder", rs_kp)):
            sx, sy = to_screen(kp.x, kp.y)
            kps[name] = GameKeypoint(sx, sy, kp.x - cam_ax, name)

        # ── Hips ───────────────────────────────────────────────────────────
        if lh_kp and rh_kp:
            for name, kp in (("left_hip", lh_kp), ("right_hip", rh_kp)):
                hx, hy = to_screen(kp.x, kp.y)
                kps[name] = GameKeypoint(hx, hy, kp.x - cam_ax, name)
        else:
            # Synthesise hips directly below shoulder anchor.
            # Use a FIXED pixel width (not proportional to shoulder width) so hips
            # don't go comically wide when person is very close to camera.
            hip_y = scr_ay + self.TARGET_SHOULDER_TO_HIP
            kps["left_hip"]  = GameKeypoint(scr_ax - self.SYNTH_HIP_HALF_WIDTH, hip_y, -0.2, "left_hip")
            kps["right_hip"] = GameKeypoint(scr_ax + self.SYNTH_HIP_HALF_WIDTH, hip_y,  0.2, "right_hip")

        # ── Arms (clamped segments + z-extension for jab/cross) ──────────
        # For hooks/uppercuts: lateral x and vertical y movement is directly
        # visible in camera → to_screen() captures it perfectly.
        # For jab/cross: wrist moves toward camera (z decreases) but x/y barely
        # change. We add a z-contribution so the fist visibly extends forward.
        # rel_z = wrist_z - shoulder_z (negative when arm extends toward camera)
        # extension_px = -rel_z * Z_FORWARD_SCALE  (positive when extending)
        sh_kp_map = {"left": ls_kp, "right": rs_kp}

        for side in ("left", "right"):
            sh_gk = kps.get(f"{side}_shoulder")
            sh_raw = sh_kp_map[side]
            if not sh_gk or not sh_raw:
                continue
            el_kp = pose.get(f"{side}_elbow")
            wr_kp = pose.get(f"{side}_wrist")

            # Direction of z-extension: right arm extends RIGHT, left arm extends LEFT.
            # sign(sh_gk.game_x - scr_ax): +1 for right shoulder, -1 for left shoulder.
            arm_side_sign = 1.0 if sh_gk.game_x >= scr_ax else -1.0

            if el_kp:
                ex, ey = to_screen(el_kp.x, el_kp.y)
                el_rel_z = el_kp.z - sh_raw.z
                # -rel_z: when rel_z negative (arm extending toward camera), offset is positive
                # * arm_side_sign: right arm extends right (+), left arm extends left (-)
                ex += -el_rel_z * self.Z_FORWARD_SCALE * arm_side_sign
                ex, ey = clamp_seg(sh_gk.game_x, sh_gk.game_y, ex, ey, self.ARM_UPPER_MAX)
                kps[f"{side}_elbow"] = GameKeypoint(ex, ey, el_kp.x - cam_ax, f"{side}_elbow")

            if wr_kp:
                wx, wy = to_screen(wr_kp.x, wr_kp.y)
                wr_rel_z = wr_kp.z - sh_raw.z
                wx += -wr_rel_z * self.Z_FORWARD_SCALE * arm_side_sign
                parent = kps.get(f"{side}_elbow") or sh_gk
                maxl   = self.ARM_LOWER_MAX if f"{side}_elbow" in kps else (self.ARM_UPPER_MAX + self.ARM_LOWER_MAX)
                wx, wy = clamp_seg(parent.game_x, parent.game_y, wx, wy, maxl)
                kps[f"{side}_wrist"] = GameKeypoint(wx, wy, wr_kp.x - cam_ax, f"{side}_wrist")

        # ── Legs (synthesised, ankle snapped to ground_y) ─────────────────
        for side, xsign in (("left", -1), ("right", 1)):
            hip = kps.get(f"{side}_hip")
            if hip:
                ankle_y = min(self.ground_y, hip.game_y + 109.9)
                knee_y  = (hip.game_y + ankle_y) / 2
                off = xsign * 8
                kps[f"{side}_knee"]  = GameKeypoint(hip.game_x + off,       knee_y,  hip.depth, f"{side}_knee")
                kps[f"{side}_ankle"] = GameKeypoint(hip.game_x + off * 1.5, ankle_y, hip.depth, f"{side}_ankle")

        return GamePose(keypoints=kps, facing_right=facing_right, valid=True)

    def set_player_position(self, base_x: float, ground_y: Optional[float] = None):
        self.player_base_x = base_x
        if ground_y is not None:
            self.ground_y = ground_y
