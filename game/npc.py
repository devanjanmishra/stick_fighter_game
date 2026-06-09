"""
NPC (Non-Player Character) with scripted AI for fighting.

The NPC has the same stick figure representation as the player but is
controlled by a simple behavior tree: approach → attack → recover → repeat.
It can throw jabs, crosses, hooks, and uppercuts with configurable timing.

Hitbox collision is rectangle-based: each limb segment generates a hitbox
during active attack frames.
"""

import random
from enum import Enum
from dataclasses import dataclass, field
from core.pose_estimator import Keypoint
from core.coordinate_transformer import GameKeypoint, GamePose


class NPCState(Enum):
    IDLE = "idle"
    APPROACH = "approach"
    ATTACK = "attack"
    RECOVER = "recover"
    BLOCK = "block"
    RETREAT = "retreat"


class NPCAttackType(Enum):
    JAB = "jab"
    CROSS = "cross"
    HOOK = "hook"
    UPPERCUT = "uppercut"


@dataclass
class Hitbox:
    """Axis-aligned bounding box for collision detection."""
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def overlaps(self, other: "Hitbox") -> bool:
        return (
            self.left < other.right
            and self.right > other.left
            and self.top < other.bottom
            and self.bottom > other.top
        )


@dataclass
class NPCConfig:
    """Configuration for NPC behavior."""
    walk_speed: float = 2.5
    preferred_distance: float = 120.0
    attack_range: float = 150.0
    retreat_distance: float = 200.0
    attack_cooldown_frames: int = 30
    attack_duration_frames: int = 12
    block_chance: float = 0.2
    block_duration_frames: int = 15
    attack_weights: dict[str, float] = field(default_factory=lambda: {
        "jab": 0.4,
        "cross": 0.3,
        "hook": 0.2,
        "uppercut": 0.1,
    })
    # Damage per attack type
    damage: dict[str, int] = field(default_factory=lambda: {
        "jab": 5,
        "cross": 8,
        "hook": 12,
        "uppercut": 15,
    })


@dataclass
class NPCPose:
    """Simple pose for NPC rendering in game coordinates."""
    head: tuple[float, float] = (0, 0)
    neck: tuple[float, float] = (0, 0)
    left_shoulder: tuple[float, float] = (0, 0)
    right_shoulder: tuple[float, float] = (0, 0)
    left_elbow: tuple[float, float] = (0, 0)
    right_elbow: tuple[float, float] = (0, 0)
    left_wrist: tuple[float, float] = (0, 0)
    right_wrist: tuple[float, float] = (0, 0)
    hip_left: tuple[float, float] = (0, 0)
    hip_right: tuple[float, float] = (0, 0)
    knee_left: tuple[float, float] = (0, 0)
    knee_right: tuple[float, float] = (0, 0)
    ankle_left: tuple[float, float] = (0, 0)
    ankle_right: tuple[float, float] = (0, 0)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "head": self.head,
            "neck": self.neck,
            "left_shoulder": self.left_shoulder,
            "right_shoulder": self.right_shoulder,
            "left_elbow": self.left_elbow,
            "right_elbow": self.right_elbow,
            "left_wrist": self.left_wrist,
            "right_wrist": self.right_wrist,
            "hip_left": self.hip_left,
            "hip_right": self.hip_right,
            "knee_left": self.knee_left,
            "knee_right": self.knee_right,
            "ankle_left": self.ankle_left,
            "ankle_right": self.ankle_right,
        }

    def to_game_pose(self, facing_right: bool = False) -> GamePose:
        """Convert to a GamePose so the same StickFigureRenderer can draw it."""
        def _gk(name: str, xy: tuple[float, float]) -> GameKeypoint:
            return GameKeypoint(game_x=xy[0], game_y=xy[1], depth=0.0, name=name)

        kps: dict[str, GameKeypoint] = {
            "nose": _gk("nose", self.head),
            "left_shoulder": _gk("left_shoulder", self.left_shoulder),
            "right_shoulder": _gk("right_shoulder", self.right_shoulder),
            "left_elbow": _gk("left_elbow", self.left_elbow),
            "right_elbow": _gk("right_elbow", self.right_elbow),
            "left_wrist": _gk("left_wrist", self.left_wrist),
            "right_wrist": _gk("right_wrist", self.right_wrist),
            "left_hip": _gk("left_hip", self.hip_left),
            "right_hip": _gk("right_hip", self.hip_right),
            "left_knee": _gk("left_knee", self.knee_left),
            "right_knee": _gk("right_knee", self.knee_right),
            "left_ankle": _gk("left_ankle", self.ankle_left),
            "right_ankle": _gk("right_ankle", self.ankle_right),
        }
        return GamePose(keypoints=kps, facing_right=facing_right, valid=True)


class NPC:
    """
    NPC fighter with scripted behavior tree AI.

    The NPC faces left (toward the player) by default at game_x position.
    Its pose is generated procedurally based on current state and attack phase.
    """

    # Body proportions (pixels)
    HEAD_RADIUS = 16
    HEAD_TO_SHOULDER = 30
    SHOULDER_WIDTH = 35
    SHOULDER_TO_ELBOW = 40
    ELBOW_TO_WRIST = 35
    SHOULDER_TO_HIP = 70
    HIP_WIDTH = 25
    HIP_TO_KNEE = 50
    KNEE_TO_ANKLE = 45

    def __init__(self, config: NPCConfig | None = None, game_x: float = 900.0, ground_y: float = 580.0):
        self.config = config or NPCConfig()
        self.game_x = game_x
        self.ground_y = ground_y
        self.facing_right = False  # NPC faces left toward player

        self._state = NPCState.IDLE
        self._attack_type: NPCAttackType | None = None
        self._state_timer = 0
        self._cooldown_timer = 0
        self._attack_progress = 0.0  # 0.0 to 1.0 during attack
        self._frame_count = 0

        # Combat state
        self._is_hit = False
        self._hit_timer = 0
        self._blocking = False

    @property
    def state(self) -> NPCState:
        return self._state

    @property
    def attack_type(self) -> NPCAttackType | None:
        return self._attack_type if self._state == NPCState.ATTACK else None

    @property
    def is_attacking(self) -> bool:
        return self._state == NPCState.ATTACK

    @property
    def is_blocking(self) -> bool:
        return self._state == NPCState.BLOCK

    @property
    def attack_progress(self) -> float:
        return self._attack_progress

    def _choose_attack(self) -> NPCAttackType:
        """Weighted random attack selection."""
        weights = self.config.attack_weights
        types = list(weights.keys())
        probs = [weights[t] for t in types]
        total = sum(probs)
        probs = [p / total for p in probs]

        r = random.random()
        cumulative = 0.0
        for t, p in zip(types, probs):
            cumulative += p
            if r <= cumulative:
                return NPCAttackType(t)
        return NPCAttackType.JAB

    def update(self, player_x: float, player_attacking: bool = False) -> NPCState:
        """
        Update NPC behavior based on player position and state.
        Call once per frame.
        """
        self._frame_count += 1
        distance = abs(self.game_x - player_x)

        # Handle hit stun
        if self._is_hit:
            self._hit_timer -= 1
            if self._hit_timer <= 0:
                self._is_hit = False
            return self._state

        # Cooldown
        if self._cooldown_timer > 0:
            self._cooldown_timer -= 1

        # State machine
        if self._state == NPCState.IDLE:
            if distance > self.config.preferred_distance + 30:
                self._state = NPCState.APPROACH
            elif distance < self.config.attack_range and self._cooldown_timer <= 0:
                # Decide: attack or block
                if player_attacking and random.random() < self.config.block_chance:
                    self._state = NPCState.BLOCK
                    self._state_timer = self.config.block_duration_frames
                else:
                    self._start_attack()
            self._state_timer = 0

        elif self._state == NPCState.APPROACH:
            direction = -1 if self.game_x > player_x else 1
            self.game_x += direction * self.config.walk_speed
            if distance <= self.config.preferred_distance:
                self._state = NPCState.IDLE

        elif self._state == NPCState.ATTACK:
            self._state_timer += 1
            self._attack_progress = self._state_timer / self.config.attack_duration_frames
            if self._state_timer >= self.config.attack_duration_frames:
                self._state = NPCState.RECOVER
                self._state_timer = 0
                self._cooldown_timer = self.config.attack_cooldown_frames
                self._attack_progress = 0.0

        elif self._state == NPCState.RECOVER:
            self._state_timer += 1
            if self._state_timer >= 10:
                self._state = NPCState.IDLE
                self._state_timer = 0

        elif self._state == NPCState.BLOCK:
            self._state_timer -= 1
            self._blocking = True
            if self._state_timer <= 0:
                self._state = NPCState.IDLE
                self._blocking = False

        elif self._state == NPCState.RETREAT:
            direction = 1 if self.game_x > player_x else -1
            self.game_x += direction * self.config.walk_speed
            self._state_timer += 1
            if self._state_timer >= 20 or distance > self.config.retreat_distance:
                self._state = NPCState.IDLE
                self._state_timer = 0

        return self._state

    def _start_attack(self):
        self._state = NPCState.ATTACK
        self._attack_type = self._choose_attack()
        self._state_timer = 0
        self._attack_progress = 0.0

    def receive_hit(self, damage: int = 0):
        """Called when the NPC is hit by the player."""
        self._is_hit = True
        self._hit_timer = 10
        # Push back
        push_dir = 1 if not self.facing_right else -1
        self.game_x += push_dir * 15
        # Cancel current attack
        if self._state == NPCState.ATTACK:
            self._state = NPCState.RECOVER
            self._state_timer = 0
            self._attack_progress = 0.0

    def get_pose(self) -> NPCPose:
        """Generate the NPC's current pose in game coordinates."""
        x = self.game_x
        y = self.ground_y

        # Base skeleton (facing left)
        sign = -1 if not self.facing_right else 1

        ankle_y = y
        knee_y = y - self.KNEE_TO_ANKLE
        hip_y = knee_y - self.HIP_TO_KNEE
        shoulder_y = hip_y - self.SHOULDER_TO_HIP
        neck_y = shoulder_y - 10
        head_y = shoulder_y - self.HEAD_TO_SHOULDER

        # Hip and shoulder positions (mirrored for facing direction)
        hip_cx = x
        shoulder_cx = x

        # Default guard pose
        l_shoulder = (shoulder_cx - self.SHOULDER_WIDTH // 2, shoulder_y)
        r_shoulder = (shoulder_cx + self.SHOULDER_WIDTH // 2, shoulder_y)

        # Arms in guard position
        l_elbow = (l_shoulder[0] + sign * 15, shoulder_y + 25)
        r_elbow = (r_shoulder[0] + sign * 15, shoulder_y + 25)
        l_wrist = (l_elbow[0] + sign * 10, l_elbow[1] - 20)
        r_wrist = (r_elbow[0] + sign * 10, r_elbow[1] - 20)

        # Apply attack animation
        if self._state == NPCState.ATTACK and self._attack_type:
            l_wrist, r_wrist, l_elbow, r_elbow = self._apply_attack_pose(
                l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, sign
            )

        # Apply block pose
        if self._state == NPCState.BLOCK:
            l_wrist = (l_shoulder[0] + sign * 25, shoulder_y - 10)
            r_wrist = (r_shoulder[0] + sign * 25, shoulder_y + 10)
            l_elbow = (l_shoulder[0] + sign * 15, shoulder_y + 5)
            r_elbow = (r_shoulder[0] + sign * 15, shoulder_y + 15)

        # Legs
        hip_l = (hip_cx - self.HIP_WIDTH // 2, hip_y)
        hip_r = (hip_cx + self.HIP_WIDTH // 2, hip_y)
        knee_l = (hip_l[0] - 5, knee_y)
        knee_r = (hip_r[0] + 5, knee_y)
        ankle_l = (knee_l[0], ankle_y)
        ankle_r = (knee_r[0], ankle_y)

        return NPCPose(
            head=(x, head_y),
            neck=(x, neck_y),
            left_shoulder=l_shoulder,
            right_shoulder=r_shoulder,
            left_elbow=l_elbow,
            right_elbow=r_elbow,
            left_wrist=l_wrist,
            right_wrist=r_wrist,
            hip_left=hip_l,
            hip_right=hip_r,
            knee_left=knee_l,
            knee_right=knee_r,
            ankle_left=ankle_l,
            ankle_right=ankle_r,
        )

    def _apply_attack_pose(
        self,
        l_shoulder: tuple[float, float],
        r_shoulder: tuple[float, float],
        l_elbow: tuple[float, float],
        r_elbow: tuple[float, float],
        l_wrist: tuple[float, float],
        r_wrist: tuple[float, float],
        sign: int,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Apply attack animation to arm positions."""
        assert self._attack_type is not None
        p = self._attack_progress
        # Smooth ease-out curve
        t = 1.0 - (1.0 - min(p * 2, 1.0)) ** 2  # quick extend
        retract = max(0.0, (p - 0.5) * 2)  # retract in second half

        extend = t * (1.0 - retract)
        reach = 80 * extend

        # Lead arm attacks (lead = left when facing left)
        lead_shoulder = l_shoulder if not self.facing_right else r_shoulder
        rear_shoulder = r_shoulder if not self.facing_right else l_shoulder

        if self._attack_type == NPCAttackType.JAB:
            new_lead_elbow = (lead_shoulder[0] + sign * (20 + reach * 0.5), lead_shoulder[1])
            new_lead_wrist = (lead_shoulder[0] + sign * (30 + reach), lead_shoulder[1])
            if not self.facing_right:
                return new_lead_wrist, r_wrist, new_lead_elbow, r_elbow
            else:
                return l_wrist, new_lead_wrist, l_elbow, new_lead_elbow

        elif self._attack_type == NPCAttackType.CROSS:
            new_rear_elbow = (rear_shoulder[0] + sign * (20 + reach * 0.5), rear_shoulder[1])
            new_rear_wrist = (rear_shoulder[0] + sign * (30 + reach), rear_shoulder[1])
            if not self.facing_right:
                return l_wrist, new_rear_wrist, l_elbow, new_rear_elbow
            else:
                return new_rear_wrist, r_wrist, new_rear_elbow, r_elbow

        elif self._attack_type == NPCAttackType.HOOK:
            arc_x = sign * (20 + reach * 0.8)
            arc_y = -20 * extend
            new_lead_elbow = (lead_shoulder[0] + arc_x * 0.5, lead_shoulder[1] + arc_y * 0.5)
            new_lead_wrist = (lead_shoulder[0] + arc_x, lead_shoulder[1] + arc_y)
            if not self.facing_right:
                return new_lead_wrist, r_wrist, new_lead_elbow, r_elbow
            else:
                return l_wrist, new_lead_wrist, l_elbow, new_lead_elbow

        elif self._attack_type == NPCAttackType.UPPERCUT:
            up_y = -40 * extend
            fwd_x = sign * (15 + reach * 0.4)
            new_rear_elbow = (rear_shoulder[0] + fwd_x * 0.5, rear_shoulder[1] + 10 + up_y * 0.5)
            new_rear_wrist = (rear_shoulder[0] + fwd_x, rear_shoulder[1] + up_y)
            if not self.facing_right:
                return l_wrist, new_rear_wrist, l_elbow, new_rear_elbow
            else:
                return new_rear_wrist, r_wrist, new_rear_elbow, r_elbow

        return l_wrist, r_wrist, l_elbow, r_elbow

    def get_attack_hitbox(self) -> Hitbox | None:
        """Get the active attack hitbox (only valid during attack)."""
        if self._state != NPCState.ATTACK or self._attack_progress < 0.3:
            return None

        pose = self.get_pose()
        # Hitbox around the attacking wrist
        if self._attack_type in (NPCAttackType.JAB, NPCAttackType.HOOK):
            wrist = pose.left_wrist if not self.facing_right else pose.right_wrist
        else:
            wrist = pose.right_wrist if not self.facing_right else pose.left_wrist

        return Hitbox(
            x=wrist[0] - 20,
            y=wrist[1] - 20,
            width=40,
            height=40,
        )

    def get_body_hitbox(self) -> Hitbox:
        """Get the NPC's body hitbox for receiving hits."""
        pose = self.get_pose()
        return Hitbox(
            x=self.game_x - 30,
            y=pose.head[1] - self.HEAD_RADIUS,
            width=60,
            height=self.ground_y - pose.head[1] + self.HEAD_RADIUS,
        )

    def reset(self, game_x: float = 900.0):
        """Reset NPC state for a new round."""
        self.game_x = game_x
        self._state = NPCState.IDLE
        self._attack_type = None
        self._state_timer = 0
        self._cooldown_timer = 0
        self._attack_progress = 0.0
        self._is_hit = False
        self._hit_timer = 0
        self._blocking = False
