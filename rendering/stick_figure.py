"""
Side-view stick figure renderer using Pygame.
Draws a fighting game character from game-space keypoints.
"""

import pygame
from typing import Optional
from core.coordinate_transformer import GamePose, GameKeypoint


# Color constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PLAYER_COLOR = (50, 120, 255)  # blue
NPC_COLOR = (255, 70, 70)      # red
HEAD_COLOR_PLAYER = (70, 140, 255)
HEAD_COLOR_NPC = (255, 90, 90)
OUTLINE_COLOR = (30, 30, 30)


# Stick figure bone connections: (start_keypoint, end_keypoint)
BODY_BONES = [
    # Neck
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Left arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # Right arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Legs
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


class StickFigureRenderer:
    """Renders a stick figure character from game-space keypoints."""

    def __init__(
        self,
        color: tuple[int, int, int] = PLAYER_COLOR,
        head_color: tuple[int, int, int] = HEAD_COLOR_PLAYER,
        line_width: int = 4,
        head_radius: int = 18,
        joint_radius: int = 5,
        fist_radius: int = 8,
    ):
        self.color = color
        self.head_color = head_color
        self.line_width = line_width
        self.head_radius = head_radius
        self.joint_radius = joint_radius
        self.fist_radius = fist_radius
        self._bones = BODY_BONES

    def draw(self, surface: pygame.Surface, game_pose: GamePose):
        """Draw the stick figure on the given surface."""
        if not game_pose.valid:
            return

        kps = game_pose.keypoints

        # Draw bones (body lines)
        for start_name, end_name in self._bones:
            start = kps.get(start_name)
            end = kps.get(end_name)
            if start and end:
                p1 = (int(start.game_x), int(start.game_y))
                p2 = (int(end.game_x), int(end.game_y))
                # Outline
                pygame.draw.line(surface, OUTLINE_COLOR, p1, p2, self.line_width + 2)
                # Main line
                pygame.draw.line(surface, self.color, p1, p2, self.line_width)

        # Draw joints
        joint_names = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]

        for name in joint_names:
            kp = kps.get(name)
            if kp:
                pos = (int(kp.game_x), int(kp.game_y))
                pygame.draw.circle(surface, OUTLINE_COLOR, pos, self.joint_radius + 1)
                pygame.draw.circle(surface, self.color, pos, self.joint_radius)

        # Draw fists (wrists rendered as larger circles)
        for wrist_name in ["left_wrist", "right_wrist"]:
            kp = kps.get(wrist_name)
            if kp:
                pos = (int(kp.game_x), int(kp.game_y))
                pygame.draw.circle(surface, OUTLINE_COLOR, pos, self.fist_radius + 1)
                pygame.draw.circle(surface, self.color, pos, self.fist_radius)

        # Draw head
        nose = kps.get("nose")
        if nose:
            head_pos = (int(nose.game_x), int(nose.game_y))
            pygame.draw.circle(surface, OUTLINE_COLOR, head_pos, self.head_radius + 2)
            pygame.draw.circle(surface, self.head_color, head_pos, self.head_radius)

            # Simple face features (eyes, mouth based on facing direction)
            if game_pose.facing_right:
                eye_offset_x = 5
            else:
                eye_offset_x = -5

            # Eyes
            eye_y = head_pos[1] - 3
            pygame.draw.circle(
                surface, BLACK,
                (head_pos[0] + eye_offset_x, eye_y),
                3,
            )

    def draw_ground_shadow(self, surface: pygame.Surface, game_pose: GamePose, ground_y: int = 580):
        """Draw a simple ground shadow ellipse."""
        if not game_pose.valid:
            return

        # Use hip midpoint for shadow center
        lh = game_pose.keypoints.get("left_hip")
        rh = game_pose.keypoints.get("right_hip")
        if lh and rh:
            center_x = int((lh.game_x + rh.game_x) / 2)
            shadow_surface = pygame.Surface((60, 12), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surface, (0, 0, 0, 40), (0, 0, 60, 12))
            surface.blit(shadow_surface, (center_x - 30, ground_y))
