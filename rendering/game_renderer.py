"""
Main game renderer that composites the scene:
- Background/arena
- Stick figures (player + NPC)
- UI elements (HP bars, round info, etc.)
"""

import pygame
from typing import Optional
from core.coordinate_transformer import GamePose
from rendering.stick_figure import StickFigureRenderer, PLAYER_COLOR, NPC_COLOR, HEAD_COLOR_PLAYER, HEAD_COLOR_NPC


# Arena colors
BG_COLOR = (25, 25, 35)
GROUND_COLOR = (45, 45, 55)
GROUND_LINE_COLOR = (60, 60, 70)


class GameRenderer:
    """Composites the full game scene."""

    def __init__(self, screen_width: int = 1280, screen_height: int = 720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.ground_y = 580

        self.player_renderer = StickFigureRenderer(
            color=PLAYER_COLOR,
            head_color=HEAD_COLOR_PLAYER,
            line_width=4,
            head_radius=18,
        )
        self.npc_renderer = StickFigureRenderer(
            color=NPC_COLOR,
            head_color=HEAD_COLOR_NPC,
            line_width=4,
            head_radius=18,
        )

    def draw_background(self, surface: pygame.Surface):
        """Draw the arena background."""
        surface.fill(BG_COLOR)

        # Ground plane
        pygame.draw.rect(
            surface, GROUND_COLOR,
            (0, self.ground_y, self.screen_width, self.screen_height - self.ground_y),
        )

        # Ground line
        pygame.draw.line(
            surface, GROUND_LINE_COLOR,
            (0, self.ground_y), (self.screen_width, self.ground_y), 2,
        )

        # Subtle grid lines on ground for depth perception
        for i in range(0, self.screen_width, 80):
            alpha = max(30, 60 - abs(i - self.screen_width // 2) // 10)
            pygame.draw.line(
                surface, (40, 40, 50),
                (i, self.ground_y), (i, self.screen_height), 1,
            )

    def draw_scene(
        self,
        surface: pygame.Surface,
        player_pose: Optional[GamePose] = None,
        npc_pose: Optional[GamePose] = None,
    ):
        """Draw the complete scene."""
        self.draw_background(surface)

        # Draw shadows first
        if player_pose:
            self.player_renderer.draw_ground_shadow(surface, player_pose, self.ground_y)
        if npc_pose:
            self.npc_renderer.draw_ground_shadow(surface, npc_pose, self.ground_y)

        # Draw stick figures
        if player_pose:
            self.player_renderer.draw(surface, player_pose)
        if npc_pose:
            self.npc_renderer.draw(surface, npc_pose)

    def draw_debug_info(self, surface: pygame.Surface, info: dict, font: pygame.font.Font):
        """Draw debug information overlay."""
        y = 10
        for key, value in info.items():
            text = font.render(f"{key}: {value}", True, (180, 180, 180))
            surface.blit(text, (10, y))
            y += 20
