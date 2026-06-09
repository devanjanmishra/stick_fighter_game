"""
Combat UI rendering: HP bars, round indicators, timer, and overlays.

Renders the fighting game HUD elements on top of the game scene.
"""

import pygame
from game.combat_system import CombatSystem, GamePhase, RoundResult


class CombatUI:
    """Renders combat HUD elements."""

    # Layout constants
    HP_BAR_WIDTH = 400
    HP_BAR_HEIGHT = 25
    HP_BAR_Y = 30
    HP_BAR_BORDER = 3
    PLAYER_HP_X = 50
    NPC_HP_X = 830  # 1280 - 50 - 400

    TIMER_Y = 20
    ROUND_INDICATOR_Y = 65

    # Colors
    HP_BG = (40, 40, 50)
    HP_BORDER = (200, 200, 200)
    PLAYER_HP_COLOR = (50, 200, 100)
    PLAYER_HP_LOW = (255, 80, 50)
    NPC_HP_COLOR = (255, 80, 80)
    NPC_HP_LOW = (200, 50, 50)
    TIMER_COLOR = (255, 255, 255)
    ROUND_WON_COLOR = (255, 215, 0)
    ROUND_LOST_COLOR = (80, 80, 80)
    OVERLAY_BG = (0, 0, 0, 180)

    def __init__(self, screen_width: int = 1280, screen_height: int = 720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.NPC_HP_X = screen_width - 50 - self.HP_BAR_WIDTH

    def draw(self, screen: pygame.Surface, combat: CombatSystem):
        """Draw all combat UI elements."""
        self._draw_hp_bars(screen, combat)
        self._draw_timer(screen, combat)
        self._draw_round_indicators(screen, combat)
        self._draw_names(screen)

        if combat.phase == GamePhase.COUNTDOWN:
            self._draw_countdown(screen, combat)
        elif combat.phase == GamePhase.ROUND_END:
            self._draw_round_end(screen, combat)
        elif combat.phase == GamePhase.MATCH_END:
            self._draw_match_end(screen, combat)

    def _draw_hp_bars(self, screen: pygame.Surface, combat: CombatSystem):
        """Draw player and NPC HP bars."""
        # Player HP (left side, fills left to right)
        self._draw_hp_bar(
            screen,
            x=self.PLAYER_HP_X,
            y=self.HP_BAR_Y,
            ratio=combat.player.hp_ratio,
            color=self.PLAYER_HP_COLOR if combat.player.hp_ratio > 0.25 else self.PLAYER_HP_LOW,
            fill_right=True,
        )

        # NPC HP (right side, fills right to left)
        self._draw_hp_bar(
            screen,
            x=self.NPC_HP_X,
            y=self.HP_BAR_Y,
            ratio=combat.npc.hp_ratio,
            color=self.NPC_HP_COLOR if combat.npc.hp_ratio > 0.25 else self.NPC_HP_LOW,
            fill_right=False,
        )

    def _draw_hp_bar(
        self, screen: pygame.Surface,
        x: float, y: float, ratio: float,
        color: tuple[int, int, int], fill_right: bool,
    ):
        b = self.HP_BAR_BORDER
        w = self.HP_BAR_WIDTH
        h = self.HP_BAR_HEIGHT

        # Background
        pygame.draw.rect(screen, self.HP_BG, (x, y, w, h))

        # Fill
        fill_w = int(w * ratio)
        if fill_right:
            pygame.draw.rect(screen, color, (x, y, fill_w, h))
        else:
            pygame.draw.rect(screen, color, (x + w - fill_w, y, fill_w, h))

        # Border
        pygame.draw.rect(screen, self.HP_BORDER, (x, y, w, h), b)

        # HP text
        font = pygame.font.SysFont("monospace", 16)
        hp_text = f"{int(ratio * 100)}%"
        text_surf = font.render(hp_text, True, (255, 255, 255))
        text_x = x + w // 2 - text_surf.get_width() // 2
        screen.blit(text_surf, (text_x, y + 3))

    def _draw_timer(self, screen: pygame.Surface, combat: CombatSystem):
        """Draw the round timer in the center top."""
        font = pygame.font.SysFont("monospace", 36, bold=True)
        seconds = combat.round_timer_seconds

        color = self.TIMER_COLOR
        if seconds <= 10:
            color = (255, 80, 50)

        timer_text = f"{seconds}"
        text_surf = font.render(timer_text, True, color)
        x = self.screen_width // 2 - text_surf.get_width() // 2
        screen.blit(text_surf, (x, self.TIMER_Y))

    def _draw_round_indicators(self, screen: pygame.Surface, combat: CombatSystem):
        """Draw round win indicators (circles) for both fighters."""
        max_rounds = combat.config.rounds_to_win
        cx_base_player = self.PLAYER_HP_X + self.HP_BAR_WIDTH // 2
        cx_base_npc = self.NPC_HP_X + self.HP_BAR_WIDTH // 2

        for i in range(max_rounds):
            offset = (i - max_rounds // 2) * 25

            # Player round indicators
            px = cx_base_player + offset
            if i < combat.player.rounds_won:
                pygame.draw.circle(screen, self.ROUND_WON_COLOR, (px, self.ROUND_INDICATOR_Y), 8)
            else:
                pygame.draw.circle(screen, self.ROUND_LOST_COLOR, (px, self.ROUND_INDICATOR_Y), 8)
            pygame.draw.circle(screen, self.HP_BORDER, (px, self.ROUND_INDICATOR_Y), 8, 1)

            # NPC round indicators
            nx = cx_base_npc + offset
            if i < combat.npc.rounds_won:
                pygame.draw.circle(screen, self.ROUND_WON_COLOR, (nx, self.ROUND_INDICATOR_Y), 8)
            else:
                pygame.draw.circle(screen, self.ROUND_LOST_COLOR, (nx, self.ROUND_INDICATOR_Y), 8)
            pygame.draw.circle(screen, self.HP_BORDER, (nx, self.ROUND_INDICATOR_Y), 8, 1)

    def _draw_names(self, screen: pygame.Surface):
        """Draw player and NPC names."""
        font = pygame.font.SysFont("monospace", 18, bold=True)

        player_name = font.render("PLAYER", True, (50, 200, 255))
        screen.blit(player_name, (self.PLAYER_HP_X, self.HP_BAR_Y - 22))

        npc_name = font.render("NPC", True, (255, 80, 80))
        screen.blit(npc_name, (self.NPC_HP_X + self.HP_BAR_WIDTH - npc_name.get_width(), self.HP_BAR_Y - 22))

    def _draw_countdown(self, screen: pygame.Surface, combat: CombatSystem):
        """Draw the 3-2-1-FIGHT countdown overlay."""
        value = combat.countdown_value

        font = pygame.font.SysFont("monospace", 120, bold=True)
        if value > 0:
            text = str(value)
            color = (255, 255, 100)
        else:
            text = "FIGHT!"
            color = (255, 50, 50)

        text_surf = font.render(text, True, color)
        x = self.screen_width // 2 - text_surf.get_width() // 2
        y = self.screen_height // 2 - text_surf.get_height() // 2 - 50
        screen.blit(text_surf, (x, y))

        # Round number
        round_font = pygame.font.SysFont("monospace", 30)
        round_text = round_font.render(f"Round {combat.current_round}", True, (200, 200, 200))
        rx = self.screen_width // 2 - round_text.get_width() // 2
        screen.blit(round_text, (rx, y - 50))

    def _draw_round_end(self, screen: pygame.Surface, combat: CombatSystem):
        """Draw round end overlay."""
        if not combat.round_results:
            return

        last_result = combat.round_results[-1]

        font = pygame.font.SysFont("monospace", 60, bold=True)
        if last_result == RoundResult.PLAYER_WIN:
            text = "K.O.!"
            color = (50, 255, 100)
        elif last_result == RoundResult.NPC_WIN:
            text = "K.O.!"
            color = (255, 50, 50)
        else:
            text = "TIME UP"
            color = (255, 255, 100)

        text_surf = font.render(text, True, color)
        x = self.screen_width // 2 - text_surf.get_width() // 2
        y = self.screen_height // 2 - text_surf.get_height() // 2 - 30
        screen.blit(text_surf, (x, y))

    def _draw_match_end(self, screen: pygame.Surface, combat: CombatSystem):
        """Draw match end screen."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        font = pygame.font.SysFont("monospace", 72, bold=True)
        sub_font = pygame.font.SysFont("monospace", 30)

        if combat.match_result == RoundResult.PLAYER_WIN:
            text = "YOU WIN!"
            color = (50, 255, 100)
        elif combat.match_result == RoundResult.NPC_WIN:
            text = "YOU LOSE"
            color = (255, 50, 50)
        else:
            text = "DRAW"
            color = (255, 255, 100)

        text_surf = font.render(text, True, color)
        x = self.screen_width // 2 - text_surf.get_width() // 2
        y = self.screen_height // 2 - text_surf.get_height() // 2 - 40
        screen.blit(text_surf, (x, y))

        # Score
        score_text = f"Score: {combat.player.rounds_won} - {combat.npc.rounds_won}"
        score_surf = sub_font.render(score_text, True, (200, 200, 200))
        sx = self.screen_width // 2 - score_surf.get_width() // 2
        screen.blit(score_surf, (sx, y + 80))
