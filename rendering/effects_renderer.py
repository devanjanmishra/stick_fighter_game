"""
Effects renderer: draws particles, damage numbers, combo labels, and hit flash.
"""

import pygame
from game.effects import EffectsManager
from game.combo_tracker import ComboTracker


class EffectsRenderer:
    """Renders visual combat effects on screen."""

    def __init__(self, screen_width: int = 1280, screen_height: int = 720):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def draw(self, screen: pygame.Surface, effects: EffectsManager,
             combo: ComboTracker | None = None):
        """Draw all active effects."""
        self._draw_hit_flash(screen, effects)
        self._draw_particles(screen, effects)
        self._draw_damage_numbers(screen, effects)
        if combo and combo.state.active and combo.state.count >= 2:
            self._draw_combo_label(screen, combo)

    def apply_screen_shake(self, effects: EffectsManager) -> tuple[int, int]:
        """Get screen shake offset to apply to rendering."""
        return effects.screen_shake.offset

    def _draw_hit_flash(self, screen: pygame.Surface, effects: EffectsManager):
        """Draw full-screen hit flash overlay."""
        alpha = effects.hit_flash.current_alpha
        if alpha <= 0:
            return
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        r, g, b = effects.hit_flash.color
        overlay.fill((r, g, b, alpha))
        screen.blit(overlay, (0, 0))

    def _draw_particles(self, screen: pygame.Surface, effects: EffectsManager):
        """Draw hit spark particles."""
        for p in effects.particles:
            alpha = int(255 * p.alpha)
            size = max(1, int(p.size))
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            r, g, b = p.color
            pygame.draw.circle(surf, (r, g, b, alpha), (size, size), size)
            screen.blit(surf, (int(p.x) - size, int(p.y) - size))

    def _draw_damage_numbers(self, screen: pygame.Surface, effects: EffectsManager):
        """Draw floating damage numbers."""
        font = pygame.font.SysFont("monospace", 24, bold=True)
        for dn in effects.damage_numbers:
            alpha = int(255 * dn.alpha)
            r, g, b = dn.color
            text_surf = font.render(str(dn.value), True, (r, g, b))
            # Apply alpha via a temp surface
            temp = pygame.Surface(text_surf.get_size(), pygame.SRCALPHA)
            temp.blit(text_surf, (0, 0))
            temp.set_alpha(alpha)
            screen.blit(temp, (int(dn.x), int(dn.y)))

    def _draw_combo_label(self, screen: pygame.Surface, combo: ComboTracker):
        """Draw combo counter and label."""
        label = combo.state.label
        count = combo.state.count
        if not label:
            return

        # Combo count
        count_font = pygame.font.SysFont("monospace", 48, bold=True)
        label_font = pygame.font.SysFont("monospace", 28, bold=True)

        # Color scales with combo length
        if count <= 2:
            color = (255, 255, 100)
        elif count <= 3:
            color = (255, 180, 50)
        elif count <= 4:
            color = (255, 100, 30)
        else:
            color = (255, 50, 50)

        count_text = f"{count} HIT"
        count_surf = count_font.render(count_text, True, color)
        label_surf = label_font.render(label, True, color)

        # Position on left side of screen
        x = 60
        y = 150

        screen.blit(count_surf, (x, y))
        screen.blit(label_surf, (x, y + 50))

        # Multiplier
        mult_font = pygame.font.SysFont("monospace", 20)
        mult_text = f"x{combo.state.multiplier:.1f}"
        mult_surf = mult_font.render(mult_text, True, (200, 200, 200))
        screen.blit(mult_surf, (x, y + 85))
