"""
Combo tracking system: detects sequential move chains and awards bonus damage.

A combo is a sequence of moves landed within a time window.
Longer combos multiply damage and trigger special effects.
"""

from dataclasses import dataclass, field
import time


@dataclass
class ComboState:
    count: int = 0
    moves: list[str] = field(default_factory=list)
    last_hit_frame: int = -999
    total_bonus_damage: int = 0

    @property
    def active(self) -> bool:
        return self.count > 0

    @property
    def multiplier(self) -> float:
        """Damage multiplier based on combo length."""
        if self.count <= 1:
            return 1.0
        elif self.count == 2:
            return 1.2
        elif self.count == 3:
            return 1.5
        elif self.count == 4:
            return 1.8
        else:
            return 2.0

    @property
    def label(self) -> str:
        if self.count < 2:
            return ""
        elif self.count == 2:
            return "DOUBLE!"
        elif self.count == 3:
            return "TRIPLE!"
        elif self.count == 4:
            return "QUAD!"
        elif self.count >= 5:
            return "ULTRA!"
        return ""

    def reset(self):
        self.count = 0
        self.moves.clear()
        self.last_hit_frame = -999
        self.total_bonus_damage = 0


class ComboTracker:
    """Tracks combos for a single fighter."""

    def __init__(self, combo_window: int = 45):
        """
        Args:
            combo_window: Max frames between hits to continue a combo.
                         At 30fps, 45 frames = 1.5 seconds.
        """
        self.combo_window = combo_window
        self.state = ComboState()
        self._current_frame: int = 0
        self._best_combo: int = 0

    @property
    def best_combo(self) -> int:
        return self._best_combo

    def register_hit(self, move_type: str, base_damage: int) -> tuple[int, float]:
        """
        Register a successful hit.

        Returns:
            (actual_damage, multiplier) after combo scaling.
        """
        frames_since_last = self._current_frame - self.state.last_hit_frame

        if frames_since_last > self.combo_window:
            # Combo dropped, start fresh
            self.state.reset()

        self.state.count += 1
        self.state.moves.append(move_type)
        self.state.last_hit_frame = self._current_frame

        multiplier = self.state.multiplier
        actual_damage = int(base_damage * multiplier)
        bonus = actual_damage - base_damage
        self.state.total_bonus_damage += bonus

        if self.state.count > self._best_combo:
            self._best_combo = self.state.count

        return actual_damage, multiplier

    def update(self):
        """Call once per frame to advance the internal clock."""
        self._current_frame += 1

        # Auto-drop combo if window expired
        if self.state.active:
            frames_since = self._current_frame - self.state.last_hit_frame
            if frames_since > self.combo_window:
                self.state.reset()

    def reset(self):
        """Full reset for new round/match."""
        self.state.reset()
        self._current_frame = 0
        self._best_combo = 0
