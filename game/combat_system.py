"""
Combat system: HP, rounds, timer, and game state management.

Manages the full fight lifecycle:
  - Round start/end
  - HP tracking for player and NPC
  - Round timer (countdown)
  - Win/loss conditions
  - Score tracking across rounds
"""

from enum import Enum
from dataclasses import dataclass, field


class GamePhase(Enum):
    MENU = "menu"
    COUNTDOWN = "countdown"  # 3-2-1 before round starts
    FIGHTING = "fighting"
    ROUND_END = "round_end"
    MATCH_END = "match_end"
    PAUSED = "paused"


class RoundResult(Enum):
    PLAYER_WIN = "player_win"
    NPC_WIN = "npc_win"
    DRAW = "draw"
    TIME_UP = "time_up"


@dataclass
class FighterStats:
    """HP and combat stats for a fighter."""
    max_hp: int = 100
    current_hp: int = 100
    rounds_won: int = 0

    def take_damage(self, amount: int):
        self.current_hp = max(0, self.current_hp - amount)

    def heal(self, amount: int):
        self.current_hp = min(self.max_hp, self.current_hp + amount)

    @property
    def hp_ratio(self) -> float:
        return self.current_hp / self.max_hp if self.max_hp > 0 else 0.0

    @property
    def is_ko(self) -> bool:
        return self.current_hp <= 0

    def reset_hp(self):
        self.current_hp = self.max_hp


@dataclass
class CombatConfig:
    """Configuration for the combat system."""
    max_rounds: int = 3
    rounds_to_win: int = 2
    round_time_seconds: int = 60
    countdown_seconds: int = 3
    round_end_pause_seconds: float = 2.0
    fps: int = 30

    # Damage values per move type
    damage_table: dict[str, int] = field(default_factory=lambda: {
        "jab": 5,
        "cross": 8,
        "hook": 12,
        "uppercut": 15,
    })


class CombatSystem:
    """
    Manages the full combat game state.

    Tracks HP, rounds, timer, and transitions between game phases.
    Call update() once per frame.
    """

    def __init__(self, config: CombatConfig | None = None):
        self.config = config or CombatConfig()

        self.player = FighterStats(max_hp=100)
        self.npc = FighterStats(max_hp=100)

        self._phase = GamePhase.COUNTDOWN
        self._current_round = 1
        self._round_timer_frames = 0
        self._countdown_frames = 0
        self._round_end_frames = 0
        self._match_result: RoundResult | None = None
        self._round_results: list[RoundResult] = []

        self._hit_registered_this_frame = False
        self._total_frames = 0

        self._start_countdown()

    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def current_round(self) -> int:
        return self._current_round

    @property
    def round_timer_seconds(self) -> int:
        """Remaining time in the current round."""
        if self._phase != GamePhase.FIGHTING:
            return self.config.round_time_seconds
        remaining_frames = max(0, self._round_timer_frames)
        return remaining_frames // self.config.fps

    @property
    def countdown_value(self) -> int:
        """Current countdown number (3, 2, 1, or 0 for FIGHT!)."""
        if self._phase != GamePhase.COUNTDOWN:
            return 0
        return max(0, self._countdown_frames // self.config.fps) + 1

    @property
    def match_result(self) -> RoundResult | None:
        return self._match_result

    @property
    def round_results(self) -> list[RoundResult]:
        return list(self._round_results)

    def _start_countdown(self):
        self._phase = GamePhase.COUNTDOWN
        self._countdown_frames = self.config.countdown_seconds * self.config.fps

    def _start_round(self):
        self._phase = GamePhase.FIGHTING
        self._round_timer_frames = self.config.round_time_seconds * self.config.fps
        self.player.reset_hp()
        self.npc.reset_hp()

    def apply_damage_to_npc(self, move_type: str) -> int:
        """Apply damage from player to NPC. Returns damage dealt."""
        if self._phase != GamePhase.FIGHTING:
            return 0
        damage = self.config.damage_table.get(move_type, 5)
        self.npc.take_damage(damage)
        return damage

    def apply_damage_to_player(self, move_type: str) -> int:
        """Apply damage from NPC to player. Returns damage dealt."""
        if self._phase != GamePhase.FIGHTING:
            return 0
        damage = self.config.damage_table.get(move_type, 5)
        self.player.take_damage(damage)
        return damage

    def update(self) -> GamePhase:
        """
        Update the combat system. Call once per frame.
        Returns the current game phase.
        """
        self._total_frames += 1

        if self._phase == GamePhase.COUNTDOWN:
            self._countdown_frames -= 1
            if self._countdown_frames <= 0:
                self._start_round()

        elif self._phase == GamePhase.FIGHTING:
            self._round_timer_frames -= 1

            # Check KO
            if self.player.is_ko:
                self._end_round(RoundResult.NPC_WIN)
            elif self.npc.is_ko:
                self._end_round(RoundResult.PLAYER_WIN)
            elif self._round_timer_frames <= 0:
                # Time up — whoever has more HP wins
                if self.player.current_hp > self.npc.current_hp:
                    self._end_round(RoundResult.PLAYER_WIN)
                elif self.npc.current_hp > self.player.current_hp:
                    self._end_round(RoundResult.NPC_WIN)
                else:
                    self._end_round(RoundResult.DRAW)

        elif self._phase == GamePhase.ROUND_END:
            self._round_end_frames -= 1
            if self._round_end_frames <= 0:
                # Check if match is over
                if self._check_match_end():
                    self._phase = GamePhase.MATCH_END
                else:
                    self._current_round += 1
                    self._start_countdown()

        return self._phase

    def _end_round(self, result: RoundResult):
        self._phase = GamePhase.ROUND_END
        self._round_results.append(result)
        self._round_end_frames = int(self.config.round_end_pause_seconds * self.config.fps)

        if result == RoundResult.PLAYER_WIN:
            self.player.rounds_won += 1
        elif result == RoundResult.NPC_WIN:
            self.npc.rounds_won += 1

    def _check_match_end(self) -> bool:
        if self.player.rounds_won >= self.config.rounds_to_win:
            self._match_result = RoundResult.PLAYER_WIN
            return True
        if self.npc.rounds_won >= self.config.rounds_to_win:
            self._match_result = RoundResult.NPC_WIN
            return True
        if self._current_round >= self.config.max_rounds:
            if self.player.rounds_won > self.npc.rounds_won:
                self._match_result = RoundResult.PLAYER_WIN
            elif self.npc.rounds_won > self.player.rounds_won:
                self._match_result = RoundResult.NPC_WIN
            else:
                self._match_result = RoundResult.DRAW
            return True
        return False

    def reset_match(self):
        """Reset everything for a new match."""
        self.player = FighterStats(max_hp=100)
        self.npc = FighterStats(max_hp=100)
        self._phase = GamePhase.COUNTDOWN
        self._current_round = 1
        self._round_timer_frames = 0
        self._round_end_frames = 0
        self._match_result = None
        self._round_results = []
        self._total_frames = 0
        self._start_countdown()
