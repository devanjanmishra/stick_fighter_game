"""
Visual hit effects: screen shake, hit sparks, flash, and floating damage numbers.
"""

import random
import math
from dataclasses import dataclass, field


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: int
    max_life: int
    color: tuple[int, int, int]
    size: float = 4.0

    @property
    def alpha(self) -> float:
        return max(0.0, self.life / self.max_life)

    @property
    def alive(self) -> bool:
        return self.life > 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.3  # gravity
        self.vx *= 0.95  # drag
        self.size *= 0.97
        self.life -= 1


@dataclass
class DamageNumber:
    x: float
    y: float
    value: int
    life: int = 40
    max_life: int = 40
    color: tuple[int, int, int] = (255, 255, 100)
    vy: float = -2.0

    @property
    def alpha(self) -> float:
        return max(0.0, self.life / self.max_life)

    @property
    def alive(self) -> bool:
        return self.life > 0

    def update(self):
        self.y += self.vy
        self.vy *= 0.96
        self.life -= 1


@dataclass
class ScreenShake:
    intensity: float = 0.0
    duration: int = 0
    _frame: int = 0

    @property
    def active(self) -> bool:
        return self._frame < self.duration and self.intensity > 0

    @property
    def offset(self) -> tuple[int, int]:
        if not self.active:
            return (0, 0)
        decay = 1.0 - (self._frame / self.duration)
        mag = self.intensity * decay
        ox = int(random.uniform(-mag, mag))
        oy = int(random.uniform(-mag, mag))
        return (ox, oy)

    def trigger(self, intensity: float, duration: int):
        self.intensity = intensity
        self.duration = duration
        self._frame = 0

    def update(self):
        if self.active:
            self._frame += 1


@dataclass
class HitFlash:
    duration: int = 0
    _frame: int = 0
    color: tuple[int, int, int] = (255, 255, 255)
    alpha: int = 80

    @property
    def active(self) -> bool:
        return self._frame < self.duration

    def trigger(self, duration: int = 4, color: tuple[int, int, int] = (255, 255, 255)):
        self.duration = duration
        self._frame = 0
        self.color = color

    def update(self):
        if self.active:
            self._frame += 1

    @property
    def current_alpha(self) -> int:
        if not self.active:
            return 0
        decay = 1.0 - (self._frame / self.duration)
        return int(self.alpha * decay)


# Hit spark colors by move type
SPARK_COLORS = {
    "jab": [(255, 255, 200), (255, 230, 150), (200, 200, 150)],
    "cross": [(255, 200, 50), (255, 180, 30), (255, 150, 0)],
    "hook": [(255, 100, 50), (255, 80, 30), (255, 60, 0)],
    "uppercut": [(255, 50, 50), (255, 30, 30), (200, 0, 0)],
}


class EffectsManager:
    """Manages all visual combat effects."""

    def __init__(self):
        self.particles: list[Particle] = []
        self.damage_numbers: list[DamageNumber] = []
        self.screen_shake = ScreenShake()
        self.hit_flash = HitFlash()
        self._hitstop_frames: int = 0

    @property
    def hitstop_active(self) -> bool:
        return self._hitstop_frames > 0

    def spawn_hit_sparks(self, x: float, y: float, move_type: str, count: int = 12):
        """Spawn particle burst at hit location."""
        colors = SPARK_COLORS.get(move_type, SPARK_COLORS["jab"])
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2.0, 8.0)
            p = Particle(
                x=x + random.uniform(-5, 5),
                y=y + random.uniform(-5, 5),
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed - 2.0,
                life=random.randint(10, 25),
                max_life=25,
                color=random.choice(colors),
                size=random.uniform(3.0, 6.0),
            )
            self.particles.append(p)

    def spawn_damage_number(self, x: float, y: float, damage: int, is_player: bool = False):
        """Spawn floating damage number."""
        color = (255, 80, 80) if is_player else (255, 255, 100)
        dn = DamageNumber(
            x=x + random.uniform(-10, 10),
            y=y - 30,
            value=damage,
            color=color,
        )
        self.damage_numbers.append(dn)

    def trigger_hit(self, x: float, y: float, move_type: str, damage: int, is_player_hit: bool = False):
        """Trigger all effects for a hit."""
        self.spawn_hit_sparks(x, y, move_type)
        self.spawn_damage_number(x, y, damage, is_player=is_player_hit)

        # Scale shake and hitstop by move strength
        shake_map = {"jab": (3, 4), "cross": (5, 6), "hook": (8, 8), "uppercut": (10, 10)}
        intensity, duration = shake_map.get(move_type, (3, 4))
        self.screen_shake.trigger(intensity, duration)

        hitstop_map = {"jab": 2, "cross": 3, "hook": 4, "uppercut": 5}
        self._hitstop_frames = hitstop_map.get(move_type, 2)

        self.hit_flash.trigger(duration=3)

    def update(self):
        """Update all effects by one frame."""
        if self._hitstop_frames > 0:
            self._hitstop_frames -= 1
            return  # freeze game during hitstop but still render

        self.screen_shake.update()
        self.hit_flash.update()

        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.alive]

        for dn in self.damage_numbers:
            dn.update()
        self.damage_numbers = [dn for dn in self.damage_numbers if dn.alive]

    def clear(self):
        """Clear all active effects."""
        self.particles.clear()
        self.damage_numbers.clear()
        self.screen_shake = ScreenShake()
        self.hit_flash = HitFlash()
        self._hitstop_frames = 0
