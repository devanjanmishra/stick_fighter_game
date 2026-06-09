"""
NPC fighting styles and difficulty levels.

Provides preset NPCConfig configurations for different fighting archetypes
and difficulty scaling. Each style has a distinct personality in combat.
"""

from dataclasses import dataclass
from enum import Enum
from game.npc import NPCConfig


class FightingStyle(Enum):
    BOXER = "boxer"
    BRAWLER = "brawler"
    COUNTER = "counter"
    SPEEDSTER = "speedster"
    TANK = "tank"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    NIGHTMARE = "nightmare"


@dataclass
class StyleProfile:
    """Describes a fighting style for display/UI."""
    name: str
    style: FightingStyle
    description: str
    color: tuple[int, int, int]  # character tint
    hp_multiplier: float = 1.0
    damage_multiplier: float = 1.0


# Difficulty modifiers applied on top of style configs
DIFFICULTY_MODIFIERS = {
    Difficulty.EASY: {
        "walk_speed_mult": 0.7,
        "attack_cooldown_mult": 1.5,
        "block_chance_mult": 0.5,
        "damage_mult": 0.7,
        "reaction_delay": 15,  # extra frames before reacting
    },
    Difficulty.MEDIUM: {
        "walk_speed_mult": 1.0,
        "attack_cooldown_mult": 1.0,
        "block_chance_mult": 1.0,
        "damage_mult": 1.0,
        "reaction_delay": 5,
    },
    Difficulty.HARD: {
        "walk_speed_mult": 1.2,
        "attack_cooldown_mult": 0.7,
        "block_chance_mult": 1.5,
        "damage_mult": 1.2,
        "reaction_delay": 2,
    },
    Difficulty.NIGHTMARE: {
        "walk_speed_mult": 1.4,
        "attack_cooldown_mult": 0.5,
        "block_chance_mult": 2.0,
        "damage_mult": 1.5,
        "reaction_delay": 0,
    },
}


# Base style configurations
STYLE_CONFIGS: dict[FightingStyle, NPCConfig] = {
    FightingStyle.BOXER: NPCConfig(
        walk_speed=2.5,
        preferred_distance=110.0,
        attack_range=140.0,
        retreat_distance=200.0,
        attack_cooldown_frames=25,
        attack_duration_frames=10,
        block_chance=0.25,
        block_duration_frames=12,
        attack_weights={"jab": 0.45, "cross": 0.35, "hook": 0.15, "uppercut": 0.05},
        damage={"jab": 5, "cross": 8, "hook": 12, "uppercut": 15},
    ),
    FightingStyle.BRAWLER: NPCConfig(
        walk_speed=2.0,
        preferred_distance=90.0,
        attack_range=130.0,
        retreat_distance=160.0,
        attack_cooldown_frames=20,
        attack_duration_frames=14,
        block_chance=0.10,
        block_duration_frames=8,
        attack_weights={"jab": 0.15, "cross": 0.25, "hook": 0.35, "uppercut": 0.25},
        damage={"jab": 6, "cross": 10, "hook": 15, "uppercut": 18},
    ),
    FightingStyle.COUNTER: NPCConfig(
        walk_speed=1.8,
        preferred_distance=140.0,
        attack_range=160.0,
        retreat_distance=220.0,
        attack_cooldown_frames=35,
        attack_duration_frames=8,
        block_chance=0.45,
        block_duration_frames=20,
        attack_weights={"jab": 0.20, "cross": 0.40, "hook": 0.25, "uppercut": 0.15},
        damage={"jab": 4, "cross": 10, "hook": 14, "uppercut": 16},
    ),
    FightingStyle.SPEEDSTER: NPCConfig(
        walk_speed=3.5,
        preferred_distance=100.0,
        attack_range=135.0,
        retreat_distance=190.0,
        attack_cooldown_frames=15,
        attack_duration_frames=8,
        block_chance=0.15,
        block_duration_frames=10,
        attack_weights={"jab": 0.50, "cross": 0.30, "hook": 0.15, "uppercut": 0.05},
        damage={"jab": 4, "cross": 6, "hook": 9, "uppercut": 12},
    ),
    FightingStyle.TANK: NPCConfig(
        walk_speed=1.5,
        preferred_distance=80.0,
        attack_range=120.0,
        retreat_distance=150.0,
        attack_cooldown_frames=40,
        attack_duration_frames=16,
        block_chance=0.35,
        block_duration_frames=25,
        attack_weights={"jab": 0.10, "cross": 0.20, "hook": 0.30, "uppercut": 0.40},
        damage={"jab": 7, "cross": 12, "hook": 18, "uppercut": 22},
    ),
}


STYLE_PROFILES: dict[FightingStyle, StyleProfile] = {
    FightingStyle.BOXER: StyleProfile(
        name="The Boxer",
        style=FightingStyle.BOXER,
        description="Balanced fighter with quick jabs and solid fundamentals.",
        color=(255, 70, 70),
        hp_multiplier=1.0,
        damage_multiplier=1.0,
    ),
    FightingStyle.BRAWLER: StyleProfile(
        name="The Brawler",
        style=FightingStyle.BRAWLER,
        description="Aggressive close-range fighter. Heavy hooks and uppercuts.",
        color=(255, 140, 50),
        hp_multiplier=1.1,
        damage_multiplier=1.2,
    ),
    FightingStyle.COUNTER: StyleProfile(
        name="The Counter",
        style=FightingStyle.COUNTER,
        description="Patient defensive fighter. Waits for openings to punish.",
        color=(100, 200, 255),
        hp_multiplier=0.9,
        damage_multiplier=1.1,
    ),
    FightingStyle.SPEEDSTER: StyleProfile(
        name="The Speedster",
        style=FightingStyle.SPEEDSTER,
        description="Lightning-fast jabs and movement. Death by a thousand cuts.",
        color=(255, 255, 100),
        hp_multiplier=0.85,
        damage_multiplier=0.8,
    ),
    FightingStyle.TANK: StyleProfile(
        name="The Tank",
        style=FightingStyle.TANK,
        description="Slow but devastating. Each hit is a wrecking ball.",
        color=(180, 80, 255),
        hp_multiplier=1.3,
        damage_multiplier=1.4,
    ),
}


def get_npc_config(style: FightingStyle, difficulty: Difficulty) -> NPCConfig:
    """Build an NPCConfig for a given style and difficulty."""
    base = STYLE_CONFIGS[style]
    mods = DIFFICULTY_MODIFIERS[difficulty]
    profile = STYLE_PROFILES[style]

    walk_speed = base.walk_speed * mods["walk_speed_mult"]
    cooldown = max(5, int(base.attack_cooldown_frames * mods["attack_cooldown_mult"]))
    block_chance = min(0.8, base.block_chance * mods["block_chance_mult"])
    dmg_mult = mods["damage_mult"] * profile.damage_multiplier

    scaled_damage = {}
    for move, dmg in base.damage.items():
        scaled_damage[move] = max(1, int(dmg * dmg_mult))

    return NPCConfig(
        walk_speed=walk_speed,
        preferred_distance=base.preferred_distance,
        attack_range=base.attack_range,
        retreat_distance=base.retreat_distance,
        attack_cooldown_frames=cooldown,
        attack_duration_frames=base.attack_duration_frames,
        block_chance=block_chance,
        block_duration_frames=base.block_duration_frames,
        attack_weights=dict(base.attack_weights),
        damage=scaled_damage,
    )


def get_npc_hp(style: FightingStyle, base_hp: int = 100) -> int:
    """Get NPC max HP adjusted for style."""
    profile = STYLE_PROFILES[style]
    return int(base_hp * profile.hp_multiplier)


def list_styles() -> list[StyleProfile]:
    """Return all available fighting styles."""
    return list(STYLE_PROFILES.values())


def list_difficulties() -> list[Difficulty]:
    """Return all difficulty levels."""
    return list(Difficulty)
