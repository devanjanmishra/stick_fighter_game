"""
Hitbox collision detection between player and NPC.

The player's attack hitbox is generated from their wrist position
during active attack frames. The NPC has both a body hitbox (for
receiving hits) and an attack hitbox (during NPC attacks).
"""

from dataclasses import dataclass
from game.npc import Hitbox


@dataclass
class CollisionResult:
    """Result of a collision check."""
    player_hit_npc: bool = False
    npc_hit_player: bool = False
    player_damage: int = 0
    npc_damage: int = 0


def check_collision(
    player_attack_hitbox: Hitbox | None,
    player_body_hitbox: Hitbox,
    npc_attack_hitbox: Hitbox | None,
    npc_body_hitbox: Hitbox,
    player_damage: int = 10,
    npc_damage: int = 10,
    npc_blocking: bool = False,
) -> CollisionResult:
    """
    Check for collisions between player and NPC hitboxes.

    Returns a CollisionResult with which hits landed.
    """
    result = CollisionResult()

    # Player attacks NPC
    if player_attack_hitbox is not None:
        if player_attack_hitbox.overlaps(npc_body_hitbox):
            if npc_blocking:
                result.player_hit_npc = False
                result.npc_damage = player_damage // 4  # chip damage
            else:
                result.player_hit_npc = True
                result.npc_damage = player_damage

    # NPC attacks player
    if npc_attack_hitbox is not None:
        if npc_attack_hitbox.overlaps(player_body_hitbox):
            result.npc_hit_player = True
            result.player_damage = npc_damage

    return result


def get_player_attack_hitbox(
    wrist_x: float, wrist_y: float, is_attacking: bool,
) -> Hitbox | None:
    """Generate a hitbox around the player's attacking wrist."""
    if not is_attacking:
        return None
    return Hitbox(
        x=wrist_x - 20,
        y=wrist_y - 20,
        width=40,
        height=40,
    )


def get_player_body_hitbox(
    game_x: float, head_y: float, ground_y: float,
) -> Hitbox:
    """Generate the player's body hitbox."""
    return Hitbox(
        x=game_x - 30,
        y=head_y - 16,
        width=60,
        height=ground_y - head_y + 16,
    )
