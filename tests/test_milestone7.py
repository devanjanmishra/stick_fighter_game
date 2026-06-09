"""
Milestone 7 Test: NPC scripted AI + hitbox collision.
- Tests NPC state machine (idle, approach, attack, recover)
- Tests NPC attack animations and hitbox generation
- Tests collision detection between player and NPC
- Tests NPC blocking
- Renders NPC combat scene
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from game.npc import NPC, NPCConfig, NPCState, NPCAttackType, Hitbox
from game.collision import check_collision, get_player_attack_hitbox, get_player_body_hitbox
from rendering.stick_figure import StickFigureRenderer
from rendering.game_renderer import GameRenderer

SCREEN_W = 1280
SCREEN_H = 720
GROUND_Y = 580
OUTPUT_DIR = "/home/ubuntu/stick_fighter/test_output"


def test_npc_initial_state():
    """NPC should start in idle state at specified position."""
    npc = NPC(game_x=900.0, ground_y=GROUND_Y)
    assert npc.state == NPCState.IDLE
    assert npc.game_x == 900.0
    assert not npc.is_attacking
    assert not npc.is_blocking
    print("[PASS] NPC initial state: idle at x=900")


def test_npc_approach():
    """NPC should approach the player when too far away."""
    npc = NPC(NPCConfig(preferred_distance=120, walk_speed=5.0, attack_range=150), game_x=900.0)

    initial_x = npc.game_x
    # Player is at x=300 — NPC should approach (600px at 5px/frame needs ~120 frames)
    for _ in range(250):
        npc.update(player_x=300.0)

    distance = abs(npc.game_x - 300.0)
    assert npc.game_x < initial_x, f"NPC should have moved toward player"
    assert distance <= 200, f"NPC should be close to player, distance={distance:.1f}"
    print(f"[PASS] NPC approach: moved to x={npc.game_x:.1f} (distance={distance:.1f})")


def test_npc_attack_cycle():
    """NPC should attack when in range, then recover."""
    npc = NPC(NPCConfig(
        preferred_distance=120,
        attack_range=150,
        attack_cooldown_frames=5,
        attack_duration_frames=10,
    ), game_x=400.0)

    # Place player close enough to trigger attack
    states_seen = set()
    attack_detected = False

    for i in range(100):
        state = npc.update(player_x=300.0)
        states_seen.add(state)
        if state == NPCState.ATTACK:
            attack_detected = True

    assert attack_detected, f"NPC should attack when in range, states seen: {states_seen}"
    assert NPCState.RECOVER in states_seen, "NPC should recover after attack"
    print(f"[PASS] NPC attack cycle: states seen = {[s.value for s in states_seen]}")


def test_npc_generates_pose():
    """NPC should generate a valid pose with all body parts."""
    npc = NPC(game_x=900.0, ground_y=GROUND_Y)
    pose = npc.get_pose()

    pose_dict = pose.as_dict()
    required_parts = [
        "head", "neck", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "hip_left", "hip_right", "knee_left", "knee_right",
        "ankle_left", "ankle_right",
    ]
    for part in required_parts:
        assert part in pose_dict, f"Missing body part: {part}"
        x, y = pose_dict[part]
        assert isinstance(x, (int, float)), f"{part} x should be numeric"
        assert isinstance(y, (int, float)), f"{part} y should be numeric"

    print(f"[PASS] NPC pose: {len(pose_dict)} body parts generated")


def test_npc_attack_hitbox():
    """NPC should generate a hitbox during active attack frames."""
    npc = NPC(NPCConfig(
        attack_range=200,
        attack_duration_frames=10,
        attack_cooldown_frames=5,
    ), game_x=400.0)

    # Force attack by placing player nearby
    hitbox_found = False
    for i in range(60):
        npc.update(player_x=300.0)
        hb = npc.get_attack_hitbox()
        if hb is not None:
            hitbox_found = True
            assert hb.width > 0 and hb.height > 0
            break

    assert hitbox_found, "NPC should generate attack hitbox during attack"
    print(f"[PASS] NPC attack hitbox generated: {hb.width}x{hb.height} at ({hb.x:.0f},{hb.y:.0f})")


def test_hitbox_overlap():
    """Test basic hitbox overlap detection."""
    box_a = Hitbox(x=100, y=100, width=50, height=50)
    box_b = Hitbox(x=130, y=120, width=50, height=50)
    box_c = Hitbox(x=300, y=300, width=50, height=50)

    assert box_a.overlaps(box_b), "Overlapping boxes should detect collision"
    assert not box_a.overlaps(box_c), "Non-overlapping boxes should not collide"
    print("[PASS] Hitbox overlap detection works correctly")


def test_collision_player_hits_npc():
    """Player attack should hit NPC when hitboxes overlap."""
    player_hitbox = get_player_attack_hitbox(450, 400, is_attacking=True)
    player_body = get_player_body_hitbox(400, 350, GROUND_Y)
    npc_body = Hitbox(x=430, y=350, width=60, height=230)

    result = check_collision(
        player_attack_hitbox=player_hitbox,
        player_body_hitbox=player_body,
        npc_attack_hitbox=None,
        npc_body_hitbox=npc_body,
        player_damage=10,
    )

    assert result.player_hit_npc, "Player should hit NPC"
    assert result.npc_damage == 10
    assert not result.npc_hit_player
    print(f"[PASS] Player hits NPC: damage={result.npc_damage}")


def test_collision_npc_blocked():
    """NPC blocking should reduce damage to chip damage."""
    player_hitbox = get_player_attack_hitbox(450, 400, is_attacking=True)
    player_body = get_player_body_hitbox(400, 350, GROUND_Y)
    npc_body = Hitbox(x=430, y=350, width=60, height=230)

    result = check_collision(
        player_attack_hitbox=player_hitbox,
        player_body_hitbox=player_body,
        npc_attack_hitbox=None,
        npc_body_hitbox=npc_body,
        player_damage=12,
        npc_blocking=True,
    )

    assert not result.player_hit_npc, "Blocked hit should not count as full hit"
    assert result.npc_damage == 3, f"Chip damage should be 12//4=3, got {result.npc_damage}"
    print(f"[PASS] NPC blocks: chip damage={result.npc_damage}")


def test_collision_mutual_hit():
    """Both player and NPC can hit each other simultaneously."""
    player_hitbox = get_player_attack_hitbox(450, 400, is_attacking=True)
    player_body = get_player_body_hitbox(400, 350, GROUND_Y)
    npc_attack = Hitbox(x=380, y=380, width=40, height=40)
    npc_body = Hitbox(x=430, y=350, width=60, height=230)

    result = check_collision(
        player_attack_hitbox=player_hitbox,
        player_body_hitbox=player_body,
        npc_attack_hitbox=npc_attack,
        npc_body_hitbox=npc_body,
        player_damage=10,
        npc_damage=8,
    )

    assert result.player_hit_npc, "Player should hit NPC"
    assert result.npc_hit_player, "NPC should hit player"
    assert result.npc_damage == 10
    assert result.player_damage == 8
    print(f"[PASS] Mutual hit: player_dmg={result.player_damage}, npc_dmg={result.npc_damage}")


def test_npc_receive_hit():
    """NPC should react to being hit (push back, cancel attack)."""
    npc = NPC(game_x=500.0, ground_y=GROUND_Y)
    initial_x = npc.game_x
    npc.receive_hit(damage=10)

    assert npc.game_x != initial_x, "NPC should be pushed back on hit"
    print(f"[PASS] NPC receive hit: pushed from {initial_x:.0f} to {npc.game_x:.0f}")


def _draw_npc_stick_figure(screen: pygame.Surface, npc: NPC, color: tuple[int, int, int]):
    """Draw the NPC as a stick figure on screen."""
    pose = npc.get_pose()
    pd = pose.as_dict()

    # Draw connections
    connections = [
        ("head", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "hip_left"), ("right_shoulder", "hip_right"),
        ("hip_left", "knee_left"), ("knee_left", "ankle_left"),
        ("hip_right", "knee_right"), ("knee_right", "ankle_right"),
        ("hip_left", "hip_right"),
    ]

    for a, b in connections:
        if a in pd and b in pd:
            pygame.draw.line(screen, color, (int(pd[a][0]), int(pd[a][1])),
                           (int(pd[b][0]), int(pd[b][1])), 3)

    # Draw joints
    for name, (x, y) in pd.items():
        radius = 14 if name == "head" else 5
        pygame.draw.circle(screen, color, (int(x), int(y)), radius)
        if name == "head":
            pygame.draw.circle(screen, (40, 40, 50), (int(x), int(y)), radius, 2)


def test_render_npc_combat(screen: pygame.Surface):
    """Render a combat scene with player and NPC."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    renderer = GameRenderer(SCREEN_W, SCREEN_H)
    renderer.draw_background(screen)

    font = pygame.font.SysFont("monospace", 20)

    # Draw player stick figure (static jab pose)
    from core.coordinate_transformer import CoordinateTransformer
    from tests.synthetic_data import generate_jab_sequence
    jab_seq = generate_jab_sequence(10)
    t = CoordinateTransformer(SCREEN_W, SCREEN_H, player_base_x=350, ground_y=GROUND_Y)
    game_pose = t.transform(jab_seq[6], facing_right=True)
    player_fig = StickFigureRenderer(
        color=(50, 200, 255), head_color=(50, 200, 255),
        line_width=3, head_radius=14,
    )
    player_fig.draw(screen, game_pose)

    # Draw NPC in different states
    npc_configs = [
        ("Idle", NPCState.IDLE, 600),
        ("Guard", NPCState.BLOCK, 750),
        ("Attacking", NPCState.ATTACK, 950),
    ]

    for label, forced_state, x_pos in npc_configs:
        npc = NPC(game_x=x_pos, ground_y=GROUND_Y)
        if forced_state == NPCState.ATTACK:
            npc._state = NPCState.ATTACK
            npc._attack_type = NPCAttackType.JAB
            npc._attack_progress = 0.7
        elif forced_state == NPCState.BLOCK:
            npc._state = NPCState.BLOCK
        npc_color = (255, 80, 80)
        _draw_npc_stick_figure(screen, npc, npc_color)

        label_surf = font.render(label, True, (255, 80, 80))
        screen.blit(label_surf, (x_pos - 25, GROUND_Y + 20))

    # Draw hitbox visualization for attacking NPC
    atk_npc = NPC(game_x=950, ground_y=GROUND_Y)
    atk_npc._state = NPCState.ATTACK
    atk_npc._attack_type = NPCAttackType.JAB
    atk_npc._attack_progress = 0.7
    hb = atk_npc.get_attack_hitbox()
    if hb:
        pygame.draw.rect(screen, (255, 255, 0, 128),
                        (int(hb.x), int(hb.y), int(hb.width), int(hb.height)), 2)

    # Labels
    player_label = font.render("PLAYER", True, (50, 200, 255))
    screen.blit(player_label, (320, GROUND_Y + 20))

    title = font.render("NPC COMBAT SCENE", True, (200, 200, 200))
    screen.blit(title, (SCREEN_W // 2 - 120, 20))

    path = os.path.join(OUTPUT_DIR, "m7_npc_combat.png")
    pygame.image.save(screen, path)
    print(f"[PASS] NPC combat scene rendered -> {path}")
    return path


if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 7 TESTS: NPC Scripted AI + Hitbox Collision")
    print("=" * 60)

    test_npc_initial_state()
    test_npc_approach()
    test_npc_attack_cycle()
    test_npc_generates_pose()
    test_npc_attack_hitbox()
    test_hitbox_overlap()
    test_collision_player_hits_npc()
    test_collision_npc_blocked()
    test_collision_mutual_hit()
    test_npc_receive_hit()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    test_render_npc_combat(screen)
    pygame.quit()

    print("=" * 60)
    print("ALL MILESTONE 7 TESTS PASSED")
    print("=" * 60)
