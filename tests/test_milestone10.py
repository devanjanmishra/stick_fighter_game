"""
Milestone 10 Test: Multiple NPC fighting styles + difficulty levels.
- Tests all 5 fighting styles have valid configs
- Tests difficulty scaling modifies stats correctly
- Tests each style has distinct behavior characteristics
- Tests NPC creation from style configs
- Tests style profiles for UI display
- Renders visual comparison of all styles
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from game.npc import NPC, NPCConfig, NPCState
from game.npc_styles import (
    FightingStyle, Difficulty, StyleProfile,
    get_npc_config, get_npc_hp, list_styles, list_difficulties,
    STYLE_CONFIGS, STYLE_PROFILES, DIFFICULTY_MODIFIERS,
)
from rendering.game_renderer import GameRenderer

SCREEN_W = 1280
SCREEN_H = 720
OUTPUT_DIR = "/home/ubuntu/stick_fighter/test_output"


def test_all_styles_have_configs():
    """Every FightingStyle should have a valid NPCConfig."""
    for style in FightingStyle:
        assert style in STYLE_CONFIGS, f"Missing config for {style.value}"
        cfg = STYLE_CONFIGS[style]
        assert cfg.walk_speed > 0
        assert cfg.attack_range > 0
        assert cfg.attack_cooldown_frames > 0
        assert sum(cfg.attack_weights.values()) > 0
        assert all(d > 0 for d in cfg.damage.values())
    print(f"[PASS] All {len(FightingStyle)} styles have valid configs")


def test_all_styles_have_profiles():
    """Every FightingStyle should have a StyleProfile for UI."""
    for style in FightingStyle:
        assert style in STYLE_PROFILES, f"Missing profile for {style.value}"
        profile = STYLE_PROFILES[style]
        assert len(profile.name) > 0
        assert len(profile.description) > 0
        assert len(profile.color) == 3
        assert profile.hp_multiplier > 0
    print(f"[PASS] All {len(FightingStyle)} styles have UI profiles")


def test_difficulty_scaling():
    """Higher difficulty should make NPC faster, more aggressive, and harder."""
    style = FightingStyle.BOXER

    easy = get_npc_config(style, Difficulty.EASY)
    medium = get_npc_config(style, Difficulty.MEDIUM)
    hard = get_npc_config(style, Difficulty.HARD)
    nightmare = get_npc_config(style, Difficulty.NIGHTMARE)

    # Walk speed increases with difficulty
    assert easy.walk_speed < medium.walk_speed < hard.walk_speed < nightmare.walk_speed, \
        f"Walk speed should increase: {easy.walk_speed}, {medium.walk_speed}, {hard.walk_speed}, {nightmare.walk_speed}"

    # Attack cooldown decreases with difficulty (faster attacks)
    assert easy.attack_cooldown_frames > medium.attack_cooldown_frames > hard.attack_cooldown_frames, \
        f"Cooldown should decrease: {easy.attack_cooldown_frames}, {medium.attack_cooldown_frames}, {hard.attack_cooldown_frames}"

    # Block chance increases with difficulty
    assert easy.block_chance < medium.block_chance < hard.block_chance, \
        f"Block chance should increase: {easy.block_chance}, {medium.block_chance}, {hard.block_chance}"

    print(f"[PASS] Difficulty scaling: speed {easy.walk_speed:.1f}->{nightmare.walk_speed:.1f}, "
          f"cooldown {easy.attack_cooldown_frames}->{nightmare.attack_cooldown_frames}")


def test_style_distinct_damage():
    """Each style should have different damage profiles."""
    damages = {}
    for style in FightingStyle:
        cfg = get_npc_config(style, Difficulty.MEDIUM)
        damages[style.value] = dict(cfg.damage)

    # Brawler should hit harder than Speedster
    assert damages["brawler"]["hook"] > damages["speedster"]["hook"], \
        f"Brawler hook ({damages['brawler']['hook']}) should beat Speedster ({damages['speedster']['hook']})"

    # Tank should have highest uppercut
    tank_upper = damages["tank"]["uppercut"]
    for style_name, dmg in damages.items():
        if style_name != "tank":
            assert tank_upper >= dmg["uppercut"], \
                f"Tank uppercut ({tank_upper}) should be >= {style_name} ({dmg['uppercut']})"

    # Speedster should have lowest damage overall
    speedster_total = sum(damages["speedster"].values())
    for style_name, dmg in damages.items():
        if style_name != "speedster":
            total = sum(dmg.values())
            assert speedster_total <= total, \
                f"Speedster total ({speedster_total}) should be <= {style_name} ({total})"

    print(f"[PASS] Style damage profiles are distinct:")
    for name, dmg in damages.items():
        print(f"       {name}: {dmg}")


def test_style_distinct_behavior():
    """Each style should exhibit different behavior patterns."""
    # Speedster should be fastest
    speedster = get_npc_config(FightingStyle.SPEEDSTER, Difficulty.MEDIUM)
    tank = get_npc_config(FightingStyle.TANK, Difficulty.MEDIUM)
    assert speedster.walk_speed > tank.walk_speed
    assert speedster.attack_cooldown_frames < tank.attack_cooldown_frames

    # Counter should block more
    counter = get_npc_config(FightingStyle.COUNTER, Difficulty.MEDIUM)
    brawler = get_npc_config(FightingStyle.BRAWLER, Difficulty.MEDIUM)
    assert counter.block_chance > brawler.block_chance

    # Brawler prefers close range
    assert brawler.preferred_distance < counter.preferred_distance

    print("[PASS] Style behaviors are distinct: speedster fast, counter blocks, brawler close-range")


def test_npc_creation_from_style():
    """NPC should work correctly when created from a style config."""
    for style in FightingStyle:
        for diff in Difficulty:
            cfg = get_npc_config(style, diff)
            npc = NPC(config=cfg, game_x=900.0)
            assert npc.state == NPCState.IDLE

            # Should be able to update without errors
            for _ in range(30):
                npc.update(player_x=300.0)

            # Should generate valid pose
            pose = npc.get_pose()
            d = pose.as_dict()
            assert len(d) == 14

    print(f"[PASS] NPC created and simulated for all {len(FightingStyle)}x{len(Difficulty)} combinations")


def test_hp_scaling():
    """HP should scale based on style profile."""
    tank_hp = get_npc_hp(FightingStyle.TANK)
    speedster_hp = get_npc_hp(FightingStyle.SPEEDSTER)
    boxer_hp = get_npc_hp(FightingStyle.BOXER)

    assert tank_hp > boxer_hp > speedster_hp, \
        f"HP should scale: tank={tank_hp} > boxer={boxer_hp} > speedster={speedster_hp}"
    print(f"[PASS] HP scaling: tank={tank_hp}, boxer={boxer_hp}, speedster={speedster_hp}")


def test_list_functions():
    """list_styles and list_difficulties should return all entries."""
    styles = list_styles()
    assert len(styles) == 5
    assert all(isinstance(s, StyleProfile) for s in styles)

    diffs = list_difficulties()
    assert len(diffs) == 4
    print(f"[PASS] list_styles={len(styles)}, list_difficulties={len(diffs)}")


def test_nightmare_still_beatable():
    """Even nightmare difficulty should have reasonable parameters."""
    for style in FightingStyle:
        cfg = get_npc_config(style, Difficulty.NIGHTMARE)
        # Cooldown should still be positive
        assert cfg.attack_cooldown_frames >= 5, \
            f"{style.value} nightmare cooldown too low: {cfg.attack_cooldown_frames}"
        # Block chance should be capped
        assert cfg.block_chance <= 0.8, \
            f"{style.value} nightmare block chance too high: {cfg.block_chance}"
        # Damage should be high but not instant-kill
        max_dmg = max(cfg.damage.values())
        assert max_dmg <= 50, \
            f"{style.value} nightmare max damage too high: {max_dmg}"
    print("[PASS] Nightmare difficulty is challenging but beatable")


def test_speedster_attacks_more(screen: pygame.Surface = None):
    """Speedster should attack more frequently than Tank in same timeframe."""
    speedster_npc = NPC(config=get_npc_config(FightingStyle.SPEEDSTER, Difficulty.MEDIUM), game_x=450.0)
    tank_npc = NPC(config=get_npc_config(FightingStyle.TANK, Difficulty.MEDIUM), game_x=450.0)

    speedster_attacks = 0
    tank_attacks = 0

    for _ in range(300):
        s = speedster_npc.update(player_x=300.0)
        if s == NPCState.ATTACK:
            speedster_attacks += 1
        t = tank_npc.update(player_x=300.0)
        if t == NPCState.ATTACK:
            tank_attacks += 1

    assert speedster_attacks > tank_attacks, \
        f"Speedster should attack more ({speedster_attacks}) than Tank ({tank_attacks})"
    print(f"[PASS] Speedster attacks {speedster_attacks}x vs Tank {tank_attacks}x in 300 frames")


def test_render_all_styles(screen: pygame.Surface):
    """Render all NPC styles side by side for visual comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    renderer = GameRenderer(SCREEN_W, SCREEN_H)

    from rendering.stick_figure import StickFigureRenderer
    from core.coordinate_transformer import GamePose, GameKeypoint

    renderer.draw_background(screen)

    font = pygame.font.SysFont("monospace", 16, bold=True)
    title_font = pygame.font.SysFont("monospace", 28, bold=True)

    title = title_font.render("NPC FIGHTING STYLES", True, (255, 255, 255))
    screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 20))

    styles = list(FightingStyle)
    spacing = SCREEN_W // (len(styles) + 1)

    for i, style in enumerate(styles):
        profile = STYLE_PROFILES[style]
        cfg = get_npc_config(style, Difficulty.MEDIUM)
        npc = NPC(config=cfg, game_x=spacing * (i + 1))

        # Get NPC in attack pose for visual interest
        for _ in range(5):
            npc.update(player_x=npc.game_x - 100)
        npc._state = NPCState.ATTACK
        npc._attack_type = npc._choose_attack()
        npc._attack_progress = 0.6

        pose = npc.get_pose()

        # Convert NPCPose to GamePose for rendering
        kps = {}
        mapping = {
            "nose": pose.head, "left_shoulder": pose.left_shoulder,
            "right_shoulder": pose.right_shoulder, "left_elbow": pose.left_elbow,
            "right_elbow": pose.right_elbow, "left_wrist": pose.left_wrist,
            "right_wrist": pose.right_wrist, "left_hip": pose.hip_left,
            "right_hip": pose.hip_right, "left_knee": pose.knee_left,
            "right_knee": pose.knee_right, "left_ankle": pose.ankle_left,
            "right_ankle": pose.ankle_right,
        }
        for name, (gx, gy) in mapping.items():
            kps[name] = GameKeypoint(game_x=gx, game_y=gy, depth=0.0, name=name)

        game_pose = GamePose(keypoints=kps, facing_right=False, valid=True)

        # Draw with style color
        sfr = StickFigureRenderer(color=profile.color, head_color=profile.color, line_width=4, head_radius=16)
        sfr.draw(screen, game_pose)

        # Labels
        name_surf = font.render(profile.name, True, profile.color)
        x_center = spacing * (i + 1)
        screen.blit(name_surf, (x_center - name_surf.get_width() // 2, 70))

        hp = get_npc_hp(style)
        stats_text = f"HP:{hp} SPD:{cfg.walk_speed:.1f} BLK:{cfg.block_chance:.0%}"
        stats_surf = pygame.font.SysFont("monospace", 12).render(stats_text, True, (180, 180, 180))
        screen.blit(stats_surf, (x_center - stats_surf.get_width() // 2, 90))

        atk_type = npc._attack_type.value if npc._attack_type else "idle"
        atk_surf = pygame.font.SysFont("monospace", 14).render(atk_type.upper(), True, (255, 255, 100))
        screen.blit(atk_surf, (x_center - atk_surf.get_width() // 2, 108))

    path = os.path.join(OUTPUT_DIR, "m10_npc_styles.png")
    pygame.image.save(screen, path)
    print(f"[PASS] All styles rendered -> {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 10 TESTS: Multiple NPC Fighting Styles + Difficulty")
    print("=" * 60)

    test_all_styles_have_configs()
    test_all_styles_have_profiles()
    test_difficulty_scaling()
    test_style_distinct_damage()
    test_style_distinct_behavior()
    test_npc_creation_from_style()
    test_hp_scaling()
    test_list_functions()
    test_nightmare_still_beatable()
    test_speedster_attacks_more()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    test_render_all_styles(screen)
    pygame.quit()

    print("=" * 60)
    print("ALL MILESTONE 10 TESTS PASSED")
    print("=" * 60)
