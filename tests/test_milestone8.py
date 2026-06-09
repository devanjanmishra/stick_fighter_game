"""
Milestone 8 Test: HP, rounds, timer, basic combat UI.
- Tests combat system state machine (countdown, fighting, round end, match end)
- Tests HP damage and KO
- Tests round timer
- Tests multi-round match flow
- Renders combat UI
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from game.combat_system import CombatSystem, CombatConfig, GamePhase, RoundResult
from rendering.combat_ui import CombatUI
from rendering.game_renderer import GameRenderer

SCREEN_W = 1280
SCREEN_H = 720
OUTPUT_DIR = "/home/ubuntu/stick_fighter/test_output"


def test_initial_state():
    """Combat system should start in countdown phase."""
    combat = CombatSystem()
    assert combat.phase == GamePhase.COUNTDOWN
    assert combat.current_round == 1
    assert combat.player.current_hp == 100
    assert combat.npc.current_hp == 100
    assert combat.player.rounds_won == 0
    assert combat.npc.rounds_won == 0
    print("[PASS] Initial state: countdown, round 1, HP=100/100")


def test_countdown_to_fighting():
    """System should transition from countdown to fighting."""
    combat = CombatSystem(CombatConfig(countdown_seconds=1, fps=30))

    # Advance through countdown (1 second = 30 frames)
    for _ in range(35):
        combat.update()

    assert combat.phase == GamePhase.FIGHTING, f"Should be fighting, got {combat.phase}"
    print(f"[PASS] Countdown -> Fighting transition after {35} frames")


def test_damage_and_ko():
    """Dealing enough damage should KO and end the round."""
    combat = CombatSystem(CombatConfig(countdown_seconds=0, fps=30))
    # Skip countdown
    for _ in range(5):
        combat.update()

    assert combat.phase == GamePhase.FIGHTING

    # Deal damage to NPC
    total_damage = 0
    for _ in range(20):
        dmg = combat.apply_damage_to_npc("uppercut")  # 15 damage each
        total_damage += dmg
        combat.update()
        if combat.npc.is_ko:
            break

    assert combat.npc.is_ko, f"NPC should be KO, HP={combat.npc.current_hp}"
    assert combat.phase in (GamePhase.ROUND_END, GamePhase.FIGHTING)
    # Update one more to trigger round end
    combat.update()
    assert combat.phase == GamePhase.ROUND_END
    print(f"[PASS] NPC KO after {total_damage} damage, round ended")


def test_timer_countdown():
    """Round timer should count down."""
    combat = CombatSystem(CombatConfig(countdown_seconds=0, fps=30, round_time_seconds=5))
    # Skip countdown
    for _ in range(5):
        combat.update()

    initial_time = combat.round_timer_seconds
    # Advance 60 frames (2 seconds)
    for _ in range(60):
        combat.update()

    later_time = combat.round_timer_seconds
    assert later_time < initial_time, f"Timer should decrease: {initial_time} -> {later_time}"
    print(f"[PASS] Timer countdown: {initial_time}s -> {later_time}s")


def test_time_up_higher_hp_wins():
    """When time runs out, the fighter with more HP wins the round."""
    combat = CombatSystem(CombatConfig(countdown_seconds=0, fps=30, round_time_seconds=2))
    # Skip countdown
    for _ in range(5):
        combat.update()

    # Damage NPC slightly
    combat.apply_damage_to_npc("jab")  # 5 damage

    # Run out the clock (2 seconds = 60 frames)
    for _ in range(70):
        combat.update()

    assert len(combat.round_results) > 0, "Round should have ended"
    assert combat.round_results[-1] == RoundResult.PLAYER_WIN, \
        f"Player had more HP, should win: {combat.round_results[-1]}"
    print(f"[PASS] Time up: player wins with more HP ({combat.player.current_hp} vs {combat.npc.current_hp})")


def test_multi_round_match():
    """Full match should play through multiple rounds."""
    combat = CombatSystem(CombatConfig(
        countdown_seconds=0, fps=30, rounds_to_win=2, max_rounds=3,
        round_end_pause_seconds=0.1,
    ))

    # Play through rounds by KOing NPC twice
    for round_num in range(2):
        # Skip countdown
        for _ in range(5):
            combat.update()

        # KO the NPC
        while combat.phase == GamePhase.FIGHTING:
            combat.apply_damage_to_npc("uppercut")
            combat.update()

        # Wait through round end pause
        for _ in range(30):
            combat.update()

    assert combat.player.rounds_won == 2, f"Player should have 2 wins, got {combat.player.rounds_won}"
    assert combat.phase == GamePhase.MATCH_END, f"Should be match end, got {combat.phase}"
    assert combat.match_result == RoundResult.PLAYER_WIN
    print(f"[PASS] Multi-round match: player wins 2-0, match ended")


def test_npc_wins_match():
    """NPC should be able to win the match."""
    combat = CombatSystem(CombatConfig(
        countdown_seconds=0, fps=30, rounds_to_win=2, max_rounds=3,
        round_end_pause_seconds=0.1,
    ))

    for round_num in range(2):
        for _ in range(5):
            combat.update()

        while combat.phase == GamePhase.FIGHTING:
            combat.apply_damage_to_player("cross")
            combat.update()

        for _ in range(30):
            combat.update()

    assert combat.npc.rounds_won == 2
    assert combat.match_result == RoundResult.NPC_WIN
    print(f"[PASS] NPC wins match 2-0")


def test_damage_table():
    """Different moves should deal different damage."""
    combat = CombatSystem(CombatConfig(countdown_seconds=0, fps=30))
    for _ in range(5):
        combat.update()

    jab_dmg = combat.apply_damage_to_npc("jab")
    assert jab_dmg == 5, f"Jab should do 5 damage, got {jab_dmg}"

    combat.npc.reset_hp()
    cross_dmg = combat.apply_damage_to_npc("cross")
    assert cross_dmg == 8, f"Cross should do 8 damage, got {cross_dmg}"

    combat.npc.reset_hp()
    hook_dmg = combat.apply_damage_to_npc("hook")
    assert hook_dmg == 12, f"Hook should do 12 damage, got {hook_dmg}"

    combat.npc.reset_hp()
    upper_dmg = combat.apply_damage_to_npc("uppercut")
    assert upper_dmg == 15, f"Uppercut should do 15 damage, got {upper_dmg}"

    print(f"[PASS] Damage table: jab={jab_dmg}, cross={cross_dmg}, hook={hook_dmg}, uppercut={upper_dmg}")


def test_reset_match():
    """Resetting should restore everything to initial state."""
    combat = CombatSystem(CombatConfig(countdown_seconds=0, fps=30))
    for _ in range(5):
        combat.update()

    combat.apply_damage_to_npc("hook")
    combat.player.rounds_won = 1

    combat.reset_match()

    assert combat.player.current_hp == 100
    assert combat.npc.current_hp == 100
    assert combat.player.rounds_won == 0
    assert combat.current_round == 1
    assert combat.phase == GamePhase.COUNTDOWN
    print("[PASS] Match reset: all state restored")


def test_render_combat_ui(screen: pygame.Surface):
    """Render the combat UI with various states."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    renderer = GameRenderer(SCREEN_W, SCREEN_H)
    ui = CombatUI(SCREEN_W, SCREEN_H)

    # Scene 1: Mid-fight with some damage
    renderer.draw_background(screen)
    combat = CombatSystem(CombatConfig(countdown_seconds=0, fps=30))
    for _ in range(5):
        combat.update()
    combat.apply_damage_to_npc("hook")
    combat.apply_damage_to_npc("jab")
    combat.apply_damage_to_player("cross")
    combat.player.rounds_won = 1
    ui.draw(screen, combat)

    path = os.path.join(OUTPUT_DIR, "m8_combat_ui_fighting.png")
    pygame.image.save(screen, path)
    print(f"[PASS] Combat UI (fighting) rendered -> {path}")

    # Scene 2: Match end (player wins)
    renderer.draw_background(screen)
    combat2 = CombatSystem(CombatConfig(countdown_seconds=0, fps=30, rounds_to_win=2, round_end_pause_seconds=0.1))
    for _ in range(5):
        combat2.update()
    for _ in range(2):
        while combat2.phase == GamePhase.FIGHTING:
            combat2.apply_damage_to_npc("uppercut")
            combat2.update()
        for _ in range(30):
            combat2.update()

    ui.draw(screen, combat2)
    path2 = os.path.join(OUTPUT_DIR, "m8_combat_ui_match_end.png")
    pygame.image.save(screen, path2)
    print(f"[PASS] Combat UI (match end) rendered -> {path2}")


if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 8 TESTS: HP, Rounds, Timer, Basic Combat UI")
    print("=" * 60)

    test_initial_state()
    test_countdown_to_fighting()
    test_damage_and_ko()
    test_timer_countdown()
    test_time_up_higher_hp_wins()
    test_multi_round_match()
    test_npc_wins_match()
    test_damage_table()
    test_reset_match()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    test_render_combat_ui(screen)
    pygame.quit()

    print("=" * 60)
    print("ALL MILESTONE 8 TESTS PASSED")
    print("=" * 60)
