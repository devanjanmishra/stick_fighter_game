"""
Milestone 9 Test: Combat feel — hit effects, combos, sound.
- Tests particle system (spawn, update, decay)
- Tests screen shake and hit flash
- Tests hitstop (frame freeze on hit)
- Tests combo tracker (sequential hits, multiplier, timeout)
- Tests sound manager initialization and synthesis
- Tests effects renderer visual output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from game.effects import EffectsManager, Particle, ScreenShake, HitFlash
from game.combo_tracker import ComboTracker, ComboState
from game.sound_manager import SoundManager, SoundConfig, _generate_samples
from game.combat_system import CombatSystem, CombatConfig, GamePhase
from rendering.effects_renderer import EffectsRenderer
from rendering.game_renderer import GameRenderer
from rendering.combat_ui import CombatUI

SCREEN_W = 1280
SCREEN_H = 720
OUTPUT_DIR = "/home/ubuntu/stick_fighter/test_output"


def test_particle_lifecycle():
    """Particles should spawn, move, and die."""
    p = Particle(x=100, y=100, vx=5, vy=-3, life=10, max_life=10,
                 color=(255, 200, 100), size=5.0)
    assert p.alive
    assert p.alpha == 1.0

    for _ in range(5):
        p.update()

    assert p.alive
    assert p.alpha == 0.5
    assert p.x > 100  # moved right
    assert p.size < 5.0  # shrinking

    for _ in range(5):
        p.update()

    assert not p.alive
    assert p.alpha == 0.0
    print("[PASS] Particle lifecycle: spawn -> move -> decay -> die")


def test_effects_manager_hit():
    """EffectsManager should spawn particles and damage numbers on hit."""
    em = EffectsManager()
    assert len(em.particles) == 0
    assert len(em.damage_numbers) == 0

    em.trigger_hit(x=500, y=300, move_type="hook", damage=12)

    assert len(em.particles) == 12  # default count
    assert len(em.damage_numbers) == 1
    assert em.damage_numbers[0].value == 12
    assert em.screen_shake.active
    assert em.hit_flash.active
    assert em.hitstop_active  # hook = 4 frames hitstop
    print("[PASS] EffectsManager: hit triggers particles, damage number, shake, flash, hitstop")


def test_hitstop_freezes_effects():
    """During hitstop, effects should not update (game freezes briefly)."""
    em = EffectsManager()
    em.trigger_hit(x=500, y=300, move_type="uppercut", damage=15)

    initial_particle_pos = [(p.x, p.y) for p in em.particles]

    # Update during hitstop — particles should NOT move
    em.update()
    after_hitstop_pos = [(p.x, p.y) for p in em.particles]
    assert initial_particle_pos == after_hitstop_pos, "Particles should freeze during hitstop"

    # Advance past hitstop (uppercut = 5 frames)
    for _ in range(5):
        em.update()

    after_pos = [(p.x, p.y) for p in em.particles]
    assert after_pos != initial_particle_pos, "Particles should move after hitstop ends"
    print("[PASS] Hitstop freezes effects, then they resume")


def test_screen_shake():
    """Screen shake should produce offsets that decay over time."""
    shake = ScreenShake()
    assert not shake.active
    assert shake.offset == (0, 0)

    shake.trigger(intensity=10.0, duration=10)
    assert shake.active

    offsets = []
    for _ in range(10):
        offsets.append(shake.offset)
        shake.update()

    assert not shake.active
    assert shake.offset == (0, 0)
    # At least some offsets should be non-zero
    non_zero = [o for o in offsets if o != (0, 0)]
    assert len(non_zero) > 0, "Shake should produce non-zero offsets"
    print(f"[PASS] Screen shake: {len(non_zero)}/10 non-zero offsets, decays to zero")


def test_hit_flash():
    """Hit flash should have alpha that decays."""
    flash = HitFlash()
    assert not flash.active
    assert flash.current_alpha == 0

    flash.trigger(duration=4)
    assert flash.active
    assert flash.current_alpha > 0

    alphas = []
    for _ in range(4):
        alphas.append(flash.current_alpha)
        flash.update()

    assert not flash.active
    assert flash.current_alpha == 0
    # Alpha should decrease over time
    assert alphas[0] >= alphas[-1], "Flash alpha should decay"
    print(f"[PASS] Hit flash: alphas={alphas}, decays correctly")


def test_effects_cleanup():
    """Dead particles and damage numbers should be cleaned up."""
    em = EffectsManager()
    em.trigger_hit(x=500, y=300, move_type="jab", damage=5)

    initial_count = len(em.particles)
    assert initial_count > 0

    # Run for many frames until all particles die
    for _ in range(100):
        em.update()

    assert len(em.particles) == 0, "Dead particles should be removed"
    assert len(em.damage_numbers) == 0, "Dead damage numbers should be removed"
    print("[PASS] Effects cleanup: all particles and damage numbers removed after death")


def test_combo_basic():
    """Combo should track sequential hits within the window."""
    combo = ComboTracker(combo_window=30)

    # First hit — no combo yet
    dmg1, mult1 = combo.register_hit("jab", 5)
    assert dmg1 == 5
    assert mult1 == 1.0
    assert combo.state.count == 1
    assert combo.state.label == ""

    # Advance a few frames
    for _ in range(10):
        combo.update()

    # Second hit within window
    dmg2, mult2 = combo.register_hit("cross", 8)
    assert mult2 == 1.2
    assert dmg2 == int(8 * 1.2)  # 9
    assert combo.state.count == 2
    assert combo.state.label == "DOUBLE!"

    for _ in range(10):
        combo.update()

    # Third hit
    dmg3, mult3 = combo.register_hit("hook", 12)
    assert mult3 == 1.5
    assert combo.state.count == 3
    assert combo.state.label == "TRIPLE!"

    print(f"[PASS] Combo: 3-hit combo, multipliers 1.0 -> 1.2 -> 1.5")


def test_combo_timeout():
    """Combo should reset if too many frames pass between hits."""
    combo = ComboTracker(combo_window=20)

    combo.register_hit("jab", 5)
    assert combo.state.count == 1

    # Advance beyond the combo window
    for _ in range(25):
        combo.update()

    assert combo.state.count == 0, "Combo should have timed out"

    # New hit starts fresh combo
    dmg, mult = combo.register_hit("cross", 8)
    assert mult == 1.0
    assert combo.state.count == 1
    print("[PASS] Combo timeout: resets after window expires")


def test_combo_max_multiplier():
    """Combo multiplier should cap at 2.0 for 5+ hits."""
    combo = ComboTracker(combo_window=100)

    for i in range(6):
        combo.register_hit("jab", 5)
        for _ in range(5):
            combo.update()

    assert combo.state.count == 6
    assert combo.state.multiplier == 2.0
    assert combo.state.label == "ULTRA!"
    assert combo.best_combo == 6
    print("[PASS] Combo max: 6-hit ULTRA, multiplier capped at 2.0")


def test_sound_synthesis():
    """Sound samples should generate valid PCM data."""
    raw = _generate_samples(freq=440, duration=0.1, sample_rate=22050, decay=5.0, noise_mix=0.0)
    expected_bytes = int(22050 * 0.1) * 2  # 16-bit = 2 bytes per sample
    assert len(raw) == expected_bytes, f"Expected {expected_bytes} bytes, got {len(raw)}"

    # With noise mix
    raw2 = _generate_samples(freq=200, duration=0.2, sample_rate=22050, decay=10.0, noise_mix=0.5)
    expected2 = int(22050 * 0.2) * 2
    assert len(raw2) == expected2
    print(f"[PASS] Sound synthesis: clean={len(raw)} bytes, noisy={len(raw2)} bytes")


def test_sound_manager_init():
    """SoundManager should initialize and create all sounds."""
    pygame.mixer.quit()  # ensure clean state
    sm = SoundManager(SoundConfig(master_volume=0.5))
    success = sm.initialize()

    if success:
        assert sm.initialized
        assert not sm.is_muted
        sm.mute()
        assert sm.is_muted
        sm.unmute()
        assert not sm.is_muted
        print("[PASS] SoundManager: initialized, mute/unmute works")
    else:
        # No audio device (expected in headless env)
        assert not sm.initialized
        sm.play_hit("jab")  # should not crash
        print("[PASS] SoundManager: gracefully handles missing audio device")


def test_render_effects(screen: pygame.Surface):
    """Render combat effects visually."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    renderer = GameRenderer(SCREEN_W, SCREEN_H)
    effects_renderer = EffectsRenderer(SCREEN_W, SCREEN_H)
    ui = CombatUI(SCREEN_W, SCREEN_H)

    em = EffectsManager()
    combo = ComboTracker(combo_window=100)

    # Set up a combat system in fighting state
    combat = CombatSystem(CombatConfig(countdown_seconds=0, fps=30))
    for _ in range(5):
        combat.update()

    # Simulate a 3-hit combo with effects
    hits = [("jab", 5, 550, 350), ("cross", 8, 560, 340), ("hook", 12, 540, 330)]
    for move, base_dmg, hx, hy in hits:
        actual_dmg, _ = combo.register_hit(move, base_dmg)
        em.trigger_hit(hx, hy, move, actual_dmg)
        combat.apply_damage_to_npc(move)
        for _ in range(8):
            em.update()
            combo.update()

    # Render the scene with effects
    renderer.draw_background(screen)
    ui.draw(screen, combat)
    effects_renderer.draw(screen, em, combo)

    path = os.path.join(OUTPUT_DIR, "m9_combat_effects.png")
    pygame.image.save(screen, path)
    print(f"[PASS] Combat effects rendered -> {path}")

    # Scene 2: big uppercut hit with particles and screen shake
    em.clear()
    combo.reset()

    renderer.draw_background(screen)
    combo.register_hit("jab", 5)
    for _ in range(5):
        combo.update()
    combo.register_hit("cross", 8)
    for _ in range(5):
        combo.update()
    combo.register_hit("hook", 12)
    for _ in range(5):
        combo.update()
    actual_dmg, mult = combo.register_hit("uppercut", 15)
    em.trigger_hit(600, 320, "uppercut", actual_dmg)

    # Advance a couple frames so particles spread
    for _ in range(3):
        em.update()

    combat.apply_damage_to_npc("uppercut")
    combat.apply_damage_to_npc("hook")
    combat.apply_damage_to_player("cross")

    ui.draw(screen, combat)
    effects_renderer.draw(screen, em, combo)

    path2 = os.path.join(OUTPUT_DIR, "m9_combo_effects.png")
    pygame.image.save(screen, path2)
    print(f"[PASS] Combo effects rendered (4-hit QUAD, x{mult:.1f}) -> {path2}")


if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 9 TESTS: Combat Feel - Hit Effects, Combos, Sound")
    print("=" * 60)

    test_particle_lifecycle()
    test_effects_manager_hit()
    test_hitstop_freezes_effects()
    test_screen_shake()
    test_hit_flash()
    test_effects_cleanup()
    test_combo_basic()
    test_combo_timeout()
    test_combo_max_multiplier()
    test_sound_synthesis()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))

    test_sound_manager_init()
    test_render_effects(screen)

    pygame.quit()

    print("=" * 60)
    print("ALL MILESTONE 9 TESTS PASSED")
    print("=" * 60)
