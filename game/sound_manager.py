"""
Sound manager: generates and plays combat sound effects using pygame.mixer.

Since we don't have audio files, we synthesize simple sounds programmatically
using numpy waveforms converted to pygame Sound objects.
"""

import math
import struct
from dataclasses import dataclass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pygame
    HAS_PYGAME_MIXER = True
except ImportError:
    HAS_PYGAME_MIXER = False


@dataclass
class SoundConfig:
    master_volume: float = 0.5
    hit_volume: float = 0.7
    whoosh_volume: float = 0.3
    ui_volume: float = 0.5
    sample_rate: int = 22050


def _generate_samples_pure(freq: float, duration: float, sample_rate: int,
                           decay: float = 5.0, noise_mix: float = 0.0) -> bytes:
    """Generate sound samples using pure Python (no numpy)."""
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        envelope = math.exp(-decay * t)
        sine = math.sin(2 * math.pi * freq * t)

        noise = 0.0
        if noise_mix > 0:
            # Simple pseudo-noise via multiple sines
            noise = (math.sin(2 * math.pi * 1337 * t + 0.7) * 0.5 +
                     math.sin(2 * math.pi * 2671 * t + 1.3) * 0.3 +
                     math.sin(2 * math.pi * 4001 * t + 2.1) * 0.2)

        value = envelope * ((1 - noise_mix) * sine + noise_mix * noise)
        clamped = max(-1.0, min(1.0, value))
        samples.append(int(clamped * 32767))

    return struct.pack(f"<{n_samples}h", *samples)


def _generate_samples_np(freq: float, duration: float, sample_rate: int,
                         decay: float = 5.0, noise_mix: float = 0.0) -> bytes:
    """Generate sound samples using numpy (faster)."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    envelope = np.exp(-decay * t)
    sine = np.sin(2 * np.pi * freq * t)

    if noise_mix > 0:
        noise = (np.sin(2 * np.pi * 1337 * t + 0.7) * 0.5 +
                 np.sin(2 * np.pi * 2671 * t + 1.3) * 0.3 +
                 np.sin(2 * np.pi * 4001 * t + 2.1) * 0.2)
        signal = envelope * ((1 - noise_mix) * sine + noise_mix * noise)
    else:
        signal = envelope * sine

    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767).astype(np.int16)
    return pcm.tobytes()


def _generate_samples(freq: float, duration: float, sample_rate: int,
                      decay: float = 5.0, noise_mix: float = 0.0) -> bytes:
    if HAS_NUMPY:
        return _generate_samples_np(freq, duration, sample_rate, decay, noise_mix)
    return _generate_samples_pure(freq, duration, sample_rate, decay, noise_mix)


# Sound specifications per event type
SOUND_SPECS = {
    "jab_hit":      {"freq": 300, "duration": 0.12, "decay": 15, "noise_mix": 0.6},
    "cross_hit":    {"freq": 250, "duration": 0.15, "decay": 12, "noise_mix": 0.7},
    "hook_hit":     {"freq": 200, "duration": 0.20, "decay": 10, "noise_mix": 0.8},
    "uppercut_hit": {"freq": 180, "duration": 0.25, "decay": 8,  "noise_mix": 0.85},
    "block":        {"freq": 400, "duration": 0.08, "decay": 20, "noise_mix": 0.3},
    "whoosh":       {"freq": 800, "duration": 0.10, "decay": 25, "noise_mix": 0.2},
    "round_bell":   {"freq": 660, "duration": 0.50, "decay": 3,  "noise_mix": 0.0},
    "ko":           {"freq": 120, "duration": 0.60, "decay": 2,  "noise_mix": 0.5},
    "countdown":    {"freq": 440, "duration": 0.15, "decay": 8,  "noise_mix": 0.0},
    "fight":        {"freq": 550, "duration": 0.30, "decay": 4,  "noise_mix": 0.0},
}


class SoundManager:
    """Manages combat sound effects."""

    def __init__(self, config: SoundConfig | None = None):
        self.config = config or SoundConfig()
        self._sounds: dict[str, "pygame.mixer.Sound"] = {}
        self._initialized = False
        self._muted = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> bool:
        """Initialize the mixer and pre-generate all sounds."""
        if not HAS_PYGAME_MIXER:
            return False

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=self.config.sample_rate,
                    size=-16,
                    channels=1,
                    buffer=512,
                )
        except pygame.error:
            return False

        for name, spec in SOUND_SPECS.items():
            raw = _generate_samples(
                freq=spec["freq"],
                duration=spec["duration"],
                sample_rate=self.config.sample_rate,
                decay=spec["decay"],
                noise_mix=spec["noise_mix"],
            )
            sound = pygame.mixer.Sound(buffer=raw)
            # Set volume based on category
            if "hit" in name or name == "ko":
                sound.set_volume(self.config.hit_volume * self.config.master_volume)
            elif name == "whoosh":
                sound.set_volume(self.config.whoosh_volume * self.config.master_volume)
            else:
                sound.set_volume(self.config.ui_volume * self.config.master_volume)
            self._sounds[name] = sound

        self._initialized = True
        return True

    def play(self, sound_name: str):
        """Play a named sound effect."""
        if self._muted or not self._initialized:
            return
        sound = self._sounds.get(sound_name)
        if sound:
            sound.play()

    def play_hit(self, move_type: str):
        """Play the appropriate hit sound for a move."""
        self.play(f"{move_type}_hit")

    def play_whoosh(self):
        self.play("whoosh")

    def play_block(self):
        self.play("block")

    def play_round_bell(self):
        self.play("round_bell")

    def play_ko(self):
        self.play("ko")

    def play_countdown(self):
        self.play("countdown")

    def play_fight(self):
        self.play("fight")

    def mute(self):
        self._muted = True

    def unmute(self):
        self._muted = False

    @property
    def is_muted(self) -> bool:
        return self._muted

    def set_master_volume(self, vol: float):
        self.config.master_volume = max(0.0, min(1.0, vol))
        if self._initialized:
            for name, sound in self._sounds.items():
                if "hit" in name or name == "ko":
                    sound.set_volume(self.config.hit_volume * self.config.master_volume)
                elif name == "whoosh":
                    sound.set_volume(self.config.whoosh_volume * self.config.master_volume)
                else:
                    sound.set_volume(self.config.ui_volume * self.config.master_volume)
