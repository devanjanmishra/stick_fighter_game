"""
Voice recognizer for move labeling in Dojo training mode.

Runs speech recognition in a background thread, listening for move keywords
(jab, cross, hook, uppercut, walk/walking, idle/stop). When a keyword is
detected it is timestamped and queued for the main game loop to consume.

Supports two backends:
  - "google" (default): Google Web Speech API via SpeechRecognition library.
    Requires internet. More accurate.
  - "sphinx" : CMU Sphinx offline recognizer. No internet needed but less
    accurate.  Requires pocketsphinx installed.

Usage:
    vr = VoiceRecognizer()
    vr.start()
    ...
    while True:
        labels = vr.get_labels()   # non-blocking, returns list of (time, label)
        ...
    vr.stop()
"""
import queue
import threading
import time
from dataclasses import dataclass, field


# Move keywords we listen for, mapped to canonical move names.
KEYWORD_MAP: dict[str, str] = {
    "jab": "jab",
    "jabs": "jab",
    "job": "jab",       # common misrecognition
    "jump": "jab",      # common misrecognition
    "cross": "cross",
    "crosses": "cross",
    "across": "cross",  # common misrecognition
    "hook": "hook",
    "hooks": "hook",
    "who": "hook",      # common misrecognition
    "uppercut": "uppercut",
    "upper cut": "uppercut",
    "upper": "uppercut",
    "cut": "uppercut",
    "walk": "walking",
    "walking": "walking",
    "forward": "walking",
    "backward": "walking",
    "idle": "idle",
    "stop": "idle",
    "nothing": "idle",
}

VALID_LABELS = {"jab", "cross", "hook", "uppercut", "walking", "idle"}


@dataclass
class VoiceLabel:
    """A single voice-recognized move label with timestamp."""
    timestamp: float          # time.time() when recognized
    label: str                # canonical move name
    raw_text: str = ""        # raw recognized text
    confidence: float = 1.0   # recognition confidence (0-1)


class VoiceRecognizer:
    """
    Background threaded speech recognizer that listens for move keywords.

    Parameters
    ----------
    backend : str
        "google" for Google Web Speech API, "sphinx" for offline.
    energy_threshold : int
        Microphone energy threshold for speech detection.
        Higher = less sensitive to background noise.
    pause_threshold : float
        Seconds of non-speaking audio before a phrase is considered complete.
        Lower = faster response but more fragmented phrases.
    phrase_time_limit : float
        Max seconds for a single phrase. Shorter = faster turnaround.
    """

    def __init__(
        self,
        backend: str = "google",
        energy_threshold: int = 300,
        pause_threshold: float = 0.5,
        phrase_time_limit: float = 2.0,
    ):
        self._backend = backend
        self._energy_threshold = energy_threshold
        self._pause_threshold = pause_threshold
        self._phrase_time_limit = phrase_time_limit

        self._label_queue: queue.Queue[VoiceLabel] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._error: str | None = None
        self._last_label_time: float = 0.0
        self._min_label_gap: float = 0.4  # min seconds between labels

        # Diagnostics
        self._listen_count: int = 0        # total listen attempts
        self._recognize_count: int = 0     # successful recognitions
        self._unknown_count: int = 0       # speech not understood
        self._api_error_count: int = 0     # API errors
        self._last_raw_text: str = ""      # last raw recognized text

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> bool:
        """
        Start the background listener thread.
        Returns True if started successfully, False on error.
        """
        if self._running:
            return True

        self._error = None
        self._stop_event.clear()

        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = self._energy_threshold
            self._recognizer.pause_threshold = self._pause_threshold
            self._recognizer.dynamic_energy_threshold = True

            self._mic = sr.Microphone()
            # Ambient noise calibration
            print("[VOICE] Calibrating microphone for ambient noise (1s)...")
            with self._mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print(f"[VOICE] Mic ready. Energy threshold: {self._recognizer.energy_threshold:.0f}")

        except (ImportError, OSError, AttributeError) as exc:
            self._error = f"Mic init failed: {exc}"
            print(f"[VOICE] ERROR: {self._error}")
            return False

        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="VoiceRecognizer"
        )
        self._running = True
        self._thread.start()
        print("[VOICE] Listening started. Shout move names!")
        return True

    def stop(self) -> None:
        """Stop the background listener."""
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        print(f"[VOICE] Stopped. Stats: {self._listen_count} listens, "
              f"{self._recognize_count} recognized, {self._unknown_count} unknown, "
              f"{self._api_error_count} API errors")

    def get_labels(self) -> list[VoiceLabel]:
        """
        Non-blocking: return all labels recognized since last call.
        Returns empty list if nothing new.
        """
        labels = []
        while True:
            try:
                labels.append(self._label_queue.get_nowait())
            except queue.Empty:
                break
        return labels

    def _listen_loop(self) -> None:
        """Background loop: listen → recognize → enqueue labels."""
        import speech_recognition as sr

        while not self._stop_event.is_set():
            self._listen_count += 1
            try:
                with self._mic as source:
                    audio = self._recognizer.listen(
                        source,
                        timeout=3.0,
                        phrase_time_limit=self._phrase_time_limit,
                    )
                capture_time = time.time()
            except sr.WaitTimeoutError:
                continue
            except Exception as exc:
                self._error = f"Listen error: {exc}"
                print(f"[VOICE] Listen error: {exc}")
                continue

            # Recognize
            try:
                if self._backend == "google":
                    text = self._recognizer.recognize_google(audio).lower().strip()
                elif self._backend == "sphinx":
                    text = self._recognizer.recognize_sphinx(audio).lower().strip()
                else:
                    text = self._recognizer.recognize_google(audio).lower().strip()
            except sr.UnknownValueError:
                self._unknown_count += 1
                print(f"[VOICE] (could not understand audio — try louder/clearer)")
                continue
            except sr.RequestError as exc:
                self._api_error_count += 1
                self._error = f"API error: {exc}"
                print(f"[VOICE] API ERROR: {exc}")
                continue

            self._recognize_count += 1
            self._last_raw_text = text
            print(f"[VOICE] Heard: \"{text}\"")

            # Match keywords
            label = self._match_keyword(text)
            if label is not None:
                now = time.time()
                if now - self._last_label_time >= self._min_label_gap:
                    self._label_queue.put(VoiceLabel(
                        timestamp=capture_time,
                        label=label,
                        raw_text=text,
                    ))
                    self._last_label_time = now
                    print(f"[VOICE] >>> LABEL: {label.upper()} (from \"{text}\")")
            else:
                print(f"[VOICE] No keyword match for: \"{text}\"")

    @staticmethod
    def _match_keyword(text: str) -> str | None:
        """Match recognized text to a canonical move label."""
        # Direct match first
        if text in KEYWORD_MAP:
            return KEYWORD_MAP[text]

        # Check if any keyword appears as a substring
        words = text.split()
        for word in words:
            if word in KEYWORD_MAP:
                return KEYWORD_MAP[word]

        # Fuzzy: check if any keyword is contained in the text
        for keyword, label in KEYWORD_MAP.items():
            if keyword in text:
                return label

        return None
