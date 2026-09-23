"""
voice/voice_input.py

Non-blocking speech-to-text command listener for the AI-Based Assistive
Spatial Memory and Guidance System.

This module listens to the microphone in a background thread and puts
recognised commands into a thread-safe queue. The main navigation loop
drains the queue once per frame — zero blocking.

Architecture
------------
  VoiceInput (background thread)
    → microphone capture
    → Google STT (online, primary)
    → Windows SAPI STT (offline, fallback)
    → command queue (maxsize=5)

The navigation loop calls:
    command = voice_input.get_command()   # non-blocking, returns None or str

Dependencies already in .venv
------------------------------
  SpeechRecognition==3.10.4
  pyaudio==0.2.14

No new pip packages required.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Supported command keywords (used for validation after STT)
# ---------------------------------------------------------------------------

_FIND_VERBS = {"find", "search", "locate", "where"}

_CLEAR_CMDS = {"clear", "stop", "stop search", "cancel"}

_HELP_CMDS  = {"help"}


# ---------------------------------------------------------------------------
# VoiceInput
# ---------------------------------------------------------------------------

class VoiceInput:
    """
    Background speech recognition feeding a command queue.

    Parameters
    ----------
    device_index : int or None
        Microphone device index. None = system default.
        Use list_microphones() to discover indices.
    energy_threshold : int
        Ambient noise threshold for SpeechRecognition.
        Lower = more sensitive. Default 300 works well indoors.
    pause_threshold : float
        Seconds of silence that end a phrase. Default 0.8 s.
    queue_maxsize : int
        Maximum queued commands. Oldest is dropped if full.
    """

    def __init__(
        self,
        device_index    : Optional[int] = None,
        energy_threshold: int   = 300,
        pause_threshold : float = 0.8,
        queue_maxsize   : int   = 5,
    ) -> None:

        if not _SR_AVAILABLE:
            raise RuntimeError(
                "SpeechRecognition is not installed. "
                "Run: pip install SpeechRecognition pyaudio"
            )

        self._device_index     = device_index
        self._energy_threshold = energy_threshold
        self._pause_threshold  = pause_threshold
        self._command_queue: queue.Queue[str] = queue.Queue(
            maxsize=queue_maxsize
        )
        self._running  = False
        self._thread   : Optional[threading.Thread] = None
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background listening thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="VoiceInputThread",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._running = False

    def get_command(self) -> Optional[str]:
        """
        Non-blocking drain of one command from the queue.

        Returns the next recognised command string, or None if
        nothing is queued.

        The returned string is always lowercase and stripped.
        """
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None

    def get_last_error(self) -> Optional[str]:
        """Return the last recognition error message, or None."""
        return self._last_error

    @staticmethod
    def list_microphones() -> list[str]:
        """Return list of available microphone device names."""
        if not _SR_AVAILABLE:
            return []
        return sr.Microphone.list_microphone_names()

    # ------------------------------------------------------------------
    # Background listening loop
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        """Continuously listen and recognise in the background."""

        recognizer = sr.Recognizer()
        recognizer.energy_threshold      = self._energy_threshold
        recognizer.pause_threshold       = self._pause_threshold
        recognizer.dynamic_energy_threshold = True

        # Calibrate ambient noise once on startup
        try:
            with sr.Microphone(
                device_index=self._device_index
            ) as source:
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
        except Exception as e:
            self._last_error = f"Microphone init error: {e}"
            self._running = False
            return

        while self._running:
            try:
                with sr.Microphone(
                    device_index=self._device_index
                ) as source:
                    # Listen with a reasonable timeout so the thread
                    # remains responsive to stop() calls
                    audio = recognizer.listen(
                        source,
                        timeout=5.0,
                        phrase_time_limit=6.0,
                    )

                text = self._recognise(recognizer, audio)

                if text:
                    self._last_error = None
                    self._enqueue(text)

            except sr.WaitTimeoutError:
                # No speech detected in this window — continue
                continue

            except sr.UnknownValueError:
                # Speech detected but not intelligible — skip silently
                continue

            except Exception as e:
                self._last_error = str(e)
                # Brief back-off before retrying to avoid tight loops
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # Speech recognition (Google primary, SAPI fallback)
    # ------------------------------------------------------------------

    def _recognise(
        self,
        recognizer: "sr.Recognizer",
        audio: "sr.AudioData",
    ) -> Optional[str]:
        """
        Attempt recognition. Returns lowercase text or None.

        Primary:  Google Web Speech API (free, requires internet).
        Fallback: Windows SAPI recogniser via Sphinx or native COM
                  — only attempted if Google raises RequestError.
        """
        # --- Google (online) ---
        try:
            text = recognizer.recognize_google(audio)
            return text.lower().strip()

        except sr.RequestError:
            # Network unavailable — try offline fallback
            pass
        except sr.UnknownValueError:
            return None

        # --- Windows SAPI offline fallback via Sphinx ---
        try:
            text = recognizer.recognize_sphinx(audio)
            return text.lower().strip()
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _enqueue(self, text: str) -> None:
        """
        Put a recognised command into the queue.
        Drop the oldest item if the queue is full (keeps queue fresh).
        """
        text = text.strip().lower()
        if not text:
            return

        # Only queue items that look like valid commands
        if not self._is_relevant(text):
            return

        if self._command_queue.full():
            try:
                self._command_queue.get_nowait()   # discard oldest
            except queue.Empty:
                pass

        try:
            self._command_queue.put_nowait(text)
        except queue.Full:
            pass

    @staticmethod
    def _is_relevant(text: str) -> bool:
        """
        Return True only for text that contains a known command verb.
        Filters out random background speech.
        """
        words = set(text.split())

        if words & _CLEAR_CMDS:
            return True

        if words & _HELP_CMDS:
            return True

        # "find X", "search for X", "locate X", "where is X"
        if words & _FIND_VERBS:
            return True

        # Bare object names are also valid (single-word fallback)
        return False
