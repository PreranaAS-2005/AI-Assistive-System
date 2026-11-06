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


_FIND_VERBS = {"find", "search", "locate", "where"}
_CLEAR_CMDS = {"clear", "stop", "cancel"}
_HELP_CMDS = {"help"}


class VoiceInput:

    def __init__(
        self,
        device_index: Optional[int] = None,
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
        queue_maxsize: int = 5,
    ) -> None:

        if not _SR_AVAILABLE:
            raise RuntimeError(
                "SpeechRecognition is not installed. "
                "Run: pip install SpeechRecognition pyaudio"
            )

        self._device_index = device_index
        self._energy_threshold = energy_threshold
        self._pause_threshold = pause_threshold

        self._command_queue: queue.Queue[str] = queue.Queue(
            maxsize=queue_maxsize
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None

    # ---------------------------------------------------------
    # START
    # ---------------------------------------------------------

    def start(self) -> None:

        if self._running:
            return

        print("[VOICE] Starting voice listener...")

        self._running = True

        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="VoiceInputThread",
        )

        self._thread.start()

    # ---------------------------------------------------------
    # STOP
    # ---------------------------------------------------------

    def stop(self) -> None:

        print("[VOICE] Stopping voice listener...")

        self._running = False

    # ---------------------------------------------------------
    # GET COMMAND
    # ---------------------------------------------------------

    def get_command(self) -> Optional[str]:

        try:
            command = self._command_queue.get_nowait()

            print(f"[VOICE] Command received from queue: {command}")

            return command

        except queue.Empty:
            return None

    # ---------------------------------------------------------
    # ERROR
    # ---------------------------------------------------------

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    # ---------------------------------------------------------
    # MICROPHONES
    # ---------------------------------------------------------

    @staticmethod
    def list_microphones() -> list[str]:

        if not _SR_AVAILABLE:
            return []

        return sr.Microphone.list_microphone_names()

    # ---------------------------------------------------------
    # LISTEN LOOP
    # ---------------------------------------------------------

    def _listen_loop(self) -> None:

        print("[VOICE] Voice thread started.")

        recognizer = sr.Recognizer()

        recognizer.energy_threshold = self._energy_threshold
        recognizer.pause_threshold = self._pause_threshold
        recognizer.dynamic_energy_threshold = True

        # -----------------------------------------------------
        # MICROPHONE TEST
        # -----------------------------------------------------

        try:

            print("[VOICE] Opening microphone...")

            with sr.Microphone(
                device_index=self._device_index
            ) as source:

                print("[VOICE] Calibrating microphone...")

                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=1.0
                )

                print(
                    f"[VOICE] Calibration complete. "
                    f"Energy threshold = {recognizer.energy_threshold}"
                )

        except Exception as e:

            self._last_error = f"Microphone init error: {e}"

            print(f"[VOICE ERROR] {self._last_error}")

            self._running = False

            return

        # -----------------------------------------------------
        # CONTINUOUS LISTENING
        # -----------------------------------------------------

        print("[VOICE] Listening for commands...")

        while self._running:

            try:

                with sr.Microphone(
                    device_index=self._device_index
                ) as source:

                    print("[VOICE] Listening...")

                    audio = recognizer.listen(
                        source,
                        timeout=5.0,
                        phrase_time_limit=6.0,
                    )

                print("[VOICE] Speech detected. Recognizing...")

                text = self._recognise(
                    recognizer,
                    audio
                )

                if text:

                    print(f"[VOICE] Recognized: '{text}'")

                    self._last_error = None

                    self._enqueue(text)

                else:

                    print("[VOICE] Could not recognize speech.")

            except sr.WaitTimeoutError:

                # No speech during the timeout.
                continue

            except sr.UnknownValueError:

                print("[VOICE] Speech was not understandable.")

                continue

            except Exception as e:

                self._last_error = str(e)

                print(f"[VOICE ERROR] {e}")

                time.sleep(1)

    # ---------------------------------------------------------
    # SPEECH RECOGNITION
    # ---------------------------------------------------------

    def _recognise(
        self,
        recognizer: "sr.Recognizer",
        audio: "sr.AudioData",
    ) -> Optional[str]:

        # -----------------------------------------------------
        # GOOGLE STT
        # -----------------------------------------------------

        try:

            print("[VOICE] Trying Google Speech Recognition...")

            text = recognizer.recognize_google(audio)

            return text.lower().strip()

        except sr.UnknownValueError:

            print("[VOICE] Google could not understand speech.")

            return None

        except sr.RequestError as e:

            print(
                f"[VOICE] Google Speech Recognition unavailable: {e}"
            )

        # -----------------------------------------------------
        # SPHINX FALLBACK
        # -----------------------------------------------------

        try:

            print("[VOICE] Trying offline recognition...")

            text = recognizer.recognize_sphinx(audio)

            return text.lower().strip()

        except Exception as e:

            print(
                f"[VOICE] Offline recognition unavailable: {e}"
            )

        return None

    # ---------------------------------------------------------
    # QUEUE
    # ---------------------------------------------------------

    def _enqueue(self, text: str) -> None:

        text = text.strip().lower()

        if not text:
            return

        if not self._is_relevant(text):

            print(
                f"[VOICE] Ignored non-command speech: '{text}'"
            )

            return

        print(
            f"[VOICE] Valid command detected: '{text}'"
        )

        if self._command_queue.full():

            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                pass

        try:

            self._command_queue.put_nowait(text)

            print(
                f"[VOICE] Command added to queue: '{text}'"
            )

        except queue.Full:

            pass

    # ---------------------------------------------------------
    # COMMAND VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _is_relevant(text: str) -> bool:

        words = set(text.split())

        # clear / stop / cancel
        if words & _CLEAR_CMDS:
            return True

        # help
        if words & _HELP_CMDS:
            return True

        # find / search / locate / where
        if words & _FIND_VERBS:
            return True

        return False