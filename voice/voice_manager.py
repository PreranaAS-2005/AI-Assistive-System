import subprocess
import threading
import queue


class VoiceManager:

    def __init__(self):
        self.speech_queue = queue.Queue(maxsize=1)

        self.voice_thread = threading.Thread(
            target=self._voice_worker,
            daemon=True
        )

        self.voice_thread.start()

    def _voice_worker(self):

        while True:

            message = self.speech_queue.get()

            if message is None:
                self.speech_queue.task_done()
                break

            print(f"Speaking: {message}")

            subprocess.run([
                "powershell",
                "-Command",
                f'Add-Type -AssemblyName System.Speech; '
                f'$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$speaker.Speak("{message}")'
            ])

            self.speech_queue.task_done()

    def speak(self, message):

        # If a message is already waiting, don't add another one.
        if self.speech_queue.full():
            return

        self.speech_queue.put(message)