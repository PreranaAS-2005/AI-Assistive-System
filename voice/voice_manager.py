import subprocess


class VoiceManager:

    def speak(self, message):

        print(f"Speaking: {message}")

        subprocess.run([
            "powershell",
            "-Command",
            f'Add-Type -AssemblyName System.Speech; '
            f'$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            f'$speaker.Speak("{message}")'
        ])