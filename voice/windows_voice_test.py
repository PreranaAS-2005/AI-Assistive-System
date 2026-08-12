import subprocess

messages = [
    "Person detected",
    "Bottle detected",
    "Cell phone detected"
]

for message in messages:
    print("Speaking:", message)

    subprocess.run([
        "powershell",
        "-Command",
        f'Add-Type -AssemblyName System.Speech; '
        f'$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
        f'$speaker.Speak("{message}")'
    ])

print("Voice test completed.")