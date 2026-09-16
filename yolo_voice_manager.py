import sys
import time
import subprocess
import speech_recognition as sr
from voice.voice_manager import VoiceManager

voice = VoiceManager()


def listen_and_route():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    with sr.Microphone() as source:
        print("🎤 Calibrating microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=2)

        print("\n" + "=" * 50)
        print("🎯 ASSISTIVE SYSTEM VOICE CONTROLLER")
        print("Voice Commands:")
        print(" • 'Face recognition' -> Identifies enrolled people")
        print(" • 'Find <item>'       -> Guides you to an item (e.g. bottle, chair)")
        print(" • 'Add person'        -> Enrolls a new person's face")
        print(" • 'Exit' or 'Quit'    -> Stops the program")
        print("=" * 50 + "\n")

        while True:
            try:
                print("🎤 Listening for command...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=4)
                command = recognizer.recognize_google(audio).lower().strip()
                print(f"🗣️ Heard: '{command}'")

                # 1. Face Recognition Mode
                if any(phrase in command for phrase in ["face recognition", "face", "who is", "recognize"]):
                    print("\n🚀 Launching Face Recognition Module...")
                    voice.speak("Starting face recognition.")
                    subprocess.run([sys.executable, "face_recognition_manager.py"])
                    print("📷 Returned to Voice Controller.\n")

                # 2. Add Person Mode
                elif any(phrase in command for phrase in ["add person", "new person", "add someone"]):
                    print("\n🚀 Launching Add Person Module...")
                    voice.speak("Starting add person mode.")
                    subprocess.run([sys.executable, "add_person.py"])
                    print("📷 Returned to Voice Controller.\n")

                # 3. Targeted Object Navigation Mode
                elif any(command.startswith(trig) or f" {trig} " in f" {command} " for trig in ["find", "search", "where is", "locate"]):
                    clean_target = command
                    for filler in [
                        "where is the", "where is my", "where is",
                        "find my", "find the", "find",
                        "search for", "search my", "search",
                        "locate my", "locate the", "locate",
                    ]:
                        clean_target = clean_target.replace(filler, "")

                    target_item = clean_target.strip().split()[0] if clean_target.strip() else None
                    if target_item:
                        print(f"\n🚀 Launching Navigation for '{target_item}'...")
                        subprocess.run([sys.executable, "navigation_main.py", target_item])
                        print("📷 Returned to Voice Controller.\n")

                elif any(term in command for term in ["exit", "quit", "stop system"]):
                    print("👋 Shutting down Voice Controller.")
                    voice.speak("Exiting system.")
                    break

            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"❌ Speech service error: {e}")
            except Exception as e:
                print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    listen_and_route()