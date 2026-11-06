import speech_recognition as sr
import time

recognizer = sr.Recognizer()

device_index = 12


print("======================================")
print("       MICROPHONE TEST")
print("======================================")

print("\nMicrophone:")
print(sr.Microphone.list_microphone_names()[device_index])

with sr.Microphone(device_index=device_index) as source:

    print("\n🔧 Calibrating microphone...")
    recognizer.adjust_for_ambient_noise(source, duration=2)

    print("Energy threshold:", recognizer.energy_threshold)

    # Give you time to get ready
    print("\n--------------------------------------")
    print("GET READY")
    print("--------------------------------------")

    for i in range(3, 0, -1):
        print(f"🎤 Speak in {i}...")
        time.sleep(1)

    print("\n🔴 NOW SPEAK!")
    print("Say: FIND BOTTLE")
    print("You have 8 seconds.\n")

    audio = recognizer.listen(
        source,
        timeout=10,
        phrase_time_limit=8
    )

print("\n✅ AUDIO CAPTURED!")
print("Sending to Google...")

try:

    text = recognizer.recognize_google(
        audio,
        language="en-IN"
    )

    print("\n======================================")
    print("🎉 SUCCESS!")
    print("YOU SAID:", text)
    print("======================================")

except sr.UnknownValueError:

    print("\n❌ Audio was captured, but Google couldn't understand it.")

except sr.RequestError as e:

    print("\n❌ Google Speech Recognition error:")
    print(e)

except Exception as e:

    print("\n❌ Error:")
    print(type(e).__name__)
    print(e)