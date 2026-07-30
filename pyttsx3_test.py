import pyttsx3
import time

engine = pyttsx3.init()

while True:
    print("Speaking...")
    engine.say("Hello")
    engine.runAndWait()
    time.sleep(2)