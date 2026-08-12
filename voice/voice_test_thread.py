import pyttsx3
import threading
import queue
import time

speech_queue = queue.Queue()


def voice_worker():
    engine = pyttsx3.init()

    while True:
        text = speech_queue.get()

        if text is None:
            break

        print("Speaking:", text)

        engine.say(text)
        engine.runAndWait()

        speech_queue.task_done()


# Start voice thread
voice_thread = threading.Thread(target=voice_worker, daemon=True)
voice_thread.start()


# Test messages
speech_queue.put("Person detected")
time.sleep(2)

speech_queue.put("Bottle detected")
time.sleep(2)

speech_queue.put("Cell phone detected")

# Wait for all speech to finish
speech_queue.join()

# Stop voice thread
speech_queue.put(None)

print("Voice test completed.")