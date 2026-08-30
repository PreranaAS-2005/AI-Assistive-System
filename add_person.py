import cv2
import os
import time
import pyttsx3
import speech_recognition as sr


# =================================
# SETTINGS
# =================================

KNOWN_FACES_PATH = "known_faces"

TOTAL_IMAGES = 10
CAPTURE_DELAY = 1


# =================================
# SPEAK
# =================================

def speak(text):

    print(f"🔊 {text}")

    engine = pyttsx3.init()

    engine.say(text)

    engine.runAndWait()

    engine.stop()

    # Small pause before microphone starts
    time.sleep(1)


# =================================
# LISTEN
# =================================

def listen_to_voice(recognizer, phrase_time_limit=6):

    try:

        with sr.Microphone() as source:

            print("🎤 Listening now...")

            audio = recognizer.listen(
                source,
                timeout=20,
                phrase_time_limit=phrase_time_limit
            )

        print("Processing voice...")

        text = recognizer.recognize_google(
            audio
        )

        text = text.lower().strip()

        print(f"Recognized: {text}")

        return text


    except sr.WaitTimeoutError:

        print("⚠️ No speech detected.")

        return None


    except sr.UnknownValueError:

        print("⚠️ Could not understand speech.")

        return None


    except sr.RequestError as e:

        print(f"❌ Speech recognition error: {e}")

        speak(
            "Speech recognition service is unavailable."
        )

        return None


# =================================
# SETUP MICROPHONE
# =================================

def setup_microphone(recognizer):

    with sr.Microphone() as source:

        print("🎤 Adjusting for background noise...")

        speak(
            "Please wait while I adjust the microphone."
        )

        recognizer.adjust_for_ambient_noise(
            source,
            duration=2
        )

    print(
        f"Microphone energy threshold: "
        f"{recognizer.energy_threshold}"
    )


# =================================
# GET CONFIRMATION
# =================================

def get_confirmation(recognizer):

    while True:

        speak(
            "Please say yes correct, "
            "or no try again."
        )

        confirmation = listen_to_voice(
            recognizer,
            phrase_time_limit=5
        )


        if not confirmation:

            speak(
                "I could not understand. "
                "Please try again."
            )

            continue


        print(
            f"Confirmation received: {confirmation}"
        )


        # YES / CORRECT

        yes_words = [

            "yes",
            "correct",
            "yeah",
            "yep",
            "right",
            "okay",
            "ok"

        ]


        # NO / WRONG

        no_words = [

            "no",
            "wrong",
            "incorrect",
            "again"

        ]


        for word in yes_words:

            if word in confirmation:

                return True


        for word in no_words:

            if word in confirmation:

                return False


        speak(
            "Please say yes correct, "
            "or no try again."
        )


# =================================
# GET PERSON NAME
# =================================

def get_person_name():

    recognizer = sr.Recognizer()

    recognizer.dynamic_energy_threshold = True

    recognizer.pause_threshold = 1

    recognizer.non_speaking_duration = 0.5


    # Calibrate microphone ONCE
    setup_microphone(
        recognizer
    )


    while True:

        speak(
            "Please say the new person's name."
        )

        person_name = listen_to_voice(
            recognizer,
            phrase_time_limit=6
        )


        if not person_name:

            speak(
                "I could not understand the name. "
                "Please try again."
            )

            continue


        speak(
            f"I heard {person_name}. "
            "Please confirm."
        )


        confirmation = get_confirmation(
            recognizer
        )


        if confirmation:

            speak(
                f"Okay. Adding {person_name}."
            )

            return person_name


        else:

            speak(
                "Okay. Please say the name again."
            )


# =================================
# GET NAME
# =================================

person_name = get_person_name()


# =================================
# CLEAN NAME
# =================================

person_name = person_name.replace(
    " ",
    "_"
)


# =================================
# CREATE FOLDER
# =================================

person_path = os.path.join(
    KNOWN_FACES_PATH,
    person_name
)

os.makedirs(
    person_path,
    exist_ok=True
)


print(
    f"\nAdding new person: {person_name}"
)


speak(
    f"Please look at the camera. "
    f"I will capture ten images of {person_name}. "
    "Please slowly move your face."
)


# =================================
# START CAMERA
# =================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("❌ Could not open camera")

    speak(
        "Could not open the camera."
    )

    exit()


print("\nCamera started.")

print("Capturing face images...")


# =================================
# CAPTURE IMAGES
# =================================

image_count = 0

last_capture_time = 0


while image_count < TOTAL_IMAGES:

    success, frame = camera.read()


    if not success:

        print(
            "❌ Could not read camera frame."
        )

        break


    current_time = time.time()


    # Capture every second

    if (
        current_time - last_capture_time
        >= CAPTURE_DELAY
    ):

        image_count += 1


        image_name = (
            f"image_{image_count}.jpg"
        )


        image_path = os.path.join(
            person_path,
            image_name
        )


        cv2.imwrite(
            image_path,
            frame
        )


        print(
            f"✅ Saved {image_name}"
        )


        last_capture_time = current_time


    # Display progress

    cv2.putText(

        frame,

        f"Capturing {image_count}/{TOTAL_IMAGES}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 0),

        2

    )


    cv2.imshow(
        "Adding New Person - Press Q to Cancel",
        frame
    )


    # Cancel

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print(
            "\n❌ Person addition cancelled."
        )

        speak(
            "Person addition cancelled."
        )

        break


# =================================
# CLEANUP
# =================================

camera.release()

cv2.destroyAllWindows()


# =================================
# COMPLETION
# =================================

if image_count == TOTAL_IMAGES:

    print(
        f"\n🎉 {person_name} added successfully!"
    )

    speak(
        f"{person_name} has been added "
        "to known people successfully."
    )


else:

    print(
        "\n⚠️ Not enough images were captured."
    )