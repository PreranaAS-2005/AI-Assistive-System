import speech_recognition as sr


def is_add_person_command(command):

    command = command.lower().strip()

    # Words/phrases that can mean
    # "add new person"
    person_words = [
        "person",
        "version",
        "patient",
        "people",
        "individual",
        "somebody",
        "someone"
    ]

    # Check if the command contains "add"
    has_add = (
        "add" in command
        or "at" in command
    )

    # Check for person-related words
    has_person_word = any(
        word in command
        for word in person_words
    )

    # Accept variations such as:
    # "add new person"
    # "add new version"
    # "add person"
    # "add patient"
    if has_add and has_person_word:

        return True


    # Also allow these complete phrases
    accepted_phrases = [

        "new person",

        "new version",

        "add someone",

        "add somebody"

    ]

    for phrase in accepted_phrases:

        if phrase in command:

            return True


    return False


def listen_for_command():

    recognizer = sr.Recognizer()

    recognizer.dynamic_energy_threshold = True

    recognizer.pause_threshold = 0.8


    try:

        with sr.Microphone() as source:

            print(
                "🎤 Adjusting for background noise..."
            )

            recognizer.adjust_for_ambient_noise(
                source,
                duration=2
            )

            print(
                "🎤 Listening for a command..."
            )

            audio = recognizer.listen(
                source,
                timeout=15,
                phrase_time_limit=5
            )


        command = recognizer.recognize_google(
            audio
        )

        command = command.lower().strip()


        print(
            f"🗣️ You said: {command}"
        )


        if is_add_person_command(command):

            print(
                "✅ ADD NEW PERSON COMMAND DETECTED!"
            )

        else:

            print(
                "❌ Command not recognized."
            )


    except sr.WaitTimeoutError:

        print(
            "⚠️ No command detected."
        )


    except sr.UnknownValueError:

        print(
            "⚠️ Could not understand the command."
        )


    except sr.RequestError as e:

        print(
            f"❌ Speech recognition error: {e}"
        )


if __name__ == "__main__":

    listen_for_command()