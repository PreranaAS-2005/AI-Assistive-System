from ultralytics import YOLO

import cv2

import os

import sys

import time

import numpy as np

import speech_recognition as sr

import subprocess

import threading

from collections import deque, Counter



from deepface import DeepFace

from voice.voice_manager import VoiceManager

from navigation.object_memory import ObjectMemory





# ==========================================

# STARTUP MENU

# ==========================================



def show_menu():

    print("\n" + "=" * 45)

    print("     AI ASSISTIVE SYSTEM — MAIN MENU")

    print("=" * 45)

    print("  1. Navigation")

    print("  2. Face Recognition")

    print("  3. Add Person")

    print("  4. Exit")

    print("=" * 45)





def run_startup_menu():
    """Voice-controlled main menu. Say 'cancel' in a module to return here."""
    while True:
        show_menu()
        print("\n🎤 Listening for your command...")
        speak_and_pause(
            "Please say a command. Say navigation to start navigation. "
            "Say face recognition to start face recognition. "
            "Say add person to add a new person. Say exit to close the system."
        )

        while True:
            if command_queue:
                command = command_queue.pop(0)
                print(f"\n🗣️ Voice command received: {command}")
                if command == "navigation":
                    result = run_module_with_cancel("navigation_main.py", "Navigation")
                    if result == "exit":
                        speak_and_pause("Closing the assistive system. Goodbye.")
                        sys.exit(0)
                    break
                elif command == "face_recognition":
                    result = run_module_with_cancel("face_recognition_manager.py", "Face Recognition")
                    if result == "exit":
                        speak_and_pause("Closing the assistive system. Goodbye.")
                        sys.exit(0)
                    break
                elif command == "add_person":
                    result = run_module_with_cancel("add_person.py", "Add Person")
                    if result == "exit":
                        speak_and_pause("Closing the assistive system. Goodbye.")
                        sys.exit(0)
                    break
                elif command == "cancel":
                    print("\n↩️ Already at the main menu.")
                    break
                elif command == "exit":
                    print("\n👋 Goodbye!")
                    sys.exit(0)
            time.sleep(0.1)


def run_module_with_cancel(script_name, module_name):
    """Run a module and announce transitions aloud for blind users."""
    while "cancel" in command_queue:
        command_queue.remove("cancel")
    while "exit" in command_queue:
        command_queue.remove("exit")

    # Announce before launching so the user knows which mode is starting.
    print(f"[INFO] Starting {module_name}...")
    speak_and_pause(f"Starting {module_name} mode. Say cancel at any time to return to the main menu.")
    command_queue.clear()  # only accept commands spoken after the prompt

    child = subprocess.Popen([sys.executable, script_name])
    print(f"[INFO] {module_name} is running. Say 'Cancel' to return to the menu.")

    while child.poll() is None:
        if "cancel" in command_queue:
            command_queue.remove("cancel")
            print(f"[INFO] Cancel heard. Stopping {module_name} and returning to menu...")
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            speak_and_pause(f"Cancelling {module_name}. Returning to the main menu.")
            command_queue.clear()
            time.sleep(0.3)
            return "cancelled"

        if "exit" in command_queue:
            command_queue.remove("exit")
            print(f"[INFO] Exit heard. Stopping {module_name}...")
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            speak_and_pause(f"Exiting {module_name}. Closing the assistive system.")
            command_queue.clear()
            return "exit"
        time.sleep(0.1)

    print(f"[INFO] {module_name} finished. Returning to the main menu.")
    speak_and_pause(f"{module_name} finished. Returning to the main menu.")
    command_queue.clear()
    time.sleep(0.3)
    return "finished"


# SETTINGS

# ==========================================



KNOWN_FACES_PATH = "known_faces"



MODEL_NAME = "Facenet512"



DISTANCE_THRESHOLD = 0.35





# Face prediction stability

PREDICTION_HISTORY_SIZE = 5

MIN_STABLE_COUNT = 3





# YOLO settings

CONFIDENCE_THRESHOLD = 0.60



REQUIRED_FRAMES = 3



MAX_MISSING_FRAMES = 10





# Run face recognition every few frames

FACE_RECOGNITION_INTERVAL = 5





# ==========================================

# SMART ANNOUNCEMENT SETTINGS

# ==========================================



# Minimum seconds before the same name/object

# can be announced again.

ANNOUNCEMENT_COOLDOWN = 8





# ==========================================

# LOAD YOLO MODEL

# ==========================================



print("Loading YOLO model...")



model = YOLO("yolov8n.pt")





# ==========================================

# VOICE MANAGER

# ==========================================



voice = VoiceManager()

object_memory = ObjectMemory()

# Suppress commands captured while the system is speaking.
speech_active = threading.Event()
speech_generation = 0
speech_state_lock = threading.Lock()

def speak_and_pause(message):
    """
    Speak a prompt while the command listener is paused.

    The listener may already be inside recognizer.listen() when speech starts.
    speech_generation makes it discard that overlapping recording. The extra
    cooldown also covers TTS implementations that return just before audio
    playback has completely finished.
    """
    global speech_generation

    with speech_state_lock:
        speech_generation += 1
        speech_active.set()

    try:
        result = voice.speak(message)

        # Keep the microphone listener paused until the spoken prompt should
        # have finished, even if TTS returns before playback fully ends.
        word_count = max(1, len(str(message).split()))
        estimated_playback_seconds = (word_count * 0.38) + 0.8
        time.sleep(estimated_playback_seconds)
        return result
    finally:
        # Invalidate any recording captured during the prompt/cooldown.
        with speech_state_lock:
            speech_generation += 1
        speech_active.clear()





# ==========================================

# KNOWN FACE DATA

# ==========================================



known_embeddings = []

known_names = []





# ==========================================

# SMART ANNOUNCEMENT MEMORY

# ==========================================



# Stores when each thing was last announced.

# Example:

# {

#     "person_rashmi": 12345.5,

#     "object_cell phone": 12348.2

# }

last_announced_time = {}





# ==========================================

# LOAD KNOWN FACES

# ==========================================



def load_known_faces():



    global known_embeddings

    global known_names



    known_embeddings = []

    known_names = []



    print("\nLoading known faces...\n")



    if not os.path.exists(KNOWN_FACES_PATH):



        print(

            f"❌ Folder not found: {KNOWN_FACES_PATH}"

        )



        return





    for person_name in os.listdir(KNOWN_FACES_PATH):



        person_path = os.path.join(

            KNOWN_FACES_PATH,

            person_name

        )



        if not os.path.isdir(person_path):



            continue





        print(

            f"Processing: {person_name}"

        )



        images_loaded = 0





        for filename in os.listdir(person_path):



            image_path = os.path.join(

                person_path,

                filename

            )



            try:



                embedding_objects = DeepFace.represent(



                    img_path=image_path,



                    model_name=MODEL_NAME,



                    detector_backend="opencv",



                    enforce_detection=False



                )





                embedding = embedding_objects[0]["embedding"]





                known_embeddings.append(

                    np.array(embedding)

                )





                known_names.append(

                    person_name

                )





                images_loaded += 1





            except Exception as e:



                print(

                    f"⚠️ Could not process {filename}"

                )



                print(

                    f"Error: {e}"

                )





        print(

            f"✅ Loaded {images_loaded} image(s)"

        )





    print(

        f"\n👤 Total known face images: "

        f"{len(known_embeddings)}"

    )





# ==========================================

# COSINE DISTANCE

# ==========================================



def get_object_position(x_center, frame_width):



    # Divide screen into 3 parts

    left_boundary = frame_width / 3

    right_boundary = 2 * frame_width / 3



    if x_center < left_boundary:

        return "on your left"



    elif x_center > right_boundary:

        return "on your right"



    else:

        return "ahead"





def cosine_distance(a, b):



    return 1 - np.dot(

        a,

        b

    ) / (



        np.linalg.norm(a)

        *

        np.linalg.norm(b)



    )





# ==========================================

# RECOGNIZE PERSON

# ==========================================



def recognize_person(frame):



    global known_embeddings

    global known_names





    if len(known_embeddings) == 0:



        return "Unknown", None





    try:



        detected_faces = DeepFace.extract_faces(



            img_path=frame,



            detector_backend="opencv",



            enforce_detection=False



        )





        if len(detected_faces) == 0:



            return "Unknown", None





        detected_face = detected_faces[0]





        face = detected_face["face"]





        face = (

            face * 255

        ).astype(np.uint8)





        result = DeepFace.represent(



            img_path=face,



            model_name=MODEL_NAME,



            detector_backend="skip",



            enforce_detection=False



        )





        live_embedding = np.array(



            result[0]["embedding"]



        )





        distances = []





        for known_embedding in known_embeddings:



            distance = cosine_distance(



                live_embedding,



                known_embedding



            )



            distances.append(

                distance

            )





        best_index = np.argmin(

            distances

        )





        best_distance = distances[

            best_index

        ]





        if best_distance < DISTANCE_THRESHOLD:



            predicted_name = known_names[

                best_index

            ]



        else:



            predicted_name = "Unknown"





        return predicted_name, best_distance





    except Exception as e:



        print(

            f"⚠️ Face recognition error: {e}"

        )



        return "Unknown", None





# ==========================================

# GET OBJECT DIRECTION

# ==========================================



def get_direction(

    x1,

    x2,

    frame_width

):



    object_center = (

        x1 + x2

    ) / 2





    left_boundary = (

        frame_width / 3

    )





    right_boundary = (

        2 * frame_width / 3

    )





    if object_center < left_boundary:



        return "on your left"





    elif object_center > right_boundary:



        return "on your right"





    else:



        return "in front of you"





# ==========================================

# GET APPROXIMATE DISTANCE

# ==========================================



def get_distance_description(

    x1,

    y1,

    x2,

    y2,

    frame_width,

    frame_height

):



    object_width = (

        x2 - x1

    )





    object_height = (

        y2 - y1

    )





    object_area = (

        object_width

        *

        object_height

    )





    frame_area = (

        frame_width

        *

        frame_height

    )





    area_ratio = (

        object_area

        /

        frame_area

    )





    if area_ratio > 0.30:



        return "very close"





    elif area_ratio > 0.10:



        return "nearby"





    else:



        return "far away"





# ==========================================

# CHECK SMART ANNOUNCEMENT

# ==========================================



def can_announce(key):



    current_time = time.time()





    if key not in last_announced_time:



        last_announced_time[

            key

        ] = current_time



        return True





    elapsed_time = (



        current_time

        -

        last_announced_time[key]



    )





    if elapsed_time >= ANNOUNCEMENT_COOLDOWN:



        last_announced_time[

            key

        ] = current_time



        return True





    return False





# ==========================================

# FLEXIBLE ADD PERSON COMMAND

# ==========================================



def is_add_person_command(command):



    command = command.lower().strip()





    person_words = [



        "person",



        "version",



        "patient",



        "people",



        "individual",



        "someone",



        "somebody"



    ]





    has_add = (



        "add" in command



        or



        "at" in command



    )





    has_person_word = any(



        word in command



        for word in person_words



    )





    if has_add and has_person_word:



        return True





    accepted_phrases = [



        "new person",



        "new version",



        "add someone",



        "add somebody"



    ]





    return any(



        phrase in command



        for phrase in accepted_phrases



    )





# ==========================================

# NAVIGATION COMMAND DETECTION

# ==========================================



def is_navigation_command(command):



    command = command.lower().strip()



    navigation_phrases = [

        "navigation",

        "navigate",

        "start navigation",

        "open navigation",

        "navigation mode",

        "start navigating",

    ]



    return any(

        phrase in command

        for phrase in navigation_phrases

    )





# ==========================================

# FACE RECOGNITION COMMAND DETECTION

# ==========================================



def is_face_recognition_command(command):



    command = command.lower().strip()



    face_phrases = [

        "face recognition",

        "face recognise",

        "face recognize",

        "start face",

        "face detection",

        "recognize face",

        "recognise face",

        "identify face",

        "face mode",

    ]



    return any(

        phrase in command

        for phrase in face_phrases

    )





# ==========================================

# EXIT COMMAND DETECTION

# ==========================================



def is_cancel_command(command):
    """Recognize universal cancellation phrases without treating them as Exit."""
    normalized = " ".join(command.lower().strip().split())
    return normalized in {
        "cancel", "please cancel", "cancel please", "go back", "back to menu",
        "return to menu", "main menu", "stop navigation", "stop navigating",
        "exit navigation", "quit navigation", "cancel navigation",
        "cancel face recognition", "cancel face recognition mode",
        "cancel add person", "cancel adding person",
    }


def is_exit_command(command):



    command = command.lower().strip()

    navigation_phrases = (
        "stop navigation",
        "stop navigating",
        "exit navigation",
        "quit navigation",
        "back to menu",
        "return to menu",
    )
    if any(phrase in command for phrase in navigation_phrases):
        return False



    exit_phrases = [

        "exit",

        "quit",

        "stop",

        "close",

        "shutdown",

        "shut down",

        "turn off",

        "bye",

        "goodbye",

    ]



    return any(

        phrase in command

        for phrase in exit_phrases

    )





# ==========================================

# VOICE COMMAND LISTENER

# ==========================================



command_queue = []



add_person_mode = False
voice_listener_stop = threading.Event()





def listen_for_commands():



    global add_person_mode





    recognizer = sr.Recognizer()



    recognizer.dynamic_energy_threshold = True



    recognizer.pause_threshold = 0.8





    with sr.Microphone(device_index=12) as source:





        print(

            "🎤 Voice command system started."

        )





        print(

            "🎤 Adjusting for background noise..."

        )





        recognizer.adjust_for_ambient_noise(



            source,



            duration=2



        )





        print(

            "🎤 Listening for commands..."

        )





        while True:





            # Wait until system speech finishes before opening the microphone.
            if speech_active.is_set():
                time.sleep(0.1)
                continue

            with speech_state_lock:
                generation_at_listen_start = speech_generation

            if add_person_mode:



                time.sleep(0.1)

                continue





            try:



                audio = recognizer.listen(



                    source,



                    timeout=1,



                    phrase_time_limit=5



                )





                # Reject audio if a system prompt overlapped microphone capture.
                with speech_state_lock:
                    speech_overlapped_recording = (generation_at_listen_start != speech_generation)
                if speech_overlapped_recording or speech_active.is_set():
                    continue

                command = (
                    recognizer
                    .recognize_google(audio)
                    .lower()
                    .strip()
                )

                # TTS may have started while speech recognition was processing.
                with speech_state_lock:
                    speech_overlapped_recording = (generation_at_listen_start != speech_generation)
                if speech_overlapped_recording or speech_active.is_set():
                    continue

                print(

                    f"🗣️ Command heard: {command}"

                )





                if is_cancel_command(command):
                    print("✅ CANCEL COMMAND DETECTED!")
                    if "cancel" not in command_queue:
                        command_queue.append("cancel")

                elif is_exit_command(command):
                    print("✅ EXIT COMMAND DETECTED!")
                    if "exit" not in command_queue:
                        command_queue.append("exit")

                elif is_navigation_command(command):



                    print(

                        "✅ NAVIGATION COMMAND DETECTED!"

                    )



                    if "navigation" not in command_queue:



                        command_queue.append(

                            "navigation"

                        )





                elif is_face_recognition_command(command):



                    print(

                        "✅ FACE RECOGNITION COMMAND DETECTED!"

                    )



                    if "face_recognition" not in command_queue:



                        command_queue.append(

                            "face_recognition"

                        )





                elif is_add_person_command(command):



                    print(

                        "✅ ADD NEW PERSON COMMAND DETECTED!"

                    )



                    if "add_person" not in command_queue:



                        command_queue.append(

                            "add_person"

                        )





            except sr.WaitTimeoutError:

                continue



            except sr.UnknownValueError:



                pass





            except sr.RequestError as e:



                print(

                    f"❌ Speech recognition error: {e}"

                )





            except Exception as e:



                print(

                    f"⚠️ Voice listener error: {e}"

                )





def launch_navigation():
    """Compatibility wrapper for any older call sites."""
    print("\n🧭 Starting Navigation mode...")
    return run_module_with_cancel("navigation_main.py", "Navigation")


# ==========================================

# START VOICE COMMAND THREAD

# ==========================================



voice_listener = threading.Thread(



    target=listen_for_commands,



    daemon=True



)



voice_listener.start()





# ==========================================

# START VOICE-CONTROLLED STARTUP MENU

# ==========================================



run_startup_menu()





# ==========================================

# LOAD KNOWN FACES

# ==========================================



load_known_faces()





# ==========================================

# YOLO TRACKING DATA

# ==========================================



track_counts = {}



missing_counts = {}



announced_tracks = set()





# Separate prediction history for every person

prediction_histories = {}





# Stable recognition result for every track

recognized_tracks = {}





# Frame counter

frame_count = 0





# ==========================================

# START CAMERA

# ==========================================



camera = cv2.VideoCapture(



    0,



    cv2.CAP_DSHOW



)





if not camera.isOpened():



    print(

        "❌ Could not open camera."

    )



    exit()





print(

    "\n📷 Integrated Assistive System started."

)





# ==========================================

# MAIN LOOP

# ==========================================



while True:





    # --------------------------------------

    # EXIT COMMAND

    # --------------------------------------



    if "exit" in command_queue:



        command_queue.remove("exit")



        print("\n👋 Exit command received. Shutting down...")



        speak_and_pause("Goodbye. Shutting down.")



        import time

        time.sleep(2)



        break





    # --------------------------------------

    # NAVIGATION COMMAND

    # --------------------------------------



    if "navigation" in command_queue:



        command_queue.remove("navigation")



        print("\n🧭 Starting Navigation mode...")



        speak_and_pause("Starting navigation mode.")



        import time

        time.sleep(1)



        camera.release()

        cv2.destroyAllWindows()



        subprocess.run([

            sys.executable,

            "navigation_main.py"

        ])



        print("\n🔄 Returned from Navigation. Resuming assistive system...")



        track_counts.clear()

        missing_counts.clear()

        announced_tracks.clear()

        prediction_histories.clear()

        recognized_tracks.clear()

        last_announced_time.clear()



        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)



        continue





    # --------------------------------------

    # FACE RECOGNITION COMMAND

    # --------------------------------------



    if "face_recognition" in command_queue:



        command_queue.remove("face_recognition")



        print("\n👤 Starting Face Recognition mode...")



        speak_and_pause("Starting face recognition mode.")



        import time

        time.sleep(1)



        camera.release()

        cv2.destroyAllWindows()



        subprocess.run([

            sys.executable,

            "face_recognition_manager.py"

        ])



        print("\n🔄 Returned from Face Recognition. Resuming assistive system...")



        track_counts.clear()

        missing_counts.clear()

        announced_tracks.clear()

        prediction_histories.clear()

        recognized_tracks.clear()

        last_announced_time.clear()



        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)



        continue





    # --------------------------------------

    # ADD NEW PERSON COMMAND

    # --------------------------------------



    if "add_person" in command_queue:





        command_queue.remove(

            "add_person"

        )





        print(

            "\n🚀 Starting Add New Person mode..."

        )





        add_person_mode = True





        speak_and_pause(

            "Starting add new person mode."

        )





        time.sleep(1)





        camera.release()



        cv2.destroyAllWindows()





        subprocess.run([



            sys.executable,



            "add_person.py"



        ])





        print(

            "\n🔄 Reloading known faces..."

        )





        load_known_faces()





        print(

            "\n📷 Returning to assistive system..."

        )





        track_counts.clear()



        missing_counts.clear()



        announced_tracks.clear()



        prediction_histories.clear()



        recognized_tracks.clear()



        last_announced_time.clear()





        camera = cv2.VideoCapture(



            0,



            cv2.CAP_DSHOW



        )





        add_person_mode = False





        continue





    # --------------------------------------

    # READ CAMERA

    # --------------------------------------



    success, frame = camera.read()





    if not success:



        print(

            "❌ Could not read camera frame."

        )



        break





    frame_count += 1





    frame_height, frame_width = frame.shape[:2]





    # --------------------------------------

    # YOLO TRACKING

    # --------------------------------------



    results = model.track(



        frame,



        persist=True,



        verbose=False



    )





    boxes = results[0].boxes





    current_track_ids = set()





    if boxes is not None:





        for box in boxes:





            confidence = float(

                box.conf[0]

            )





            if confidence < CONFIDENCE_THRESHOLD:



                continue





            class_id = int(

                box.cls[0]

            )





            object_name = model.names[

                class_id

            ]





            if box.id is None:



                continue





            track_id = int(

                box.id[0]

            )





            # ----------------------------------

            # GET BOUNDING BOX

            # ----------------------------------



            x1, y1, x2, y2 = map(



                int,



                box.xyxy[0]



            )





            # ----------------------------------

            # GET DIRECTION

            # ----------------------------------



            direction = get_direction(



                x1,



                x2,



                frame_width



            )





            # ----------------------------------

            # GET DISTANCE

            # ----------------------------------



            distance_description = (

                get_distance_description(



                    x1,



                    y1,



                    x2,



                    y2,



                    frame_width,



                    frame_height



                )

            )



            # ----------------------------------

            # UPDATE OBJECT MEMORY

            # ----------------------------------

            object_memory.update(

                label=object_name,

                direction=direction,

                distance=distance_description,

                confidence=confidence,

                bbox=[x1, y1, x2, y2]

            )







            current_track_ids.add(

                track_id

            )





            missing_counts[

                track_id

            ] = 0





            track_counts[

                track_id

            ] = (



                track_counts.get(

                    track_id,

                    0

                )



                + 1



            )





            # ==================================

            # PERSON DETECTION

            # ==================================



            if object_name == "person":





                if (



                    frame_count

                    % FACE_RECOGNITION_INTERVAL

                    == 0



                ):





                    predicted_name, face_distance = (



                        recognize_person(

                            frame

                        )



                    )





                    if (



                        track_id

                        not in prediction_histories



                    ):



                        prediction_histories[

                            track_id

                        ] = deque(



                            maxlen=

                            PREDICTION_HISTORY_SIZE



                        )





                    prediction_histories[

                        track_id

                    ].append(

                        predicted_name

                    )





                    prediction_counts = Counter(



                        prediction_histories[

                            track_id

                        ]



                    )





                    most_common_name, count = (



                        prediction_counts

                        .most_common(1)[0]



                    )





                    if (



                        count

                        >= MIN_STABLE_COUNT



                    ):



                        stable_name = (

                            most_common_name

                        )



                    else:



                        stable_name = (

                            "Recognizing..."

                        )





                    recognized_tracks[

                        track_id

                    ] = stable_name





                    if face_distance is not None:



                        print(



                            f"👤 Person ID {track_id} | "

                            f"Raw: {predicted_name} | "

                            f"Stable: {stable_name} | "

                            f"Face Distance: "

                            f"{face_distance:.3f} | "

                            f"Direction: {direction} | "

                            f"Position: "

                            f"{distance_description}"



                        )





                    # ----------------------------------

                    # KNOWN PERSON ANNOUNCEMENT

                    # ----------------------------------



                    if (



                        stable_name

                        != "Recognizing..."



                        and



                        stable_name

                        != "Unknown"



                        and



                        track_id

                        not in announced_tracks



                    ):





                        announcement_key = (

                            f"person\_{stable_name}"

                        )





                        if can_announce(

                            announcement_key

                        ):





                            message = (



                                f"{stable_name} detected "

                                f"{direction}, "

                                f"{distance_description}"



                            )





                            print(

                                f"🔊 {message}"

                            )





                            speak_and_pause(

                                message

                            )





                        announced_tracks.add(

                            track_id

                        )





                    # ----------------------------------

                    # UNKNOWN PERSON ANNOUNCEMENT

                    # ----------------------------------



                    elif (



                        stable_name

                        == "Unknown"



                        and



                        track_id

                        not in announced_tracks



                        and



                        count

                        >= MIN_STABLE_COUNT



                    ):





                        announcement_key = (

                            "person_unknown"

                        )





                        if can_announce(

                            announcement_key

                        ):





                            message = (



                                f"Unknown person detected "

                                f"{direction}, "

                                f"{distance_description}"



                            )





                            print(

                                f"🔊 {message}"

                            )





                            speak_and_pause(

                                message

                            )





                        announced_tracks.add(

                            track_id

                        )





            # ==================================

            # NORMAL OBJECT DETECTION

            # ==================================



            else:





                if (



                    track_counts[

                        track_id

                    ]



                    >= REQUIRED_FRAMES



                    and



                    track_id

                    not in announced_tracks



                ):





                    announcement_key = (

                        f"object\_{object_name}"

                    )





                    if can_announce(

                        announcement_key

                    ):





                        message = (



                            f"{object_name} detected "

                            f"{direction}, "

                            f"{distance_description}"



                        )





                        print(

                            f"🔊 {message}"

                        )





                        speak_and_pause(

                            message

                        )





                    announced_tracks.add(

                        track_id

                    )





    # --------------------------------------

    # HANDLE DISAPPEARED TRACKS

    # --------------------------------------



    for track_id in list(

        track_counts.keys()

    ):





        if track_id not in current_track_ids:





            missing_counts[

                track_id

            ] = (



                missing_counts.get(

                    track_id,

                    0

                )



                + 1



            )





            if (



                missing_counts[

                    track_id

                ]



                >= MAX_MISSING_FRAMES



            ):





                track_counts.pop(



                    track_id,



                    None



                )





                missing_counts.pop(



                    track_id,



                    None



                )





                announced_tracks.discard(

                    track_id

                )





                prediction_histories.pop(



                    track_id,



                    None



                )





                recognized_tracks.pop(



                    track_id,



                    None



                )





    # --------------------------------------

    # DRAW RESULTS

    # --------------------------------------



    annotated_frame = results[0].plot()





    left_x = frame_width // 3

    right_x = 2 * frame_width // 3





    cv2.line(



        annotated_frame,



        (left_x, 0),



        (left_x, frame_height),



        (255, 255, 255),



        1



    )





    cv2.line(



        annotated_frame,



        (right_x, 0),



        (right_x, frame_height),



        (255, 255, 255),



        1



    )





    cv2.putText(



        annotated_frame,



        "LEFT",



        (30, 40),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.7,



        (255, 255, 255),



        2



    )





    cv2.putText(



        annotated_frame,



        "CENTER",



        (frame_width // 2 - 50, 40),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.7,



        (255, 255, 255),



        2



    )





    cv2.putText(



        annotated_frame,



        "RIGHT",



        (frame_width - 100, 40),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.7,



        (255, 255, 255),



        2



    )





    # --------------------------------------

    # SHOW CAMERA

    # --------------------------------------



    cv2.imshow(



        "AI Assistive System",



        annotated_frame



    )





    # --------------------------------------

    # PRESS Q TO QUIT

    # --------------------------------------



    if cv2.waitKey(1) & 0xFF == ord("q"):



        break





# ==========================================

# CLEANUP

# ==========================================



camera.release()



cv2.destroyAllWindows()
