"""
Phase 3 Navigation Main

AI-Based Assistive Spatial Memory and Guidance System

NORMAL MODE
-----------
Objects are detected silently unless they are a close/near
safety obstacle.

TARGET MODE
-----------
User types:

    find bottle
    find chair
    find laptop

The system then:

    1. Finds the requested object.
    2. Determines LEFT / CENTER / RIGHT.
    3. Estimates approximate distance.
    4. Speaks one combined instruction.

Example:

    Your bottle is on your right, about 1.2 meters away.

The system does NOT continuously announce unrelated objects.
"""

from ultralytics import YOLO
import cv2
import threading

from voice.voice_manager import VoiceManager

from navigation.spatial_analyzer import (
    analyze_detection,
    Priority,
    Zone,
)

from navigation.obstacle_guidance import ObstacleGuidance
from navigation.direction_assistant import DirectionAssistant
from navigation.target_search import TargetSearch
from navigation.target_guidance import TargetGuidance
from navigation.detection_filter import DetectionFilter
from navigation.distance_estimator import DistanceEstimator
from navigation.distance_smoother import DistanceSmoother
from navigation.guidance_selector import GuidanceSelector


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.60

CAMERA_INDEX = 2

IMAGE_SIZE = 320


# ============================================================
# LOAD YOLO
# ============================================================

print("[INFO] Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("[INFO] YOLO model loaded.")


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW,
)

if not camera.isOpened():

    print(
        f"[ERROR] Could not open camera "
        f"(index {CAMERA_INDEX})."
    )

    raise SystemExit(1)

print("[INFO] Camera connected.")


# ============================================================
# MODULES
# ============================================================

voice = VoiceManager()

obstacle_guidance = ObstacleGuidance()

direction_assistant = DirectionAssistant()

target_search = TargetSearch()

target_guidance = TargetGuidance()

detection_filter = DetectionFilter(
    min_confidence=CONFIDENCE_THRESHOLD,
    stability_frames=3,
    history_size=5,
)

distance_estimator = DistanceEstimator(
    focal_length_px=700.0,
)

# Larger threshold prevents tiny distance changes from
# generating repeated voice messages.
distance_smoother = DistanceSmoother(
    window_size=5,
    change_threshold_m=0.60,
)

guidance_selector = GuidanceSelector(
    switch_distance_margin_m=0.75,
)


# ============================================================
# GLOBAL STATE
# ============================================================

pending_target_message = None


# ============================================================
# TERMINAL INPUT
# ============================================================

def input_listener():
    """
    Terminal commands:

        find bottle
        find chair
        find laptop

        clear
        help
    """

    print(
        "[INFO] Target search commands: "
        "find <object> | clear | help"
    )

    while True:

        try:
            command = input().strip()

        except EOFError:
            break

        if not command:
            continue

        command_lower = command.lower()


        # ====================================================
        # CLEAR
        # ====================================================

        if command_lower in (
            "clear",
            "stop",
            "stop search",
        ):

            target_search.clear_target()

            target_guidance.stop()

            distance_smoother.reset()

            print(
                "[SEARCH] Target search/guidance cleared."
            )

            continue


        # ====================================================
        # HELP
        # ====================================================

        if command_lower == "help":

            print()
            print("[SEARCH] Commands:")
            print("         find <object>")
            print("         find my <object>")
            print("         clear")
            print()

            continue


        # ====================================================
        # FIND
        # ====================================================

        if (
            command_lower.startswith("find")
            or command_lower.startswith("search")
            or command_lower.startswith("locate")
            or command_lower.startswith("where")
        ):

            # Stop previous target.
            target_guidance.stop()

            # Reset previous distance state.
            distance_smoother.reset()

            target_name = target_search.set_target(
                command
            )

            if target_name:

                print(
                    f"[SEARCH] Searching for: "
                    f"{target_name}"
                )

                voice.speak(
                    f"Searching for {target_name}."
                )

            else:

                error_message = (
                    target_search.get_error_message()
                )

                if error_message:

                    print(
                        f"[SEARCH] {error_message}"
                    )

            continue


        # ====================================================
        # UNKNOWN COMMAND
        # ====================================================

        print(
            "[INFO] Unknown command. "
            "Type 'help' for commands."
        )


# ============================================================
# START INPUT THREAD
# ============================================================

input_thread = threading.Thread(
    target=input_listener,
    daemon=True,
)

input_thread.start()


# ============================================================
# START
# ============================================================

print(
    "[INFO] Phase 3 Navigation started. "
    "Press 'q' to quit."
)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # 1. READ CAMERA
        # ====================================================

        success, frame = camera.read()

        if not success:

            print(
                "[ERROR] Failed to read camera frame."
            )

            break


        frame_height = float(
            frame.shape[0]
        )

        frame_width = float(
            frame.shape[1]
        )


        # ====================================================
        # 2. YOLO
        # ====================================================

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
        )

        boxes = results[0].boxes


        # ====================================================
        # 3. DETECTION LISTS
        # ====================================================

        all_detections = []

        obstacle_detections = []

        current_track_ids = set()


        # ====================================================
        # 4. PROCESS DETECTIONS
        # ====================================================

        if boxes is not None:

            for box in boxes:

                # ------------------------------------------------
                # Confidence
                # ------------------------------------------------

                confidence = float(
                    box.conf[0]
                )

                if confidence < CONFIDENCE_THRESHOLD:
                    continue


                # ------------------------------------------------
                # Class
                # ------------------------------------------------

                class_id = int(
                    box.cls[0]
                )

                object_name = model.names[
                    class_id
                ]


                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                coords = box.xyxy[0]

                x1 = float(coords[0])
                y1 = float(coords[1])
                x2 = float(coords[2])
                y2 = float(coords[3])


                # ------------------------------------------------
                # Track ID
                # ------------------------------------------------

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                    current_track_ids.add(
                        track_id
                    )

                else:

                    # Temporary ID.
                    #
                    # Do not discard the detection just because
                    # YOLO did not provide a persistent track ID.

                    track_id = -(
                        100000
                        + class_id
                        + int(x1)
                        + int(y1)
                    )


                # ------------------------------------------------
                # Spatial analysis
                # ------------------------------------------------

                detection = analyze_detection(
                    track_id=track_id,
                    object_name=object_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )


                if detection is None:
                    continue


                # ------------------------------------------------
                # Keep ALL detections.
                #
                # This is important because target search needs
                # to find far objects too.
                # ------------------------------------------------

                all_detections.append(
                    detection
                )


                # =================================================
                # SAFETY FILTER
                # =================================================
                #
                # Normal objects remain silent.
                #
                # Only close/near HIGH or MEDIUM priority
                # objects directly in the center are safety
                # obstacles.
                # =================================================

                if (
                    detection.is_obstacle
                    and detection.priority in (
                        Priority.HIGH,
                        Priority.MEDIUM,
                    )
                    and detection.zone == Zone.CENTER
                    and detection.proximity in (
                        "NEAR",
                        "CLOSE",
                    )
                ):

                    obstacle_detections.append(
                        detection
                    )


        # ====================================================
        # 5. SAFETY GUIDANCE
        # ====================================================

        obstacle_decisions = (
            obstacle_guidance.process_frame(
                obstacle_detections,
                current_track_ids,
            )
        )


        selected_obstacle = (
            guidance_selector.select(
                obstacle_decisions
            )
        )


        obstacle_spoke = False


        # ----------------------------------------------------
        # If user is searching for an object, that requested
        # object gets target priority.
        #
        # Example:
        #
        # find chair
        #
        # Do NOT say:
        #
        # chair ahead
        #
        # Target guidance will handle the chair.
        # ----------------------------------------------------

        active_target_name = (
            target_guidance.target_name
            if target_guidance.is_active
            else None
        )


        if selected_obstacle is not None:

            obstacle_detection = (
                selected_obstacle.detection
            )

            obstacle_name = (
                obstacle_detection.object_name
                .lower()
                .strip()
            )


            same_as_target = (
                active_target_name is not None
                and obstacle_name
                == active_target_name.lower().strip()
            )


            if not same_as_target:

                message = (
                    direction_assistant.build_message(
                        selected_obstacle
                    )
                )


                if message is not None:

                    voice.speak(
                        message
                    )

                    obstacle_spoke = True


        # ====================================================
        # 6. TARGET SEARCH
        # ====================================================

        if target_search.is_active:

            search_result = (
                target_search.process_frame(
                    all_detections
                )
            )


            if search_result.message is not None:

                # ------------------------------------------------
                # IMPORTANT:
                #
                # TargetSearch should NOT announce the target
                # direction when it finds the target.
                #
                # TargetGuidance handles direction.
                # ------------------------------------------------

                if search_result.found:

                    pending_target_message = None

                else:

                    if obstacle_spoke:

                        pending_target_message = (
                            search_result.message
                        )

                    else:

                        voice.speak(
                            search_result.message
                        )

                        pending_target_message = None


                # ------------------------------------------------
                # Target found
                # ------------------------------------------------

                if (
                    search_result.found
                    and search_result.object_name
                ):

                    target_guidance.start(
                        search_result.object_name,
                        "",
                    )

                    distance_smoother.reset()


        # ====================================================
        # 7. TARGET DIRECTION + DISTANCE
        # ====================================================

        target_guidance_spoke = False


        if (
            target_guidance.is_active
            and not obstacle_spoke
        ):

            target_name = (
                target_guidance.target_name
            )


            if target_name:

                # ------------------------------------------------
                # Find requested object
                # ------------------------------------------------

                target_candidates = [
                    detection
                    for detection in all_detections
                    if (
                        detection.object_name
                        .lower()
                        == target_name.lower()
                    )
                ]


                if target_candidates:

                    # ------------------------------------------------
                    # Strongest matching detection
                    # ------------------------------------------------

                    target_detection = max(
                        target_candidates,
                        key=lambda detection: (
                            detection.confidence,
                            detection.box_area,
                        ),
                    )


                    # ------------------------------------------------
                    # Distance
                    # ------------------------------------------------

                    distance_estimate = (
                        distance_estimator.estimate(
                            target_detection
                        )
                    )


                    if distance_estimate is not None:

                        (
                            distance_state,
                            distance_changed,
                        ) = (
                            distance_smoother.update(
                                distance_estimate
                            )
                        )

                    else:

                        distance_state = None

                        distance_changed = False


                    # ------------------------------------------------
                    # Direction state
                    # ------------------------------------------------

                    guidance_result = (
                        target_guidance.process_frame(
                            all_detections
                        )
                    )


                    # ------------------------------------------------
                    # Target still found
                    # ------------------------------------------------

                    if guidance_result.found:

                        stable_zone = (
                            guidance_result.zone
                        )


                        # =================================================
                        # DIRECTION TEXT
                        # =================================================

                        if stable_zone == Zone.LEFT:

                            direction_text = (
                                "on your left"
                            )

                        elif stable_zone == Zone.RIGHT:

                            direction_text = (
                                "on your right"
                            )

                        else:

                            direction_text = (
                                "ahead"
                            )


                        # =================================================
                        # DISTANCE TEXT
                        # =================================================

                        distance_text = ""

                        if distance_state is not None:

                            steps = max(1, round(distance_state.distance_m / 0.5))

                            distance_text = (
                                f", about "
                                f"{steps} "
                                f"steps away"
                            )




                        


                        # =================================================
                        # SPEAK WHEN:
                        #
                        # 1. Direction changed/stabilized
                        #
                        # OR
                        #
                        # 2. Meaningful distance change happened.
                        # =================================================

                        should_speak = False


                        if (
                            guidance_result.message
                            is not None
                        ):

                            should_speak = True


                        elif distance_changed:

                            should_speak = True


                        if should_speak:

                            if distance_text:

                                message = (
                                    f"Your {target_name} "
                                    f"is {direction_text}"
                                    f"{distance_text}."
                                )

                            voice.speak(
                                message
                            )

                            target_guidance_spoke = True


        # ====================================================
        # 8. PENDING SEARCH MESSAGE
        # ====================================================

        if (
            pending_target_message is not None
            and not obstacle_spoke
            and not target_guidance_spoke
        ):

            voice.speak(
                pending_target_message
            )

            pending_target_message = None


        # ====================================================
        # 9. DETECTION FILTER
        # ====================================================

        # Keep this running as part of the Phase 3 pipeline.
        #
        # Target guidance does not wait for this filter.

        stable_detections = (
            detection_filter.process_frame(
                all_detections
            )
        )

        _ = stable_detections


        # ====================================================
        # 10. DISPLAY
        # ====================================================

        annotated_frame = (
            results[0].plot()
        )


        cv2.imshow(
            "Phase 3 Navigation",
            annotated_frame
        )


        # ====================================================
        # 11. QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            print(
                "[INFO] Quit key pressed. "
                "Stopping."
            )

            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    camera.release()

    cv2.destroyAllWindows()

    print(
        "[INFO] Camera released. "
        "Phase 3 Navigation stopped."
    )