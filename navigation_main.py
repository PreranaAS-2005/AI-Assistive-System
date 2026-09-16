"""
Phase 3 Navigation Module
AI-Based Assistive Spatial Memory and Guidance System
"""

from ultralytics import YOLO
import cv2
import threading
import sys

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

CONFIDENCE_THRESHOLD = 0.30
IMAGE_SIZE = 640

# Average human stride length
AVERAGE_STEP_METERS = 0.70

# DroidCam / camera index
CAMERA_INDEX = 1


# ============================================================
# INITIALIZATION
# ============================================================

print("[INFO] Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("[INFO] YOLO model loaded.")


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


distance_smoother = DistanceSmoother(
    window_size=5,
    change_threshold_m=0.60,
)


guidance_selector = GuidanceSelector(
    switch_distance_margin_m=0.75,
)


pending_target_message = None


# ============================================================
# CAMERA SETUP
# ============================================================

print(
    f"[INFO] Connecting to camera index {CAMERA_INDEX}..."
)

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_MSMF
)


if not camera.isOpened():

    print(
        f"[ERROR] Could not open camera index "
        f"{CAMERA_INDEX}."
    )

    print(
        "[ERROR] Make sure DroidCam is running "
        "and connected."
    )

    sys.exit(1)


print(
    f"[INFO] Camera index {CAMERA_INDEX} connected."
)


# ============================================================
# KEYBOARD INPUT LISTENER
# ============================================================

def input_listener():

    global pending_target_message

    while True:

        try:

            user_input = input().strip()

            if not user_input:
                continue


            # ------------------------------------------------
            # CANCEL TARGET
            # ------------------------------------------------

            if user_input.lower() == "cancel":

                target_search.cancel()

                target_guidance.stop()

                distance_smoother.reset()

                pending_target_message = None

                print(
                    "[SEARCH] Target guidance cancelled."
                )

                voice.speak(
                    "Target cancelled."
                )


            # ------------------------------------------------
            # FIND TARGET
            # ------------------------------------------------

            else:

                target_guidance.stop()

                distance_smoother.reset()

                pending_target_message = None

                clean_target = (
                    user_input
                    .lower()
                    .strip()
                )

                set_name = target_search.set_target(
                    f"find {clean_target}"
                )


                if set_name:

                    print(
                        f"[SEARCH] Searching for: "
                        f"{set_name}"
                    )

                    voice.speak(
                        f"Searching for {set_name}."
                    )


        except (EOFError, KeyboardInterrupt):

            break


input_thread = threading.Thread(
    target=input_listener,
    daemon=True,
)

input_thread.start()


# ============================================================
# AUTO-APPLY SPOKEN TARGET
# ============================================================

if len(sys.argv) > 1:

    target_arg = (
        sys.argv[1]
        .lower()
        .strip()
    )

    target_search.set_target(
        f"find {target_arg}"
    )

    voice.speak(
        f"Searching for {target_arg}."
    )


# ============================================================
# MAIN NAVIGATION LOOP
# ============================================================

print(
    "[INFO] Navigation running on DroidCam."
)

print(
    "[INFO] Type 'find <object>' in the terminal."
)

print(
    "[INFO] Press 'q' on the camera window to quit."
)


try:

    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        success, frame = camera.read()


        if not success or frame is None:

            print(
                "[ERROR] Failed to read frame "
                "from camera."
            )

            break


        frame_height = float(
            frame.shape[0]
        )

        frame_width = float(
            frame.shape[1]
        )


        # ====================================================
        # YOLO TRACKING
        # ====================================================

        results = model.track(

            frame,

            persist=True,

            verbose=False,

            imgsz=IMAGE_SIZE,

            conf=CONFIDENCE_THRESHOLD,

        )


        boxes = results[0].boxes


        all_detections = []

        obstacle_detections = []

        current_track_ids = set()


        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        if boxes is not None:

            for box in boxes:

                confidence = float(
                    box.conf[0]
                )


                if (
                    confidence
                    < CONFIDENCE_THRESHOLD
                ):

                    continue


                class_id = int(
                    box.cls[0]
                )


                object_name = (
                    model.names[class_id]
                )


                # ------------------------------------------------
                # YOLO DIAGNOSTIC
                # ------------------------------------------------

                print(
                    f"[YOLO] {object_name} | "
                    f"confidence={confidence:.2f}"
                )


                if (
                    object_name.lower().strip()
                    == "bottle"
                ):

                    print(
                        f"[BOTTLE YOLO] "
                        f"Bottle detected by YOLO | "
                        f"confidence={confidence:.2f}"
                    )


                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                coords = box.xyxy[0]


                x1 = float(
                    coords[0]
                )

                y1 = float(
                    coords[1]
                )

                x2 = float(
                    coords[2]
                )

                y2 = float(
                    coords[3]
                )


                # ------------------------------------------------
                # TRACK ID
                # ------------------------------------------------

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                    current_track_ids.add(
                        track_id
                    )

                else:

                    track_id = -(
                        100000
                        + class_id
                        + int(x1)
                        + int(y1)
                    )


                # ------------------------------------------------
                # SPATIAL ANALYSIS
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

                    print(
                        f"[REJECTED] "
                        f"{object_name} "
                        f"was rejected by spatial analyzer."
                    )

                    continue


                # ------------------------------------------------
                # ACCEPTED DETECTION DIAGNOSTIC
                # ------------------------------------------------

                print(
                    f"[ACCEPTED] "
                    f"{detection.object_name} | "
                    f"confidence={detection.confidence:.2f} | "
                    f"zone={detection.zone} | "
                    f"proximity={detection.proximity}"
                )


                all_detections.append(
                    detection
                )


                if (
                    detection.object_name
                    .lower()
                    .strip()
                    == "bottle"
                ):

                    print(
                        f"[BOTTLE ACCEPTED] "
                        f"Bottle reached all_detections | "
                        f"confidence="
                        f"{detection.confidence:.2f} | "
                        f"zone={detection.zone}"
                    )


                # ------------------------------------------------
                # OBSTACLE FILTER
                # ------------------------------------------------

                if (

                    detection.is_obstacle

                    and detection.priority
                    in (
                        Priority.HIGH,
                        Priority.MEDIUM,
                    )

                    and detection.zone
                    == Zone.CENTER

                    and detection.proximity
                    in (
                        "NEAR",
                        "CLOSE",
                    )

                ):

                    obstacle_detections.append(
                        detection
                    )


        # ====================================================
        # SAFETY & OBSTACLE HANDLING
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


        active_target_name = (

            target_guidance.target_name

            if target_guidance.is_active

            else None

        )


        if selected_obstacle is not None:

            obs_name = (
                selected_obstacle
                .detection
                .object_name
                .lower()
                .strip()
            )


            same_as_target = (

                active_target_name

                and

                obs_name
                == active_target_name
                .lower()
                .strip()

            )


            if not same_as_target:

                message = (
                    direction_assistant
                    .build_message(
                        selected_obstacle
                    )
                )


                if message:

                    voice.speak(
                        message
                    )

                    obstacle_spoke = True


        # ====================================================
        # TARGET SEARCH
        # ====================================================

        target_guidance_spoke = False


        if target_search.is_active:

            search_result = (
                target_search.process_frame(
                    all_detections
                )
            )


            # ------------------------------------------------
            # TARGET SEARCH DIAGNOSTIC
            # ------------------------------------------------

            if target_search.target_name:

                target_in_frame = any(

                    d.object_name.lower().strip()
                    ==
                    target_search.target_name.lower().strip()

                    for d in all_detections

                )


                print(
                    f"[SEARCH CHECK] "
                    f"target="
                    f"{target_search.target_name} | "
                    f"visible="
                    f"{target_in_frame}"
                )


            if search_result.message is not None:

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
            # TARGET FOUND
            # ------------------------------------------------

            if (

                search_result.found

                and

                search_result.object_name

            ):

                target_guidance.start(

                    search_result.object_name,

                    "",

                )

                distance_smoother.reset()


        # ====================================================
        # CONTINUOUS TARGET GUIDANCE
        # ====================================================

        if (

            target_guidance.is_active

            and

            not obstacle_spoke

        ):

            target_name = (
                target_guidance.target_name
            )


            if target_name:

                candidates = [

                    d

                    for d in all_detections

                    if (
                        d.object_name.lower()
                        == target_name.lower()
                    )

                ]


                if candidates:

                    target_det = max(

                        candidates,

                        key=lambda d: (
                            d.confidence,
                            d.box_area,
                        ),

                    )


                    # ------------------------------------------------
                    # DISTANCE
                    # ------------------------------------------------

                    dist_estimate = (
                        distance_estimator
                        .estimate(
                            target_det
                        )
                    )


                    if dist_estimate:

                        (
                            dist_state,
                            dist_changed,
                        ) = (
                            distance_smoother.update(
                                dist_estimate
                            )
                        )

                    else:

                        dist_state = None

                        dist_changed = False


                    # ------------------------------------------------
                    # TARGET GUIDANCE
                    # ------------------------------------------------

                    guidance_res = (
                        target_guidance
                        .process_frame(
                            all_detections
                        )
                    )


                    if guidance_res.found:

                        if (

                            guidance_res.message
                            is not None

                            or

                            dist_changed

                        ):

                            dist_m = (

                                dist_state
                                .distance_m

                                if dist_state

                                else None

                            )


                            # ------------------------------------------------
                            # VERY CLOSE
                            # ------------------------------------------------

                            if (

                                dist_m

                                and

                                dist_m < 0.6

                            ):

                                message = (

                                    f"Your "
                                    f"{target_name} "
                                    f"is right in "
                                    f"front of you. "
                                    f"Reach out."

                                )


                            else:

                                # --------------------------------------------
                                # HUMAN STEPS
                                # --------------------------------------------

                                if dist_m:

                                    steps = max(

                                        1,

                                        round(
                                            dist_m
                                            / AVERAGE_STEP_METERS
                                        ),

                                    )

                                else:

                                    steps = None


                                if steps:

                                    step_str = (

                                        f"{steps} "

                                        f"{'step' if steps == 1 else 'steps'}"

                                    )

                                else:

                                    step_str = ""


                                if dist_m:

                                    dist_str = (

                                        f", about "
                                        f"{dist_m:.1f} "
                                        f"meters away"

                                    )

                                else:

                                    dist_str = ""


                                # ------------------------------------------------
                                # LEFT
                                # ------------------------------------------------

                                if (

                                    guidance_res.zone
                                    == Zone.LEFT

                                ):

                                    if step_str:

                                        action = (
                                            f"Turn left "
                                            f"and take "
                                            f"{step_str}."
                                        )

                                    else:

                                        action = (
                                            "Turn left."
                                        )


                                    message = (

                                        f"Your "
                                        f"{target_name} "
                                        f"is on your "
                                        f"left"
                                        f"{dist_str}. "
                                        f"{action}"

                                    )


                                # ------------------------------------------------
                                # RIGHT
                                # ------------------------------------------------

                                elif (

                                    guidance_res.zone
                                    == Zone.RIGHT

                                ):

                                    if step_str:

                                        action = (
                                            f"Turn right "
                                            f"and take "
                                            f"{step_str}."
                                        )

                                    else:

                                        action = (
                                            "Turn right."
                                        )


                                    message = (

                                        f"Your "
                                        f"{target_name} "
                                        f"is on your "
                                        f"right"
                                        f"{dist_str}. "
                                        f"{action}"

                                    )


                                # ------------------------------------------------
                                # CENTER
                                # ------------------------------------------------

                                else:

                                    if step_str:

                                        action = (
                                            f"Walk straight "
                                            f"{step_str}."
                                        )

                                    else:

                                        action = (
                                            "Walk straight "
                                            "ahead."
                                        )


                                    message = (

                                        f"Your "
                                        f"{target_name} "
                                        f"is ahead"
                                        f"{dist_str}. "
                                        f"{action}"

                                    )


                            print(
                                f"[GUIDANCE] "
                                f"{message}"
                            )


                            voice.speak(
                                message
                            )


                            target_guidance_spoke = True


        # ====================================================
        # PENDING SEARCH MESSAGE
        # ====================================================

        if (

            pending_target_message
            is not None

            and

            not obstacle_spoke

            and

            not target_guidance_spoke

        ):

            voice.speak(
                pending_target_message
            )

            pending_target_message = None


        # ====================================================
        # DISPLAY
        # ====================================================

        annotated_frame = (
            results[0].plot()
        )


        cv2.imshow(
            "Navigation Mode",
            annotated_frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):

            print(
                "[INFO] Quit key pressed."
            )

            break


finally:

    camera.release()

    cv2.destroyAllWindows()

    print(
        "[INFO] Navigation stopped."
    )
