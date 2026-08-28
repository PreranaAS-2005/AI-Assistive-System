"""
navigation_main.py

Phase 3 standalone entry point.

Pipeline:
  Camera
    -> YOLO tracking
    -> spatial analysis
    -> obstacle guidance
    -> target search / continuous target guidance
    -> stable detection filtering
    -> approximate distance estimation + smoothing
    -> VoiceManager
"""

from ultralytics import YOLO
import cv2
import threading

from voice.voice_manager import VoiceManager
from navigation.spatial_analyzer import analyze_detection
from navigation.obstacle_guidance import ObstacleGuidance
from navigation.direction_assistant import DirectionAssistant
from navigation.target_search import TargetSearch
from navigation.target_guidance import TargetGuidance
from navigation.detection_filter import DetectionFilter
from navigation.distance_estimator import DistanceEstimator
from navigation.distance_smoother import DistanceSmoother


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60
CAMERA_INDEX = 2
IMAGE_SIZE = 320


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW,
)

if not camera.isOpened():
    print(
        f"[ERROR] Could not open camera "
        f"(index {CAMERA_INDEX}, CAP_DSHOW)."
    )
    print(
        "        Ensure DroidCam/OBS virtual camera is running "
        "and the phone is connected."
    )
    raise SystemExit(1)


voice = VoiceManager()

guidance = ObstacleGuidance()
assistant = DirectionAssistant()

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
    change_threshold_m=0.30,
)

pending_target_message = None


# ---------------------------------------------------------------------------
# Background terminal input
# ---------------------------------------------------------------------------

def _input_listener() -> None:
    """Read target-search commands from the terminal."""

    print(
        "[INFO] Target search commands: "
        "find <object> | clear | help"
    )

    while True:

        try:
            line = input().strip()

        except EOFError:
            break

        if not line:
            continue

        cmd = line.lower()

        if cmd in ("clear", "stop", "stop search"):

            target_search.clear_target()
            target_guidance.stop()

            print("[SEARCH] Target search/guidance cleared.")

        elif cmd == "help":

            print("[SEARCH] Commands:")
            print("         find <object>   e.g. find bottle")
            print("         find my <object> e.g. find my phone")
            print("         clear           stop searching/guidance")

        elif (
            cmd.startswith("find")
            or cmd.startswith("search")
            or cmd.startswith("locate")
            or cmd.startswith("where")
        ):

            # A new search cancels any previous continuous guidance.
            target_guidance.stop()

            name = target_search.set_target(line)

            if name:
                print(f"[SEARCH] Searching for: {name}")

            else:
                err = target_search.get_error_message()

                if err:
                    print(f"[SEARCH] {err}")

        else:

            target_guidance.stop()

            name = target_search.set_target(line)

            if name:
                print(f"[SEARCH] Searching for: {name}")

            else:
                err = target_search.get_error_message()

                if err:
                    print(f"[SEARCH] {err}")


_input_thread = threading.Thread(
    target=_input_listener,
    daemon=True,
)

_input_thread.start()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print("[INFO] Phase 3 Navigation started. Press 'q' to quit.")

try:

    while True:

        # ---------------------------------------------------------------
        # 1. Camera
        # ---------------------------------------------------------------

        success, frame = camera.read()

        if not success:
            print("[ERROR] Failed to read camera frame. Stopping.")
            break

        frame_height = float(frame.shape[0])
        frame_width = float(frame.shape[1])

        # ---------------------------------------------------------------
        # 2. YOLO tracking
        # ---------------------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            imgsz=IMAGE_SIZE,
        )

        boxes = results[0].boxes

        detections = []
        all_detections = []
        current_track_ids = set()

        # ---------------------------------------------------------------
        # 3. Convert YOLO boxes to DetectionResult
        # ---------------------------------------------------------------

        if boxes is not None:

            for box in boxes:

                confidence = float(box.conf[0])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(box.cls[0])
                object_name = model.names[class_id]

                if box.id is None:
                    continue

                track_id = int(box.id[0])
                current_track_ids.add(track_id)

                coords = box.xyxy[0]

                x1 = float(coords[0])
                y1 = float(coords[1])
                x2 = float(coords[2])
                y2 = float(coords[3])

                result = analyze_detection(
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

                if result is None:
                    continue

                # All valid objects are available to target search/guidance.
                all_detections.append(result)

                # Obstacle guidance only receives obstacles.
                if result.is_obstacle:
                    detections.append(result)

        # ---------------------------------------------------------------
        # 4. Existing obstacle guidance
        # ---------------------------------------------------------------

        decisions = guidance.process_frame(
            detections,
            current_track_ids,
        )

        obstacle_spoke = False

        if decisions:

            message = assistant.build_message(
                decisions[0]
            )

            if message is not None:

                voice.speak(message)
                obstacle_spoke = True

        # ---------------------------------------------------------------
        # 5. One-shot target search
        # ---------------------------------------------------------------

        if target_search.is_active:

            search_result = target_search.process_frame(
                all_detections
            )

            if search_result.message is not None:

                if obstacle_spoke:

                    pending_target_message = (
                        search_result.message
                    )

                else:

                    voice.speak(
                        search_result.message
                    )

                    pending_target_message = None

                # When search FOUND the object, begin continuous
                # left/center/right guidance for that object.
                if (
                    search_result.found
                    and search_result.object_name
                ):

                    target_guidance.start(
                        search_result.object_name,
                        search_result.zone,
                    )

        # ---------------------------------------------------------------
        # 6. Continuous target guidance
        # ---------------------------------------------------------------

        target_guidance_spoke = False

        if (
            target_guidance.is_active
            and not obstacle_spoke
        ):

            guidance_result = (
                target_guidance.process_frame(
                    all_detections
                )
            )

            if guidance_result.message is not None:

                voice.speak(
                    guidance_result.message
                )

                target_guidance_spoke = True

        # ---------------------------------------------------------------
        # 7. Deferred one-shot search message
        # ---------------------------------------------------------------

        if (
            pending_target_message is not None
            and not obstacle_spoke
            and not target_guidance_spoke
        ):

            voice.speak(
                pending_target_message
            )

            pending_target_message = None

        # ---------------------------------------------------------------
        # 8. Stable detections for distance
        #
        # DetectionFilter is intentionally used here rather than replacing
        # the existing obstacle-guidance state machine. This prevents
        # double temporal filtering from making obstacle warnings sluggish.
        # ---------------------------------------------------------------

        stable_detections = (
            detection_filter.process_frame(
                all_detections
            )
        )

        # Distance has the lowest speaking priority:
        # obstacle > target guidance > search > distance.
        if (
            not obstacle_spoke
            and not target_guidance_spoke
            and pending_target_message is None
        ):

            for stable in stable_detections:

                estimate = distance_estimator.estimate(
                    stable.detection
                )

                if estimate is None:
                    continue

                state, should_report = (
                    distance_smoother.update(
                        estimate
                    )
                )

                if should_report:

                    message = (
                        distance_smoother.spoken_message(
                            state
                        )
                    )

                    voice.speak(message)

                    # At most one voice message per frame.
                    break

        # ---------------------------------------------------------------
        # 9. Display
        # ---------------------------------------------------------------

        annotated_frame = results[0].plot()

        cv2.imshow(
            "Phase 3 Navigation",
            annotated_frame,
        )

        # ---------------------------------------------------------------
        # 10. Quit
        # ---------------------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("[INFO] Quit key pressed. Stopping.")
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print(
        "[INFO] Camera released. "
        "Phase 3 Navigation stopped."
    )
