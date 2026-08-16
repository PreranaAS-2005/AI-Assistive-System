"""
navigation_main.py

Phase 3 standalone entry point for the AI-Based Assistive Spatial Memory
and Guidance System.

This file connects the existing YOLO tracking pipeline with the three
Phase 3 navigation modules and the existing VoiceManager.

Data flow:
  Camera
    → YOLOv8 tracking  (ultralytics)
    → spatial_analyzer  (zone, proximity, priority)
    → obstacle_guidance (when to announce)
    → direction_assistant (what to say)
    → VoiceManager (speaks it aloud)

Run this file to start Phase 3 navigation.
Do NOT modify Phase 1 files (yolo_voice_manager.py etc.) to add this logic.
"""

from ultralytics import YOLO
import cv2
import threading

from voice.voice_manager import VoiceManager
from navigation.spatial_analyzer import analyze_detection
from navigation.obstacle_guidance import ObstacleGuidance
from navigation.direction_assistant import DirectionAssistant
from navigation.target_search import TargetSearch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Confidence threshold — consistent with yolo_voice_manager.py
CONFIDENCE_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

# Load YOLOv8 nano model (same model used by Phase 1)
model = YOLO("yolov8n.pt")

# Camera configuration
# DroidCam Video is DirectShow index 0 on this machine.
# Integrated Webcam is DirectShow index 1.
# If DroidCam is not installed, use: cv2.VideoCapture(0, cv2.CAP_MSMF)
camera = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Check that the camera opened successfully before entering the loop
if not camera.isOpened():
    print("[ERROR] Could not open OBS Virtual Camera (index 2, CAP_DSHOW).")
    print("        Ensure DroidCam desktop client is running and the phone is connected.")
    exit(1)

# Initialize voice output (non-blocking, background thread)
voice = VoiceManager()

# Phase 3 navigation modules
guidance  = ObstacleGuidance()    # decides when to announce
assistant = DirectionAssistant()  # builds the spoken message
target_search = TargetSearch()    # target object search
pending_target_message = None       # deferred if obstacle guidance speaks first


# ---------------------------------------------------------------------------
# Background stdin thread for target search commands
# ---------------------------------------------------------------------------

def _input_listener() -> None:
    """Read search commands from the terminal in a background thread."""
    print("[INFO] Target search commands:  find <object>  |  clear  |  help")
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
            print("[SEARCH] Target search cleared.")

        elif cmd == "help":
            print("[SEARCH] Commands:")
            print("         find <object>   e.g.  find bottle")
            print("         find my <object> e.g.  find my phone")
            print("         clear           stop searching")

        elif cmd.startswith("find") or cmd.startswith("search") or cmd.startswith("locate") or cmd.startswith("where"):
            name = target_search.set_target(line)
            if name:
                print(f"[SEARCH] Searching for: {name}")
            else:
                err = target_search.get_error_message()
                if err:
                    print(f"[SEARCH] {err}")

        else:
            # Treat bare object name as a search request
            name = target_search.set_target(line)
            if name:
                print(f"[SEARCH] Searching for: {name}")
            else:
                err = target_search.get_error_message()
                if err:
                    print(f"[SEARCH] {err}")


_input_thread = threading.Thread(target=_input_listener, daemon=True)
_input_thread.start()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print("[INFO] Phase 3 Navigation started. Press 'q' to quit.")

try:

    while True:

        # ------------------------------------------------------------------
        # 1. Read one frame from the camera.
        # ------------------------------------------------------------------
        success, frame = camera.read()

        if not success:
            print("[ERROR] Failed to read frame from camera. Stopping.")
            break

        # ------------------------------------------------------------------
        # 2. Get frame dimensions
        #    frame.shape = (height, width, channels)
        #    Always use shape[0] for height and shape[1] for width.
        # ------------------------------------------------------------------
        frame_height = float(frame.shape[0])
        frame_width  = float(frame.shape[1])

        # ------------------------------------------------------------------
        # 3. Run YOLO tracking
        #    persist=True keeps track IDs stable across frames.
        #    verbose=False suppresses YOLO's per-frame console output.
        # ------------------------------------------------------------------
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            imgsz=320,
        )

        boxes = results[0].boxes

        # ------------------------------------------------------------------
        # 4. Process detections
        # ------------------------------------------------------------------

        # All valid DetectionResult objects for this frame (obstacles only)
        detections: list = []

        # All valid DetectionResult objects including IGNORE-priority
        # objects - used by target search
        all_detections: list = []

        # All track IDs visible in this frame (including non-obstacles)
        # Used by ObstacleGuidance to detect disappearances
        current_track_ids: set = set()

        if boxes is not None:

            for box in boxes:

                # (a) Get confidence score
                confidence = float(box.conf[0])

                # (b) Skip low-confidence detections
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # (c) Get object class and name
                class_id    = int(box.cls[0])
                object_name = model.names[class_id]

                # (d) Skip if the tracker has not assigned an ID yet
                if box.id is None:
                    continue

                # (e) Get the persistent tracking ID
                track_id = int(box.id[0])

                # (f) Record this ID as visible this frame
                #     Done before any obstacle filtering so disappearance
                #     detection works correctly for all tracked objects
                current_track_ids.add(track_id)

                # (g) Get bounding-box coordinates from the tensor
                #     box.xyxy[0] = [x1, y1, x2, y2] as a PyTorch tensor
                #     Cast each value to float for spatial_analyzer
                coords = box.xyxy[0]
                x1 = float(coords[0])
                y1 = float(coords[1])
                x2 = float(coords[2])
                y2 = float(coords[3])

                # (h) Run spatial analysis
                #     Returns None for invalid/zero-size boxes
                result = analyze_detection(
                    track_id     = track_id,
                    object_name  = object_name,
                    confidence   = confidence,
                    x1           = x1,
                    y1           = y1,
                    x2           = x2,
                    y2           = y2,
                    frame_width  = frame_width,
                    frame_height = frame_height,
                )

                # (i) Skip invalid boxes
                if result is None:
                    continue

                # (i-b) Keep every valid detection for target search
                all_detections.append(result)

                # (j) Skip IGNORE-priority objects
                #     Track ID is already in current_track_ids above,
                #     so disappearance tracking still works for them.
                if not result.is_obstacle:
                    continue

                # (k) Add valid obstacle to this frame's detection list
                detections.append(result)

        # ------------------------------------------------------------------
        # 5. Get guidance decisions
        #    Decisions are returned sorted HIGH priority first.
        # ------------------------------------------------------------------
        decisions = guidance.process_frame(
            detections,
            current_track_ids,
        )

        # ------------------------------------------------------------------
        # 6. Speak the highest-priority guidance message (at most one)
        #    Obstacle guidance takes priority over target search.
        # ------------------------------------------------------------------
        obstacle_spoke = False

        if decisions:

            # decisions[0] is highest priority due to sort in process_frame
            message = assistant.build_message(decisions[0])

            if message is not None:
                voice.speak(message)
                obstacle_spoke = True

        # ------------------------------------------------------------------
        # 6b. Target object search (lower priority than obstacle guidance)
        # ------------------------------------------------------------------
        if target_search.is_active:
            search_result = target_search.process_frame(all_detections)

            if search_result.message is not None:
                if obstacle_spoke:
                    # Preserve the one-shot result until it is safe to speak.
                    pending_target_message = search_result.message
                else:
                    voice.speak(search_result.message)
                    pending_target_message = None

        # If a target result was deferred because of an obstacle warning,
        # announce it once the obstacle guidance is quiet.
        if pending_target_message is not None and not obstacle_spoke:
            voice.speak(pending_target_message)
            pending_target_message = None

        # ------------------------------------------------------------------
        # 7. Display the annotated frame for visual debugging
        # ------------------------------------------------------------------
        annotated_frame = results[0].plot()
        cv2.imshow("Phase 3 Navigation", annotated_frame)

        # ------------------------------------------------------------------
        # 8. Check for quit key
        # ------------------------------------------------------------------
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit key pressed. Stopping.")
            break

finally:
    # ------------------------------------------------------------------
    # 9. Clean shutdown — always runs even if an error occurs
    # ------------------------------------------------------------------
    camera.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Phase 3 Navigation stopped.")