from ultralytics import YOLO
import cv2

from navigation.spatial_analyzer import analyze_detection
from navigation.distance_estimator import DistanceEstimator
from navigation.distance_smoother import DistanceSmoother


CAMERA_INDEX = 2
CONFIDENCE = 0.60
IMAGE_SIZE = 320


print("[INFO] Loading YOLO...")
model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not camera.isOpened():
    print("[ERROR] Camera could not be opened.")
    raise SystemExit(1)

estimator = DistanceEstimator(
    focal_length_px=700.0
)

smoother = DistanceSmoother(
    window_size=5,
    change_threshold_m=0.30,
)

print("[INFO] Camera connected.")
print()
print("Show ONE object at a time.")
print("Test person, bottle, laptop and chair.")
print("Move closer and farther slowly.")
print("Press Q to quit.")
print()

last_report = None


try:

    while True:

        success, frame = camera.read()

        if not success:
            print("[ERROR] Failed to read camera frame.")
            break

        height, width = frame.shape[:2]

        results = model(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False,
        )

        current_states = []

        if results[0].boxes is not None:

            for box in results[0].boxes:

                confidence = float(box.conf[0])

                if confidence < CONFIDENCE:
                    continue

                class_id = int(box.cls[0])

                object_name = model.names[class_id]

                coords = box.xyxy[0]

                x1 = float(coords[0])
                y1 = float(coords[1])
                x2 = float(coords[2])
                y2 = float(coords[3])

                detection = analyze_detection(
                    track_id=0,
                    object_name=object_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    frame_width=float(width),
                    frame_height=float(height),
                )

                if detection is None:
                    continue

                estimate = estimator.estimate(
                    detection
                )

                if estimate is None:
                    continue

                state, should_report = smoother.update(
                    estimate
                )

                if should_report:

                    current_states.append(
                        (
                            state.object_name,
                            round(state.distance_m, 1),
                            state.band,
                        )
                    )

        current_states.sort()

        state_signature = tuple(
            current_states
        )

        if state_signature != last_report:

            if current_states:

                for name, distance, band in current_states:

                    print(
                        f"[DISTANCE] {name}: "
                        f"approximately {distance} m "
                        f"({band})"
                    )

            last_report = state_signature

        annotated = results[0].plot()

        cv2.imshow(
            "Distance Smoother Test",
            annotated
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Q pressed.")
            break


finally:

    camera.release()
    cv2.destroyAllWindows()

    print("[INFO] Camera released.")
    print("[INFO] Distance smoother test stopped.")