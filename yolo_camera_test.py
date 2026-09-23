from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("✅ Camera connected")
print("Point the camera at a person, bottle, chair, or laptop.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    results = model(frame, imgsz=320, conf=0.60, verbose=False)

    for box in results[0].boxes:
        confidence = float(box.conf[0])

        if confidence >= 0.60:
            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            print(
                f"Detected: {object_name} "
                f"(confidence={confidence:.2f})"
            )

    annotated = results[0].plot()

    cv2.imshow("YOLO Camera Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()