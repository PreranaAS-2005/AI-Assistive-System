from ultralytics import YOLO
import cv2
import pyttsx3

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize voice engine
engine = pyttsx3.init()

# Open webcam
camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    results = model(frame,conf=0.6)

    boxes = results[0].boxes

    for box in boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        object_name = model.names[class_id]

        print(f"{object_name} : {confidence:.2f}")

        engine.stop()
        engine.say(f"{object_name} detected")
        engine.runAndWait()

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Voice", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

camera.release()
cv2.destroyAllWindows()