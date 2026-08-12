import cv2

cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

if not cap.isOpened():
    print("❌ Could not open DroidCam")
    exit()

print("✅ DroidCam camera connected!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    cv2.imshow("Phone Camera - DroidCam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()