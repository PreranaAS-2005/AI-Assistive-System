"""
diag_camera.py — Temporary diagnostic script.

Tests ONLY raw OpenCV camera capture from DroidCam (index 1, CAP_MSMF).
No YOLO, no model, no navigation code.

Purpose: determine whether the green frame comes from OpenCV/DroidCam
         or from YOLO/results[0].plot() downstream.

Run from project root:
    python diag_camera.py

Press 'q' to quit.
Delete this file after diagnosis is complete.
"""

import cv2

print("[DIAG] Opening camera index 1 with CAP_MSMF ...")
camera = cv2.VideoCapture(1, cv2.CAP_MSMF)

if not camera.isOpened():
    print("[DIAG] ERROR: Camera did not open.")
else:
    print("[DIAG] Camera opened successfully.")

frame_count = 0

while True:
    success, frame = camera.read()

    frame_count += 1

    if not success or frame is None:
        print(f"[DIAG] Frame {frame_count}: read() returned success={success}, frame=None")
        break

    # Print diagnostic values every 5 frames to avoid console flood
    if frame_count % 5 == 1:
        print(
            f"[DIAG] Frame {frame_count}: "
            f"success={success} | "
            f"shape={frame.shape} | "
            f"dtype={frame.dtype} | "
            f"min={int(frame.min())} | "
            f"max={int(frame.max())} | "
            f"mean={frame.mean():.1f}"
        )

    # Show the RAW frame — no YOLO, no plot(), no annotation
    cv2.imshow("RAW DROIDCAM (no YOLO)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[DIAG] Quit key pressed.")
        break

camera.release()
cv2.destroyAllWindows()
print("[DIAG] Done.")
