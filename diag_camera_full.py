"""
diag_camera_full.py — Temporary diagnostic: tests all index/backend combinations
while DroidCam client is open. Run once, then delete.
"""
import cv2

combos = [
    (0, cv2.CAP_MSMF,  "CAP_MSMF  idx=0"),
    (1, cv2.CAP_MSMF,  "CAP_MSMF  idx=1"),
    (2, cv2.CAP_MSMF,  "CAP_MSMF  idx=2"),
    (0, cv2.CAP_DSHOW, "CAP_DSHOW idx=0"),
    (1, cv2.CAP_DSHOW, "CAP_DSHOW idx=1"),
    (2, cv2.CAP_DSHOW, "CAP_DSHOW idx=2"),
]

for idx, backend, label in combos:
    try:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            print(f"  {label}: DID NOT OPEN")
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  {label}: OK | mean={frame.mean():.1f} | std={frame.std():.1f}")
        else:
            print(f"  {label}: OPENED but read failed")
        cap.release()
    except Exception as e:
        print(f"  {label}: EXCEPTION: {e}")
