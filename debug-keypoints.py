"""
Howard — Debug Keypoints
Shows all 6 facial keypoints and their coordinates in the terminal.
Useful for tuning the frontal orientation filter.

Run:
    python debug_keypoints.py

Press Q to quit.
"""

import cv2
import mediapipe as mp
import urllib.request
import os

MODEL_PATH = "blaze_face_full_range.tflite"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_full_range/float16/1/"
    "blaze_face_full_range.tflite"
)

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

options = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    min_detection_confidence=0.5
)
detector = mp.tasks.vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

KP_LABELS  = ["R.eye", "L.eye", "Nose", "Mouth", "L.ear", "R.ear"]
KP_COLOURS = [
    (0,200,255),(0,200,255),(0,120,255),
    (200,200,0),(255,60,180),(255,60,180),
]

print("Face the camera directly. Press Q to quit.")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results  = detector.detect(mp_image)
    h, w     = frame.shape[:2]

    if results.detections:
        d = results.detections[0]
        print(f"Face width: {d.bounding_box.width}px  |  Keypoints: {len(d.keypoints)}")
        for i, kp in enumerate(d.keypoints):
            label = KP_LABELS[i] if i < len(KP_LABELS) else f"kp{i}"
            print(f"  {label:<6}  x={kp.x:.3f}  y={kp.y:.3f}")

            cx, cy = int(kp.x * w), int(kp.y * h)
            col = KP_COLOURS[i] if i < len(KP_COLOURS) else (255,255,255)
            cv2.circle(frame, (cx, cy), 5, col, -1)
            cv2.putText(frame, label, (cx+6, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        bbox = d.bounding_box
        cv2.rectangle(frame,
                      (bbox.origin_x, bbox.origin_y),
                      (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                      (0, 255, 120), 2)
    else:
        print("No face detected", end="\r")

    cv2.imshow("Debug keypoints — press Q", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()