"""
Howard — Camera Calibration
Stand exactly 1 metre from the camera and note the face width in pixels.
Set that value as FACE_1M_PX in face_detection.py.

Run:
    python calibrate.py

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
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

options = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    min_detection_confidence=0.5
)
detector = mp.tasks.vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nStand exactly 1 metre from the camera.")
print("The face width value shown is what you should set as FACE_1M_PX.")
print("Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results  = detector.detect(mp_image)

    if results.detections:
        w = results.detections[0].bounding_box.width
        msg = f"Face width at this distance: {w}px  ->  set FACE_1M_PX = {w}"
        print(msg, end="\r")
        cv2.putText(frame, f"FACE_1M_PX = {w}px", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 120), 2)
    else:
        print("No face detected                              ", end="\r")
        cv2.putText(frame, "No face detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Calibrate — press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
print("\nDone. Update FACE_1M_PX in face_detection.py with the value above.")