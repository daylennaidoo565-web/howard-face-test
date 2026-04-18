"""
Howard — Face Detection with Live Benchmarks
Tests MediaPipe BlazeFace on the Pi camera with real-time performance stats.

Run:
    python face_detection.py

Press Q to quit.
"""

import cv2
import mediapipe as mp
import urllib.request
import os
import time
import collections
import threading
import psutil

# ── Config ─────────────────────────────────────────────────────────────────────
FACE_1M_PX        = 80        # face width in pixels at 1m — run calibrate.py to set this
HYSTERESIS        = 20
FRONTAL_TOLERANCE = 0.28
CONFIRM_FRAMES    = 8
RELEASE_FRAMES    = 10
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_full_range/float16/1/"
    "blaze_face_full_range.tflite"
)
MODEL_PATH = "blaze_face_full_range.tflite"

# ── Download model if needed ────────────────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[detector] Downloading face detection model (~1 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[detector] Model downloaded.")

# ── Frontal check ───────────────────────────────────────────────────────────────
def check_frontal(detection):
    kps = detection.keypoints
    if len(kps) < 6:
        return False
    nose      = kps[2]
    left_ear  = kps[4]
    right_ear = kps[5]
    ear_w     = abs(right_ear.x - left_ear.x)
    ear_mid_x = (left_ear.x + right_ear.x) / 2.0
    if ear_w < 0.01:
        return False
    deviation = abs(nose.x - ear_mid_x) / ear_w
    return deviation <= FRONTAL_TOLERANCE

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    ensure_model()

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.6
    )
    detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[detector] ERROR: Cannot open camera. Check camera index.")
        return

    process = psutil.Process(os.getpid())

    # ── Benchmark tracking ──────────────────────────────────────────────────────
    frame_times    = collections.deque(maxlen=30)
    detect_times   = collections.deque(maxlen=30)
    ram_samples    = collections.deque(maxlen=30)
    session_start  = time.time()
    frame_count    = 0
    confirm_count  = 0
    release_count  = 0
    state          = "idle"

    # ── Benchmark summary at end ────────────────────────────────────────────────
    all_fps     = []
    all_detect  = []
    all_ram     = []

    print("\n" + "=" * 60)
    print("  Howard — Face Detection Benchmark")
    print("  Press Q to quit and see summary")
    print("=" * 60)
    print(f"  Model    : BlazeFace Full Range")
    print(f"  Camera   : {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  FACE_1M  : {FACE_1M_PX}px  (run calibrate.py to update)")
    print("=" * 60 + "\n")

    ram_before = process.memory_info().rss / (1024 ** 2)

    while True:
        t_frame_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_count += 1

        # ── Detection ───────────────────────────────────────────────────────────
        t_detect_start = time.perf_counter()

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = detector.detect(mp_image)

        t_detect_end = time.perf_counter()
        detect_ms = (t_detect_end - t_detect_start) * 1000
        detect_times.append(detect_ms)
        all_detect.append(detect_ms)

        # ── Parse detections ────────────────────────────────────────────────────
        face_px   = 0
        is_frontal = False
        valid_dets = []

        if results.detections:
            for d in results.detections:
                bbox = d.bounding_box
                if bbox.height == 0:
                    continue
                aspect = bbox.width / bbox.height
                if 0.5 < aspect < 1.8:
                    valid_dets.append(d)

            if valid_dets:
                best      = max(valid_dets, key=lambda d: d.bounding_box.width)
                face_px   = best.bounding_box.width
                is_frontal = check_frontal(best)

        # ── State machine ───────────────────────────────────────────────────────
        in_range = face_px >= FACE_1M_PX - HYSTERESIS
        is_valid = in_range and is_frontal

        if state == "idle":
            if is_valid:
                confirm_count += 1
                release_count  = 0
            else:
                confirm_count  = 0
            if confirm_count >= CONFIRM_FRAMES:
                state         = "detected"
                confirm_count = 0
                print(f"\n[detector] ✓ Face confirmed — would wake Howard")
        else:
            if not is_valid:
                release_count += 1
                confirm_count  = 0
            else:
                release_count  = 0
            if release_count >= RELEASE_FRAMES:
                state         = "idle"
                release_count = 0
                print(f"\n[detector] Person gone — returning to idle")

        # ── RAM ─────────────────────────────────────────────────────────────────
        ram_now = process.memory_info().rss / (1024 ** 2)
        ram_samples.append(ram_now)
        all_ram.append(ram_now)

        # ── FPS ─────────────────────────────────────────────────────────────────
        t_frame_end = time.perf_counter()
        frame_ms = (t_frame_end - t_frame_start) * 1000
        frame_times.append(frame_ms)
        all_fps.append(1000 / frame_ms if frame_ms > 0 else 0)

        fps_now     = 1000 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        detect_avg  = sum(detect_times) / len(detect_times) if detect_times else 0
        ram_avg     = sum(ram_samples) / len(ram_samples) if ram_samples else 0

        # ── Draw overlay ────────────────────────────────────────────────────────
        vis = frame.copy()
        h, w = vis.shape[:2]

        box_colour = (0, 255, 120) if state == "idle" else (255, 180, 0)

        KP_COLOURS = [
            (0,200,255),(0,200,255),(0,120,255),
            (200,200,0),(255,60,180),(255,60,180),
        ]
        KP_LABELS = ["R.eye","L.eye","Nose","Mouth","L.ear","R.ear"]

        for det in valid_dets:
            bbox = det.bounding_box
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2, y2 = x1 + bbox.width, y1 + bbox.height
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_colour, 2)
            cv2.putText(vis, f"{bbox.width}px", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colour, 1)

            for i, kp in enumerate(det.keypoints):
                cx, cy = int(kp.x * w), int(kp.y * h)
                col = KP_COLOURS[i] if i < len(KP_COLOURS) else (255,255,255)
                cv2.circle(vis, (cx, cy), 4, col, -1)
                cv2.putText(vis, KP_LABELS[i], (cx+5, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

        # ── HUD ─────────────────────────────────────────────────────────────────
        elapsed = int(time.time() - session_start)
        ram_drift = ram_now - ram_before

        hud = [
            f"FPS      : {fps_now:.1f}",
            f"Detect   : {detect_avg:.1f}ms avg",
            f"Frame    : {frame_ms:.1f}ms",
            f"CPU%     : {psutil.cpu_percent():.0f}%",
            f"RAM      : {ram_now:.0f}MB  (drift: {ram_drift:+.1f}MB)",
            f"Faces    : {len(valid_dets)}",
            f"Face px  : {face_px}px",
            f"Frontal  : {is_frontal}",
            f"State    : {state.upper()}",
            f"Confirm  : {confirm_count}/{CONFIRM_FRAMES}",
            f"Release  : {release_count}/{RELEASE_FRAMES}",
            f"Uptime   : {elapsed}s  Frames: {frame_count}",
        ]

        overlay_bg = vis.copy()
        cv2.rectangle(overlay_bg, (0, 0), (260, len(hud) * 22 + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.5, vis, 0.5, 0, vis)

        for i, line in enumerate(hud):
            col = (0, 255, 120) if "DETECT" in line else (200, 200, 200)
            if "State" in line:
                col = (255, 180, 0) if state == "detected" else (200, 200, 200)
            cv2.putText(vis, line, (8, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

        cv2.imshow("Howard — Face Detection Benchmark (Q to quit)", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Summary ─────────────────────────────────────────────────────────────────
    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    total_time = time.time() - session_start
    ram_drift  = (max(all_ram) if all_ram else 0) - ram_before

    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Session duration     : {total_time:.1f}s")
    print(f"  Total frames         : {frame_count}")
    print()
    if all_fps:
        print(f"  FPS — average        : {sum(all_fps)/len(all_fps):.1f}")
        print(f"  FPS — min            : {min(all_fps):.1f}")
        print(f"  FPS — max            : {max(all_fps):.1f}")
    if all_detect:
        print(f"  Detection ms — avg   : {sum(all_detect)/len(all_detect):.1f}ms")
        print(f"  Detection ms — min   : {min(all_detect):.1f}ms")
        print(f"  Detection ms — max   : {max(all_detect):.1f}ms")
    print(f"  RAM at start         : {ram_before:.0f}MB")
    print(f"  RAM peak             : {max(all_ram) if all_ram else 0:.0f}MB")
    print(f"  RAM drift            : {ram_drift:+.1f}MB")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()