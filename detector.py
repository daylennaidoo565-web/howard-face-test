import cv2
import mediapipe as mp
import threading
import time
import urllib.request
import os

# ── Calibration ────────────────────────────────────────────────────────────────
FACE_1M_PX        = 80
HYSTERESIS        = 20
COOLDOWN_S        = 2.0
FRONTAL_TOLERANCE = 0.28
CONFIRM_FRAMES    = 8
RELEASE_FRAMES    = 10

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_full_range/float16/1/"
    "blaze_face_full_range.tflite"
)
MODEL_PATH = "blaze_face_full_range.tflite"


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[detector] Downloading face detection model (~1 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[detector] Model downloaded.")


class ProximityDetector:
    def __init__(self, camera_index=0, frame_width=1280, frame_height=720):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.camera_index = camera_index
        self.face_1m_px   = FACE_1M_PX

        _ensure_model()

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.6
        )
        self._detector_ctx = mp.tasks.vision.FaceDetector.create_from_options(options)

        self._lock          = threading.Lock()
        self._state         = "idle"
        self._last_trigger  = 0.0
        self._confirm_count = 0
        self._release_count = 0
        self._running       = False
        self._thread        = None
        self._paused        = False          # ← added

        self.on_enter_detected = None
        self.on_enter_idle     = None
        self.on_frame          = None

    # ── Pause / Resume ─────────────────────────────────────────────────────────

    def pause(self):
        """Stop processing frames — saves CPU during conversation."""
        self._paused = True
        print("[detector] Paused")

    def resume(self):
        """Resume processing frames."""
        self._paused = False
        print("[detector] Resumed")

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def confirm_count(self):
        return self._confirm_count

    @property
    def release_count(self):
        return self._release_count

    @property
    def state(self):
        with self._lock:
            return self._state

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    # ── Internal loop ──────────────────────────────────────────────────────────

    def _loop(self):
        cap = None

        def _open_cap():
            nonlocal cap
            c = cv2.VideoCapture(self.camera_index)
            c.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_width)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not c.isOpened():
                raise RuntimeError("Cannot open camera. Check index and permissions.")
            cap = c

        _open_cap()

        try:
            while self._running:
                # ── Paused — release camera so the hardware light turns off
                if self._paused:
                    if cap is not None:
                        cap.release()
                        cap = None
                    time.sleep(0.1)
                    continue

                # ── Reopen camera after coming back from pause
                if cap is None:
                    _open_cap()

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                face_px, is_frontal, detections = self._analyse_frame(frame)

                if self.on_frame:
                    self.on_frame(frame, detections)

                self._update_state(face_px, is_frontal)
                time.sleep(0.033)

        finally:
            if cap is not None:
                cap.release()
            self._detector_ctx.close()

    # ── Frame analysis ─────────────────────────────────────────────────────────

    def _analyse_frame(self, frame):
        """Return (face_width_px, is_frontal, detections)."""
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = self._detector_ctx.detect(mp_image)

        if not results.detections:
            return 0, False, []

        valid = []
        for d in results.detections:
            bbox = d.bounding_box
            if bbox.height == 0:
                continue
            aspect = bbox.width / bbox.height
            if 0.5 < aspect < 1.8:
                valid.append(d)

        if not valid:
            return 0, False, []

        best      = max(valid, key=lambda d: d.bounding_box.width)
        face_w_px = best.bounding_box.width
        frontal   = self._check_frontal(best)

        return face_w_px, frontal, valid

    def _check_frontal(self, detection):
        """
        Returns True only if the face is roughly pointing at the camera.
        kp[0]=right eye, kp[1]=left eye, kp[2]=nose tip,
        kp[3]=mouth,     kp[4]=left ear, kp[5]=right ear
        """
        kps = detection.keypoints
        if len(kps) < 6:
            return False

        nose      = kps[2]
        left_ear  = kps[4]
        right_ear = kps[5]

        ear_to_ear_w   = abs(right_ear.x - left_ear.x)
        ear_midpoint_x = (left_ear.x + right_ear.x) / 2.0

        if ear_to_ear_w < 0.01:
            return False

        deviation = abs(nose.x - ear_midpoint_x) / ear_to_ear_w
        return deviation <= FRONTAL_TOLERANCE

    # ── State machine ──────────────────────────────────────────────────────────

    def _update_state(self, face_px, is_frontal):
        now = time.time()
        with self._lock:
            current = self._state

        in_range = face_px >= FACE_1M_PX - HYSTERESIS
        is_valid = in_range and is_frontal

        if current == "idle":
            if is_valid:
                self._confirm_count += 1
                self._release_count  = 0
            else:
                self._confirm_count  = 0

            print(
                f"[detector] IDLE  | face={face_px}px "
                f"in_range={in_range} frontal={is_frontal} "
                f"confirm={self._confirm_count}/{CONFIRM_FRAMES}    ",
                end="\r"
            )

            if self._confirm_count >= CONFIRM_FRAMES:
                if now - self._last_trigger >= COOLDOWN_S:
                    print(f"\n[detector] Confirmed — waking after "
                          f"{CONFIRM_FRAMES} frames")
                    self._confirm_count = 0
                    self._last_trigger  = now
                    self._set_state("detected")

        elif current == "detected":
            if not is_valid:
                self._release_count += 1
                self._confirm_count  = 0
            else:
                self._release_count  = 0

            print(
                f"[detector] ACTIVE | face={face_px}px "
                f"in_range={in_range} frontal={is_frontal} "
                f"release={self._release_count}/{RELEASE_FRAMES}    ",
                end="\r"
            )

            if self._release_count >= RELEASE_FRAMES:
                print(f"\n[detector] Person gone — returning to idle")
                self._release_count = 0
                self._set_state("idle")

    def _set_state(self, new_state):
        with self._lock:
            old_state   = self._state
            self._state = new_state

        if new_state == "detected" and old_state == "idle":
            if self.on_enter_detected:
                threading.Thread(
                    target=self.on_enter_detected, daemon=True
                ).start()

        elif new_state == "idle" and old_state == "detected":
            if self.on_enter_idle:
                threading.Thread(
                    target=self.on_enter_idle, daemon=True
                ).start()