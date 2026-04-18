"""
Howard — Face Detection Flask App
Turns screen on when a face is detected, off when person leaves.

Run:
    python app.py

Then open http://localhost:5000 in Chromium.
"""

import time
import queue
import threading
from flask import Flask, Response, render_template
from detector import ProximityDetector
from display import screen_off, screen_on, disable_screensaver

app = Flask(__name__)

# ── State ──────────────────────────────────────────────────────────────────────
_state      = "idle"
_state_lock = threading.Lock()
_sse_clients = []
_sse_lock    = threading.Lock()

IDLE_TIMEOUT_S = 30

_latest_frame      = None
_latest_frame_lock = threading.Lock()


# ── SSE ────────────────────────────────────────────────────────────────────────

def _push_event(event: str, data: str):
    msg = f"event: {event}\ndata: {data}\n\n"
    with _sse_lock:
        for q in list(_sse_clients):
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass


def _set_state(new_state: str):
    global _state
    with _state_lock:
        _state = new_state
    _push_event("state", new_state)
    print(f"[app] State → {new_state}")


def _register_client():
    q = queue.Queue(maxsize=20)
    with _sse_lock:
        _sse_clients.append(q)
    return q


def _unregister_client(q):
    with _sse_lock:
        if q in _sse_clients:
            _sse_clients.remove(q)


# ── Face detection callbacks ───────────────────────────────────────────────────

def _on_person_detected():
    with _state_lock:
        if _state != "idle":
            return
    screen_on()
    _set_state("greeting")
    print("[app] Face detected — screen on")


def _on_person_left():
    threading.Timer(IDLE_TIMEOUT_S, _check_idle_timeout).start()


def _check_idle_timeout():
    with _state_lock:
        current = _state
    if detector.state != "detected" and current != "idle":
        _set_state("idle")
        screen_off()
        print("[app] Person gone — screen off")


# ── Video feed ─────────────────────────────────────────────────────────────────

import cv2

def _on_frame(frame, detections):
    global _latest_frame
    import cv2
    vis = frame.copy()
    h, w = vis.shape[:2]

    with _state_lock:
        current_state = _state

    box_colour = (0, 255, 120) if current_state == "idle" else (255, 180, 0)

    KP_LABELS  = ["R.eye", "L.eye", "Nose", "Mouth", "L.ear", "R.ear"]
    KP_COLOURS = [
        (0,200,255),(0,200,255),(0,120,255),
        (200,200,0),(255,60,180),(255,60,180),
    ]

    for det in detections:
        bbox = det.bounding_box
        x1, y1 = bbox.origin_x, bbox.origin_y
        x2, y2 = x1 + bbox.width, y1 + bbox.height
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_colour, 2)

        for i, kp in enumerate(det.keypoints):
            cx, cy = int(kp.x * w), int(kp.y * h)
            col = KP_COLOURS[i] if i < len(KP_COLOURS) else (255,255,255)
            cv2.circle(vis, (cx, cy), 4, col, -1)
            cv2.putText(vis, KP_LABELS[i], (cx+5, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    hud = [
        f"State   : {current_state.upper()}",
        f"Confirm : {detector.confirm_count} / 8",
        f"Release : {detector.release_count} / 10",
    ]
    for i, line in enumerate(hud):
        cv2.putText(vis, line, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
    with _latest_frame_lock:
        _latest_frame = buf.tobytes()


def _mjpeg_generator():
    while True:
        with _latest_frame_lock:
            frame = _latest_frame
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.04)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    from flask import jsonify
    with _state_lock:
        return jsonify({"state": _state})


@app.route("/api/video")
def api_video():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/events")
def sse_stream():
    q = _register_client()

    def generate():
        with _state_lock:
            current = _state
        yield f"event: state\ndata: {current}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            _unregister_client(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Startup ────────────────────────────────────────────────────────────────────

disable_screensaver()
screen_off()

detector = ProximityDetector(camera_index=0)
detector.on_enter_detected = _on_person_detected
detector.on_enter_idle     = _on_person_left
detector.on_frame          = _on_frame
detector.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)