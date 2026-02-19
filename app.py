import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import csv
import os

from utils.landmark_utils import get_landmark_coords
from features.blink import detect_blink
from features.lip_tension import lip_tension
from features.eyebrow import eyebrow_movement
from features.head_nod import head_nod
from features.symmetry import facial_symmetry
from models.state_model import compute_state


# ----------------------------
# CSV Logging Setup
# ----------------------------
LOG_FILE = "stress_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "blink",
            "lip_tension",
            "eyebrow",
            "head_nod",
            "symmetry",
            "score",
            "classification"
        ])

# 🔁 Logging Toggle Switch
ENABLE_LOGGING = True   # Set False to disable logging


# ----------------------------
# MediaPipe Setup
# ----------------------------
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

prev_nose_y = 0

# Alert system
last_alert_time = 0
ALERT_COOLDOWN = 3
alert_count = 0

# Analytics data
score_history = []
MAX_HISTORY = 120
frame_count = 0
start_time = time.time()

# Precision Enhancements
smoothed_score = None
ALPHA = 0.2
high_stress_frames = 0
STABILITY_THRESHOLD = 15


# ----------------------------
# Helper Functions
# ----------------------------

def get_color(value):
    if value < 0.25:
        return (0, 200, 0)
    elif value < 0.42:
        return (0, 165, 255)
    else:
        return (0, 0, 255)


def play_alert():
    global last_alert_time, alert_count
    current_time = time.time()
    if current_time - last_alert_time > ALERT_COOLDOWN:
        winsound.Beep(1000, 400)
        last_alert_time = current_time
        alert_count += 1


def draw_metric(panel, y, label, value, max_val=1.0):
    norm_value = min(value / max_val, 1.0)
    bar_width = int(norm_value * 250)

    color = get_color(norm_value)

    cv2.rectangle(panel, (20, y), (280, y + 20), (40, 40, 40), -1)
    cv2.rectangle(panel, (20, y), (20 + bar_width, y + 20), color, -1)
    cv2.rectangle(panel, (20, y), (280, y + 20), (100, 100, 100), 1)

    cv2.putText(panel, label, (20, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(panel, f"{value:.2f}", (230, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def draw_graph(panel, values, x_offset, y_offset):
    if len(values) < 2:
        return

    graph_w = 280
    graph_h = 120

    cv2.rectangle(panel,
                  (x_offset, y_offset),
                  (x_offset + graph_w, y_offset + graph_h),
                  (30, 30, 30), -1)

    scaled = [int(v * graph_h) for v in values[-MAX_HISTORY:]]

    for i in range(1, len(scaled)):
        x1 = x_offset + int((i - 1) * graph_w / MAX_HISTORY)
        y1 = y_offset + graph_h - scaled[i - 1]
        x2 = x_offset + int(i * graph_w / MAX_HISTORY)
        y2 = y_offset + graph_h - scaled[i]

        cv2.line(panel, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.putText(panel, "Stress Trend",
                (x_offset, y_offset - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)


# ----------------------------
# Main Loop
# ----------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    panel = np.zeros((h, 320, 3), dtype=np.uint8)

    cv2.rectangle(panel, (0, 0), (320, 110), (0, 150, 0), -1)
    cv2.putText(panel, "AI MICRO-EXPRESSION ANALYZER", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(panel, "Behavioral Alert System Active", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                face,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 215, 0), thickness=1, circle_radius=1
                ),
            )

            landmarks = get_landmark_coords(face.landmark, w, h)

            blink_val = detect_blink(landmarks)
            lip_val = lip_tension(landmarks) / 50.0
            brow_val = eyebrow_movement(landmarks) / 50.0
            nod_val = abs(head_nod(prev_nose_y, landmarks) / 50.0)
            sym_val = facial_symmetry(landmarks) / 100.0

            prev_nose_y = landmarks[1][1]

            state, score = compute_state(
                blink_val, lip_val, brow_val, nod_val, sym_val
            )

            score_norm = score

            # EMA smoothing
            if smoothed_score is None:
                smoothed_score = score_norm
            else:
                smoothed_score = ALPHA * score_norm + (1 - ALPHA) * smoothed_score

            filtered_score = smoothed_score

            # 🔔 Beep when STRESS
            if state == "STRESS":
                play_alert()

            # Stable High Stress Detection
            if filtered_score >= 0.65:
                high_stress_frames += 1
            else:
                high_stress_frames = 0

            if high_stress_frames >= STABILITY_THRESHOLD:
                play_alert()

            # CSV Logging (Toggle Controlled)
            if ENABLE_LOGGING:
                with open(LOG_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        time.time(),
                        blink_val,
                        lip_val,
                        brow_val,
                        nod_val,
                        sym_val,
                        filtered_score,
                        state
                    ])

            score_history.append(filtered_score)
            if len(score_history) > MAX_HISTORY:
                score_history.pop(0)

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed

            state_color = get_color(filtered_score)

            cv2.putText(panel, f"Status: {state}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            cv2.putText(panel, f"Score: {filtered_score:.2f}", (200, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            draw_metric(panel, 130, "Blink Score", blink_val)
            draw_metric(panel, 180, "Eyebrow Movement", brow_val)
            draw_metric(panel, 230, "Lip Tension", lip_val)
            draw_metric(panel, 280, "Head Micro-Nod", nod_val)
            draw_metric(panel, 330, "Symmetry Shift", sym_val)

    analytics_panel = np.zeros((h, 320, 3), dtype=np.uint8)

    cv2.putText(analytics_panel, "ANALYTICS", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

    draw_graph(analytics_panel, score_history, 20, 80)

    rolling_avg = sum(score_history) / len(score_history) if score_history else 0

    cv2.putText(analytics_panel, f"Rolling Avg: {rolling_avg:.2f}",
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(analytics_panel, f"Alerts Triggered: {alert_count}",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(analytics_panel, f"FPS: {fps:.1f}",
                (20, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    combined = np.hstack((frame, panel, analytics_panel))
    cv2.imshow("AI Micro-Expression Analyzer", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('l'):
        ENABLE_LOGGING = not ENABLE_LOGGING

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
