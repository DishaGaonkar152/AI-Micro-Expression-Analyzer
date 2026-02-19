import numpy as np

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def detect_blink(landmarks):
    leftEAR = eye_aspect_ratio(landmarks[LEFT_EYE])
    rightEAR = eye_aspect_ratio(landmarks[RIGHT_EYE])
    return (leftEAR + rightEAR) / 2.0
