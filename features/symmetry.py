import numpy as np

def facial_symmetry(landmarks):
    left = landmarks[:234]
    right = landmarks[234:]
    return abs(np.mean(left[:, 0]) - np.mean(right[:, 0]))
