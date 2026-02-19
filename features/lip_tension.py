import numpy as np

UPPER_LIP = 13
LOWER_LIP = 14

def lip_tension(landmarks):
    vertical_dist = np.linalg.norm(
        landmarks[UPPER_LIP] - landmarks[LOWER_LIP]
    )
    return vertical_dist
