import numpy as np

def get_landmark_coords(landmarks, img_w, img_h):
    coords = []
    for lm in landmarks:
        coords.append((int(lm.x * img_w), int(lm.y * img_h)))
    return np.array(coords)
