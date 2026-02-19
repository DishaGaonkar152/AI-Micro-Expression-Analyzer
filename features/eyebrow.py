EYEBROW_L = 65
EYE_TOP_L = 159

def eyebrow_movement(landmarks):
    return abs(landmarks[EYEBROW_L][1] - landmarks[EYE_TOP_L][1])
