NOSE_TIP = 1

def head_nod(prev_y, curr_landmarks):
    return curr_landmarks[NOSE_TIP][1] - prev_y
