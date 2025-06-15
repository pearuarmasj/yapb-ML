import cv2
import numpy as np

def preprocess(frame, size=(84, 84)):
    # Resize, convert to grayscale, normalize
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, size)
    frame = frame.astype(np.float32) / 255.0  # Normalize
    return frame  # Shape: (84, 84)

from collections import deque

frame_stack = deque(maxlen=4)

def reset_stack(frame):
    frame_stack.clear()
    for _ in range(4):
        frame_stack.append(frame)

def stack_frames(new_frame):
    frame_stack.append(new_frame)
    return np.stack(frame_stack, axis=0)  # Shape: (4, 84, 84)
