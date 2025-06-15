import random
import mss
import numpy as np
import cv2
import pyautogui
import time

time.sleep(2)  # Allow time to switch to the game window

def grab_screen(region=None):
    """Grab the screen and return it as a numpy array."""
    with mss.mss() as sct:
        # Define the monitor to capture
        monitor = sct.monitors[1] if region is None else {
            'top': region[1],
            'left': region[0],
            'width': region[2] - region[0],
            'height': region[3] - region[1]
        }
        
        # Capture the screen
        screenshot = sct.grab(monitor)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert BGRA to BGR (OpenCV uses BGR format)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img

class CS16Env:
    def __init__(self, region=None):
        self.region = region
        if self.region is None:
            # Default region for CS 1.6, adjust as necessary
            self.region = (0, 0, 640, 480)
        self.actions = ['w', 'a', 's', 'd', 'shoot']

    def reset(self):
        # Respawn logic if needed, or just start at initial position
        pass

    def step(self, action):
        if action in ['w', 'a', 's', 'd']:
            pyautogui.keyDown(action)
            time.sleep(0.1)
            pyautogui.keyUp(action)
        elif action == 'shoot':
            pyautogui.click(button='left')
        # Get new screen/frame
        obs = grab_screen(self.region)
        # Reward: For now, just random
        reward = random.random()
        done = False  # TODO: Detect if dead or episode ends
        return obs, reward, done, {}

    def get_observation(self):
        return grab_screen(self.region)