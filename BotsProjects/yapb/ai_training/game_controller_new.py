#!/usr/bin/env python3
"""
CS 1.6 Game Control Interface
Handles sending keyboard/mouse commands to CS 1.6 for bot control using direct injection
"""

import time
import ctypes
from ctypes import wintypes
import win32api
import win32con
import win32gui
import win32process
import numpy as np
from typing import Dict, Tuple

# C struct redefinitions for SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

class CS16GameController:
    """Interface for controlling CS 1.6 game through direct input injection"""
    
    def __init__(self):
        self.hwnd = None
        self.process_id = None
        
        # Key codes and scan codes for CS 1.6 controls
        self.key_codes = {
            'forward': (0x57, 0x11),     # W
            'backward': (0x53, 0x1F),    # S
            'left': (0x41, 0x1E),        # A
            'right': (0x44, 0x20),       # D
            'jump': (0x20, 0x39),        # Space
            'duck': (0x11, 0x1D),        # Ctrl
            'attack1': (0x01, 0),        # Left mouse
            'attack2': (0x02, 0),        # Right mouse
            'reload': (0x52, 0x13),      # R
            'use': (0x45, 0x12),         # E
        }
        
        # Constants for SendInput
        self.KEYEVENTF_KEYUP = 0x0002
        self.KEYEVENTF_SCANCODE = 0x0008
        self.INPUT_KEYBOARD = 1
        self.INPUT_MOUSE = 0
        
        # Mouse movement constants
        self.MOUSEEVENTF_MOVE = 0x0001
        self.MOUSEEVENTF_LEFTDOWN = 0x0002
        self.MOUSEEVENTF_LEFTUP = 0x0004
        self.MOUSEEVENTF_RIGHTDOWN = 0x0008
        self.MOUSEEVENTF_RIGHTUP = 0x0010
        
        # Current key states (to avoid key repeat)
        self.key_states = {key: False for key in self.key_codes}
        
        # Find and connect to CS 1.6
        self.find_cs16_window()
    
    def find_cs16_window(self):
        """Find CS 1.6 window and get handle"""
        window_titles = ["Counter-Strike", "Counter-Strike 1.6", "Half-Life"]
        
        for title in window_titles:
            self.hwnd = win32gui.FindWindow(None, title)
            if self.hwnd:
                _, self.process_id = win32process.GetWindowThreadProcessId(self.hwnd)
                print(f"Found CS 1.6 window: {title} (HWND: {self.hwnd}, PID: {self.process_id})")
                return True
        
        print("CS 1.6 window not found!")
        return False
    
    def set_window_handle(self, hwnd: int):
        """Set the CS 1.6 window handle"""
        self.hwnd = hwnd
        if hwnd:
            _, self.process_id = win32process.GetWindowThreadProcessId(hwnd)
    
    def send_key_down(self, key: str):
        """Send key down event using SendInput for direct injection"""
        if key not in self.key_codes:
            return
            
        if not self.key_states[key]:  # Only send if not already pressed
            keycode, scancode = self.key_codes[key]
            
            if key in ['attack1', 'attack2']:
                # Mouse button - use SendInput for mouse
                extra = ctypes.c_ulong(0)
                ii_ = Input_I()
                if key == 'attack1':
                    ii_.mi = MouseInput(0, 0, 0, self.MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
                else:
                    ii_.mi = MouseInput(0, 0, 0, self.MOUSEEVENTF_RIGHTDOWN, 0, ctypes.pointer(extra))
                x = Input(ctypes.c_ulong(self.INPUT_MOUSE), ii_)
                ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
            else:
                # Keyboard key - use SendInput for direct injection
                extra = ctypes.c_ulong(0)
                ii_ = Input_I()
                ii_.ki = KeyBdInput(0, scancode, self.KEYEVENTF_SCANCODE, 0, ctypes.pointer(extra))
                x = Input(ctypes.c_ulong(self.INPUT_KEYBOARD), ii_)
                ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
            
            self.key_states[key] = True
    
    def send_key_up(self, key: str):
        """Send key up event using SendInput for direct injection"""
        if key not in self.key_codes:
            return
            
        if self.key_states[key]:  # Only send if currently pressed
            keycode, scancode = self.key_codes[key]
            
            if key in ['attack1', 'attack2']:
                # Mouse button - use SendInput for mouse
                extra = ctypes.c_ulong(0)
                ii_ = Input_I()
                if key == 'attack1':
                    ii_.mi = MouseInput(0, 0, 0, self.MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
                else:
                    ii_.mi = MouseInput(0, 0, 0, self.MOUSEEVENTF_RIGHTUP, 0, ctypes.pointer(extra))
                x = Input(ctypes.c_ulong(self.INPUT_MOUSE), ii_)
                ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
            else:
                # Keyboard key - use SendInput for direct injection
                extra = ctypes.c_ulong(0)
                ii_ = Input_I()
                ii_.ki = KeyBdInput(0, scancode, self.KEYEVENTF_SCANCODE | self.KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
                x = Input(ctypes.c_ulong(self.INPUT_KEYBOARD), ii_)
                ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
            
            self.key_states[key] = False
    
    def send_mouse_move(self, dx: int, dy: int):
        """Send relative mouse movement using SendInput"""
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(dx, dy, 0, self.MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(self.INPUT_MOUSE), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def move_mouse(self, dx: int, dy: int):
        """Alias for send_mouse_move for compatibility"""
        self.send_mouse_move(dx, dy)
    
    def execute_action(self, action_vector: np.ndarray, duration: float = 0.05):
        """
        Execute action based on neural network output
        
        Action vector format:
        [0] forward/backward (-1 to 1)
        [1] left/right strafe (-1 to 1)  
        [2] turn left/right (-1 to 1)
        [3] look up/down (-1 to 1)
        [4] jump (0 or 1)
        [5] duck (0 or 1)
        [6] attack1 (0 or 1)
        [7] attack2 (0 or 1)
        """
        
        # Release all keys first
        self.release_all_keys()
        
        # Movement keys
        if action_vector[0] > 0.3:  # Forward
            self.send_key_down('forward')
        elif action_vector[0] < -0.3:  # Backward
            self.send_key_down('backward')
            
        if action_vector[1] > 0.3:  # Right strafe
            self.send_key_down('right')
        elif action_vector[1] < -0.3:  # Left strafe
            self.send_key_down('left')
        
        # Mouse movement
        if abs(action_vector[2]) > 0.1:  # Turning
            turn_speed = int(action_vector[2] * 10)  # Scale factor
            self.send_mouse_move(turn_speed, 0)
            
        if abs(action_vector[3]) > 0.1:  # Looking up/down
            look_speed = int(action_vector[3] * 10)  # Scale factor
            self.send_mouse_move(0, look_speed)
        
        # Action keys
        if action_vector[4] > 0.5:  # Jump
            self.send_key_down('jump')
            
        if action_vector[5] > 0.5:  # Duck
            self.send_key_down('duck')
            
        if action_vector[6] > 0.5:  # Attack1
            self.send_key_down('attack1')
            
        if action_vector[7] > 0.5:  # Attack2
            self.send_key_down('attack2')
        
        # Hold action for duration
        time.sleep(duration)
        
        # Release all keys
        self.release_all_keys()
    
    def release_all_keys(self):
        """Release all currently pressed keys"""
        for key in self.key_codes:
            if self.key_states[key]:
                self.send_key_up(key)
    
    def execute_discrete_action(self, action_id: int, duration: float = 0.05):
        """
        Execute discrete action by ID
        
        Action IDs:
        0: Move forward
        1: Move backward  
        2: Strafe left
        3: Strafe right
        4: Turn left
        5: Turn right
        6: Jump
        7: Duck
        """
        
        self.release_all_keys()
        
        if action_id == 0:  # Forward
            self.send_key_down('forward')
        elif action_id == 1:  # Backward
            self.send_key_down('backward')
        elif action_id == 2:  # Strafe left
            self.send_key_down('left')
        elif action_id == 3:  # Strafe right
            self.send_key_down('right')
        elif action_id == 4:  # Turn left
            self.send_mouse_move(-10, 0)
        elif action_id == 5:  # Turn right
            self.send_mouse_move(10, 0)
        elif action_id == 6:  # Jump
            self.send_key_down('jump')
        elif action_id == 7:  # Duck
            self.send_key_down('duck')
        
        time.sleep(duration)
        self.release_all_keys()
    
    def test_controls(self):
        """Test basic controls"""
        print("Testing CS 1.6 controls...")
        print("Make sure CS 1.6 window is active!")
        
        actions = [
            (0, "Moving forward"),
            (2, "Strafing left"),
            (3, "Strafing right"),
            (1, "Moving backward"),
            (4, "Turning left"),
            (5, "Turning right"),
            (6, "Jumping"),
            (7, "Ducking")
        ]
        
        for action_id, description in actions:
            print(f"{description}...")
            self.execute_discrete_action(action_id, 0.5)
            time.sleep(0.5)
        
        print("Control test completed!")

# Action space utilities
class ActionSpace:
    """Utility class for handling action spaces"""
    
    @staticmethod
    def discrete_to_vector(action_id: int) -> np.ndarray:
        """Convert discrete action ID to continuous action vector"""
        action_vector = np.zeros(8, dtype=np.float32)
        
        if action_id == 0:  # Forward
            action_vector[0] = 1.0
        elif action_id == 1:  # Backward
            action_vector[0] = -1.0
        elif action_id == 2:  # Left strafe
            action_vector[1] = -1.0
        elif action_id == 3:  # Right strafe
            action_vector[1] = 1.0
        elif action_id == 4:  # Turn left
            action_vector[2] = -1.0
        elif action_id == 5:  # Turn right
            action_vector[2] = 1.0
        elif action_id == 6:  # Jump
            action_vector[4] = 1.0
        elif action_id == 7:  # Duck
            action_vector[5] = 1.0
            
        return action_vector
    
    @staticmethod
    def vector_to_discrete(action_vector: np.ndarray) -> int:
        """Convert continuous action vector to discrete action ID"""
        # Find the action with highest absolute value
        abs_actions = np.abs(action_vector)
        max_idx = np.argmax(abs_actions)
        
        # Map back to discrete actions
        if max_idx == 0:  # Forward/backward
            return 0 if action_vector[0] > 0 else 1
        elif max_idx == 1:  # Left/right strafe
            return 2 if action_vector[1] < 0 else 3
        elif max_idx == 2:  # Turn left/right
            return 4 if action_vector[2] < 0 else 5
        elif max_idx == 4:  # Jump
            return 6
        elif max_idx == 5:  # Duck
            return 7
        else:
            return 0  # Default to forward

def main():
    """Test the game controller"""
    controller = CS16GameController()
    
    print("CS 1.6 Game Controller Test")
    print("Make sure CS 1.6 is running and the window is active")
    input("Press Enter to start control test...")
    
    controller.test_controls()

if __name__ == "__main__":
    main()
