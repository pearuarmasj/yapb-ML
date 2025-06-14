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

class CS16GameController:
    """Interface for controlling CS 1.6 game through direct window input injection"""
    
    def __init__(self):
        self.hwnd = None
        self.process_id = None
        
        # Key codes for CS 1.6 controls
        self.key_codes = {
            'forward': 0x57,     # W
            'backward': 0x53,    # S
            'left': 0x41,        # A
            'right': 0x44,       # D
            'jump': 0x20,        # Space
            'duck': 0x11,        # Ctrl
            'attack1': 0x01,     # Left mouse
            'attack2': 0x02,     # Right mouse
            'reload': 0x52,      # R
            'use': 0x45,         # E
        }
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
        """Send key down event directly to CS 1.6 window"""
        if key not in self.key_codes or not self.hwnd:
            return
            
        if not self.key_states[key]:  # Only send if not already pressed
            keycode = self.key_codes[key]
            
            if key in ['attack1', 'attack2']:
                # Mouse button - send to window
                if key == 'attack1':
                    win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, 0)
                else:
                    win32api.PostMessage(self.hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, 0)
            else:
                # Keyboard key - send WM_KEYDOWN directly to window
                lparam = (1 << 16) | (keycode << 16)  # Proper lparam construction
                win32api.PostMessage(self.hwnd, win32con.WM_KEYDOWN, keycode, lparam)
            
            self.key_states[key] = True
    
    def send_key_up(self, key: str):
        """Send key up event directly to CS 1.6 window"""
        if key not in self.key_codes or not self.hwnd:
            return
            
        if self.key_states[key]:  # Only send if currently pressed
            keycode = self.key_codes[key]
            
            if key in ['attack1', 'attack2']:
                # Mouse button
                if key == 'attack1':
                    win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, 0, 0)
                else:
                    win32api.PostMessage(self.hwnd, win32con.WM_RBUTTONUP, 0, 0)
            else:
                # Keyboard key - send WM_KEYUP directly to window
                lparam = (1 << 16) | (1 << 30) | (1 << 31) | (keycode << 16)
                win32api.PostMessage(self.hwnd, win32con.WM_KEYUP, keycode, lparam)
            
            self.key_states[key] = False
    
    def send_mouse_move(self, dx: int, dy: int):
        """Send relative mouse movement to CS 1.6 window"""
        if not self.hwnd:
            return
        
        # Scale movement for CS 1.6 sensitivity
        scaled_dx = int(dx * 10)  # Increase sensitivity
        scaled_dy = int(dy * 10)
        
        # Get current cursor position relative to window
        rect = win32gui.GetWindowRect(self.hwnd)
        center_x = (rect[2] - rect[0]) // 2
        center_y = (rect[3] - rect[1]) // 2
        
        new_x = center_x + scaled_dx
        new_y = center_y + scaled_dy
        
        # Send mouse move message to window
        lparam = (new_y << 16) | (new_x & 0xFFFF)
        win32api.PostMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
    
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
