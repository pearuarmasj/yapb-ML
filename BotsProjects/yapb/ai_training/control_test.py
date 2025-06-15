#!/usr/bin/env python3
"""
Control Test Script
Tests basic mouse and keyboard control methods to verify they work outside of game context
"""

import time
import ctypes
from ctypes import wintypes
import win32api
import win32con

class ControlTester:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        
    def test_keyboard_basic(self):
        print("Testing basic keyboard (typing 'test' in 3 seconds)...")
        time.sleep(3)
        
        win32api.keybd_event(ord('T'), 0, 0, 0)
        time.sleep(0.05)
        win32api.keybd_event(ord('T'), 0, win32con.KEYEVENTF_KEYUP, 0)
        
        win32api.keybd_event(ord('E'), 0, 0, 0)
        time.sleep(0.05)
        win32api.keybd_event(ord('E'), 0, win32con.KEYEVENTF_KEYUP, 0)
        
        win32api.keybd_event(ord('S'), 0, 0, 0)
        time.sleep(0.05)
        win32api.keybd_event(ord('S'), 0, win32con.KEYEVENTF_KEYUP, 0)
        
        win32api.keybd_event(ord('T'), 0, 0, 0)
        time.sleep(0.05)
        win32api.keybd_event(ord('T'), 0, win32con.KEYEVENTF_KEYUP, 0)
        
        print("Keyboard test completed")
    
    def test_mouse_setcursorpos(self):
        print("Testing SetCursorPos (moving to center then corners in 3 seconds)...")
        time.sleep(3)
        
        screen_width = self.user32.GetSystemMetrics(0)
        screen_height = self.user32.GetSystemMetrics(1)
        center_x = screen_width // 2
        center_y = screen_height // 2
        
        print(f"Screen: {screen_width}x{screen_height}, Center: {center_x},{center_y}")
        
        win32api.SetCursorPos((center_x, center_y))
        time.sleep(0.5)
        
        win32api.SetCursorPos((100, 100))
        time.sleep(0.5)
        
        win32api.SetCursorPos((screen_width - 100, 100))
        time.sleep(0.5)
        
        win32api.SetCursorPos((screen_width - 100, screen_height - 100))
        time.sleep(0.5)
        
        win32api.SetCursorPos((100, screen_height - 100))
        time.sleep(0.5)
        
        win32api.SetCursorPos((center_x, center_y))
        print("SetCursorPos test completed")
    
    def test_mouse_event(self):
        print("Testing mouse_event (relative movement in 3 seconds)...")
        time.sleep(3)
        
        for i in range(10):
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 10, 0, 0, 0)
            time.sleep(0.1)
        
        for i in range(10):
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, 10, 0, 0)
            time.sleep(0.1)
        
        for i in range(10):
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -10, 0, 0, 0)
            time.sleep(0.1)
        
        for i in range(10):
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, -10, 0, 0)
            time.sleep(0.1)
        
        print("mouse_event test completed")
    
    def test_mouse_click(self):
        print("Testing mouse clicks (left click in 3 seconds)...")
        time.sleep(3)
        
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        
        print("Mouse click test completed")
    
    def test_sendinput_mouse(self):
        print("Testing SendInput mouse movement (in 3 seconds)...")
        time.sleep(3)
        
        # SendInput structures
        PUL = ctypes.POINTER(ctypes.c_ulong)
        
        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long),
                        ("dy", ctypes.c_long),
                        ("mouseData", ctypes.c_ulong),
                        ("dwFlags", ctypes.c_ulong),
                        ("time", ctypes.c_ulong),
                        ("dwExtraInfo", PUL)]
        
        class Input_I(ctypes.Union):
            _fields_ = [("mi", MouseInput)]
        
        class Input(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong),
                        ("ii", Input_I)]
        
        INPUT_MOUSE = 0
        MOUSEEVENTF_MOVE = 0x0001
        
        for i in range(20):
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(5, 0, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(INPUT_MOUSE), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
            time.sleep(0.05)
        
        print("SendInput mouse test completed")
    
    def run_all_tests(self):
        print("=== Control Test Suite ===")
        print("Close this window and switch to a text editor or notepad to see results")
        print("Tests will start in 5 seconds...")
        time.sleep(5)
        
        try:
            self.test_keyboard_basic()
            time.sleep(1)
            
            self.test_mouse_setcursorpos()
            time.sleep(1)
            
            self.test_mouse_event()
            time.sleep(1)
            
            self.test_mouse_click()
            time.sleep(1)
            
            self.test_sendinput_mouse()
            
        except Exception as e:
            print(f"Error during testing: {e}")
        
        print("=== All tests completed ===")

def main():
    tester = ControlTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
