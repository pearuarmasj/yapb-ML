#!/usr/bin/env python3
"""
CS 1.6 Game Control Interface
Handles sending keyboard/mouse commands to CS 1.6 for bot control using direct injection
"""

import time
import ctypes
from ctypes import wintypes, Structure, c_long, c_ulong, POINTER, sizeof, byref, c_void_p, c_ubyte
import win32api
import win32con
import win32gui
import win32process
import numpy as np
from typing import Dict, Tuple

# Define INPUT structure for SendInput
class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

class MOUSEINPUT(Structure):
    _fields_ = [("dx", c_long),
                ("dy", c_long),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", POINTER(wintypes.ULONG))]

class INPUT(Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _fields_ = [("type", wintypes.DWORD),
                ("_input", _INPUT)]

class CS16GameController:
    """Interface for controlling CS 1.6 game through direct window input injection"""
    
    def __init__(self):
        self.hwnd = None
        self.process_id = None
        
        # Key codes for CS 1.6 controls with scan codes
        self.key_codes = {
            'forward': (0x57, 0x11),     # W key, scan code 0x11
            'backward': (0x53, 0x1F),    # S key, scan code 0x1F  
            'left': (0x41, 0x1E),        # A key, scan code 0x1E
            'right': (0x44, 0x20),       # D key, scan code 0x20
            'jump': (0x20, 0x39),        # Space, scan code 0x39
            'duck': (0x11, 0x1D),        # Ctrl, scan code 0x1D
            'attack1': (0x01, 0x00),     # Left mouse
            'attack2': (0x02, 0x00),     # Right mouse
            'reload': (0x52, 0x13),      # R key, scan code 0x13
            'use': (0x45, 0x12),         # E key, scan code 0x12
        }
        
        # Current key states (to avoid key repeat)
        self.key_states = {key: False for key in self.key_codes}
        
        # Find and connect to CS 1.6
        self.find_cs16_window()
        
        # SDL2 memory injection setup
        self.process_handle = None
        self.sdl2_base_address = None
        self.keyboard_state_addr = None
        self.mouse_state_addr = None
        
        # Initialize SDL2 memory access
        self.init_sdl2_memory_access()
    
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
        if hwnd:            _, self.process_id = win32process.GetWindowThreadProcessId(hwnd)
    
    def send_key_down(self, key: str):
        """Send key down event using SDL2 direct injection or fallback to Windows API"""
        if key not in self.key_codes or not self.hwnd:
            return
            
        if not self.key_states[key]:  # Only send if not already pressed
            keycode, scancode = self.key_codes[key]
            
            # Try SDL2 direct injection first
            if self.sdl2_base_address and self.process_handle:
                if key in ['attack1', 'attack2']:
                    button_id = 1 if key == 'attack1' else 2
                    if self.inject_mouse_state(button_id, True):
                        self.key_states[key] = True
                        return
                else:
                    if self.inject_key_state(scancode, True):
                        self.key_states[key] = True
                        return
            
            # Fallback to Windows API
            if key in ['attack1', 'attack2']:
                if key == 'attack1':
                    win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, 0)
                else:
                    win32api.PostMessage(self.hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, 0)
            else:
                lparam = (1 << 16) | (scancode << 16)
                win32api.PostMessage(self.hwnd, win32con.WM_KEYDOWN, keycode, lparam)
            
            self.key_states[key] = True
    
    def send_key_up(self, key: str):
        """Send key up event directly to CS 1.6 window"""
        if key not in self.key_codes or not self.hwnd:
            return
            
        if self.key_states[key]:  # Only send if currently pressed
            keycode, scancode = self.key_codes[key]
            
            if key in ['attack1', 'attack2']:
                # Mouse button
                if key == 'attack1':
                    win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, 0, 0)
                else:
                    win32api.PostMessage(self.hwnd, win32con.WM_RBUTTONUP, 0, 0)
            else:
                # Keyboard key - send WM_KEYUP directly to window
                lparam = (1 << 16) | (1 << 30) | (1 << 31) | (scancode << 16)
                win32api.PostMessage(self.hwnd, win32con.WM_KEYUP, keycode, lparam)
            
            self.key_states[key] = False
    
    def send_mouse_move(self, dx: int, dy: int):
        """Send relative mouse movement using SDL2 direct injection or SendInput fallback"""
        if not self.hwnd:
            return
        
        # Try SDL2 direct injection first
        if self.process_handle:
            if self.inject_mouse_look(dx, dy):
                return
        
        # Fallback to SendInput
        extra = c_ulong(0)
        ii_ = INPUT()
        ii_.type = 0  # INPUT_MOUSE
        ii_._input.mi.dx = dx
        ii_._input.mi.dy = dy
        ii_._input.mi.dwFlags = 0x0001  # MOUSEEVENTF_MOVE
        ii_._input.mi.time = 0
        ii_._input.mi.dwExtraInfo = ctypes.pointer(extra)
        
        ctypes.windll.user32.SendInput(1, byref(ii_), sizeof(ii_))
    
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
        print("Starting in 3 seconds...")
        time.sleep(3)
        
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
    
    # SDL2 memory injection methods
    def init_sdl2_memory_access(self):
        """Initialize SDL2 memory access for direct input injection"""
        if not self.process_id:
            return False
        
        try:
            # Open process with full access
            PROCESS_ALL_ACCESS = 0x1F0FFF
            self.process_handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_ALL_ACCESS, False, self.process_id
            )
            
            if not self.process_handle:
                print("Failed to open process")
                return False
            
            # Find SDL2.dll base address
            self.sdl2_base_address = self.find_sdl2_base()
            if self.sdl2_base_address:
                print(f"SDL2.dll found at base: 0x{self.sdl2_base_address:08X}")
                return True
            else:
                print("SDL2.dll not found in process")
                return False
                
        except Exception as e:
            print(f"Error initializing SDL2 memory access: {e}")
            return False
    
    def find_sdl2_base(self):
        """Since we have absolute addresses, just return a dummy base"""
        return 0x10000000
    
    def write_memory(self, address, data):
        """Write data directly to process memory"""
        if not self.process_handle:
            return False
        
        try:
            bytes_written = ctypes.c_size_t(0)
            result = ctypes.windll.kernel32.WriteProcessMemory(
                self.process_handle,
                ctypes.c_void_p(address),
                ctypes.byref(data),
                ctypes.sizeof(data),
                ctypes.byref(bytes_written)
            )
            return bool(result)
        except Exception as e:
            print(f"Memory write error: {e}")
            return False
    
    def read_memory(self, address, size):
        """Read data from process memory"""
        if not self.process_handle:
            return None
        
        try:
            buffer = (ctypes.c_ubyte * size)()
            bytes_read = ctypes.c_size_t(0)
            result = ctypes.windll.kernel32.ReadProcessMemory(
                self.process_handle,
                ctypes.c_void_p(address),
                buffer,
                size,
                ctypes.byref(bytes_read)
            )
            
            if result:
                return bytes(buffer)
            return None
        except Exception as e:
            print(f"Memory read error: {e}")
            return None
    
    def inject_key_state(self, key_scancode, pressed):
        """Directly inject key state into SDL2 memory"""
        if not self.sdl2_base_address or not self.process_handle:
            return False
        
        # This will need to be calibrated with the specific memory addresses
        # you found in Cheat Engine for the SDL2 input state
        try:
            # Placeholder - replace with actual SDL2 keyboard state offset
            keyboard_state_offset = 0x1000  # You'll need to find this with Cheat Engine
            key_address = self.sdl2_base_address + keyboard_state_offset + key_scancode
            
            # Write key state (1 for pressed, 0 for released)
            state_value = ctypes.c_ubyte(1 if pressed else 0)
            return self.write_memory(key_address, state_value)
            
        except Exception as e:
            print(f"Key injection error: {e}")
            return False
    
    def inject_mouse_state(self, button, pressed):
        """Directly inject mouse button state into SDL2 memory at 0x1043EE14"""
        if not self.process_handle:
            return False
        
        try:
            # Your discovered mouse button address
            mouse_button_address = 0x1043EE14
            
            # Current state: 0x4B normal, 0x4C when mouse pressed
            state_value = ctypes.c_ubyte(0x4C if pressed else 0x4B)
            return self.write_memory(mouse_button_address, state_value)
            
        except Exception as e:
            print(f"Mouse injection error: {e}")
            return False
    
    def inject_mouse_look(self, dx, dy):
        """Directly inject mouse look values into SDL2 memory"""
        if not self.process_handle:
            return False
        
        try:
            # Your discovered mouse look addresses
            mouse_x_address = 0x1043ED54
            mouse_y_address = 0x1043ED64
            
            # Write relative mouse movement values
            x_value = ctypes.c_long(dx)
            y_value = ctypes.c_long(dy)
            
            success_x = self.write_memory(mouse_x_address, x_value)
            success_y = self.write_memory(mouse_y_address, y_value)
            
            return success_x and success_y
            
        except Exception as e:
            print(f"Mouse look injection error: {e}")
            return False
    
    def set_sdl2_addresses(self, keyboard_offset=None, mouse_offset=None):
        """Set the SDL2 memory offsets found through Cheat Engine"""
        if keyboard_offset is not None:
            self.keyboard_state_addr = self.sdl2_base_address + keyboard_offset
            print(f"SDL2 keyboard state address set: 0x{self.keyboard_state_addr:08X}")
        
        if mouse_offset is not None:
            self.mouse_state_addr = self.sdl2_base_address + mouse_offset
            print(f"SDL2 mouse state address set: 0x{self.mouse_state_addr:08X}")
    
    def test_sdl2_injection(self):
        """Test SDL2 direct injection by writing to memory addresses"""
        if not self.sdl2_base_address:
            print("SDL2 base address not found")
            return False
        
        print("Testing SDL2 direct injection...")
        print("This will try to inject key states directly into SDL2 memory")
        print("Make sure you've set the correct offsets with set_sdl2_addresses()")
        
        # Test keyboard injection (W key)
        if self.keyboard_state_addr:
            w_key_scancode = 0x11
            test_address = self.keyboard_state_addr + w_key_scancode
            
            print(f"Testing W key injection at address: 0x{test_address:08X}")
            
            # Press W
            if self.write_memory(test_address, ctypes.c_ubyte(1)):
                print("W key pressed via SDL2 injection")
                time.sleep(1)
                
                # Release W
                if self.write_memory(test_address, ctypes.c_ubyte(0)):
                    print("W key released via SDL2 injection")
                    return True
        
        print("SDL2 injection test failed")
        return False

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
    """Test the game controller with SDL2 injection options"""
    controller = CS16GameController()
    
    print("CS 1.6 Game Controller Test")
    print("Make sure CS 1.6 is running and the window is active")
    print()
    print("Options:")
    print("1. Test standard Windows API controls")
    print("2. Test SDL2 direct injection (requires memory addresses)")
    print("3. Set SDL2 memory addresses")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        input("Press Enter to start control test...")
        controller.test_controls()
    
    elif choice == "2":
        if not controller.sdl2_base_address:
            print("SDL2 base address not found!")
            return
        
        print("SDL2 injection test")
        print("Make sure you've set the correct memory offsets first (option 3)")
        input("Press Enter to test SDL2 injection...")
        controller.test_sdl2_injection()
    
    elif choice == "3":
        if not controller.sdl2_base_address:
            print(f"SDL2 base address not found!")
            return
        
        print(f"SDL2.dll base address: 0x{controller.sdl2_base_address:08X}")
        print("Enter the memory offsets you found with Cheat Engine:")
        
        try:
            kb_offset = input("Keyboard state offset (hex, e.g. 0x1000): ").strip()
            if kb_offset:
                kb_offset = int(kb_offset, 16)
                controller.set_sdl2_addresses(keyboard_offset=kb_offset)
            
            mouse_offset = input("Mouse state offset (hex, e.g. 0x2000): ").strip()
            if mouse_offset:
                mouse_offset = int(mouse_offset, 16)
                controller.set_sdl2_addresses(mouse_offset=mouse_offset)
                
        except ValueError:
            print("Invalid hex format!")
    
    else:
        print("Invalid choice")
        main()

if __name__ == "__main__":
    main()
