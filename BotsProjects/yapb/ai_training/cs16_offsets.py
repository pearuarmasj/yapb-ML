#!/usr/bin/env python3
"""
CS 1.6 Memory Offsets and Manager
Pre-configured offsets for CS 1.6 build 6153 (most common version)
"""

import psutil
import ctypes
from ctypes import wintypes
import numpy as np
from typing import Optional, Dict, Tuple
import win32gui
import win32ui
import win32con
from PIL import Image

class CS16Offsets:
    """Memory offsets for CS 1.6 game state reading"""
    
    # Base addresses (need to be found dynamically)
    CLIENT_DLL_BASE = 0x01D00000  # Typical client.dll base
    
    # Static offsets (relative to client.dll)
    LOCAL_PLAYER_PTR = 0x00A3ACD4  # Pointer to local player entity
    ENTITY_LIST = 0x00A73FA0       # Entity list base  
    MAX_ENTITIES = 0x00A73E30      # Maximum entities
    VIEW_MATRIX = 0x00A3B09C       # View matrix for world-to-screen
    
    # Player entity offsets (relative to player entity base)
    PLAYER_ORIGIN_X = 0x04    # X coordinate
    PLAYER_ORIGIN_Y = 0x08    # Y coordinate
    PLAYER_ORIGIN_Z = 0x0C    # Z coordinate
    
    PLAYER_VELOCITY_X = 0x20  # X velocity
    PLAYER_VELOCITY_Y = 0x24  # Y velocity
    PLAYER_VELOCITY_Z = 0x28  # Z velocity
    
    PLAYER_ANGLES_X = 0x0C    # View angle X (pitch)
    PLAYER_ANGLES_Y = 0x10    # View angle Y (yaw) 
    PLAYER_ANGLES_Z = 0x14    # View angle Z (roll)
    
    PLAYER_HEALTH = 0x8C      # Health (0-100)
    PLAYER_ARMOR = 0x90       # Armor (0-100)
    PLAYER_TEAM = 0x9C        # Team number (1=T, 2=CT)
    PLAYER_CLASS = 0xA0       # Player class
    PLAYER_FLAGS = 0x94       # Player flags (ducking, etc)
    
    # Entity offsets
    ENTITY_SIZE = 0x400       # Size of each entity in bytes
    
    @classmethod
    def get_all_offsets(cls):
        """Return all offsets as dictionary"""
        return {
            "client_dll_base": cls.CLIENT_DLL_BASE,
            "local_player_ptr": cls.LOCAL_PLAYER_PTR,
            "entity_list": cls.ENTITY_LIST,
            "max_entities": cls.MAX_ENTITIES,
            "view_matrix": cls.VIEW_MATRIX,
            "player_origin_x": cls.PLAYER_ORIGIN_X,
            "player_origin_y": cls.PLAYER_ORIGIN_Y,
            "player_origin_z": cls.PLAYER_ORIGIN_Z,
            "player_velocity_x": cls.PLAYER_VELOCITY_X,
            "player_velocity_y": cls.PLAYER_VELOCITY_Y,
            "player_velocity_z": cls.PLAYER_VELOCITY_Z,
            "player_angles_x": cls.PLAYER_ANGLES_X,
            "player_angles_y": cls.PLAYER_ANGLES_Y,
            "player_angles_z": cls.PLAYER_ANGLES_Z,
            "player_health": cls.PLAYER_HEALTH,
            "player_armor": cls.PLAYER_ARMOR,
            "player_team": cls.PLAYER_TEAM,
            "player_class": cls.PLAYER_CLASS,
            "player_flags": cls.PLAYER_FLAGS,
            "entity_size": cls.ENTITY_SIZE,
        }

# Alternative offsets for different CS 1.6 versions
class CS16OffsetsAlt:
    """Alternative offsets for different CS 1.6 builds"""
    
    # Build 4554 offsets (older version)
    BUILD_4554 = {
        "local_player_ptr": 0x009E35D4,
        "entity_list": 0x00A1C3A0,
        # ... other offsets
    }
    
    # Steam version offsets
    STEAM_VERSION = {
        "local_player_ptr": 0x00B3B2A4,
        "entity_list": 0x00B63F70,
        # ... other offsets  
    }

class CS16OffsetManager:
    """Manager class for CS 1.6 memory reading and game interaction"""
    
    def __init__(self):
        self.process = None
        self.process_handle = None
        self.offsets = CS16Offsets()
        self.hwnd = None
        
    def find_cs16_process(self) -> bool:
        """Find and attach to CS 1.6 process"""
        process_names = ['hl.exe', 'cstrike.exe', 'hlds.exe']
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'].lower() in process_names:
                    self.process = proc
                    # Open process handle for memory reading
                    self.process_handle = ctypes.windll.kernel32.OpenProcess(
                        0x1F0FFF,  # PROCESS_ALL_ACCESS
                        False,
                        proc.info['pid']
                    )
                    print(f"Found CS 1.6 process: {proc.info['name']} (PID: {proc.info['pid']})")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        print("CS 1.6 process not found!")
        return False
    
    def get_cs16_window(self) -> Optional[int]:
        """Get CS 1.6 window handle"""
        if self.hwnd:
            return self.hwnd
            
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if 'counter-strike' in window_text.lower() or 'half-life' in window_text.lower():
                    windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            self.hwnd = windows[0]
            return self.hwnd
        return None
    
    def capture_window_screenshot(self, hwnd: int) -> Optional[np.ndarray]:
        """Capture screenshot of CS 1.6 window"""
        try:
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top
            
            # Create device contexts
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            # Create bitmap
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            
            # Copy window to bitmap
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
            
            # Convert to numpy array
            bmp_info = save_bitmap.GetInfo()
            bmp_str = save_bitmap.GetBitmapBits(True)
            
            # Create PIL image and convert to numpy
            img = Image.frombuffer('RGB', (bmp_info['bmWidth'], bmp_info['bmHeight']), 
                                 bmp_str, 'raw', 'BGRX', 0, 1)
            screenshot = np.array(img)
            
            # Cleanup
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            
            return screenshot
            
        except Exception as e:
            print(f"Screenshot capture error: {e}")
            return None
    
    def read_memory(self, address: int, size: int) -> Optional[bytes]:
        """Read memory from CS 1.6 process"""
        if not self.process_handle:
            return None
            
        try:
            buffer = ctypes.create_string_buffer(size)
            bytes_read = ctypes.c_size_t()
            
            success = ctypes.windll.kernel32.ReadProcessMemory(
                self.process_handle,
                ctypes.c_void_p(address),
                buffer,
                size,
                ctypes.byref(bytes_read)
            )
            
            if success:
                return buffer.raw
            return None
        except Exception as e:
            print(f"Memory read error: {e}")
            return None
    
    def read_float(self, address: int) -> Optional[float]:
        """Read float value from memory"""
        data = self.read_memory(address, 4)
        if data:
            return ctypes.c_float.from_buffer(ctypes.create_string_buffer(data)).value
        return None
    
    def read_int(self, address: int) -> Optional[int]:
        """Read integer value from memory"""
        data = self.read_memory(address, 4)
        if data:
            return ctypes.c_int.from_buffer(ctypes.create_string_buffer(data)).value
        return None
    
    def read_game_state(self) -> Dict:
        """Read current game state from memory"""
        if not self.process_handle:
            if not self.find_cs16_process():
                return {}
        
        game_state = {}
        
        try:
            # Read local player pointer
            local_player_ptr = self.read_int(self.offsets.CLIENT_DLL_BASE + self.offsets.LOCAL_PLAYER_PTR)
            if not local_player_ptr:
                return {}
            
            # Read player position
            pos_x = self.read_float(local_player_ptr + self.offsets.PLAYER_ORIGIN_X)
            pos_y = self.read_float(local_player_ptr + self.offsets.PLAYER_ORIGIN_Y)
            pos_z = self.read_float(local_player_ptr + self.offsets.PLAYER_ORIGIN_Z)
            
            if pos_x is not None and pos_y is not None and pos_z is not None:
                game_state['position'] = [pos_x, pos_y, pos_z]
            
            # Read player velocity
            vel_x = self.read_float(local_player_ptr + self.offsets.PLAYER_VELOCITY_X)
            vel_y = self.read_float(local_player_ptr + self.offsets.PLAYER_VELOCITY_Y)
            vel_z = self.read_float(local_player_ptr + self.offsets.PLAYER_VELOCITY_Z)
            
            if vel_x is not None and vel_y is not None and vel_z is not None:
                game_state['velocity'] = [vel_x, vel_y, vel_z]
            
            # Read player angles
            angle_x = self.read_float(local_player_ptr + self.offsets.PLAYER_ANGLES_X)
            angle_y = self.read_float(local_player_ptr + self.offsets.PLAYER_ANGLES_Y)
            angle_z = self.read_float(local_player_ptr + self.offsets.PLAYER_ANGLES_Z)
            
            if angle_x is not None and angle_y is not None and angle_z is not None:
                game_state['angles'] = [angle_x, angle_y, angle_z]
            
            # Read health and armor
            health = self.read_int(local_player_ptr + self.offsets.PLAYER_HEALTH)
            armor = self.read_int(local_player_ptr + self.offsets.PLAYER_ARMOR)
            
            if health is not None:
                game_state['health'] = health
            if armor is not None:
                game_state['armor'] = armor
            
            # Read team and flags
            team = self.read_int(local_player_ptr + self.offsets.PLAYER_TEAM)
            flags = self.read_int(local_player_ptr + self.offsets.PLAYER_FLAGS)
            
            if team is not None:
                game_state['team'] = team
            if flags is not None:
                game_state['flags'] = flags
                
        except Exception as e:
            print(f"Game state read error: {e}")
        
        return game_state
    
    def cleanup(self):
        """Cleanup resources"""
        if self.process_handle:
            ctypes.windll.kernel32.CloseHandle(self.process_handle)
            self.process_handle = None

if __name__ == "__main__":
    print("CS 1.6 Memory Offsets")
    print("="*30)
    
    offsets = CS16Offsets.get_all_offsets()
    for name, value in offsets.items():
        print(f"{name:20} = 0x{value:08X}")
