import interception.inputs as inputs
import time
import ctypes
from ctypes import wintypes
import pymem
import pymem.process
import pymem.exception

inputs.auto_capture_devices = True

print("Wait 3 seconds before starting the test...")
time.sleep(3)

def disable_mouse_acceleration():
    user32 = ctypes.windll.user32
    original_params = (ctypes.c_int * 3)()
    user32.SystemParametersInfoW(4, 0, ctypes.byref(original_params), 0)
    
    new_params = (ctypes.c_int * 3)(0, 0, 0)
    user32.SystemParametersInfoW(4, 0, ctypes.byref(new_params), 2)
    
    return original_params

def restore_mouse_acceleration(original_params):
    user32 = ctypes.windll.user32
    user32.SystemParametersInfoW(4, 0, ctypes.byref(original_params), 2)

def get_ac_process():
    try:
        pm = pymem.Pymem("ac_client.exe")
        return pm
    except pymem.exception.PymemError:
        print("AssaultCube process not found")
        return None

def test_wasd_movement():
    pm = get_ac_process()
    if not pm:
        print("AssaultCube not running, using global input")
    
    keys = ['w', 'a', 's', 'd']
    
    for key in keys:
        print(f"Pressing {key.upper()}")
        inputs.key_down(key)
        time.sleep(0.1)
        inputs.key_up(key)
        time.sleep(0.5)

def test_mouse_look():
    pm = get_ac_process()
    if not pm:
        print("AssaultCube not running, using global input")
    
    movements = [
        ("up", 0, -50),
        ("down", 0, 50),
        ("left", -50, 0),
        ("right", 50, 0)
    ]
    
    for direction, x, y in movements:
        print(f"Mouse look {direction}")
        inputs.move_relative(x, y)
        time.sleep(1)

if __name__ == "__main__":
    original_params = disable_mouse_acceleration()
    try:
        print("Testing WASD movement...")
        test_wasd_movement()
        
        print("\nTesting mouse look...")
        test_mouse_look()
        
        print("Test complete.")
    finally:
        restore_mouse_acceleration(original_params)

