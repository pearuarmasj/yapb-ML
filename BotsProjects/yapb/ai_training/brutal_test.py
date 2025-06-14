#!/usr/bin/env python3
"""
🔥 BRUTAL REALITY CHECK for CS 1.6 ML Bot
No more bullshit "tests passed" - let's see what ACTUALLY works
"""

import time
import numpy as np
import win32gui
import win32api
import win32con
from working_ml_bot import WorkingCS16Environment

def brutal_keyboard_test():
    """Test if keyboard input actually works"""
    print("🎹 BRUTAL Keyboard Test")
    print("=" * 30)
    print("This will send keys to CS 1.6 for 10 seconds.")
    print("You should see your character move around.")
    print("If nothing happens, keyboard input is BROKEN.")
    print()
    
    # Find CS 1.6 window
    hwnd = win32gui.FindWindow(None, "Counter-Strike")
    if not hwnd:
        hwnd = win32gui.FindWindow(None, "Counter-Strike 1.6")
    
    if not hwnd:
        print("❌ Can't find CS 1.6 window!")
        return False
    
    print(f"✅ Found CS 1.6 window: {hwnd}")
    
    # Make sure CS 1.6 is active
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1)
    
    print("🎮 Sending keyboard commands for 10 seconds...")
    print("Watch your character - it should move!")
    
    key_codes = {
        'W': 0x57,  # Forward
        'A': 0x41,  # Left
        'S': 0x53,  # Backward
        'D': 0x44,  # Right
        'Space': 0x20,  # Jump
    }
    
    actions = [
        ('W', 2, "Moving FORWARD"),
        ('A', 1, "Moving LEFT"),
        ('S', 1, "Moving BACKWARD"),
        ('D', 1, "Moving RIGHT"),
        ('Space', 0.5, "JUMPING"),
        ('W', 2, "Moving FORWARD again"),
    ]
    
    total_moved = False
    
    for key, duration, description in actions:
        print(f"  {description}...")
        
        # Key down
        win32api.keybd_event(key_codes[key], 0, 0, 0)
        time.sleep(duration)
        # Key up
        win32api.keybd_event(key_codes[key], 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.5)
    
    print("\n❓ Did your character move around? (y/n): ", end="")
    response = input().strip().lower()
    
    if response.startswith('y'):
        print("✅ Keyboard input WORKS!")
        return True
    else:
        print("❌ Keyboard input BROKEN!")
        print("Possible fixes:")
        print("- Run as administrator")
        print("- Check if CS 1.6 is actually the active window")
        print("- Try different key bindings")
        return False

def brutal_memory_test():
    """Test if memory reading gives real values"""
    print("\n🧠 BRUTAL Memory Test")
    print("=" * 30)
    print("Testing if we can read REAL position data...")
    
    try:
        env = WorkingCS16Environment(debug=False)
        obs, info = env.reset()
        
        print(f"Initial position: {obs['position']}")
        
        print("\nNow MANUALLY move your character around!")
        print("Move forward, backward, left, right...")
        print("We'll check if the position values change.")
        
        input("Press Enter when you've moved around...")
        
        # Take several readings
        positions = []
        for i in range(5):
            obs, reward, done, truncated, info = env.step(np.array([0, 0, 0, 0, 0, 0, 0]))
            positions.append(obs['position'].copy())
            print(f"Reading {i+1}: {obs['position']}")
            time.sleep(1)
        
        env.close()
        
        # Check if values changed
        unique_positions = len(set(tuple(pos) for pos in positions))
        
        if unique_positions > 1:
            print("✅ Memory reading WORKS! Positions changed.")
            return True
        else:
            print("❌ Memory reading BROKEN! All positions the same.")
            print("All readings were identical - memory offsets are wrong!")
            return False
            
    except Exception as e:
        print(f"❌ Memory test CRASHED: {e}")
        return False

def brutal_mouse_test():
    """Test if mouse movement works"""
    print("\n🖱️  BRUTAL Mouse Test")
    print("=" * 30)
    
    # Find CS 1.6 window
    hwnd = win32gui.FindWindow(None, "Counter-Strike")
    if not hwnd:
        hwnd = win32gui.FindWindow(None, "Counter-Strike 1.6")
    
    if not hwnd:
        print("❌ Can't find CS 1.6 window!")
        return False
    
    # Make sure CS 1.6 is active
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1)
    
    print("🎮 Moving mouse for 5 seconds...")
    print("Your view should turn left, then right, then up, then down.")
    
    movements = [
        (-100, 0, "Turning LEFT"),
        (200, 0, "Turning RIGHT"),
        (0, -50, "Looking UP"),
        (0, 100, "Looking DOWN"),
        (-100, -50, "Back to center-ish"),
    ]
    
    for dx, dy, description in movements:
        print(f"  {description}...")
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)
        time.sleep(1)
    
    print("\n❓ Did your view turn around? (y/n): ", end="")
    response = input().strip().lower()
    
    if response.startswith('y'):
        print("✅ Mouse movement WORKS!")
        return True
    else:
        print("❌ Mouse movement BROKEN!")
        return False

def brutal_overall_test():
    """Overall brutal reality check"""
    print("\n🔥 BRUTAL OVERALL TEST")
    print("=" * 50)
    print("This will test if the AI can actually control your player")
    print("in a way that's not completely useless.")
    print()
    
    try:
        env = WorkingCS16Environment(debug=True)
        obs, info = env.reset()
        
        print("🤖 AI will now try to control your player for 30 seconds...")
        print("If it works, you should see purposeful movement.")
        print("If it's broken, nothing will happen or it'll be random twitches.")
        
        # Simple movement pattern
        actions = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Forward
            np.array([1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),   # Forward + turn right
            np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Strafe left
            np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),  # Turn left
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),   # Forward + jump
        ]
        
        initial_pos = obs['position'].copy()
        
        for i in range(30):  # 30 steps
            action = actions[i % len(actions)]
            obs, reward, done, truncated, info = env.step(action)
            
            if i % 5 == 0:
                print(f"  Step {i}: Pos={obs['position'][:2]}, Reward={reward:.2f}")
            
            time.sleep(1)  # 1 second per action
        
        final_pos = obs['position']
        distance_moved = np.linalg.norm(final_pos - initial_pos)
        
        env.close()
        
        print(f"\n📊 Results:")
        print(f"  Initial position: {initial_pos}")
        print(f"  Final position: {final_pos}")
        print(f"  Distance moved: {distance_moved:.2f}")
        
        if distance_moved > 10:  # Moved more than 10 units
            print("✅ AI control WORKS! Character moved significantly.")
            return True
        else:
            print("❌ AI control BROKEN! Character barely moved.")
            return False
            
    except Exception as e:
        print(f"❌ Overall test CRASHED: {e}")
        return False

def run_brutal_tests():
    """Run all brutal tests"""
    print("🔥 BRUTAL CS 1.6 ML Bot Reality Check")
    print("=" * 60)
    print("No more bullshit. Let's see what ACTUALLY works.")
    print()
    print("REQUIREMENTS:")
    print("- CS 1.6 must be running")
    print("- You must be IN-GAME (not in menus)")
    print("- You should be able to move around manually")
    print("- CS 1.6 should be the active window")
    print()
    
    input("Press Enter when you're ready for the brutal truth...")
    
    results = {}
    
    # Test 1: Keyboard
    results['keyboard'] = brutal_keyboard_test()
    
    # Test 2: Memory  
    results['memory'] = brutal_memory_test()
    
    # Test 3: Mouse
    results['mouse'] = brutal_mouse_test()
    
    # Test 4: Overall
    results['overall'] = brutal_overall_test()
    
    # Brutal summary
    print("\n" + "="*60)
    print("🔥 BRUTAL TRUTH SUMMARY")
    print("="*60)
    
    working_count = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ WORKS" if passed else "❌ BROKEN"
        print(f"{test_name.upper():12} : {status}")
    
    print("="*60)
    
    if working_count == total_tests:
        print("🎉 HOLY SHIT! Everything actually works!")
        print("🚀 You can proceed with training.")
    elif working_count >= 2:
        print(f"😐 {working_count}/{total_tests} tests passed. Might work with fixes.")
        print("🔧 Fix the broken parts before training.")
    else:
        print(f"💀 Only {working_count}/{total_tests} tests passed. This is fucked.")
        print("🔨 Major debugging needed.")
    
    print("\n🎯 What to do next:")
    if not results['memory']:
        print("- Run memory_hunter.py to find correct offsets")
    if not results['keyboard']:
        print("- Check CS 1.6 key bindings and window focus")
    if not results['mouse']:
        print("- Verify mouse settings in CS 1.6")
    if not results['overall']:
        print("- Fix individual components first")

if __name__ == "__main__":
    run_brutal_tests()
