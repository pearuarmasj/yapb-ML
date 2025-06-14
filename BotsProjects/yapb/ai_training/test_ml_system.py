#!/usr/bin/env python3
"""
🧪 CS 1.6 ML System Test
Test if everything is working before full training
"""

import time
import numpy as np
from working_ml_bot import WorkingCS16Environment

def test_basic_connection():
    """Test basic connection to CS 1.6"""
    print("🔌 Testing CS 1.6 Connection...")
    
    try:
        env = WorkingCS16Environment(debug=True)
        obs, info = env.reset()
        
        print("✅ Connection successful!")
        print(f"📍 Position: {obs['position']}")
        print(f"❤️  Health: {obs['health'][0]}")
        print(f"🖼️  Screen captured: {obs['screen'].shape}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_movement_actions():
    """Test if movement actions work"""
    print("\n🏃 Testing Movement Actions...")
    
    try:
        env = WorkingCS16Environment(debug=True)
        obs, info = env.reset()
        
        # Test different actions
        actions = [
            (np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "Move Forward"),
            (np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "Strafe Left"),
            (np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), "Turn Right"),
            (np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), "Jump"),
        ]
        
        for i, (action, description) in enumerate(actions):
            print(f"  Testing: {description}")
            
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"    Reward: {reward:.2f}")
            print(f"    Position: {obs['position'][:2]}")
            
            time.sleep(1)  # Wait between actions
        
        env.close()
        print("✅ Movement test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Movement test failed: {e}")
        return False

def test_memory_reading():
    """Test memory reading capabilities"""
    print("\n🧠 Testing Memory Reading...")
    
    try:
        env = WorkingCS16Environment(debug=True)
        obs, info = env.reset()
        
        # Test for 10 steps to see if values change
        for i in range(10):
            action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Slow forward
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"  Step {i+1}: Pos={obs['position'][:2]}, Health={obs['health'][0]}")
            time.sleep(0.5)
        
        env.close()
        print("✅ Memory reading test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory reading test failed: {e}")
        return False

def test_screen_capture():
    """Test screen capture functionality"""
    print("\n📷 Testing Screen Capture...")
    
    try:
        env = WorkingCS16Environment(image_size=(64, 64), debug=True)
        obs, info = env.reset()
        
        # Capture a few frames
        for i in range(5):
            action = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0])  # Slow turn
            obs, reward, done, truncated, info = env.step(action)
            
            screen = obs['screen']
            print(f"  Frame {i+1}: Shape={screen.shape}, Min={screen.min():.2f}, Max={screen.max():.2f}")
            
            time.sleep(0.5)
        
        env.close()
        print("✅ Screen capture test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Screen capture test failed: {e}")
        return False

def run_all_tests():
    """Run all system tests"""
    print("🧪 CS 1.6 ML System Tests")
    print("="*50)
    print("This will test if your system is ready for training!")
    print()
    print("BEFORE RUNNING:")
    print("✅ Start Counter-Strike 1.6")
    print("✅ Load de_survivor map")
    print("✅ Make sure you can move around")
    print("✅ Make CS 1.6 the active window")
    print()
    
    input("Press Enter when CS 1.6 is ready...")
    
    tests = [
        ("Basic Connection", test_basic_connection),
        ("Movement Actions", test_movement_actions),
        ("Memory Reading", test_memory_reading),
        ("Screen Capture", test_screen_capture),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
        
        time.sleep(2)  # Brief pause between tests
    
    # Summary
    print("\n" + "="*60)
    print("🏁 TEST RESULTS SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 Your system is ready for training!")
        print()
        print("Next steps:")
        print("1. Run: python train_ml_bot.py")
        print("2. Watch your player learn de_survivor!")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("Please check your CS 1.6 setup and try again.")
        print()
        print("Troubleshooting:")
        print("- Make sure CS 1.6 is running")
        print("- Make sure you're on de_survivor map")
        print("- Try running CS 1.6 as administrator")
        print("- Check that you can move around manually")

if __name__ == "__main__":
    run_all_tests()
