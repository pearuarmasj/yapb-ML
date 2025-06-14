#!/usr/bin/env python3
"""
CS 1.6 Bot Controller Demo
Shows the bot actually controlling the game!
"""

import time
import numpy as np
from working_ml_bot import WorkingCS16Environment

def demo_bot_control():
    """Demonstrate bot actually controlling CS 1.6"""
    print("🎮 CS 1.6 Bot Controller Demo")
    print("=" * 40)
    print("This will make the bot actually move in CS 1.6!")
    print("Make sure CS 1.6 is the active window.")
    print()
    
    input("Press Enter when CS 1.6 is active and ready...")
    
    # Create environment
    env = WorkingCS16Environment(image_size=(64, 64))
    obs, info = env.reset()
    
    print("🤖 Bot is now controlling CS 1.6!")
    print("Watch your character move!")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        for step in range(100):  # Run for 100 steps
            # Create some demo actions
            if step < 20:
                # Move forward
                action = np.array([0.8, 0, 0, 0, 0, 0, 0], dtype=np.float32)
                print(f"Step {step+1}: Moving FORWARD")
            elif step < 40:
                # Turn right and move
                action = np.array([0.5, 0, 0.5, 0, 0, 0, 0], dtype=np.float32)
                print(f"Step {step+1}: Moving FORWARD + turning RIGHT")
            elif step < 60:
                # Strafe left
                action = np.array([0, -0.8, 0, 0, 0, 0, 0], dtype=np.float32)
                print(f"Step {step+1}: Strafing LEFT")
            elif step < 80:
                # Jump and move
                action = np.array([0.6, 0, 0, 0, 1, 0, 0], dtype=np.float32)
                print(f"Step {step+1}: JUMPING forward")
            else:
                # Duck and turn
                action = np.array([0, 0, -0.3, 0, 0, 1, 0], dtype=np.float32)
                print(f"Step {step+1}: DUCKING and turning LEFT")
            
            # Execute the action
            obs, reward, done, truncated, info = env.step(action)
            
            # Show some feedback
            print(f"  Reward: {reward:.2f}, Health: {obs['health'][0]:.1f}")
            
            # Pause between actions
            time.sleep(0.5)
            
            if done:
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    
    finally:
        env.close()
        print("🎉 Demo completed!")
        print()
        print("What you just saw:")
        print("✅ Bot actually moving your character")
        print("✅ Real keyboard inputs sent to CS 1.6")
        print("✅ Mouse movement for turning")
        print("✅ Jump and duck actions")
        print()
        print("Now imagine this trained by ML to play intelligently!")

def test_individual_actions():
    """Test individual actions to see what each does"""
    print("\n🔧 Testing Individual Actions")
    print("=" * 30)
    
    env = WorkingCS16Environment()
    obs, info = env.reset()
    
    actions = [
        ([1, 0, 0, 0, 0, 0, 0], "Move FORWARD"),
        ([-1, 0, 0, 0, 0, 0, 0], "Move BACKWARD"), 
        ([0, 1, 0, 0, 0, 0, 0], "Strafe RIGHT"),
        ([0, -1, 0, 0, 0, 0, 0], "Strafe LEFT"),
        ([0, 0, 1, 0, 0, 0, 0], "Turn RIGHT"),
        ([0, 0, -1, 0, 0, 0, 0], "Turn LEFT"),
        ([0, 0, 0, 0, 1, 0, 0], "JUMP"),
        ([0, 0, 0, 0, 0, 1, 0], "DUCK"),
        ([0, 0, 0, 0, 0, 0, 1], "SHOOT"),
    ]
    
    print("Testing each action for 2 seconds...")
    input("Make sure CS 1.6 is active, then press Enter...")
    
    for action_values, description in actions:
        print(f"\n🎯 Testing: {description}")
        action = np.array(action_values, dtype=np.float32)
        
        for i in range(4):  # 4 steps = ~2 seconds
            obs, reward, done, truncated, info = env.step(action)
            time.sleep(0.5)
        
        print("  ✅ Action completed")
        time.sleep(1)  # Pause between different actions
    
    env.close()
    print("\n🎉 All actions tested!")

def main():
    """Main demo"""
    print("🤖 CS 1.6 ML Bot - ACTUAL GAME CONTROL")
    print("=" * 50)
    print("This will show you the bot actually controlling CS 1.6!")
    print()
    print("Choose demo:")
    print("1. Full movement demo")
    print("2. Test individual actions")
    print("3. Exit")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        demo_bot_control()
    elif choice == "2":
        test_individual_actions()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")
        main()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")
