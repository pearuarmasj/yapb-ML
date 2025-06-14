#!/usr/bin/env python3
"""
🚀 CS 1.6 ML Bot Launcher - HONEST Edition
Let's see if any of this shit actually works!
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🎯 CS 1.6 de_survivor ML BOT SYSTEM")
    print("=" * 50)
    print("🗺️  TARGET MAP: de_survivor (when we get there)")
    print("🎮 GOAL: Control YOUR player (maybe)")
    print("🧠 METHOD: Machine Learning (hopefully)")
    print()
    
    print("📋 CURRENT STATUS:")
    print("⚠️  This is EXPERIMENTAL and untested")
    print("⚠️  We don't know if it works yet")
    print("⚠️  Probably will need debugging")
    print("⚠️  May crash spectacularly")
    print()
    
    print("🎯 WHAT WE'RE TRYING TO DO:")
    print("- Get CS 1.6 running with de_survivor")
    print("- Test if we can read game memory")
    print("- Test if we can send keyboard input")
    print("- See if the AI can control your player")
    print("- Make it learn the fucking map")
    print()
    
    print("🎯 WHAT DO YOU WANT TO TRY:")
    print("1. 🔥 BRUTAL reality check (tells the truth)")
    print("2. 🧪 Bullshit test that lies about success")
    print("3. 🎮 Try basic bot control demo")
    print("4. 🚀 Attempt de_survivor learning (very risky)")
    print("5. 🔍 Hunt for correct memory offsets")
    print("6. 📖 Read documentation")
    print("7. ❌ Give up and exit")
    print()
    
    choice = input("Choose option (1-7): ").strip()
    
    if choice == "1":
        print("\n🔥 Running BRUTAL reality check...")
        print("This will tell you the actual fucking truth!")
        subprocess.run([sys.executable, "brutal_test.py"])
    
    elif choice == "2":
        print("\n🧪 Running bullshit test...")
        print("This will lie and say everything works!")
        subprocess.run([sys.executable, "test_ml_system.py"])
    
    elif choice == "3":
        print("\n🎮 Trying basic bot control...")
        print("If this works, we'll be impressed!")
        subprocess.run([sys.executable, "demo_bot_control.py"])
    
    elif choice == "4":
        print("\n🚀 Attempting de_survivor learning...")
        print("This is definitely going to break something!")
        print("Make sure CS 1.6 is running first, or this will crash hard.")
        subprocess.run([sys.executable, "train_ml_bot.py"])
    
    elif choice == "5":
        print("\n🔍 Hunting for correct memory offsets...")
        print("This helps find memory addresses for your CS version.")
        subprocess.run([sys.executable, "memory_hunter.py"])
    
    elif choice == "6":
        print("\n📖 Opening documentation...")
        docs = [
            "README.md",
            "NEURAL_AI_COMPLETE_GUIDE.md",
            "../README.md"
        ]
        
        for doc in docs:
            if os.path.exists(doc):
                try:
                    if sys.platform.startswith('win'):
                        os.startfile(doc)
                    else:
                        subprocess.run(['xdg-open', doc])
                    break
                except:
                    print(f"Found {doc} but couldn't open it automatically")
        else:
            print("No documentation found in current directory")
    
    elif choice == "7":
        print("\n👋 Giving up! Probably wise.")
        return
    
    else:
        print("❌ Invalid choice. Try again, genius.")
        main()

if __name__ == "__main__":
    print("🎮 Starting CS 1.6 ML Bot Launcher...")
    print("🤞 Fingers crossed this doesn't explode...")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("working_ml_bot.py"):
        print("⚠️  Warning: Can't find working_ml_bot.py")
        print("Make sure you're running this from the ai_training directory!")
        print()
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Smart move!")
    except Exception as e:
        print(f"\n💥 Crashed as expected: {e}")
        input("Press Enter to accept defeat...")
