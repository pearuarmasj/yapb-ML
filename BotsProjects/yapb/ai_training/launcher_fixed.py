#!/usr/bin/env python3
"""
🚀 CS 1.6 ML Bot Launcher - de_survivor Edition
Easy way to start learning de_survivor map!
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🎯 CS 1.6 de_survivor ML BOT SYSTEM")
    print("=" * 50)
    print("🗺️  TARGET MAP: de_survivor")
    print("🎮 GOAL: Learn to control YOUR player")
    print("🧠 METHOD: Reinforcement Learning")
    print()
    
    print("📋 SETUP CHECKLIST:")
    print("✅ Counter-Strike 1.6 is running")
    print("✅ Loaded de_survivor map")
    print("✅ You can move around normally")
    print("✅ CS 1.6 is the active window")
    print()
    
    print("🎯 WHAT DO YOU WANT TO DO:")
    print("1. 🧪 Test System (recommended first)")
    print("2. 🚀 Start Learning de_survivor")
    print("3. 🎮 Demo Bot Control")
    print("4. 🔍 Find Memory Offsets")
    print("5. 📖 View Documentation")
    print("6. ❌ Exit")
    print()
    
    choice = input("Choose option (1-6): ").strip()
    
    if choice == "1":
        print("\n🧪 Running System Tests...")
        print("This will verify everything works before training.")
        subprocess.run([sys.executable, "test_ml_system.py"])
    
    elif choice == "2":
        print("\n🚀 Starting de_survivor Learning...")
        print("The AI will now control your player to learn the map!")
        subprocess.run([sys.executable, "train_ml_bot.py"])
    
    elif choice == "3":
        print("\n🎮 Starting Bot Control Demo...")
        print("Watch the bot move your character around!")
        subprocess.run([sys.executable, "demo_bot_control.py"])
    
    elif choice == "4":
        print("\n🔍 Starting Offset Finder...")
        print("This helps find memory addresses for different CS versions.")
        subprocess.run([sys.executable, "offset_finder.py"])
    
    elif choice == "5":
        print("\n📖 Opening Documentation...")
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
    
    elif choice == "6":
        print("\n👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    print("🎮 Starting CS 1.6 ML Bot Launcher...")
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
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to exit...")
