#!/usr/bin/env python3
"""
🚀 CS 1.6 ML Bot Launcher
The easy way to start training your ML bot!
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🤖 CS 1.6 PROPER ML BOT SYSTEM")
    print("=" * 50)
    print("✅ PyTorch + Stable-Baselines3 + Computer Vision")
    print("✅ Memory Reading + Screen Capture")
    print("✅ Real ML Training (Not Amateur BS)")
    print()
    
    print("📋 WHAT YOU NEED:")
    print("1. Counter-Strike 1.6 running")
    print("2. Join de_survivor map")
    print("3. Make sure you can move around")
    print()
    
    print("🎯 TRAINING OPTIONS:")
    print("1. Run full ML training (1M steps)")
    print("2. Run system tests only")
    print("3. Find memory offsets")
    print("4. View README")
    print("5. Exit")
    print()
    
    choice = input("Choose option (1-5): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting ML Training...")
        subprocess.run([sys.executable, "train_ml_bot.py"])
    
    elif choice == "2":
        print("\n🧪 Running System Tests...")
        subprocess.run([sys.executable, "test_ml_system.py"])
    
    elif choice == "3":
        print("\n🔍 Starting Offset Finder...")
        subprocess.run([sys.executable, "offset_finder.py"])
    
    elif choice == "4":
        print("\n📖 Opening README...")
        if os.path.exists("README.md"):
            os.startfile("README.md")
        else:
            print("README.md not found!")
    
    elif choice == "5":
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("\n❌ Invalid choice!")
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to exit...")
