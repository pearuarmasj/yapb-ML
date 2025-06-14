
#!/usr/bin/env python3
"""
Automated Training Pipeline
1. Collect data for X minutes
2. Train neural network
3. Export weights
4. Repeat
"""

import time
import subprocess
import os
import glob
import shutil

def automated_pipeline():
    """Run continuous data collection + training"""
    
    while True:
        print("PHASE 1: Data Collection (30 minutes)")
        print("Make sure CS 1.6 is running with:")
        print("neural_data_collection 1")
        print("neural_use_for_decisions 0")
        print("yb_quota 31")
        
        # Wait for data collection
        input("Press Enter when you have new data to train on...")
        
        print("PHASE 2: Training Neural Network")
        
        # Check data size
        data_files = glob.glob("data/training_data_*.csv")
        total_samples = 0
        for file in data_files:
            with open(file, 'r') as f:
                total_samples += len(f.readlines()) - 1  # Minus header
        
        print(f"Training on {total_samples:,} samples")
        
        if total_samples < 10000:
            print("WARNING: Less than 10,000 samples - need more data!")
            continue
        
        # Run training
        os.system("python train_continuous.py")
        
        print("PHASE 3: Deploy to Game")
        # Copy weights to game directory
        shutil.copy("models/neural_weights_continuous.json", 
                   "F:/SteamLibrary/steamapps/common/Half-Life/cstrike/addons/yapb/ai_training/models/")
        
        print("New neural network deployed!")
        print("Test with: neural_use_for_decisions 1")
        print()

if __name__ == "__main__":
    automated_pipeline()
