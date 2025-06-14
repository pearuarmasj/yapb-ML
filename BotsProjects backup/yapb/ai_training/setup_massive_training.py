#!/usr/bin/env python3
"""
Massive Data Collection Setup for YaPB Neural Training
Collects 100x more data with proper reward signals
"""

import os
import time
import shutil
from datetime import datetime

def setup_massive_data_collection():
    """Setup for collecting massive amounts of training data"""
    
    print("MASSIVE DATA COLLECTION SETUP")
    print("=" * 50)
    
    # Game directory paths
    yapb_dir = "F:\\SteamLibrary\\steamapps\\common\\Half-Life\\cstrike\\addons\\yapb"
    data_dir = "c:\\Users\\pearu\\BotsProjects\\yapb\\ai_training\\data"
    
    print("STEP 1: Setup ConVars for massive data collection")
    print("In CS 1.6 console, run these commands:")
    print()
    print("neural_data_collection 1        // Enable data collection")
    print("neural_use_for_decisions 0      // Use regular AI (better data)")
    print("yb_quota 31                     // Max bots for more data")
    print("mp_startmoney 16000             // More equipment variety")
    print("mp_buytime 999                  // Always allow buying")
    print("mp_roundtime 10                 // Longer rounds = more data")
    print()
    
    print("STEP 2: Collect data for at least 30 minutes of gameplay")
    print("- Play multiple maps (de_dust2, cs_office, de_inferno, etc.)")
    print("- Mix terrorist and counter-terrorist rounds")
    print("- Let bots fight each other and against players")
    print("- Target: 50,000+ samples minimum")
    print()
    
    print("STEP 3: Enhanced reward system")
    print("The current reward system is too simple. We need:")
    print("- Kill rewards (+10 points)")
    print("- Damage dealt (+1 per HP damage)")
    print("- Survival time (+0.1 per second alive)")
    print("- Objective completion (+20 for defuse/plant)")
    print("- Team win contribution (+5)")
    print("- Accuracy bonuses (+1 for headshots)")
    print()
    
    print("STEP 4: Data collection monitoring")
    print(f"Current data files in {data_dir}:")
    
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        total_size = 0
        for file in files:
            path = os.path.join(data_dir, file)
            size = os.path.getsize(path)
            total_size += size
            print(f"  {file}: {size:,} bytes")
        print(f"Total data: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    else:
        print("  No data directory found!")
    
    print()
    print("TARGET FOR GOOD TRAINING:")
    print("- 50,000+ samples minimum")
    print("- 500+ MB of CSV data")
    print("- Multiple maps and scenarios")
    print("- Mix of winning and losing situations")

def enhanced_reward_system_code():
    """Show the C++ code needed for better rewards"""
    
    print("\nENHANCED REWARD SYSTEM")
    print("=" * 50)
    
    cpp_code = '''
// Add to neural_zombie_ai.cpp in calculateReward() function:

float calculateReward(Bot* bot, const NeuralGameState& prevState, const NeuralGameState& currentState) {
    float reward = 0.0f;
    
    // KILL REWARDS - Most important!
    if (currentState.enemy_health <= 0 && prevState.enemy_health > 0) {
        reward += 20.0f;  // Big reward for kills
        g_console.print("Bot %s got KILL REWARD: +20", bot->getTeamName());
    }
    
    // DAMAGE REWARDS - Reward for dealing damage
    if (prevState.enemy_health > currentState.enemy_health && currentState.enemy_health > 0) {
        float damage = prevState.enemy_health - currentState.enemy_health;
        reward += damage * 0.5f;  // 0.5 points per HP damage
    }
    
    // SURVIVAL REWARDS - Stay alive longer
    reward += 0.1f;  // Small reward for each frame survived
    
    // HEALTH PRESERVATION - Don't take unnecessary damage
    if (currentState.health < prevState.health) {
        float damage_taken = prevState.health - currentState.health;
        reward -= damage_taken * 0.3f;  // Penalty for taking damage
    }
    
    // MOVEMENT REWARDS - Encourage good positioning
    if (currentState.has_enemy && currentState.enemy_visible) {
        // Reward for engaging visible enemies
        reward += 1.0f;
    }
    
    // ACCURACY REWARDS - Reward for shooting when enemy is in crosshairs
    if (currentState.attack_primary && currentState.has_enemy && currentState.enemy_visible) {
        if (currentState.enemy_distance < 500.0f) {
            reward += 2.0f;  // Reward for shooting at close enemies
        }
    }
    
    // OBJECTIVE REWARDS (for bomb defusal maps)
    // TODO: Add bomb plant/defuse detection
    
    // TEAM COORDINATION - More enemies nearby = higher risk/reward
    if (currentState.enemies_nearby > 1) {
        reward += currentState.enemies_nearby * 0.5f;  // Bonus for engaging multiple enemies
    }
    
    return reward;
}
'''
    
    print(cpp_code)
    print("\nThis will give much better training signals!")

def create_training_pipeline():
    """Create automated training pipeline"""
    
    pipeline_code = '''
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
'''
    
    with open("automated_pipeline.py", "w") as f:
        f.write(pipeline_code)
    
    print("Created automated_pipeline.py")

if __name__ == "__main__":
    setup_massive_data_collection()
    enhanced_reward_system_code()
    create_training_pipeline()
