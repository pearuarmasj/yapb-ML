#!/usr/bin/env python3
"""
YaPB Real Neural AI Training Script
Trains a real neural network on actual gameplay data collected from YaPB bots
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import glob
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class ZombieNeuralNet(nn.Module):
    """Neural network for zombie bot decision making"""
    
    def __init__(self, input_size=25, hidden_size1=64, hidden_size2=32, output_size=11):
        super(ZombieNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

def load_yapb_training_data():
    """Load real training data exported from YaPB"""
    data_files = glob.glob("data/training_data_*.csv")
    
    if not data_files:
        print("ERROR: No training data files found!")
        print("Make sure to:")
        print("1. Enable data collection: neural_data_collection 1")
        print("2. Play with zombie bots to generate data")
        return None
    
    print(f"Found {len(data_files)} training data files")
    
    all_data = []
    for file in data_files:
        try:
            print(f"Loading {file}...")
            
            # Read CSV and handle potential header issues
            df = pd.read_csv(file)
            
            # Check if first row contains header strings instead of numbers
            first_row = df.iloc[0]
            if first_row['pos_x'] == 'pos_x':  # Header got mixed in
                print(f"Removing header row from {file}")
                df = df[1:]  # Skip first row
            
            # Convert all numeric columns
            numeric_cols = [
                'timestamp', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                'view_yaw', 'view_pitch', 'health', 'armor', 'is_stuck', 'is_on_ladder',
                'is_in_water', 'has_enemy', 'enemy_pos_x', 'enemy_pos_y', 'enemy_pos_z',
                'enemy_distance', 'enemy_health', 'enemy_visible', 'time_since_enemy_seen',
                'team_id', 'teammates_nearby', 'enemies_nearby', 'current_task',
                'hunt_range', 'aggression_level', 'target_present', 'move_forward',
                'move_right', 'turn_yaw', 'turn_pitch', 'attack_primary',
                'attack_secondary', 'task_switch', 'jump', 'duck', 'walk',
                'confidence', 'immediate_reward', 'total_reward'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values
            original_len = len(df)
            df = df.dropna()
            if len(df) < original_len:
                print(f"Cleaned {original_len - len(df)} invalid rows from {file}")
            
            all_data.append(df)
            print(f"Loaded {len(df)} valid samples from {file}")
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        return None
        
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Total training samples: {len(combined_data)}")
    return combined_data

def prepare_training_data(df):
    """Prepare data for neural network training"""
    
    print(f"Preparing training data from {len(df)} samples...")
    
    # Input features (state)
    state_features = [
        'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
        'view_yaw', 'view_pitch', 'health', 'armor', 'is_stuck',
        'is_on_ladder', 'is_in_water', 'has_enemy', 'enemy_pos_x',
        'enemy_pos_y', 'enemy_pos_z', 'enemy_distance', 'enemy_health',
        'enemy_visible', 'time_since_enemy_seen', 'team_id',
        'teammates_nearby', 'enemies_nearby', 'current_task'
    ]
    
    # Output features (actions)
    action_features = [
        'move_forward', 'move_right', 'turn_yaw', 'turn_pitch',
        'attack_primary', 'attack_secondary', 'task_switch',
        'jump', 'duck', 'walk', 'confidence'
    ]
    
    # Extract features
    X = df[state_features].values.astype(np.float32)
    y = df[action_features].values.astype(np.float32)
    
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    # Normalize input features
    X[:, 0:3] /= 4096.0  # positions
    X[:, 3:6] /= 320.0   # velocities  
    X[:, 8:10] /= 100.0  # health, armor
    if X.shape[1] > 17:
        X[:, 17] /= 4096.0   # enemy_distance
    if X.shape[1] > 18:
        X[:, 18] /= 100.0    # enemy_health
    
    return X, y

def train_neural_network():
    """Train the neural network on real gameplay data"""
    
    print("🧠 YaPB Real Neural AI Training")
    print("=" * 50)
    
    # Load data
    df = load_yapb_training_data()
    if df is None:
        return False
    
    # Prepare training data
    X, y = prepare_training_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    model = ZombieNeuralNet(input_size=X.shape[1], output_size=y.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nNeural Network Architecture:")
    print(f"Input: {X.shape[1]} features")
    print(f"Hidden: 64 -> 32 neurons")
    print(f"Output: {y.shape[1]} actions")
    
    # Training loop
    print(f"\nTraining for 100 epochs...")
    model.train()
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                print(f"Epoch {epoch+1}/100, Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
            model.train()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        final_loss = criterion(test_outputs, y_test)
        print(f"\nFinal Test Loss: {final_loss.item():.6f}")
    
    # Save PyTorch model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/yapb_zombie_neural_net.pth")
    print("✅ PyTorch model saved to models/yapb_zombie_neural_net.pth")
    
    # Export weights for C++
    export_weights_for_cpp(model)
    
    # Save training log
    save_training_log(df, final_loss.item())
    
    return True

def export_weights_for_cpp(model):
    """Export trained weights to JSON format for C++ loading"""
    
    print("📤 Exporting weights for C++ integration...")
    
    weights_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy().tolist()
    
    with open("models/neural_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
    
    print("✅ Weights exported to models/neural_weights.json")
    
    # Show weight summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params}")
    print(f"📁 File size: {os.path.getsize('models/neural_weights.json') / 1024:.1f} KB")

def save_training_log(df, final_loss):
    """Save training details to log file"""
    
    os.makedirs("logs", exist_ok=True)
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "training_samples": len(df),
        "unique_bots": df['bot_name'].nunique(),
        "final_loss": final_loss,
        "data_timespan": f"{df['timestamp'].min():.1f} - {df['timestamp'].max():.1f}",
        "bot_names": df['bot_name'].unique().tolist()
    }
    
    with open("logs/training_log.txt", "w") as f:
        f.write("YaPB Neural AI Training Log\n")
        f.write("=" * 30 + "\n\n")
        for key, value in log_data.items():
            f.write(f"{key}: {value}\n")
    
    print("✅ Training log saved to logs/training_log.txt")

if __name__ == "__main__":
    success = train_neural_network()
    if success:
        print("\n🎉 Neural network training completed successfully!")
        print("\nNext steps:")
        print("1. Copy neural_weights.json to your C++ project")
        print("2. Implement weight loading in C++")
        print("3. Use trained neural network for bot decisions")
    else:
        print("\n❌ Training failed. Check the error messages above.")
