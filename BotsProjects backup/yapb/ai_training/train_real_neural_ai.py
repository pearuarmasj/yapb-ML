#!/usr/bin/env python3
"""
YaPB REAL Neural Network Training
Reads actual training data exported from YaPB C++ code and trains neural networks
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import glob
import json

class ZombieNeuralNet(nn.Module):
    def __init__(self, input_size=25, hidden_size=128, output_size=13):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
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
            # Read CSV with explicit data types to handle mixed content
            df = pd.read_csv(file, dtype=str)  # Read everything as string first
            
            # Convert numeric columns manually
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
            
            # Convert to numeric, replacing any non-numeric values with NaN
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values (these would be header rows mixed in)
            original_len = len(df)
            df = df.dropna()
            if len(df) < original_len:
                print(f"Dropped {original_len - len(df)} invalid rows from {file}")
            
            all_data.append(df)
            print(f"Loaded {len(df)} samples from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        return None
        
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Total training samples: {len(combined_data)}")
    return combined_data

def prepare_training_data(df):
    """Prepare data for neural network training"""
    
    # Debug: Check the first few rows to see if header is mixed in
    print("First 3 rows of data:")
    print(df.head(3))
    print("\nColumn names:")
    print(df.columns.tolist())
    
    # Remove any rows that might contain header data (string values in numeric columns)
    print(f"Original data shape: {df.shape}")
    
    # Check for non-numeric data in pos_x column (which should be numeric)
    try:
        df['pos_x'] = pd.to_numeric(df['pos_x'], errors='coerce')
        df = df.dropna(subset=['pos_x'])  # Drop rows where pos_x couldn't be converted
        print(f"After cleaning: {df.shape}")
    except Exception as e:
        print(f"Error cleaning data: {e}")
        
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
      # Normalize input features
    try:
        X = df[state_features].values.astype(np.float32)
        print(f"Successfully converted features to float32, shape: {X.shape}")
    except Exception as e:
        print(f"Error converting features to float: {e}")
        print("Problematic rows:")
        for col in state_features:
            non_numeric = df[~pd.to_numeric(df[col], errors='coerce').notna()]
            if len(non_numeric) > 0:
                print(f"Column {col} has non-numeric values:")
                print(non_numeric[[col]].head())
        raise
    
    # Normalize positions and distances
    X[:, 0:3] /= 4096.0  # positions
    X[:, 3:6] /= 320.0   # velocities
    X[:, 8:10] /= 100.0  # health, armor
    X[:, 17] /= 4096.0   # enemy_distance
    X[:, 18] /= 100.0    # enemy_health
    
    # Actions
    y = df[action_features].values.astype(np.float32)
    
    return torch.tensor(X), torch.tensor(y)

def train_neural_network():
    """Train the neural network with real YaPB data"""
    print("🧠 Training YaPB Neural Network with REAL data")
    print("=" * 50)
    
    # Load real training data from YaPB
    df = load_yapb_training_data()
    if df is None:
        return
    
    # Prepare training data
    X, y = prepare_training_data(df)
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize model
    model = ZombieNeuralNet(input_size=X.shape[1], output_size=y.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 500
    batch_size = 64
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 50 == 0:
            # Test performance
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
            
            print(f"Epoch {epoch:3d}: Train Loss = {total_loss/len(X_train)*batch_size:.6f}, "
                  f"Test Loss = {test_loss:.6f}")
    
    # Save trained model
    model_path = "models/yapb_zombie_neural_net.pth"
    Path("models").mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X.shape[1],
        'output_size': y.shape[1],
        'training_samples': len(df)
    }, model_path)
    
    print(f"✅ Model saved to {model_path}")
    
    # Export weights for C++ integration
    export_weights_for_cpp(model, "models/neural_weights.json")
    
    return model

def export_weights_for_cpp(model, output_path):
    """Export trained weights in format that C++ can load"""
    weights_data = {}
    
    for name, param in model.named_parameters():
        weights_data[name] = param.detach().cpu().numpy().tolist()
    
    with open(output_path, 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    print(f"✅ Weights exported for C++ integration: {output_path}")

if __name__ == "__main__":
    train_neural_network()
