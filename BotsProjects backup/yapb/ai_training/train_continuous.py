#!/usr/bin/env python3
"""
Continuous Neural Network Training for YaPB
Trains continuously and shows real-time performance metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import glob
import json
import os
import time
import keyboard  # pip install keyboard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class ZombieNeuralNetGPU(nn.Module):
    def __init__(self, input_size=25, hidden_size1=128, hidden_size2=64, hidden_size3=32, output_size=11):
        super(ZombieNeuralNetGPU, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.batch_norm1 = nn.BatchNorm1d(hidden_size1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size2)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.fc4(x))
        return x

def load_all_data():
    """Load and combine all training data"""
    data_files = glob.glob("data/training_data_*.csv")
    if not data_files:
        print("No training data found!")
        return None
    
    all_data = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Convert to numeric and clean
    feature_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                   'view_yaw', 'view_pitch', 'health', 'armor',
                   'is_stuck', 'is_on_ladder', 'is_in_water', 'has_enemy',
                   'enemy_pos_x', 'enemy_pos_y', 'enemy_pos_z', 'enemy_distance',
                   'enemy_health', 'enemy_visible', 'time_since_enemy_seen',
                   'team_id', 'teammates_nearby', 'enemies_nearby', 'current_task']
    
    target_cols = ['move_forward', 'move_right', 'turn_yaw', 'turn_pitch',
                  'attack_primary', 'attack_secondary', 'task_switch',
                  'jump', 'duck', 'walk', 'confidence']
    
    # Convert to numeric
    for col in feature_cols + target_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
    
    # Clean data
    combined = combined.dropna(subset=[col for col in feature_cols + target_cols if col in combined.columns])
    
    print(f"Loaded {len(combined)} clean samples")
    return combined, feature_cols, target_cols

def continuous_training():
    """Train continuously with real-time monitoring"""
    print("CONTINUOUS NEURAL NETWORK TRAINING")
    print("Press 'q' to quit, 's' to save weights")
    print("=" * 50)
    
    # Setup GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    result = load_all_data()
    if result is None:
        print("No data available!")
        return
    
    df, feature_cols, target_cols = result
    
    # Prepare data
    available_features = [col for col in feature_cols if col in df.columns]
    available_targets = [col for col in target_cols if col in df.columns]
    
    X = df[available_features].values.astype(np.float32)
    y = df[available_targets].values.astype(np.float32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Features: {len(available_features)}")
    print(f"Targets: {len(available_targets)}")
    
    # Create model
    model = ZombieNeuralNetGPU(input_size=len(available_features), output_size=len(available_targets)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    batch_size = min(64, len(X_train) // 4)
    epoch = 0
    best_loss = float('inf')
    
    print(f"Batch size: {batch_size}")
    print("Starting continuous training...")
    print("=" * 50)
    
    try:
        while True:
            epoch_start = time.time()
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            # Shuffle data
            indices = torch.randperm(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Training batches
            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test).item()
            
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / batch_count
            
            # Update best
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'models/continuous_best.pth')
            
            # Show progress
            print(f"Epoch {epoch:5d} | Train: {avg_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Best: {best_loss:.6f} | Time: {epoch_time:.2f}s", end='')
            
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f" | GPU: {memory_used:.2f}GB")
            else:
                print()
            
            # Auto-save every 100 epochs
            if epoch > 0 and epoch % 100 == 0:
                export_weights(model, scaler, f'models/neural_weights_continuous.json')
                print(f"Auto-saved weights at epoch {epoch}")
            
            epoch += 1
            
            # Check for quit
            if keyboard.is_pressed('q'):
                print("\nQuitting...")
                break
            elif keyboard.is_pressed('s'):
                export_weights(model, scaler, f'models/neural_weights_manual_save.json')
                print(f"\nManually saved weights at epoch {epoch}")
            
    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}")
    
    # Final save
    export_weights(model, scaler, 'models/neural_weights_final.json')
    print(f"Final save completed. Best loss: {best_loss:.6f}")

def export_weights(model, scaler, filename):
    """Export weights for C++"""
    model_cpu = model.cpu()
    
    weights_dict = {
        'fc1.weight': model_cpu.fc1.weight.detach().numpy().tolist(),
        'fc1.bias': model_cpu.fc1.bias.detach().numpy().tolist(),
        'fc2.weight': model_cpu.fc2.weight.detach().numpy().tolist(),
        'fc2.bias': model_cpu.fc2.bias.detach().numpy().tolist(),
        'fc3.weight': model_cpu.fc3.weight.detach().numpy().tolist(),
        'fc3.bias': model_cpu.fc3.bias.detach().numpy().tolist(),
        'fc4.weight': model_cpu.fc4.weight.detach().numpy().tolist(),
        'fc4.bias': model_cpu.fc4.bias.detach().numpy().tolist(),
        'training_info': {
            'input_size': model_cpu.fc1.in_features,
            'output_size': model_cpu.fc4.out_features,
            'device': 'CUDA' if torch.cuda.is_available() else 'CPU',
            'export_time': datetime.now().isoformat()
        }
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    # Move model back to GPU if available
    if torch.cuda.is_available():
        model.cuda()

if __name__ == "__main__":
    continuous_training()
