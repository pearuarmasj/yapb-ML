#!/usr/bin/env python3
"""
YaPB GPU-Accelerated Neural AI Training Script - Clean Version
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def setup_gpu():
    """Setup GPU for maximum performance"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("GPU DETECTED - Training on CUDA")
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_gb:.1f} GB")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        return device, True
    else:
        print("CUDA not available - falling back to CPU")
        return torch.device('cpu'), False

class ZombieNeuralNetGPU(nn.Module):
    """GPU-optimized neural network"""
    
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

def load_training_data():
    """Load all available training data"""
    data_files = glob.glob("data/training_data_*.csv")
    
    if not data_files:
        print("ERROR: No training data files found!")
        return None
    
    print(f"Found {len(data_files)} training data files")
    
    all_data = []
    total_samples = 0
    
    for i, file in enumerate(data_files):
        print(f"Loading {file}... ({i+1}/{len(data_files)})")
        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                all_data.append(df)
                total_samples += len(df)
                print(f"  {len(df)} samples loaded")
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    if not all_data:
        print("No valid data found!")
        return None
    
    print(f"Combining {len(all_data)} datasets...")
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {total_samples:,}")
    print(f"Unique bots: {combined_data['bot_name'].nunique()}")
    
    return combined_data

def prepare_data(df, device):
    """Prepare data for training"""
    print("Preparing data for training...")
    
    # Check what columns we actually have
    print("Available columns:", list(df.columns))
    
    # Use the actual column names from the CSV
    feature_columns = [
        'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
        'view_yaw', 'view_pitch', 'health', 'armor',
        'is_stuck', 'is_on_ladder', 'is_in_water', 'has_enemy',
        'enemy_pos_x', 'enemy_pos_y', 'enemy_pos_z', 'enemy_distance',
        'enemy_health', 'enemy_visible', 'time_since_enemy_seen',
        'team_id', 'teammates_nearby', 'enemies_nearby', 'current_task'
    ]
    
    target_columns = [
        'move_forward', 'move_right', 'turn_yaw', 'turn_pitch',
        'attack_primary', 'attack_secondary', 'task_switch',
        'jump', 'duck', 'walk', 'confidence'
    ]
      # Only use columns that actually exist
    available_features = [col for col in feature_columns if col in df.columns]
    available_targets = [col for col in target_columns if col in df.columns]
    
    print(f"Using {len(available_features)} feature columns")
    print(f"Using {len(available_targets)} target columns")
    
    if len(available_features) == 0 or len(available_targets) == 0:
        print("ERROR: No valid columns found!")
        return None
    
    # Convert columns to numeric, replacing non-numeric values with NaN
    for col in available_features + available_targets:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with any NaN values
    df_clean = df.dropna(subset=available_features + available_targets)
    print(f"Cleaned data: {len(df_clean)} samples (from {len(df)})")
    
    if len(df_clean) == 0:
        print("ERROR: No valid numeric data found!")
        return None
    
    X = df_clean[available_features].values.astype(np.float32)
    y = df_clean[available_targets].values.astype(np.float32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)  
    y_train = torch.FloatTensor(y_train).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    print(f"Data prepared!")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    
    return X_train, X_test, y_train, y_test, scaler

def train_network(X_train, X_test, y_train, y_test, device):
    """Train the neural network with proper monitoring"""
    
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    model = ZombieNeuralNetGPU(input_size=input_size, output_size=output_size).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Neural network created")
    print(f"Total parameters: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # Fix batch size to be reasonable for dataset size
    data_size = len(X_train)
    batch_size = min(64, data_size // 4)  # Use smaller batches for better training
    epochs = 1000  # More epochs for better training
    
    print(f"Training starting...")
    print(f"Data size: {data_size}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {data_size // batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 50)
    
    best_loss = float('inf')
    no_improve_count = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        epoch_start = time.time()
        
        # Shuffle data each epoch
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, len(X_train_shuffled), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
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
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / batch_count
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/best_gpu_model.pth')
            no_improve_count = 0
            print(f"NEW BEST! Epoch {epoch} | Val Loss: {val_loss:.6f} | Saved model")
        else:
            no_improve_count += 1
        
        # Detailed progress reporting
        if epoch % 10 == 0 or epoch < 20:
            print(f"Epoch {epoch:4d}/{epochs} | "
                  f"Train: {avg_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.2f}s")
            
            # GPU memory info
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f"         GPU Memory: {memory_used:.2f} GB | No improve: {no_improve_count}")
        
        # Early stopping if no improvement
        if no_improve_count > 100:
            print(f"Early stopping at epoch {epoch} - no improvement for 100 epochs")
            break
        
        # Auto-export weights every 100 epochs
        if epoch > 0 and epoch % 100 == 0:
            export_weights(model, f'models/neural_weights_epoch_{epoch}.json')
            print(f"Exported weights at epoch {epoch}")
      print(f"Training completed!")
    print(f"Best validation loss: {best_loss:.6f}")
    
    return model

def export_weights(model, filename='models/neural_weights_gpu.json'):
    """Export weights for C++"""
    
    print(f"Exporting weights to {filename}...")
    
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
            'architecture': '4-layer GPU-optimized network',
            'input_size': model_cpu.fc1.in_features,
            'output_size': model_cpu.fc4.out_features,
            'device_trained': 'CUDA' if torch.cuda.is_available() else 'CPU',
            'export_time': datetime.now().isoformat()
        }
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    print(f"Weights exported successfully!")

def main():
    """Main training pipeline"""
    print("YaPB GPU Neural Training")
    print("=" * 40)
    
    device, gpu_available = setup_gpu()
    
    df = load_training_data()
    if df is None:
        return
    
    result = prepare_data(df, device)
    if result is None:
        return
    
    X_train, X_test, y_train, y_test, scaler = result
    
    model = train_network(X_train, X_test, y_train, y_test, device)
    
    export_weights(model)
    
    print("\nTraining complete!")
    print("Copy neural_weights_gpu.json to your CS 1.6 addon directory")

if __name__ == "__main__":
    main()
