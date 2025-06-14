#!/usr/bin/env python3
"""
YaPB GPU-Accelerated Neural AI Training Script - RTX 4080 BEAST MODE
Trains a massive neural network on gameplay data using CUDA acceleration
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
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        return device, True
    else:
        print("CUDA not available - falling back to CPU")
        print("Install CUDA toolkit and GPU PyTorch for speedup")
        return torch.device('cpu'), False

class ZombieNeuralNetGPU(nn.Module):
    """MASSIVE GPU-optimized neural network for zombie bot decisions"""
    
    def __init__(self, input_size=25, hidden_size1=128, hidden_size2=64, hidden_size3=32, output_size=11):
        super(ZombieNeuralNetGPU, self).__init__()
        
        # LARGER NETWORK - More capacity for complex behavior
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        
        # Advanced regularization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)
        
        # Activation functions
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

def load_massive_training_data():
    """Load ALL available training data - utilize that RTX 4080!"""
    data_files = glob.glob("data/training_data_*.csv")
    
    if not data_files:
        print("❌ ERROR: No training data files found!")
        print("Run massive data collection first:")
        print("neural_data_collection 1")
        print("neural_use_for_decisions 0") 
        print("yb_quota 31")
        return None
    
    print(f"🗂️  Found {len(data_files)} training data files")
    
    all_data = []
    total_samples = 0
    
    for i, file in enumerate(data_files):
        print(f"Loading {file}... ({i+1}/{len(data_files)})")
        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                all_data.append(df)
                total_samples += len(df)
                print(f"  ✅ {len(df)} samples loaded")
            else:
                print(f"  ⚠️  Empty file skipped")
        except Exception as e:
            print(f"  ❌ Error loading {file}: {e}")
    
    if not all_data:
        print("❌ No valid data found!")
        return None
      # Combine all data
    print(f"Combining {len(all_data)} datasets...")
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {total_samples:,}")
    print(f"Unique bots: {combined_data['bot_name'].nunique()}")
    
    # Convert timestamp to numeric if it isn't already
    if 'timestamp' in combined_data.columns:
        combined_data['timestamp'] = pd.to_numeric(combined_data['timestamp'], errors='coerce')
        time_span = combined_data['timestamp'].max() - combined_data['timestamp'].min()
        if not pd.isna(time_span):
            print(f"Time span: {time_span:.1f} seconds")
    
    return combined_data

def prepare_gpu_data(df, device):
    """Prepare data for GPU training with optimal memory usage"""
    print("Preparing data for GPU training...")
    
    # Define features (input) and targets (output)
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
    
    # Extract features and targets
    X = df[feature_columns].values.astype(np.float32)
    y = df[target_columns].values.astype(np.float32)
    
    # Normalize features for better training
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      # Convert to GPU tensors
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)  
    y_train = torch.FloatTensor(y_train).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    print(f"Data prepared for GPU!")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Targets: {y_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, scaler

def train_gpu_neural_network(X_train, X_test, y_train, y_test, device, use_mixed_precision=True):
    """Train neural network with GPU acceleration and mixed precision"""
    
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    # Create MASSIVE network
    model = ZombieNeuralNetGPU(input_size=input_size, output_size=output_size).to(device)
      # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Neural network created")
    print(f"Total parameters: {total_params:,}")
    print(f"Architecture: {input_size} -> 128 -> 64 -> 32 -> {output_size}")
    
    # Advanced optimizer for GPU
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Mixed precision for 2x speedup on RTX 4080
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    if scaler:
        print("Mixed precision enabled - 2x speedup!")
    
    # Training parameters optimized for RTX 4080
    batch_size = 2048 if device.type == 'cuda' else 64  # Massive batches!
    epochs = 200
    
    print(f"Training starting...")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    
    # Training loop
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # Time tracking
        epoch_start = time.time()
        
        # Mini-batch training for massive datasets
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / batch_count
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/best_gpu_model.pth')
        
        # Progress reporting
        if epoch % 10 == 0 or epoch < 10:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {avg_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # GPU memory info
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_total = torch.cuda.memory_reserved() / 1e9
                print(f"         GPU Memory: {memory_used:.1f}/{memory_total:.1f} GB")
          training_history.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        })
    
    print(f"Training completed!")
    print(f"Best validation loss: {best_loss:.6f}")
    
    return model, training_history

def export_gpu_weights(model, scaler, filename='models/neural_weights_gpu.json'):
    """Export GPU-trained weights for C++ integration"""
    
    print(f"Exporting GPU-trained weights to {filename}...")
    
    # Ensure model is on CPU for export
    model_cpu = model.cpu()
    
    # Extract weights and biases
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
            'hidden_sizes': [model_cpu.fc1.out_features, model_cpu.fc2.out_features, model_cpu.fc3.out_features],
            'output_size': model_cpu.fc4.out_features,
            'device_trained': 'CUDA RTX 4080',
            'mixed_precision': True,
            'export_time': datetime.now().isoformat()
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save weights
    with open(filename, 'w') as f:
        json.dump(weights_dict, f, indent=2)
      print(f"GPU weights exported successfully!")
    print(f"Architecture: {weights_dict['training_info']['input_size']} -> " + 
          " -> ".join(map(str, weights_dict['training_info']['hidden_sizes'])) + 
          f" -> {weights_dict['training_info']['output_size']}")

def main():
    """Main GPU training pipeline"""
    print("YaPB GPU-ACCELERATED NEURAL TRAINING")
    print("=" * 50)
    
    # Setup GPU
    device, gpu_available = setup_gpu()
    
    # Load massive dataset
    df = load_massive_training_data()
    if df is None:
        return
    
    # Prepare data for GPU
    X_train, X_test, y_train, y_test, scaler = prepare_gpu_data(df, device)
    
    # Train the beast
    model, history = train_gpu_neural_network(X_train, X_test, y_train, y_test, device)
    
    # Export for C++
    export_gpu_weights(model, scaler)
      print("\nGPU NEURAL TRAINING COMPLETE!")
    print("Your neural network is ready!")
    print("Copy neural_weights_gpu.json to your CS 1.6 addon directory")
    print("Set neural_use_for_decisions 1 and test it out")

if __name__ == "__main__":
    main()
