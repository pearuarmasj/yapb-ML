import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import cv2
from collections import deque
import csv
import matplotlib.pyplot as plt
from datetime import datetime

def preprocess(frame, size=(84, 84)):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, size)
    return frame.astype(np.uint8)

class CNNPolicy(nn.Module):
    def __init__(self, num_actions=12):
        super(CNNPolicy, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out_size((4, 84, 84))
        
        self.value_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )
        
    def _get_conv_out_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))
    
    def forward(self, x):
        x = x.float() / 255.0
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        value = self.value_head(conv_out)
        policy = self.policy_head(conv_out)
        
        return policy, value

def load_training_data(data_dir="training_data"):
    if not os.path.exists(data_dir):
        print("No training data found!")
        return None
    
    all_data = []
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.pkl'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    all_data.extend(data)
    
    if not all_data:
        print("No training data found!")
        return None
    
    print(f"Loaded {len(all_data)} training samples")
    return all_data

def prepare_data(data):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    for sample in data:
        state_frames = list(sample['state'])
        next_state = sample['next_state']
        
        states.append(np.stack(state_frames, axis=0))
        actions.append(sample['action'])
        rewards.append(sample['reward'])
        next_states.append(next_state)
        dones.append(sample['done'])
    
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones, dtype=np.float32)
    
    return states, actions, rewards, next_states, dones

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
    
    returns = advantages + values
    return advantages, returns

def train_offline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = load_training_data()
    if data is None:
        return
    
    states, actions, rewards, next_states, dones = prepare_data(data)
    
    model = CNNPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    batch_size = 64
    num_epochs = 10
    
    dataset = TensorDataset(
        torch.tensor(states),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(rewards),
        torch.tensor(next_states),
        torch.tensor(dones)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"training_log_batch{batch_size}_epochs{num_epochs}_{timestamp}.csv"
    
    training_history = []
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Policy_Loss', 'Value_Loss', 'Total_Loss', 'Batch_Size', 'Learning_Rate'])
    
    print("Starting offline training...")
    
    for epoch in range(num_epochs):
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        num_batches = 0
        
        for batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_rewards = batch_rewards.to(device)
            batch_next_states = batch_next_states.to(device)
            batch_dones = batch_dones.to(device)
            
            policy_probs, values = model(batch_states)
            _, next_values = model(batch_next_states)
            
            values = values.squeeze()
            next_values = next_values.squeeze()
            
            advantages, returns = compute_gae(
                batch_rewards.cpu().numpy(),
                values.detach().cpu().numpy(),
                next_values.detach().cpu().numpy(),
                batch_dones.cpu().numpy()
            )
            
            advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            
            action_probs = policy_probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
            log_probs = torch.log(action_probs + 1e-8)
            
            policy_loss = -(log_probs * advantages).mean()
            value_loss = nn.MSELoss()(values, returns)
            
            total_loss_batch = policy_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_total_loss = total_loss / num_batches
        
        training_history.append({
            'epoch': epoch + 1,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_total_loss
        })
        
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, avg_policy_loss, avg_value_loss, avg_total_loss, batch_size, 3e-4])
        
        print(f"Epoch {epoch+1}/{num_epochs} - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
    
    print(f"Training data saved to: {csv_filename}")
    
    plot_training_curves(training_history, batch_size, num_epochs, timestamp)
    
    print("Saving trained model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'assaultcube_offline_trained_batch{batch_size}_epochs{num_epochs}_{timestamp}.pth')
    
    print("Offline training complete!")

def plot_training_curves(training_history, batch_size, num_epochs, timestamp):
    epochs = [entry['epoch'] for entry in training_history]
    policy_losses = [entry['policy_loss'] for entry in training_history]
    value_losses = [entry['value_loss'] for entry in training_history]
    total_losses = [entry['total_loss'] for entry in training_history]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, policy_losses, 'b-', linewidth=2, label='Policy Loss')
    plt.title(f'Policy Loss\nBatch: {batch_size}, Epochs: {num_epochs}')
    plt.xlabel('Epoch')
    plt.ylabel('Policy Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, value_losses, 'r-', linewidth=2, label='Value Loss')
    plt.title(f'Value Loss\nBatch: {batch_size}, Epochs: {num_epochs}')
    plt.xlabel('Epoch')
    plt.ylabel('Value Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, total_losses, 'g-', linewidth=2, label='Total Loss')
    plt.title(f'Total Loss\nBatch: {batch_size}, Epochs: {num_epochs}')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plot_filename = f"training_curves_batch{batch_size}_epochs{num_epochs}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Training plots saved to: {plot_filename}")

if __name__ == "__main__":
    train_offline()
