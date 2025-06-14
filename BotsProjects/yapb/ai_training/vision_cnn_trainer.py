#!/usr/bin/env python3
"""
CNN Vision-Based Navigation Trainer for CS 1.6 Bot
Focuses on de_survivor map navigation using computer vision
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import time
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

# Custom modules
from cs16_offsets import CS16OffsetManager
from models.state_preprocessor import StatePreprocessor
from game_controller import CS16GameController, ActionSpace

@dataclass
class NavigationConfig:
    """Configuration for navigation training"""
    image_width: int = 160
    image_height: int = 120
    channels: int = 3
    sequence_length: int = 4  # Number of frames to stack
    learning_rate: float = 0.0001
    batch_size: int = 32
    memory_size: int = 50000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 10000
    target_update: int = 1000
    save_interval: int = 5000
    
class NavigationCNN(nn.Module):
    """
    Convolutional Neural Network for navigation in de_survivor
    Takes visual input and predicts movement actions
    """
    
    def __init__(self, config: NavigationConfig):
        super(NavigationCNN, self).__init__()
        self.config = config
        
        # CNN Feature Extractor
        self.conv_layers = nn.Sequential(
            # First conv block - detect basic features
            nn.Conv2d(config.channels * config.sequence_length, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Second conv block - detect edges and textures
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Third conv block - detect complex patterns
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Fourth conv block - high-level features
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.channels * config.sequence_length, 
                                    config.image_height, config.image_width)
            conv_output = self.conv_layers(dummy_input)
            self.flattened_size = conv_output.view(1, -1).size(1)
        
        # Navigation decision layers
        self.navigation_net = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Output: movement commands
            nn.Linear(128, 8)  # forward, back, left, right, turn_left, turn_right, jump, duck
        )
        
        # Value estimation (for advantage calculation)
        self.value_net = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        """Forward pass through the network"""
        batch_size = x.size(0)
        
        # Extract visual features
        conv_features = self.conv_layers(x)
        flattened = conv_features.view(batch_size, -1)
        
        # Navigation decisions
        navigation_output = self.navigation_net(flattened)
        
        # Value estimation
        value_output = self.value_net(flattened)
        
        return navigation_output, value_output

class VisionMemoryBuffer:
    """Experience replay buffer for vision-based training"""
    
    def __init__(self, capacity: int, image_shape: Tuple[int, int, int]):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pre-allocate memory for efficiency
        self.states = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=bool)
        
    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int):
        """Sample a batch of transitions"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            torch.from_numpy(self.states[indices]).float() / 255.0,
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]).float() / 255.0,
            torch.from_numpy(self.dones[indices])
        )
        
    def __len__(self):
        return self.size

class CS16VisionCapture:
    """Real-time screen capture and processing for CS 1.6"""
    
    def __init__(self, config: NavigationConfig):
        self.config = config
        self.offset_manager = CS16OffsetManager()
        self.game_controller = CS16GameController()
        self.frame_buffer = deque(maxlen=config.sequence_length)
        
        # Initialize empty frame buffer
        empty_frame = np.zeros((config.image_height, config.image_width, config.channels), dtype=np.uint8)
        for _ in range(config.sequence_length):
            self.frame_buffer.append(empty_frame)
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture and preprocess a frame from CS 1.6"""
        try:
            # Get game window and capture screen
            hwnd = self.offset_manager.get_cs16_window()
            if not hwnd:
                return None
                
            # Capture screenshot
            screenshot = self.offset_manager.capture_window_screenshot(hwnd)
            if screenshot is None:
                return None
            
            # Resize and normalize
            frame = cv2.resize(screenshot, (self.config.image_width, self.config.image_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add to frame buffer
            self.frame_buffer.append(frame)
            
            # Stack frames for temporal information
            stacked_frames = np.concatenate(list(self.frame_buffer), axis=2)
            
            return stacked_frames
            
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None
    
    def get_game_state(self) -> Dict:
        """Get current game state from memory"""
        try:
            return self.offset_manager.read_game_state()
        except Exception as e:
            print(f"Game state read error: {e}")
            return {}

class NavigationTrainer:
    """Main trainer for vision-based navigation"""
    
    def __init__(self, config: NavigationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = NavigationCNN(config).to(self.device)
        self.target_net = NavigationCNN(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        
        # Experience replay
        image_shape = (config.channels * config.sequence_length, config.image_height, config.image_width)
        self.memory = VisionMemoryBuffer(config.memory_size, image_shape)
        
        # Vision capture
        self.vision_capture = CS16VisionCapture(config)
        
        # Training state
        self.episode = 0
        self.step_count = 0
        self.epsilon = config.epsilon_start
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0
                q_values, _ = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return np.random.randint(0, 8)  # Random action
    
    def update_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - (self.step_count / self.config.epsilon_decay)
        )
    
    def calculate_reward(self, current_state: Dict, previous_state: Dict, action: int) -> float:
        """Calculate reward for navigation training"""
        reward = 0.0
        
        if not current_state or not previous_state:
            return -1.0  # Penalty for invalid state
        
        # Movement reward - encourage exploration
        prev_pos = previous_state.get('position', [0, 0, 0])
        curr_pos = current_state.get('position', [0, 0, 0])
        
        if len(prev_pos) == 3 and len(curr_pos) == 3:
            distance_moved = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
            reward += distance_moved * 0.1  # Small reward for movement
        
        # Health penalty
        health = current_state.get('health', 100)
        if health < 50:
            reward -= 0.5
        if health <= 0:
            reward -= 10.0  # Death penalty
        
        # Survival reward
        reward += 0.01  # Small reward for staying alive
        
        # Navigation-specific rewards
        if action in [0, 1]:  # Forward/backward movement
            reward += 0.05
        if action in [4, 5]:  # Turning
            reward += 0.02
        
        return float(reward)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.config.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values, current_values = self.policy_net(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values, next_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(1)[0].detach()
            target_q_values = rewards + (self.config.epsilon_decay * max_next_q_values * (~dones))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Update target network
        if self.step_count % self.config.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def run_episode(self, max_steps: int = 1000) -> Tuple[float, int]:
        """Run one training episode"""
        episode_reward = 0.0
        episode_steps = 0
        
        # Get initial state
        current_frame = self.vision_capture.capture_frame()
        if current_frame is None:
            return 0.0, 0
        
        current_game_state = self.vision_capture.get_game_state()
        previous_game_state = current_game_state.copy()
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(current_frame)
            
            # Execute action in game
            self.vision_capture.game_controller.execute_discrete_action(action, duration=0.05)
            
            # Wait for action to take effect
            time.sleep(0.05)
            
            # Get next state
            next_frame = self.vision_capture.capture_frame()
            if next_frame is None:
                break
                
            next_game_state = self.vision_capture.get_game_state()
            
            # Calculate reward
            reward = self.calculate_reward(next_game_state, current_game_state, action)
            episode_reward += reward
            
            # Check if episode is done
            done = (next_game_state.get('health', 100) <= 0 or 
                   step >= max_steps - 1)
            
            # Store transition
            self.memory.push(
                current_frame.transpose(2, 0, 1),  # CHW format
                action,
                reward,
                next_frame.transpose(2, 0, 1),
                done
            )
            
            # Training step
            self.train_step()
            
            # Update state
            current_frame = next_frame
            current_game_state = next_game_state
            
            self.step_count += 1
            episode_steps += 1
            
            # Update epsilon
            self.update_epsilon()
            
            if done:
                break
        
        return episode_reward, episode_steps
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'episode': self.episode,
            'step_count': self.step_count,
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode = checkpoint['episode']
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")
    
    def plot_training_progress(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Training loss
        if self.losses:
            axes[1, 0].plot(self.losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Epsilon decay
        episodes = range(len(self.episode_rewards))
        epsilons = [max(self.config.epsilon_end, 
                       self.config.epsilon_start - (ep * 1000 / self.config.epsilon_decay)) 
                   for ep in episodes]
        axes[1, 1].plot(episodes, epsilons)
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig(f'training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()
    
    def train(self, num_episodes: int = 1000):
        """Main training loop"""
        print(f"Starting CNN navigation training for {num_episodes} episodes")
        print(f"Target map: de_survivor")
        print(f"Device: {self.device}")
        
        for episode in range(num_episodes):
            self.episode = episode
            
            # Run episode
            episode_reward, episode_steps = self.run_episode()
            
            # Record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            
            # Progress reporting
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Steps: {episode_steps:4d} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f}")
            
            # Save model periodically
            if episode % self.config.save_interval == 0 and episode > 0:
                self.save_model(f'models/navigation_cnn_episode_{episode}.pth')
                self.plot_training_progress()
        
        # Final save
        self.save_model('models/navigation_cnn_final.pth')
        self.plot_training_progress()
        print("Training completed!")

def main():
    """Main training function"""
    print("CS 1.6 CNN Vision Navigation Trainer")
    print("=====================================")
    print("This will train a CNN to navigate de_survivor using computer vision")
    print()
    
    # Configuration
    config = NavigationConfig(
        image_width=160,
        image_height=120,
        sequence_length=4,
        learning_rate=0.0001,
        batch_size=32,
        memory_size=50000
    )
    
    # Create trainer
    trainer = NavigationTrainer(config)
    
    # Check for existing model
    if os.path.exists('models/navigation_cnn_latest.pth'):
        load_existing = input("Load existing model? (y/n): ").lower() == 'y'
        if load_existing:
            trainer.load_model('models/navigation_cnn_latest.pth')
    
    # Training parameters
    num_episodes = int(input("Number of episodes to train (default 1000): ") or "1000")
    
    print(f"\nStarting training with:")
    print(f"- Episodes: {num_episodes}")
    print(f"- Image size: {config.image_width}x{config.image_height}")
    print(f"- Frame stack: {config.sequence_length}")
    print(f"- Learning rate: {config.learning_rate}")
    print(f"- Batch size: {config.batch_size}")
    print()
    
    # Ensure CS 1.6 is running
    input("Make sure CS 1.6 is running on de_survivor map. Press Enter to start training...")
    
    # Start training
    try:
        trainer.train(num_episodes)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_model('models/navigation_cnn_interrupted.pth')
    except Exception as e:
        print(f"\nTraining error: {e}")
        trainer.save_model('models/navigation_cnn_error.pth')

if __name__ == "__main__":
    main()
