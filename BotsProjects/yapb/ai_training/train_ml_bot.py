#!/usr/bin/env python3
"""
🎯 CS 1.6 de_survivor Map Learning Script
Train AI to control YOUR player and learn the map!
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Tuple

from working_ml_bot import WorkingCS16Environment

class DeSurvivorLearner:
    """
    Simple reinforcement learning agent for de_survivor
    Focuses on map exploration and learning basic navigation
    """
    
    def __init__(self, learning_rate=0.1, epsilon=0.9, epsilon_decay=0.995):
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.05
        
        # Simple Q-table for basic actions
        # We'll use a simplified state space based on position zones
        self.q_table = {}
        
        # Track learning progress
        self.episode_rewards = []
        self.episode_steps = []
        self.positions_visited = []
        
        # de_survivor zone mapping (rough estimates)
        self.zones = {
            'spawn_t': {'center': np.array([0, 0, 0]), 'size': 200},
            'spawn_ct': {'center': np.array([800, 800, 0]), 'size': 200},
            'bridge': {'center': np.array([400, 200, 50]), 'size': 150},
            'tower': {'center': np.array([-200, 400, 100]), 'size': 100},
            'underground': {'center': np.array([200, -400, -50]), 'size': 180},
            'rooftop': {'center': np.array([300, 300, 200]), 'size': 120},
            'center': {'center': np.array([400, 400, 0]), 'size': 250}
        }
        
        # Action space
        self.actions = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),    # Forward
            np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Backward
            np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Strafe left
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),    # Strafe right
            np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),   # Turn left
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),    # Turn right
            np.array([0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),    # Jump forward
            np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),    # Forward + turn right
            np.array([0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),   # Forward + turn left
        ]
        
        print("🎯 de_survivor Learner Initialized")
        print(f"📊 Learning Rate: {learning_rate}")
        print(f"🎲 Exploration Rate: {epsilon}")
        print(f"🎮 Action Space: {len(self.actions)} actions")
        print(f"🗺️  Map Zones: {len(self.zones)} zones")
    
    def get_state_from_position(self, position: np.ndarray) -> str:
        """Convert position to discrete state (zone)"""
        min_dist = float('inf')
        closest_zone = 'unknown'
        
        for zone_name, zone_info in self.zones.items():
            dist = np.linalg.norm(position - zone_info['center'])
            if dist < zone_info['size'] and dist < min_dist:
                min_dist = dist
                closest_zone = zone_name
        
        return closest_zone
    
    def choose_action(self, state: str) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, len(self.actions))
        else:            # Exploit: best known action for this state
            if state in self.q_table:
                return int(np.argmax(self.q_table[state]))
            else:
                # Unknown state, explore
                return np.random.randint(0, len(self.actions))
    
    def update_q_table(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-table using Q-learning"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions), dtype=np.float64)
        
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions), dtype=np.float64)
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def train_episode(self, env: WorkingCS16Environment, max_steps=500) -> Tuple[float, int]:
        """Train for one episode"""
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        
        state = self.get_state_from_position(obs['position'])
        
        print(f"🎬 Episode starting in zone: {state}")
        print(f"📍 Position: {obs['position'][:2]}")
        
        for step in range(max_steps):
            # Choose and execute action
            action_idx = self.choose_action(state)
            action = self.actions[action_idx]
            
            obs, reward, done, truncated, info = env.step(action)
            
            next_state = self.get_state_from_position(obs['position'])
            
            # Update Q-table
            self.update_q_table(state, action_idx, reward, next_state)
            
            # Track progress
            total_reward += reward
            steps += 1
            self.positions_visited.append(obs['position'].copy())
            
            # Debug output
            if step % 50 == 0:
                print(f"  Step {step}: {state} -> {next_state}, R={reward:.2f}, H={obs['health'][0]}")
            
            # Move to next state
            state = next_state
            
            if done or truncated:
                print(f"💀 Episode ended: done={done}, truncated={truncated}")
                break
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_reward, steps
    
    def train(self, episodes=50):
        """Train the agent for multiple episodes"""
        print(f"🚀 Starting training for {episodes} episodes on de_survivor")
        print("="*60)
        
        env = WorkingCS16Environment(debug=True)
        
        try:
            for episode in range(episodes):
                print(f"\n🎯 Episode {episode + 1}/{episodes}")
                print(f"🎲 Exploration rate: {self.epsilon:.3f}")
                
                total_reward, steps = self.train_episode(env)
                
                self.episode_rewards.append(total_reward)
                self.episode_steps.append(steps)
                
                print(f"✅ Episode {episode + 1} completed!")
                print(f"💰 Total Reward: {total_reward:.2f}")
                print(f"👟 Steps Taken: {steps}")
                print(f"📊 Q-table size: {len(self.q_table)} states")
                
                # Save progress every 10 episodes
                if (episode + 1) % 10 == 0:
                    self.save_progress(f"episode_{episode + 1}")
                    self.plot_progress()
                
                # Brief pause between episodes
                time.sleep(2)
        
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
        
        finally:
            env.close()
            self.save_progress("final")
            self.plot_progress()
            print("🎉 Training completed!")
    
    def save_progress(self, suffix=""):
        """Save learning progress to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"de_survivor_learning_{timestamp}_{suffix}"
        
        # Save Q-table
        with open(f"data/{filename}_qtable.pkl", "wb") as f:
            pickle.dump(self.q_table, f)
        
        # Save learning stats
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'positions_visited': [pos.tolist() for pos in self.positions_visited],
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'zones': self.zones
        }
        
        with open(f"data/{filename}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 Progress saved: {filename}")
    
    def plot_progress(self):
        """Plot learning progress"""
        if not self.episode_rewards:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward over episodes
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Steps over episodes
        axes[0, 1].plot(self.episode_steps)
        axes[0, 1].set_title('Episode Steps')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Exploration map (last 1000 positions)
        if self.positions_visited:
            recent_positions = self.positions_visited[-1000:]
            x_coords = [pos[0] for pos in recent_positions]
            y_coords = [pos[1] for pos in recent_positions]
            
            axes[1, 0].scatter(x_coords, y_coords, alpha=0.5, s=1)
            axes[1, 0].set_title('Exploration Map (Recent 1000 positions)')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Y')
            
            # Add zone centers
            for zone_name, zone_info in self.zones.items():
                center = zone_info['center']
                axes[1, 0].scatter(center[0], center[1], color='red', s=50, marker='x')
                axes[1, 0].annotate(zone_name, (center[0], center[1]))
        
        # Q-table heatmap (simplified)
        if self.q_table:
            states = list(self.q_table.keys())
            actions_count = len(self.actions)
            q_matrix = np.zeros((len(states), actions_count))
            
            for i, state in enumerate(states):
                q_matrix[i] = self.q_table[state]
            
            im = axes[1, 1].imshow(q_matrix, aspect='auto', cmap='viridis')
            axes[1, 1].set_title('Q-table Heatmap')
            axes[1, 1].set_xlabel('Actions')
            axes[1, 1].set_ylabel('States')
            axes[1, 1].set_yticks(range(len(states)))
            axes[1, 1].set_yticklabels(states)
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"data/training_progress_{timestamp}.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Progress plot updated")

def main():
    """Main training function"""
    print("🎮 CS 1.6 de_survivor Map Learning")
    print("="*50)
    print("This will train an AI to control YOUR player!")
    print("Make sure:")
    print("✅ CS 1.6 is running")
    print("✅ You're on de_survivor map")
    print("✅ You can move around normally")
    print()
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    input("Press Enter when ready to start training...")
    
    # Create learner and start training
    learner = DeSurvivorLearner(
        learning_rate=0.1,
        epsilon=0.9,
        epsilon_decay=0.995
    )
    
    # Start with fewer episodes for testing
    learner.train(episodes=20)

if __name__ == "__main__":
    main()
