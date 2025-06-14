"""
Reward Shaper - Advanced Reward Engineering for Survival Training
Implements sophisticated reward shaping techniques for faster learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

class RewardType(Enum):
    """Types of rewards that can be shaped."""
    SURVIVAL_TIME = "survival_time"
    HAZARD_AVOIDANCE = "hazard_avoidance"  
    MOVEMENT_EFFICIENCY = "movement_efficiency"
    COMBAT_PERFORMANCE = "combat_performance"
    EXPLORATION = "exploration"
    LEARNING_PROGRESS = "learning_progress"

@dataclass
class RewardShapingConfig:
    """Configuration for reward shaping methods."""
    use_potential_based: bool = True      # Potential-based reward shaping
    use_curriculum: bool = True           # Curriculum learning
    use_adaptive_weights: bool = True     # Dynamic reward weights
    exploration_bonus_decay: float = 0.99 # Decay rate for exploration bonus
    difficulty_progression_rate: float = 0.001  # How fast to increase difficulty

class RewardShaper:
    """
    Advanced reward shaping for zombie survival training.
    
    Implements multiple reward shaping techniques:
    - Potential-based shaping (theoretically sound)
    - Curriculum learning (gradual difficulty increase)  
    - Adaptive reward weights (based on performance)
    - Exploration bonuses (encourage diverse behavior)
    """
    
    def __init__(self, config: Optional[RewardShapingConfig] = None):
        self.config = config or RewardShapingConfig()
        
        # Potential functions for different reward types
        self.potential_functions = {
            RewardType.SURVIVAL_TIME: self._survival_potential,
            RewardType.HAZARD_AVOIDANCE: self._hazard_potential,
            RewardType.MOVEMENT_EFFICIENCY: self._movement_potential,
            RewardType.COMBAT_PERFORMANCE: self._combat_potential
        }
        
        # Adaptive weights (learned over time)
        self.reward_weights = {
            RewardType.SURVIVAL_TIME: 1.0,
            RewardType.HAZARD_AVOIDANCE: 2.0,      # Higher weight for safety
            RewardType.MOVEMENT_EFFICIENCY: 0.5,
            RewardType.COMBAT_PERFORMANCE: 1.0,
            RewardType.EXPLORATION: 0.3,
            RewardType.LEARNING_PROGRESS: 0.5
        }
        
        # Curriculum learning state
        self.training_step = 0
        self.difficulty_level = 0.0  # 0.0 = easy, 1.0 = full difficulty
        
        # Exploration tracking
        self.state_visitation_counts = {}
        self.exploration_decay = self.config.exploration_bonus_decay
        
        # Performance tracking for adaptive weights
        self.performance_history = {reward_type: [] for reward_type in RewardType}
        self.adaptation_rate = 0.001
        
    def shape_reward(self, 
                    reward_components: Dict[RewardType, float],
                    current_state: np.ndarray,
                    next_state: Optional[np.ndarray] = None,
                    bot_id: int = 0) -> Dict[str, float]:
        """
        Apply reward shaping to raw reward components.
        
        Args:
            reward_components: Raw rewards by type
            current_state: Current game state features
            next_state: Next state (for potential-based shaping)
            bot_id: Bot identifier for per-bot tracking
            
        Returns:
            Dictionary with shaped rewards and debugging info
        """
        shaped_rewards = {}
        total_shaped_reward = 0.0
        
        # Apply potential-based shaping
        if self.config.use_potential_based and next_state is not None:
            potential_rewards = self._apply_potential_shaping(
                reward_components, current_state, next_state)
            shaped_rewards.update(potential_rewards)
            
        # Apply curriculum learning
        if self.config.use_curriculum:
            curriculum_modifier = self._get_curriculum_modifier()
            for reward_type, value in reward_components.items():
                if reward_type in shaped_rewards:
                    shaped_rewards[f"{reward_type.value}_curriculum"] = value * curriculum_modifier
                else:
                    shaped_rewards[f"{reward_type.value}_curriculum"] = value * curriculum_modifier
                    
        # Add exploration bonus
        exploration_bonus = self._calculate_exploration_bonus(current_state, bot_id)
        shaped_rewards["exploration_bonus"] = exploration_bonus
        
        # Apply adaptive weights
        if self.config.use_adaptive_weights:
            weighted_rewards = self._apply_adaptive_weights(reward_components)
            shaped_rewards.update(weighted_rewards)
        else:
            # Apply static weights
            for reward_type, value in reward_components.items():
                weight = self.reward_weights.get(reward_type, 1.0)
                shaped_rewards[f"{reward_type.value}_weighted"] = value * weight
                
        # Calculate total shaped reward
        total_shaped_reward = sum(shaped_rewards.values())
        shaped_rewards["total_shaped"] = total_shaped_reward
        
        # Update tracking
        self._update_tracking(reward_components, bot_id)
        
        return shaped_rewards
        
    def _apply_potential_shaping(self, 
                               reward_components: Dict[RewardType, float],
                               current_state: np.ndarray,
                               next_state: np.ndarray) -> Dict[str, float]:
        """Apply potential-based reward shaping (theoretically sound)."""
        potential_rewards = {}
        
        for reward_type in reward_components.keys():
            if reward_type in self.potential_functions:
                current_potential = self.potential_functions[reward_type](current_state)
                next_potential = self.potential_functions[reward_type](next_state)
                
                # F(s') - F(s) is the potential-based shaping reward
                gamma = 0.99  # Discount factor
                potential_reward = gamma * next_potential - current_potential
                potential_rewards[f"{reward_type.value}_potential"] = potential_reward
                
        return potential_rewards
        
    def _survival_potential(self, state: np.ndarray) -> float:
        """Potential function for survival time rewards."""
        # Higher potential for safer positions (further from hazards)
        if len(state) > 20:  # Ensure we have hazard distance features
            hazard_distances = state[12:20]  # Approximate hazard distance indices
            min_hazard_distance = np.min(hazard_distances)
            return np.tanh(min_hazard_distance / 100.0)  # Normalize to [0,1]
        return 0.0
        
    def _hazard_potential(self, state: np.ndarray) -> float:
        """Potential function for hazard avoidance."""
        if len(state) > 20:
            hazard_distances = state[12:20]
            # Exponential potential - gets very high when far from hazards
            avg_hazard_distance = np.mean(hazard_distances)
            return 1.0 - np.exp(-avg_hazard_distance / 50.0)
        return 0.0
        
    def _movement_potential(self, state: np.ndarray) -> float:
        """Potential function for movement efficiency."""
        if len(state) >= 6:  # Position and velocity
            velocity = state[3:6]  # vel_x, vel_y, vel_z
            speed = np.linalg.norm(velocity)
            return np.tanh(speed / 200.0)  # Encourage movement
        return 0.0
        
    def _combat_potential(self, state: np.ndarray) -> float:
        """Potential function for combat performance."""
        if len(state) > 25:  # Ensure we have enemy info
            has_enemy = state[15] if len(state) > 15 else 0  # has_enemy flag
            enemy_distance = state[20] if len(state) > 20 else 1000  # enemy distance
            
            if has_enemy > 0.5:
                # Higher potential when engaging enemies at optimal range
                optimal_range = 300.0  # Optimal engagement distance
                range_factor = 1.0 - abs(enemy_distance - optimal_range) / optimal_range
                return max(0.0, range_factor)
        return 0.0
        
    def _get_curriculum_modifier(self) -> float:
        """Get curriculum learning difficulty modifier."""
        # Gradually increase difficulty over training
        self.difficulty_level = min(1.0, self.difficulty_level + self.config.difficulty_progression_rate)
        
        # Early training: easier rewards, later training: full difficulty
        return 0.5 + 0.5 * self.difficulty_level
        
    def _calculate_exploration_bonus(self, state: np.ndarray, bot_id: int) -> float:
        """Calculate exploration bonus for visiting new states."""
        # Create state hash (simplified)
        if len(state) >= 3:
            position = state[:3]  # x, y, z coordinates
            # Discretize position for state counting
            discretized = tuple(np.round(position / 50.0).astype(int))  # 50-unit grid
            
            state_key = (bot_id, discretized)
            
            # Count visits to this state
            visits = self.state_visitation_counts.get(state_key, 0)
            self.state_visitation_counts[state_key] = visits + 1
            
            # Exploration bonus decreases with visits
            exploration_bonus = 1.0 / (1.0 + visits) * (self.exploration_decay ** self.training_step)
            return exploration_bonus
            
        return 0.0
        
    def _apply_adaptive_weights(self, reward_components: Dict[RewardType, float]) -> Dict[str, float]:
        """Apply adaptive reward weights based on performance."""
        weighted_rewards = {}
        
        for reward_type, value in reward_components.items():
            # Get current weight
            current_weight = self.reward_weights.get(reward_type, 1.0)
            
            # Adapt weight based on recent performance
            if reward_type in self.performance_history:
                recent_performance = self.performance_history[reward_type][-100:]  # Last 100 steps
                if len(recent_performance) > 10:
                    avg_performance = np.mean(recent_performance)
                    
                    # Increase weight for underperforming rewards
                    if avg_performance < 0:  # Poor performance
                        self.reward_weights[reward_type] += self.adaptation_rate
                    elif avg_performance > 10:  # Good performance  
                        self.reward_weights[reward_type] = max(0.1, 
                            self.reward_weights[reward_type] - self.adaptation_rate)
                        
            # Apply weight
            weighted_rewards[f"{reward_type.value}_adaptive"] = value * current_weight
            
        return weighted_rewards
        
    def _update_tracking(self, reward_components: Dict[RewardType, float], bot_id: int):
        """Update internal tracking for adaptive behavior."""
        self.training_step += 1
        
        # Update performance history
        for reward_type, value in reward_components.items():
            if reward_type in self.performance_history:
                self.performance_history[reward_type].append(value)
                
                # Keep only recent history
                if len(self.performance_history[reward_type]) > 1000:
                    self.performance_history[reward_type] = self.performance_history[reward_type][-1000:]
                    
    def get_shaping_stats(self) -> Dict[str, Any]:
        """Get statistics about reward shaping."""
        return {
            "training_step": self.training_step,
            "difficulty_level": self.difficulty_level,
            "reward_weights": dict(self.reward_weights),
            "exploration_states_visited": len(self.state_visitation_counts),
            "avg_exploration_visits": np.mean(list(self.state_visitation_counts.values())) if self.state_visitation_counts else 0
        }
        
    def reset_curriculum(self):
        """Reset curriculum learning progress."""
        self.difficulty_level = 0.0
        self.training_step = 0
        
    def reset_exploration(self):
        """Reset exploration tracking."""
        self.state_visitation_counts.clear()

if __name__ == "__main__":
    # Test reward shaper
    shaper = RewardShaper()
    
    # Test reward components
    rewards = {
        RewardType.SURVIVAL_TIME: 10.0,
        RewardType.HAZARD_AVOIDANCE: -50.0,
        RewardType.MOVEMENT_EFFICIENCY: 5.0,
        RewardType.COMBAT_PERFORMANCE: 20.0
    }
    
    # Test states
    current_state = np.random.randn(56)
    next_state = np.random.randn(56)
    
    # Shape rewards
    shaped = shaper.shape_reward(rewards, current_state, next_state, bot_id=0)
    
    print("Original rewards:")
    for reward_type, value in rewards.items():
        print(f"  {reward_type.value}: {value:.2f}")
        
    print("\nShaped rewards:")
    for name, value in shaped.items():
        print(f"  {name}: {value:.2f}")
        
    print(f"\nShaping stats:")
    stats = shaper.get_shaping_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
