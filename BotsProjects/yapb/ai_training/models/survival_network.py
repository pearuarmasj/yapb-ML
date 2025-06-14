"""
Survival Network - Neural Network Architecture for Zombie Survival
Lightweight network designed for real-time inference at 240 FPS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SurvivalNetwork(nn.Module):
    """
    Neural network for zombie survival decision making.
    
    Input: Game state features (position, health, hazards, enemies, etc.)
    Output: Movement and action commands
    """
    
    def __init__(self, 
                 input_size: int = 56,           # From StateEncoder::TOTAL_FEATURES
                 hidden_sizes: Tuple[int, ...] = (128, 64, 32),
                 output_size: int = 9,           # Movement + actions
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True):
        super(SurvivalNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size) if use_batch_norm else nn.Identity()
        
        # Hidden layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity()
            ])
            prev_size = hidden_size
            
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output heads
        self.movement_head = nn.Linear(prev_size, 4)    # forward, right, yaw, pitch
        self.action_head = nn.Linear(prev_size, 4)      # jump, duck, attack1, attack2  
        self.confidence_head = nn.Linear(prev_size, 1)  # decision confidence
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Output tensor [batch_size, output_size]
        """
        # Input normalization
        x = self.input_norm(x)
        
        # Hidden layers
        features = self.hidden_layers(x)
        
        # Output heads
        movement = torch.tanh(self.movement_head(features))      # -1 to 1 for movement
        actions = torch.sigmoid(self.action_head(features))      # 0 to 1 for binary actions
        confidence = torch.sigmoid(self.confidence_head(features)) # 0 to 1 for confidence
        
        # Concatenate outputs
        output = torch.cat([movement, actions, confidence], dim=1)
        
        return output
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == 1:  # confidence head
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.5)  # Start with moderate confidence
                else:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    nn.init.constant_(module.bias, 0.0)
                    
    def get_model_size(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
        
    def get_inference_time_estimate(self) -> float:
        """Estimate inference time in milliseconds (very rough)."""
        # Rule of thumb: ~1ms per 10k parameters on modern GPU
        return self.get_model_size() / 10000.0

class SurvivalNetworkLarge(SurvivalNetwork):
    """Larger network for more complex scenarios (offline training)."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('hidden_sizes', (256, 128, 64, 32))
        super().__init__(**kwargs)

class SurvivalNetworkTiny(SurvivalNetwork):
    """Tiny network for maximum performance (real-time inference)."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('hidden_sizes', (64, 32))
        kwargs.setdefault('dropout_rate', 0.0)
        kwargs.setdefault('use_batch_norm', False)
        super().__init__(**kwargs)

def create_survival_network(size: str = 'medium') -> SurvivalNetwork:
    """
    Factory function to create appropriately sized networks.
    
    Args:
        size: 'tiny', 'medium', or 'large'
        
    Returns:
        Configured SurvivalNetwork instance
    """
    if size == 'tiny':
        return SurvivalNetworkTiny()
    elif size == 'large':
        return SurvivalNetworkLarge()
    else:  # medium
        return SurvivalNetwork()

if __name__ == "__main__":
    # Test network creation and inference
    network = create_survival_network('medium')
    print(f"Network parameters: {network.get_model_size():,}")
    print(f"Estimated inference time: {network.get_inference_time_estimate():.2f}ms")
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 56)
    output = network(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
