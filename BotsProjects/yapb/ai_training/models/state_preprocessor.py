"""
State Preprocessor - Input Feature Processing for Neural Networks
Handles normalization, encoding, and feature engineering for game state data
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    """Configuration for individual feature preprocessing."""
    name: str
    min_value: float = -1.0
    max_value: float = 1.0
    normalization: str = 'minmax'  # 'minmax', 'zscore', 'none'
    clip_outliers: bool = True
    outlier_percentile: float = 0.95

class StatePreprocessor:
    """
    Preprocessor for converting raw game state to neural network input.
    
    Handles feature normalization, outlier clipping, and encoding for optimal
    neural network training and inference.
    """
    
    def __init__(self):
        self.feature_configs = self._create_default_configs()
        self.feature_stats = {}  # Running statistics for normalization
        self.is_fitted = False
        
    def _create_default_configs(self) -> Dict[str, FeatureConfig]:
        """Create default feature configurations based on C++ StateEncoder."""
        configs = {}
        
        # Position features (world coordinates)
        for feature in ['pos_x', 'pos_y', 'pos_z']:
            configs[feature] = FeatureConfig(feature, -4096, 4096, 'minmax')
            
        # Velocity features
        for feature in ['vel_x', 'vel_y', 'vel_z']:
            configs[feature] = FeatureConfig(feature, -500, 500, 'minmax')
            
        # Angles (already in reasonable range)
        for feature in ['yaw', 'pitch', 'roll']:
            configs[feature] = FeatureConfig(feature, -180, 180, 'minmax')
            
        # Health and armor
        configs['health'] = FeatureConfig('health', 0, 100, 'minmax')
        configs['armor'] = FeatureConfig('armor', 0, 100, 'minmax')
        
        # Distances (hazards, enemies)
        for i in range(8):  # hazard distances
            configs[f'hazard_dist_{i}'] = FeatureConfig(f'hazard_dist_{i}', 0, 1000, 'minmax')
            
        for i in range(6):  # enemy/teammate distances  
            configs[f'enemy_dist_{i}'] = FeatureConfig(f'enemy_dist_{i}', 0, 2000, 'minmax')
            
        # Binary features (already 0-1)
        binary_features = ['stuck', 'on_ladder', 'in_water', 'has_enemy', 
                          'enemy_visible', 'jump', 'duck', 'attack1', 'attack2']
        for feature in binary_features:
            configs[feature] = FeatureConfig(feature, 0, 1, 'none')
            
        return configs
        
    def fit(self, data: np.ndarray, feature_names: Optional[List[str]] = None) -> 'StatePreprocessor':
        """
        Fit preprocessor to training data to learn normalization parameters.
        
        Args:
            data: Training data [num_samples, num_features]
            feature_names: Optional feature names
            
        Returns:
            Self (for method chaining)
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
            
        self.feature_stats = {}
        
        for i, name in enumerate(feature_names):
            if i >= data.shape[1]:
                break
                
            feature_data = data[:, i]
            config = self.feature_configs.get(name, FeatureConfig(name))
            
            stats = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'percentile_5': np.percentile(feature_data, 5),
                'percentile_95': np.percentile(feature_data, 95)
            }
            
            self.feature_stats[name] = stats
            
        self.is_fitted = True
        return self
        
    def transform(self, data: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Transform raw features to normalized neural network input.
        
        Args:
            data: Raw feature data [num_samples, num_features] or [num_features]
            feature_names: Optional feature names
            
        Returns:
            Normalized feature data
        """
        if not self.is_fitted:
            # Auto-fit on first transform (for inference)
            return self._transform_without_fitting(data, feature_names)
            
        # Handle single sample
        single_sample = data.ndim == 1
        if single_sample:
            data = data.reshape(1, -1)
            
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
            
        normalized_data = np.zeros_like(data, dtype=np.float32)
        
        for i, name in enumerate(feature_names):
            if i >= data.shape[1]:
                break
                
            feature_data = data[:, i]
            config = self.feature_configs.get(name, FeatureConfig(name))
            stats = self.feature_stats.get(name, {})
            
            # Apply normalization
            if config.normalization == 'minmax':
                min_val = stats.get('min', config.min_value)
                max_val = stats.get('max', config.max_value)
                normalized = (feature_data - min_val) / (max_val - min_val + 1e-8)
                normalized = np.clip(normalized, 0, 1)
                
            elif config.normalization == 'zscore':
                mean = stats.get('mean', 0)
                std = stats.get('std', 1)
                normalized = (feature_data - mean) / (std + 1e-8)
                
            else:  # 'none'
                normalized = feature_data
                
            # Clip outliers if requested
            if config.clip_outliers and config.normalization != 'none':
                if config.normalization == 'minmax':
                    normalized = np.clip(normalized, 0, 1)
                else:  # zscore
                    normalized = np.clip(normalized, -3, 3)
                    
            normalized_data[:, i] = normalized
            
        return normalized_data[0] if single_sample else normalized_data
        
    def _transform_without_fitting(self, data: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Transform using default ranges without fitting to data."""
        single_sample = data.ndim == 1
        if single_sample:
            data = data.reshape(1, -1)
            
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
            
        normalized_data = np.zeros_like(data, dtype=np.float32)
        
        for i, name in enumerate(feature_names):
            if i >= data.shape[1]:
                break
                
            feature_data = data[:, i]
            config = self.feature_configs.get(name, FeatureConfig(name))
            
            # Use config defaults for normalization
            if config.normalization == 'minmax':
                normalized = (feature_data - config.min_value) / (config.max_value - config.min_value + 1e-8)
                normalized = np.clip(normalized, 0, 1)
            elif config.normalization == 'zscore':
                # Use rough defaults
                normalized = feature_data / 100.0  # Rough normalization
                normalized = np.clip(normalized, -3, 3)
            else:
                normalized = feature_data
                
            normalized_data[:, i] = normalized
            
        return normalized_data[0] if single_sample else normalized_data
        
    def fit_transform(self, data: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit preprocessor and transform data in one step."""
        return self.fit(data, feature_names).transform(data, feature_names)
        
    def to_tensor(self, data: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert preprocessed data to PyTorch tensor."""
        tensor = torch.from_numpy(data.astype(np.float32))
        if device:
            tensor = tensor.to(device)
        return tensor
        
    def get_feature_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get the expected input ranges for all features."""
        ranges = {}
        for name, config in self.feature_configs.items():
            if config.normalization == 'minmax':
                ranges[name] = (0.0, 1.0)
            elif config.normalization == 'zscore':
                ranges[name] = (-3.0, 3.0)
            else:
                ranges[name] = (config.min_value, config.max_value)
        return ranges
        
    def validate_input(self, data: np.ndarray, feature_names: Optional[List[str]] = None) -> bool:
        """Validate that input data is within expected ranges."""
        ranges = self.get_feature_ranges()
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(data.shape[-1])]
            
        single_sample = data.ndim == 1
        if single_sample:
            data = data.reshape(1, -1)
            
        for i, name in enumerate(feature_names):
            if i >= data.shape[1]:
                continue
                
            feature_data = data[:, i]
            min_range, max_range = ranges.get(name, (-np.inf, np.inf))
            
            if np.any(feature_data < min_range) or np.any(feature_data > max_range):
                return False
                
        return True

if __name__ == "__main__":
    # Test preprocessor
    preprocessor = StatePreprocessor()
    
    # Generate test data
    num_samples, num_features = 1000, 56
    test_data = np.random.randn(num_samples, num_features) * 100
    
    # Fit and transform
    normalized = preprocessor.fit_transform(test_data)
    
    print(f"Original data range: [{test_data.min():.2f}, {test_data.max():.2f}]")
    print(f"Normalized data range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    print(f"Normalized data shape: {normalized.shape}")
    
    # Test single sample
    single_sample = test_data[0]
    normalized_single = preprocessor.transform(single_sample)
    print(f"Single sample shape: {normalized_single.shape}")
    
    # Validate
    is_valid = preprocessor.validate_input(normalized_single)
    print(f"Validation passed: {is_valid}")
