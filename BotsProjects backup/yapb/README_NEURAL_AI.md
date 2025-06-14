# YaPB Neural Zombie AI System 🧠🧟

## Revolutionary AI Training for Counter-Strike Bots

This system implements a **groundbreaking machine learning approach** for YaPB zombie AI, combining the best of rule-based intelligence with neural network learning capabilities.

### 🚀 **What This System Does**

This neural network integration transforms your existing YaPB zombie AI from static rule-based behavior into a **learning, adaptive intelligence system** that:

1. **Learns from Expert Behavior**: Uses your existing rule-based zombie AI as a "teacher" to train neural networks
2. **Collects Real-Time Data**: Captures state-action pairs during actual bot gameplay
3. **Trains Deep Neural Networks**: Implements advanced Deep Q-Learning (DQN) for decision making
4. **Provides Intelligent Fallback**: Seamlessly switches between neural and rule-based decisions based on confidence
5. **Continuously Improves**: Gets better over time as more training data is collected

---

## 🎯 **Key Features**

### **Machine Learning Integration**
- **Deep Q-Network (DQN)** implementation with experience replay
- **Behavioral Cloning** from expert rule-based AI demonstrations
- **Multi-head neural network** architecture for specialized decision types
- **Confidence-based decision making** with intelligent fallback systems

### **Real-Time Data Collection**
- **State extraction** from live bot gameplay (position, health, enemy info, etc.)
- **Action recording** of all bot decisions and movements
- **Reward calculation** based on game outcomes and zombie behavior
- **Automatic data preprocessing** and normalization

### **Advanced Training Pipeline**
- **Automated bot matches** for massive data generation
- **Distributed training** support for multiple neural network models
- **Performance evaluation** against rule-based AI benchmarks
- **Model versioning** and automatic best-model selection

### **Seamless YaPB Integration**
- **Zero-impact integration** - works alongside existing zombie AI
- **Runtime model switching** - can enable/disable neural networks on the fly
- **Configurable confidence thresholds** for neural vs rule-based decisions
- **Comprehensive logging** and performance monitoring

---

## 🛠️ **Installation & Setup**

### **1. Prerequisites**
```bash
# Python 3.8+ required
python --version

# Install required packages
cd ai_training
pip install -r requirements.txt
```

### **2. YaPB Integration**
Add to your YaPB build files:
```cpp
// In inc/yapb.h
#include "neural_zombie_ai.h"

// In src/engine.cpp (add these ConVars)
ConVar cv_neural_training_enabled ("neural_training_enabled", "0");
ConVar cv_neural_data_collection ("neural_data_collection", "0");
ConVar cv_neural_use_for_decisions ("neural_use_for_decisions", "0");
```

### **3. Directory Structure**
```
yapb/
├── ai_training/
│   ├── neural_zombie_trainer.py      # Core neural network implementation
│   ├── train_zombie_ai.py           # Complete training pipeline
│   ├── zombie_ai_config.json        # Configuration settings
│   ├── requirements.txt             # Python dependencies
│   ├── data/                        # Training data storage
│   ├── models/                      # Trained neural network models
│   ├── logs/                        # Training logs and reports
│   └── checkpoints/                 # Model checkpoints
├── inc/neural_zombie_ai.h           # C++ header for integration
├── src/neural_zombie_ai.cpp         # C++ implementation
└── README_NEURAL_AI.md              # This file
```

---

## 🎮 **Usage Guide**

### **Quick Start - Automated Training**
```bash
# 1. Start automated training (30 minutes of data collection + training)
python train_zombie_ai.py

# 2. Quick training session (10 minutes)
python train_zombie_ai.py --quick-train

# 3. Custom data collection time
python train_zombie_ai.py --data-collection-time 60  # 1 hour
```

### **Manual Training Steps**

#### **Step 1: Data Collection**
```bash
# Start YaPB with neural data collection
yapb +neural_training_enabled 1 +neural_data_collection 1 +map de_dust2 +yb_quota_maintain 16
```

#### **Step 2: Train Neural Network**
```python
from neural_zombie_trainer import ZombieAITrainer

trainer = ZombieAITrainer("zombie_ai_config.json")
# Training data will be automatically loaded and processed
```

#### **Step 3: Evaluate Model**
```bash
python train_zombie_ai.py --evaluate-only --model models/best_zombie_model.pth
```

#### **Step 4: Deploy Trained Model**
```bash
# In YaPB console or config
neural_model_path "ai_training/models/best_zombie_model.pth"
neural_use_for_decisions 1
neural_confidence_threshold 0.8
```

---

## ⚙️ **Configuration**

### **Core Settings (zombie_ai_config.json)**

```json
{
  "zombie_behavior": {
    "hunt_range": 1024.0,           // Detection range for enemies
    "aggression_level": 80.0,       // Zombie aggression (0-100)
    "confidence_threshold": 0.8,     // Neural network confidence threshold
    "fallback_enabled": true         // Enable rule-based fallback
  },
  
  "training": {
    "learning_rate": 0.001,         // Neural network learning rate
    "batch_size": 64,               // Training batch size
    "max_episodes": 10000           // Maximum training episodes
  }
}
```

### **YaPB ConVars**

| ConVar | Default | Description |
|--------|---------|-------------|
| `neural_training_enabled` | 0 | Enable neural system |
| `neural_data_collection` | 0 | Collect training data |
| `neural_use_for_decisions` | 0 | Use neural network for decisions |
| `neural_confidence_threshold` | 0.8 | Minimum confidence for neural decisions |
| `neural_fallback_enabled` | 1 | Enable rule-based fallback |
| `neural_debug_output` | 0 | Show neural decision debug info |

---

## 📊 **Performance Monitoring**

### **Real-Time Statistics**
```bash
# In YaPB console
neural_stats  # Show performance metrics

# Output example:
# Neural AI Performance Metrics:
#   Accuracy: 87.3% (1247/1428)
#   Average Reward: 156.8
#   Neural Decisions: 1247 (87.3%)
#   Rule-based Decisions: 181 (12.7%)
```

### **Training Progress**
The system automatically generates:
- **Training curves** showing reward progression
- **Accuracy metrics** comparing neural vs rule-based decisions
- **Performance reports** in markdown format
- **Model evaluation** against benchmarks

---

## 🧪 **Advanced Features**

### **Multi-Model Ensemble**
Train multiple neural networks for different scenarios:
```python
# Train separate models for different aggression levels
configs = [
    {"aggression_level": 60, "model_name": "conservative"},
    {"aggression_level": 80, "model_name": "balanced"}, 
    {"aggression_level": 95, "model_name": "aggressive"}
]
```

### **Transfer Learning**
Use pre-trained models as starting points:
```python
trainer.load_pretrained_model("models/base_zombie_model.pth")
trainer.fine_tune(new_data)
```

### **Hyperparameter Optimization**
Automatic tuning of neural network parameters:
```bash
python optimize_hyperparameters.py --trials 100
```

---

## 🔬 **Technical Architecture**

### **Neural Network Structure**
```
Input Layer (25 features)
    ↓
Hidden Layer 1 (256 neurons, ReLU)
    ↓
Hidden Layer 2 (256 neurons, ReLU)  
    ↓
Hidden Layer 3 (256 neurons, ReLU)
    ↓
Output Heads:
├── Movement Head (4 outputs: forward, right, yaw, pitch)
├── Combat Head (2 outputs: primary attack, secondary attack)
├── Task Head (4 outputs: hunt, attack, seek_cover, no_change)
└── Modifier Head (3 outputs: jump, duck, walk)
```

### **State Representation**
The neural network receives 25 normalized features:
- **Position & Movement** (6): 3D position, 3D velocity
- **Health & Status** (5): health, armor, stuck state, ladder state, water state  
- **Enemy Information** (6): enemy presence, position, distance, health, visibility
- **Environment** (4): team, nearby teammates/enemies, current task
- **Zombie Settings** (4): hunt range, aggression, view angles

### **Action Space**
Neural network outputs 13 action components:
- **Movement** (4): forward/back, left/right, yaw turn, pitch turn
- **Combat** (2): primary attack, secondary attack
- **Tasks** (4): task switching probabilities
- **Modifiers** (3): jump, duck, walk

---

## 📈 **Expected Results**

### **Performance Improvements**
After training, you should observe:

1. **Smarter Decision Making**: Zombies make more contextual decisions based on game state
2. **Better Coordination**: Multiple zombies coordinate attacks more effectively  
3. **Adaptive Behavior**: AI adapts to different situations and maps over time
4. **Higher Success Rate**: Improved kill ratios and survival times

### **Behavioral Changes**
- **More human-like unpredictability** in zombie movements
- **Strategic positioning** rather than pure aggressive rushing
- **Dynamic response** to different human player behaviors
- **Learning from mistakes** through continuous training

---

## 🐛 **Troubleshooting**

### **Common Issues**

**Neural network not loading:**
```bash
# Check model file exists
ls -la ai_training/models/

# Verify file permissions
chmod 644 ai_training/models/*.pth
```

**Low neural confidence:**
```bash
# Reduce confidence threshold temporarily
neural_confidence_threshold 0.6

# Or force neural network usage
neural_fallback_enabled 0
```

**Training data not collecting:**
```bash
# Verify data collection is enabled
neural_data_collection 1
neural_training_enabled 1

# Check output directory exists
mkdir -p ai_training/data
```

### **Performance Issues**
- **GPU Acceleration**: Install CUDA version of PyTorch for faster training
- **Memory Usage**: Reduce batch size if experiencing out-of-memory errors
- **CPU Usage**: Adjust `neural_collection_interval` to reduce data collection frequency

---

## 🚀 **Future Enhancements**

### **Planned Features**
- **Attention Mechanisms**: Let zombies focus on important game elements
- **Hierarchical RL**: Multi-level decision making (strategic + tactical)
- **Meta-Learning**: Rapid adaptation to new maps and game modes
- **Adversarial Training**: Train against human players in real-time

### **Research Directions**
- **Imitation Learning**: Learn directly from professional players
- **Multi-Agent RL**: Coordinate multiple zombie bots simultaneously
- **Transfer Learning**: Apply learned behaviors across different game modes
- **Explainable AI**: Understand why neural networks make specific decisions

---

## 🤝 **Contributing**

We welcome contributions to improve the neural zombie AI system:

1. **Fork the repository**
2. **Create feature branches** for new capabilities
3. **Add comprehensive tests** for neural network components
4. **Submit pull requests** with detailed descriptions

### **Areas for Contribution**
- New neural network architectures
- Additional training algorithms (PPO, A3C, etc.)
- Performance optimizations
- Integration with other game mods
- Documentation improvements

---

## 📄 **License**

This neural AI system is released under the MIT License, same as YaPB.

---

## 🙏 **Acknowledgments**

- **YaPB Development Team** for the excellent bot framework
- **PyTorch Team** for the neural network framework
- **OpenAI** for reinforcement learning research and inspiration
- **Counter-Strike Community** for continuous feedback and testing

---

## 📞 **Support**

For questions, issues, or feature requests:

- **GitHub Issues**: Create detailed bug reports or feature requests
- **Discord**: Join the YaPB community discord
- **Email**: Contact the development team

---

**Transform your YaPB zombies from predictable bots into intelligent, learning adversaries! 🧠🧟‍♂️**
