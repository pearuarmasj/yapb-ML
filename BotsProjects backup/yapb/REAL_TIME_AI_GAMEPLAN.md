# Real-Time Neural Network Training Gameplan - Zombie Survival Mode

## Overview
Implement real-time neural network training for 2 bots in **Zombies vs Humans** mode on de_survivor, with continuous learning during gameplay at 240 FPS. Primary focus: **STOP BOTS FROM BEING SUICIDAL IDIOTS** and drowning/falling to death.

## Current Status
✅ Data collection system implemented  
✅ Basic neural network inference working  
✅ Multithreaded data writing system  
✅ RTX 4080 GPU training scripts  
❌ Real-time training during gameplay  
❌ Hierarchical reward system  
❌ Context-aware state representation  
❌ Online learning implementation  

## Target Setup
- **2 bots** (1 zombie vs 1 human OR 2 zombies hunting)
- **de_survivor map** with hazardous terrain
- **User spectating** to observe learning and rage at stupid deaths
- **240 FPS** game performance maintained
- **Real-time learning** - bots improve during the match
- **RTX 4080** handling neural network training
- **PRIMARY GOAL**: Bots learn to NOT fucking die from environmental hazards

## Implementation Plan - PROPER ML, NO BULLSHIT

### STEP 1: Basic Real-Time Training Infrastructure
1. **LibTorch C++ Integration**
   - Link LibTorch to the project
   - Basic GPU inference working
   - Simple neural network loading/saving

2. **Experience Replay Buffer**
   - Circular buffer for (state, action, reward, next_state)
   - Size: ~10,000 experiences per bot (reasonable for testing)
   - Basic sampling for training batches

3. **Hazard-Aware State Representation**
   - Add distance to hazards as neural network inputs
   - Current position relative to known danger zones
   - Movement direction relative to hazards

### STEP 2: Online Training Loop
1. **Real-Time Gradient Updates**
   - Mini-batch training every 60 frames (4 times per second at 240 FPS)
   - Batch size: 32 experiences
   - Simple Adam optimizer

2. **Survival-Focused Reward Engineering**
   - Environmental death: -1000 reward
   - Proximity to hazards: -10 to -100 based on distance
   - Survival time bonus: +1 per second
   - Movement away from hazards: +5 reward

### STEP 3: Neural Network Architecture
1. **Lightweight Network for Real-Time**
   - Input: Game state + hazard distances (50-100 features)
   - Hidden: 2-3 layers, 64-128 neurons each
   - Output: Movement actions (forward, right, turn, jump, etc.)
   - Target inference time: <0.5ms per bot

### STEP 4: Training During Gameplay
1. **Continuous Learning**
   - Bots learn while playing on de_survivor
   - Network weights update in real-time
   - No stopping for offline training sessions

2. **Multi-Bot Environment**
   - 2 zombie bots training simultaneously
   - Shared experiences or independent learning
   - User spectates and observes learning progress

## Technical Requirements - REAL ML SYSTEM

### LibTorch C++ Integration
- **LibTorch 2.0+** with CUDA support for RTX 4080
- **CMake configuration** to link LibTorch properly
- **GPU memory management** for continuous training

### Core ML Components
- **Neural Network**: 50-100 input features → 64-128 hidden → 10-15 action outputs
- **Experience Replay**: Circular buffer with 10,000 (s,a,r,s') tuples per bot
- **Optimizer**: Adam with learning rate 0.0001-0.001
- **Loss Function**: MSE for Q-learning or policy gradient loss

### Real-Time Performance
- **Inference time**: <0.5ms per bot per frame
- **Training time**: <2ms per mini-batch (32 samples)
- **Memory usage**: <500MB for both bots combined
- **Frame rate**: Maintain 240 FPS during training

### Hazard Detection Integration
- **Hardcoded hazard zones** for de_survivor (coordinates from map analysis)
- **Distance calculations** integrated into state representation
- **Dynamic reward calculation** based on proximity and movement

## File Changes - PROPER ML IMPLEMENTATION

### New Files Needed
```
yapb/
├── inc/
│   ├── real_time_trainer.h      # LibTorch integration and training loop
│   ├── experience_buffer.h      # Replay buffer for training data
│   └── hazard_zones.h           # de_survivor hazard coordinates
├── src/
│   ├── real_time_trainer.cpp    # Neural network training implementation
│   ├── experience_buffer.cpp    # Memory management for experiences
│   └── hazard_zones.cpp         # Hardcoded danger zone definitions
└── ai_training/
    ├── survival_network.py      # Network architecture definition
    ├── export_libtorch.py       # Convert PyTorch → LibTorch format
    └── monitor_training.py      # Real-time learning visualization
```

### Modified Files
- `neural_zombie_ai.cpp` - Integrate LibTorch inference and training
- `neural_zombie_ai.h` - Add neural network and buffer class declarations  
- `CMakeLists.txt` - Add LibTorch dependencies
- `botlib.cpp` - Hook training loop into main bot logic

## Testing Protocol

### Phase 1 Testing
1. **Performance benchmark** - Ensure 240 FPS maintained
2. **Memory leak testing** - Run for 30+ minutes
3. **Training convergence** - Verify learning is occurring

### Phase 2 Testing
1. **Reward validation** - Ensure rewards make strategic sense
2. **Context accuracy** - Verify game state interpretation
3. **Learning speed** - Compare to offline training

### Phase 3 Testing
1. **1v1 matches** - Bots vs each other for extended periods
2. **Adaptation testing** - Change strategies, observe bot adaptation
3. **Performance comparison** - Before/after learning metrics

## Success Metrics

### Short Term (1-2 hours of training)
- **ZERO environmental deaths** (drowning, falling)
- Bots learn basic hazard avoidance on de_survivor
- Safe movement patterns around frozen river and chasm
- Basic zombie vs human role understanding

### Medium Term (8-12 hours of training)
- Efficient safe pathfinding across entire de_survivor map
- Smart positioning relative to infection zones and hazards
- Adaptive behavior based on zombie/human role
- Tactical use of map geometry for survival/hunting

### Long Term (24+ hours of training)
- Master-level survival instincts on de_survivor
- Complex multi-step planning to avoid environmental traps
- Advanced zombie hunting/human survival strategies
- Dynamic adaptation to different infection scenarios

## Risk Mitigation

### Performance Risks
- **Fallback to rule-based AI** if training fails
- **Adjustable training frequency** to maintain FPS
- **Memory usage monitoring** to prevent crashes

### Learning Risks
- **Curriculum learning** - start with simple scenarios
- **Reward shaping** to prevent degenerate strategies
- **Regular model checkpointing** to prevent catastrophic forgetting

### Technical Risks
- **LibTorch compatibility** testing with CS 1.6 engine
- **Thread safety** for real-time training
- **GPU memory management** for continuous operation

## Next Steps - REAL ML IMPLEMENTATION

### STEP 1: LibTorch Integration (This Week)
1. **Set up LibTorch C++** in the build system
2. **Create basic neural network class** for inference
3. **Test GPU acceleration** on RTX 4080
4. **Verify performance impact** at 240 FPS

### STEP 2: Experience Replay System
1. **Implement circular buffer** for training experiences
2. **Add state/action/reward collection** during gameplay
3. **Create mini-batch sampling** for training
4. **Test memory usage** and performance

### STEP 3: Real-Time Training Loop
1. **Implement online learning** during gameplay
2. **Add hazard-aware rewards** for de_survivor
3. **Create training scheduler** (4 times per second)
4. **Monitor learning progress** in real-time

### STEP 4: Multi-Bot Training Environment
1. **Spawn 2 zombie bots** for training
2. **Set up spectator mode** for observation
3. **Begin continuous learning** experiments
4. **Measure survival improvement** over time

### SUCCESS CRITERIA:
- **Real neural network** learning during gameplay
- **Environmental deaths decrease** measurably over training time  
- **240 FPS maintained** during training
- **Bots adapt** to de_survivor hazards through learning, not hardcoded rules

## CRITICAL PRIORITIES
1. **STOP BOTS FROM DROWNING** in frozen river areas
2. **STOP BOTS FROM FALLING** into chasm/breakable bridge zones  
3. **TEACH BASIC SURVIVAL INSTINCTS** before any combat training
4. **MAP-SPECIFIC HAZARD AVOIDANCE** for de_survivor layout

**Goal**: Create bots that aren't fucking suicidal idiots and can survive on de_survivor without environmental deaths, then progressively learn zombie survival strategies.
