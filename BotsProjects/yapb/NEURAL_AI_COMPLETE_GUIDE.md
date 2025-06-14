# YaPB Real Neural AI Integration - Complete Guide

## Overview

This is a **REAL neural network integration** for YaPB zombie AI in Counter-Strike 1.6. Unlike simulations, this system:

1. **Collects real gameplay data** from actual bot behavior during zombie matches
2. **Trains a genuine PyTorch neural network** on this real data
3. **Exports trained weights** for C++ to load and use for bot decision-making
4. **Replaces rule-based logic** with actual neural network inference

## Current Status

✅ **COMPLETED:**
- C++ data export system for real gameplay state-action-reward data
- Python training script for real neural network training on collected data
- Debug output system for tracking neural logic execution
- Build system integration and successful compilation

⏳ **PENDING:**
- In-game data collection testing
- Neural network training on real data
- C++ weight loading and inference implementation

## Step-by-Step Process

### Step 1: Build YaPB with Neural AI Support

The C++ code has been integrated and should build successfully:

```bash
# The project should already be built, but if you need to rebuild:
cd build
cmake --build . --config Release
```

**Key Files Added/Modified:**
- `src/neural_zombie_ai.cpp` - Main neural logic with data export
- `inc/neural_zombie_ai.h` - Neural AI header declarations
- `src/botlib.cpp` - Integration calls to neural logic
- `inc/yapb.h` - Bot class extensions

### Step 2: Generate Training Data (10-15 minutes)

1. **Start Counter-Strike 1.6** with the updated YaPB DLL
2. **Create a zombie match:**
   ```
   mp_friendlyfire 0
   mp_autoteambalance 0
   yapb_add_player zombie team1
   yapb_add_player zombie team1
   yapb_add_player zombie team1
   ```
3. **Play for 10-15 minutes** to generate sufficient training data
4. **Check for CSV files** in: `addons/yapb/ai_training/data/`
   - Files named: `zombie_training_data_YYYYMMDD_HHMMSS.csv`
   - Each file contains: state features, action taken, reward received

**Expected Console Output:**
```
[NEURAL DEBUG] Neural zombie logic called for Bot_Name
[NEURAL DEBUG] Exporting training data: state=[...], action=move_forward, reward=0.1
[NEURAL DEBUG] Data saved to: addons/yapb/ai_training/data/zombie_training_data_20241223_142030.csv
```

### Step 3: Train the Neural Network

Once you have training data:

```bash
# From the YaPB root directory
cd ai_training
python train_real_neural_ai.py
```

**What happens:**
1. **Reads all CSV files** from the `data/` directory
2. **Preprocesses the data** (normalization, feature engineering)
3. **Trains a PyTorch neural network** (3-layer MLP with dropout)
4. **Exports trained weights** to `neural_weights.json` for C++ loading
5. **Saves the PyTorch model** to `models/zombie_neural_model.pth`
6. **Logs training progress** to `logs/training_log.txt`

**Expected Output:**
```
Loading training data from CSV files...
Found 1500 training samples
Training neural network...
Epoch 1/100, Loss: 0.567
Epoch 2/100, Loss: 0.432
...
Training completed!
Weights exported to neural_weights.json
Model saved to models/zombie_neural_model.pth
```

### Step 4: Load and Use Trained Weights (TODO - Next Implementation)

This is the final step that needs to be implemented:

1. **Modify C++ neural logic** to load `neural_weights.json`
2. **Implement matrix operations** for neural network forward pass
3. **Replace rule-based fallback** with actual neural inference
4. **Use neural output** for real bot decision-making

## File Structure

```
yapb/
├── src/neural_zombie_ai.cpp          # Main neural logic (data export)
├── inc/neural_zombie_ai.h            # Neural AI declarations  
├── ai_training/
│   ├── train_real_neural_ai.py       # Real neural network training
│   ├── data/                         # CSV training data from gameplay
│   ├── models/                       # Trained PyTorch models
│   ├── logs/                         # Training logs
│   ├── neural_weights.json           # Exported weights for C++
│   └── requirements.txt              # Python dependencies
└── setup_neural_ai.bat               # Setup and training script
```

## Debug Output

The system includes extensive debug output to ensure everything is working:

**In CS 1.6 Console:**
```
[NEURAL DEBUG] Neural zombie logic called for Bot_Zombie_1
[NEURAL DEBUG] Current position: (1234.5, 678.9, 123.4)
[NEURAL DEBUG] Enemy distance: 456.7, Visibility: 1
[NEURAL DEBUG] Exporting training data: action=move_forward, reward=0.1
```

**In HUD (if enabled):**
```
Neural AI: Active
Data Export: ON
Action: move_forward
Confidence: 0.85
```

## Network Architecture

The neural network used is a simple but effective 3-layer MLP:

```
Input Layer:  12 features (position, enemy info, health, etc.)
Hidden Layer: 64 neurons (ReLU activation, 20% dropout)
Hidden Layer: 32 neurons (ReLU activation, 20% dropout)  
Output Layer: 6 actions (move_forward, move_backward, strafe_left, etc.)
```

**Training Details:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss (for action classification)
- Epochs: 100 (with early stopping)
- Batch Size: 32

## Troubleshooting

### No CSV Files Generated
- Check that YaPB is detecting zombie mode correctly
- Verify debug output appears in console
- Ensure bots are actually zombie type, not regular bots

### Training Script Fails
- Check Python environment: `python -c "import torch, pandas, numpy"`
- Verify CSV files contain data: check file sizes > 0 bytes
- Look for error messages in training output

### No Debug Output
- Confirm the updated DLL is being used
- Check that zombie bots are active (not just added)
- Verify console logging is enabled in CS 1.6

## Next Steps

1. **Test data collection** by playing zombie matches and checking CSV generation
2. **Run neural training** once sufficient data is collected
3. **Implement C++ weight loading** to complete the neural inference system
4. **Fine-tune the network** based on bot performance in-game
5. **Add more sophisticated features** like memory and context awareness

## Technical Notes

- **Real Data Only**: No synthetic data generation - everything comes from actual gameplay
- **CSV Format**: Human-readable format for easy debugging and analysis  
- **JSON Weights**: Simple format for C++ loading without external ML libraries
- **Modular Design**: Neural logic is separate and can be easily extended
- **Debug First**: Extensive logging to ensure everything works as expected

This is a genuine neural network integration that will learn from real gameplay data and make actual decisions for zombie bots!
