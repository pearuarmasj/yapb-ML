# 🎯 CS 1.6 de_survivor ML Bot

**Learn de_survivor map with AI controlling YOUR player!**

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   # Run the setup script
   setup_quick.bat
   
   # OR manually install
   pip install -r requirements.txt
   ```

2. **Start Counter-Strike 1.6**
   - Load **de_survivor** map
   - Make sure you can move around normally
   - Keep CS 1.6 as the active window

3. **Launch the System**
   ```bash
   python launcher.py
   ```

4. **First Time Setup**
   - Choose option **1. Test System** first
   - If tests pass, choose **2. Start Learning**
   - Watch your player learn the map!

## 🎮 What This Does

- **Controls YOUR player** (not a separate bot)
- **Learns to navigate** de_survivor map
- **Uses reinforcement learning** to improve over time
- **Captures screen** for visual input
- **Reads game memory** for position/health data
- **Saves progress** automatically

## 🗺️ de_survivor Learning Goals

The AI learns to:
- ✅ Navigate different areas of the map
- ✅ Avoid falling off bridges
- ✅ Find paths between zones
- ✅ Explore efficiently
- ✅ Handle complex terrain

### Map Zones Tracked:
- **Spawn Areas**: T and CT spawn points
- **Bridge**: Main bridge area
- **Tower**: Sniper tower
- **Underground**: Lower tunnels
- **Rooftop**: Upper level areas
- **Center**: Main courtyard

## 📊 Progress Tracking

The system automatically:
- Saves Q-learning table every 10 episodes
- Plots learning curves
- Tracks exploration map
- Records position history
- Exports training data

**Files Created:**
- `data/de_survivor_learning_*.pkl` - Q-table saves
- `data/de_survivor_learning_*.json` - Statistics
- `data/training_progress_*.png` - Learning plots

## 🔧 Troubleshooting

**"Can't find CS 1.6"**
- Make sure CS 1.6 is running
- Try running CS 1.6 as administrator
- Check window title matches "Counter-Strike"

**"Memory reading failed"**
- Run the offset finder: `python offset_finder.py`
- Try different CS 1.6 version/build
- Some versions may need offset adjustments

**"Bot not moving"**
- Make sure CS 1.6 is the active window
- Check that manual movement works first
- Try running as administrator

**"Actions not working"**
- Verify keyboard layout (WASD)
- Check for conflicting key bindings
- Try the demo first: `python demo_bot_control.py`

## ⚙️ Configuration

Edit `train_ml_bot.py` to adjust:

```python
# Learning parameters
learning_rate = 0.1      # How fast to learn (0.01-0.3)
epsilon = 0.9           # Exploration rate (0.1-0.9)
epsilon_decay = 0.995   # How fast to reduce exploration

# Episode length
max_steps = 500         # Steps per episode

# Training duration
episodes = 50           # Number of episodes
```

## 🎯 Advanced Usage

**Custom Rewards:**
- Edit `_calculate_reward()` in `working_ml_bot.py`
- Add custom objectives (reach specific areas, etc.)

**Different Maps:**
- Update zone definitions in `train_ml_bot.py`
- Adjust landmark positions for your map

**Better AI:**
- Replace Q-learning with neural networks
- Add computer vision processing
- Implement more complex action spaces

## 📈 Understanding Results

**Good Learning Signs:**
- Episode rewards increasing over time
- Exploration map covers most areas
- Q-table grows to ~10+ states
- Bot survives longer each episode

**Problem Signs:**
- Rewards always negative
- Bot stuck in one area
- No exploration happening
- Health always dropping

## 🤝 Contributing

Want to improve the system?
- Add support for other maps
- Implement better learning algorithms
- Improve computer vision
- Add multi-bot coordination

## 📝 License

This project builds upon YaPB (MIT License) and adds ML components for educational purposes.

---

**🎮 Have fun watching your AI learn de_survivor!**
