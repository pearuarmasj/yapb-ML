#!/bin/bash
# YaPB Neural Network Massive Data Collection Setup
# This script sets up the bots to collect MASSIVE amounts of training data

echo "🚀 Setting up MASSIVE data collection mode..."
echo "📊 This will make ALL bots collect training data at 50x per second while playing normally"
echo ""

echo "Creating data collection directory structure..."
mkdir -p "ai_training/data"
mkdir -p "ai_training/models" 
mkdir -p "ai_training/logs"

echo "✅ Directories created"
echo ""

echo "🤖 COPY THIS DLL to your CS 1.6 addon directory:"
echo "   Source: c:\\Users\\pearu\\BotsProjects\\yapb\\build\\Release\\yapb.dll"
echo "   Target: F:\\SteamLibrary\\steamapps\\common\\Half-Life\\cstrike\\addons\\yapb\\bin\\yapb.dll"
echo ""

echo "🎮 CONSOLE COMMANDS to run in CS 1.6:"
echo ""
echo "// MASSIVE DATA COLLECTION MODE (all bots behave normally but record everything):"
echo "neural_data_collection 1         // Enable massive data collection (50x/sec per bot)"
echo "neural_use_for_decisions 0       // Disable neural decisions (let regular AI run)"
echo "neural_debug_output 1            // Show what's happening"
echo ""
echo "// Add lots of bots for massive data:"
echo "yb_quota 31                      // Maximum bots"
echo "yb_join_after_player 0           // Bots join immediately"
echo ""
echo "// Start a long match on any map:"
echo "mp_timelimit 0                   // No time limit"
echo "mp_maxrounds 0                   // No round limit" 
echo ""

echo "🔥 EXPECTED RESULTS:"
echo "• Each bot will generate ~50 training samples per second"
echo "• 31 bots = ~1,550 samples per second total"
echo "• In 10 minutes = ~930,000 training samples"
echo "• Each bot gets its own CSV file (no data corruption)"
echo "• Regular YaPB AI behavior (realistic training data)"
echo ""

echo "📈 AFTER DATA COLLECTION:"
echo "1. Stop the game"
echo "2. Run: python ai_training/train_real_neural_ai_fixed.py"
echo "3. Set: neural_data_collection 0, neural_use_for_decisions 1"
echo "4. Test the newly trained neural network!"
echo ""

echo "🎯 Ready to collect MASSIVE amounts of training data!"
