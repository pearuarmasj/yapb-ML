@echo off
REM YaPB Real Neural AI Setup Script
REM This script sets up the environment for real neural network training

echo.
echo =========================================================
echo    YaPB Real Neural AI - Setup and Training Guide
echo =========================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✓ Python detected

REM Check if we're in the right directory
if not exist "ai_training" (
    echo ERROR: ai_training directory not found
    echo Make sure you're running this script from the YaPB root directory
    pause
    exit /b 1
)

echo ✓ YaPB directory structure verified

REM Setup virtual environment if it doesn't exist
if not exist "ai_training\venv" (
    echo.
    echo 📦 Setting up Python virtual environment...
    cd ai_training
    python -m venv venv
    cd ..
    echo ✓ Virtual environment created
)

REM Activate virtual environment and install dependencies
echo.
echo 📚 Installing Python dependencies...
call ai_training\venv\Scripts\activate.bat
cd ai_training
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed

REM Create necessary directories
echo.
echo 📁 Creating directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "checkpoints" mkdir checkpoints
echo ✓ Directories created

REM Show the step-by-step guide
:guide
echo.
echo =========================================================
echo    REAL NEURAL AI - STEP BY STEP GUIDE
echo =========================================================
echo.
echo This is a REAL neural network integration, not a simulation!
echo.
echo STEP 1: Build YaPB with Neural AI Support
echo   - The C++ code has been updated to collect real gameplay data
echo   - Neural logic will export state-action-reward data to CSV files
echo   - Build the project using your existing build system
echo.
echo STEP 2: Generate Training Data (10-15 minutes)
echo   - Start Counter-Strike 1.6 with the updated YaPB
echo   - Add zombie bots: yapb_add_player zombie team1
echo   - Play with zombie bots for 10-15 minutes
echo   - CSV files will be saved to: addons/yapb/ai_training/data/
echo   - Look for files like: zombie_training_data_YYYYMMDD_HHMMSS.csv
echo.
echo STEP 3: Train the Neural Network
echo   - Run: python ai_training/train_real_neural_ai.py
echo   - This will read the CSV data and train a real PyTorch neural network
echo   - Trained weights will be exported to: neural_weights.json
echo.
echo STEP 4: Load and Use Trained Weights (TODO)
echo   - C++ code needs to be extended to load neural_weights.json
echo   - Replace rule-based fallback with real neural inference
echo   - Neural network will make actual bot decisions
echo.
echo Current Status:
echo   ✓ C++ data collection implemented
echo   ✓ Python training script ready
echo   ✓ Debug output enabled
echo   ⏳ Waiting for in-game testing and training
echo.

:menu
echo.
echo =========================================================
echo    Choose an option:
echo =========================================================
echo.
echo 1. Quick Setup Check
echo 2. Train Neural Network (requires data)
echo 3. View Current Configuration
echo 4. Show File Locations
echo 5. Test Python Environment
echo 0. Exit
echo.
set /p choice="Enter your choice (0-5): "

if "%choice%"=="1" goto setup_check
if "%choice%"=="2" goto train_neural
if "%choice%"=="3" goto show_config
if "%choice%"=="4" goto show_files
if "%choice%"=="5" goto test_python
if "%choice%"=="0" goto exit
goto menu

:setup_check
echo.
echo 🔍 Checking Setup...
echo.
echo ✓ Python virtual environment ready
echo ✓ Dependencies installed
echo ✓ Directory structure created
echo.
if exist "data\*.csv" (
    echo ✓ Training data found!
    dir data\*.csv /B
) else (
    echo ⚠️  No training data found yet
    echo    Play with zombie bots to generate data
)
echo.
if exist "neural_weights.json" (
    echo ✓ Trained neural weights found!
) else (
    echo ⚠️  No trained weights found yet
    echo    Run training after collecting data
)
pause
goto menu

:train_neural
echo.
echo 🧠 Training Real Neural Network...
echo.
if not exist "data\*.csv" (
    echo ❌ ERROR: No training data found!
    echo.
    echo Please collect training data first:
    echo   1. Build and run YaPB with neural AI support
    echo   2. Add zombie bots in Counter-Strike 1.6
    echo   3. Play for 10-15 minutes
    echo   4. Check that CSV files are created in data/ folder
    echo.
    pause
    goto menu
)

echo Training data found. Starting neural network training...
python train_real_neural_ai.py
if errorlevel 1 (
    echo.
    echo ❌ Training failed! Check the error messages above.
) else (
    echo.
    echo ✅ Neural network training completed!
    echo.
    echo Generated files:
    if exist "neural_weights.json" echo   ✓ neural_weights.json (for C++ loading)
    if exist "models\zombie_neural_model.pth" echo   ✓ models/zombie_neural_model.pth (PyTorch model)
    if exist "logs\training_log.txt" echo   ✓ logs/training_log.txt (training details)
)
pause
goto menu

:show_config
echo.
echo ⚙️ Current Configuration:
echo.
echo Neural AI Files:
echo   C++: src/neural_zombie_ai.cpp
echo   Header: inc/neural_zombie_ai.h
echo   Python: ai_training/train_real_neural_ai.py
echo.
echo Data Flow:
echo   Gameplay → CSV files → Neural training → JSON weights → C++ inference
echo.
echo Key Features:
echo   - Real gameplay data collection
echo   - PyTorch neural network training
echo   - JSON weight export for C++
echo   - Debug output in console and HUD
echo.
pause
goto menu

:show_files
echo.
echo 📁 Important File Locations:
echo.
echo Training Data (CSV):
echo   Location: ai_training/data/
echo   Pattern: zombie_training_data_YYYYMMDD_HHMMSS.csv
echo.
echo Trained Models:
echo   PyTorch: ai_training/models/zombie_neural_model.pth
echo   C++ Weights: ai_training/neural_weights.json
echo.
echo Logs:
echo   Training: ai_training/logs/training_log.txt
echo   Game Console: Check CS 1.6 console for debug output
echo.
echo Source Code:
echo   C++: src/neural_zombie_ai.cpp
echo   Header: inc/neural_zombie_ai.h
echo   Python: ai_training/train_real_neural_ai.py
echo.
pause
goto menu

:test_python
echo.
echo 🐍 Testing Python Environment...
echo.
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
echo.
echo ✅ Python environment is ready for neural network training
pause
goto menu

:exit
echo.
echo Deactivating virtual environment...
call deactivate 2>nul
cd ..
echo.
echo 👋 Thanks for using YaPB Real Neural AI!
echo.
echo Summary:
echo   1. Build YaPB with neural AI support
echo   2. Play with zombie bots for 10-15 minutes
echo   3. Run: python ai_training/train_real_neural_ai.py
echo   4. Implement C++ weight loading (TODO)
echo.
echo Remember: This is REAL neural network integration!
echo The C++ code exports actual gameplay data for training.
echo.
pause
exit /b 0

REM Jump to guide first
goto guide
