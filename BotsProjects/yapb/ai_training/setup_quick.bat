@echo off
echo 🎯 CS 1.6 de_survivor ML Bot Setup
echo ====================================
echo This will install the required Python packages
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip not found! Please install pip first.
    pause
    exit /b 1
)

echo ✅ pip found
echo.

echo 📦 Installing required packages...
echo This may take a few minutes...
echo.

REM Install packages from requirements.txt
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Installation failed! Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ✅ Installation completed successfully!
echo.
echo 🎯 NEXT STEPS:
echo 1. Start Counter-Strike 1.6
echo 2. Load de_survivor map
echo 3. Run: python launcher.py
echo.
echo 🎮 Have fun learning de_survivor!
pause
