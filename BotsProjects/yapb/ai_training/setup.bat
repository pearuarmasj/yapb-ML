@echo off
echo 🤖 CS 1.6 Proper ML Bot Setup
echo ================================
echo.

echo 📦 Installing Python dependencies...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully!
echo.

echo 🧪 Running system tests...
python test_ml_system.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ System tests failed!
    echo Check the errors above and fix them.
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo To start training:
echo   1. Start Counter-Strike 1.6
echo   2. Join de_survivor map
echo   3. Run: python train_ml_bot.py
echo.
pause
