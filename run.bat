@echo off
echo 🍽️ Restaurant Recommendation System
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python detected
echo.

REM Check if data file exists
if not exist "restaurant_data.xlsx" (
    echo ❌ restaurant_data.xlsx not found!
    echo Please place your data file in this directory
    pause
    exit /b 1
)

echo ✅ Data file found
echo.

REM Install dependencies
echo 🔄 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

REM Run training
echo 🔄 Training models...
python run_training.py
if errorlevel 1 (
    echo ❌ Training failed
    pause
    exit /b 1
)

echo ✅ Training completed
echo.

REM Start Streamlit
echo 🚀 Starting Streamlit app...
echo The app will be available at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run streamlit_app.py

pause 