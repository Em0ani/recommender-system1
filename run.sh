#!/bin/bash

echo "🍽️ Restaurant Recommendation System"
echo "===================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "✅ Python detected"
echo

# Check if data file exists
if [ ! -f "restaurant_data.xlsx" ]; then
    echo "❌ restaurant_data.xlsx not found!"
    echo "Please place your data file in this directory"
    exit 1
fi

echo "✅ Data file found"
echo

# Install dependencies
echo "🔄 Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"
echo

# Run training
echo "🔄 Training models..."
python3 run_training.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed"
    exit 1
fi

echo "✅ Training completed"
echo

# Start Streamlit
echo "🚀 Starting Streamlit app..."
echo "The app will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo
streamlit run streamlit_app.py 