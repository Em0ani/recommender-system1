#!/usr/bin/env python3
"""
Deployment script for the Restaurant Recommendation System
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Restaurant Recommendation System - Deployment")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required!")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies!")
        sys.exit(1)
    
    # Check if data file exists
    if not os.path.exists('restaurant_data.xlsx'):
        print("❌ restaurant_data.xlsx not found!")
        print("Please place your data file in the current directory.")
        sys.exit(1)
    
    # Run training
    if not run_command("python run_training.py", "Training models"):
        print("❌ Training failed!")
        sys.exit(1)
    
    # Check if models were created
    required_files = [
        'svd_model.pkl',
        'metadata_df.parquet',
        'restaurants_features.parquet',
        'tfidf_vectorizer.joblib',
        'tfidf_matrix.npz',
        'cosine_sim_matrix.npz',
        'content_indices.pkl',
        'name_to_id.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        sys.exit(1)
    
    print("✅ All model files created successfully")
    
    # Start Streamlit app
    print("\n🎉 Deployment completed successfully!")
    print("Starting Streamlit app...")
    print("The app will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run("streamlit run streamlit_app.py", shell=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

if __name__ == "__main__":
    main() 