#!/usr/bin/env python3
"""
Training script for the Restaurant Recommendation System
"""

import sys
import os
from restaurant_recommender import RestaurantRecommender

def main():
    print("🍽️ Restaurant Recommendation System - Training")
    print("=" * 50)
    
    # Check if data file exists
    if not os.path.exists('restaurant_data.xlsx'):
        print("❌ Error: restaurant_data.xlsx not found!")
        print("Please ensure the data file is in the current directory.")
        sys.exit(1)
    
    # Initialize recommender
    print("📊 Initializing recommender...")
    recommender = RestaurantRecommender()
    
    # Load data
    print("📁 Loading data...")
    if not recommender.load_data():
        print("❌ Failed to load data!")
        sys.exit(1)
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    try:
        recommender.preprocess_data()
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        sys.exit(1)
    
    # Create features
    print("⚙️ Creating features...")
    try:
        recommender.create_features()
    except Exception as e:
        print(f"❌ Error creating features: {e}")
        sys.exit(1)
    
    # Train models
    print("🤖 Training models...")
    try:
        recommender.train_models()
    except Exception as e:
        print(f"❌ Error training models: {e}")
        sys.exit(1)
    
    # Create content-based features
    print("📝 Creating content-based features...")
    try:
        recommender.create_content_based_features()
    except Exception as e:
        print(f"❌ Error creating content-based features: {e}")
        sys.exit(1)
    
    # Create restaurant features
    print("🏪 Creating restaurant features...")
    try:
        recommender.create_restaurant_features()
    except Exception as e:
        print(f"❌ Error creating restaurant features: {e}")
        sys.exit(1)
    
    # Save models
    print("💾 Saving models...")
    try:
        recommender.save_models()
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        sys.exit(1)
    
    print("\n🎉 Training completed successfully!")
    print("You can now run the Streamlit app with: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 