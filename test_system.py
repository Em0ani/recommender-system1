#!/usr/bin/env python3
"""
Test script for the Restaurant Recommendation System
"""

import sys
import os
from restaurant_recommender import RestaurantRecommender

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing data loading...")
    recommender = RestaurantRecommender()
    
    if not os.path.exists('restaurant_data.xlsx'):
        print("❌ Data file not found")
        return False
    
    if recommender.load_data():
        print("✅ Data loading successful")
        return True
    else:
        print("❌ Data loading failed")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("🧪 Testing model loading...")
    recommender = RestaurantRecommender()
    
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
        print(f"❌ Missing files: {missing_files}")
        return False
    
    if recommender.load_models():
        print("✅ Model loading successful")
        return True
    else:
        print("❌ Model loading failed")
        return False

def test_recommendations():
    """Test recommendation functionality"""
    print("🧪 Testing recommendations...")
    recommender = RestaurantRecommender()
    
    if not recommender.load_models():
        print("❌ Cannot test recommendations - models not loaded")
        return False
    
    try:
        # Test collaborative filtering
        if recommender.name_to_id:
            test_user = list(recommender.name_to_id.keys())[0]
            cf_recs = recommender.collaborative_filtering_recommendation(test_user, 3)
            print(f"✅ Collaborative filtering: {len(cf_recs)} recommendations")
        
        # Test content-based filtering
        if recommender.restaurants_features is not None and not recommender.restaurants_features.empty:
            test_restaurant = recommender.restaurants_features['restaurant_id'].iloc[0]
            cb_recs = recommender.content_based_recommendation(test_restaurant, 3)
            print(f"✅ Content-based filtering: {len(cb_recs)} recommendations")
        
        # Test popular restaurants
        popular = recommender.get_popular_restaurants(3)
        print(f"✅ Popular restaurants: {len(popular)} recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ Recommendation testing failed: {e}")
        return False

def main():
    print("🧪 Restaurant Recommendation System - Testing")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
        ("Recommendations", test_recommendations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 