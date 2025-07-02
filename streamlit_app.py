import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from scipy.sparse import load_npz
from surprise import dump
import plotly.express as px
import plotly.graph_objects as go
from restaurant_recommender import RestaurantRecommender
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Restaurant Recommender System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    """Load the recommender system"""
    recommender = RestaurantRecommender()
    if recommender.load_models():
        return recommender
    else:
        st.error("Failed to load models. Please ensure all model files are present.")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ Restaurant Recommender System</h1>', unsafe_allow_html=True)
    
    # Load recommender
    recommender = load_recommender()
    if recommender is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎯 Recommendation Options")
    
    # Recommendation type
    rec_type = st.sidebar.selectbox(
        "Choose Recommendation Type",
        ["Collaborative Filtering", "Content-Based", "Hybrid", "Popular Restaurants"]
    )
    
    # Number of recommendations
    k = st.sidebar.slider("Number of Recommendations", 3, 15, 5)
    
    # Main content
    if rec_type == "Collaborative Filtering":
        collaborative_filtering_page(recommender, k)
    elif rec_type == "Content-Based":
        content_based_page(recommender, k)
    elif rec_type == "Hybrid":
        hybrid_page(recommender, k)
    else:
        popular_restaurants_page(recommender, k)

def collaborative_filtering_page(recommender, k):
    """Collaborative filtering page"""
    st.header("🤝 Collaborative Filtering Recommendations")
    st.write("Get personalized recommendations based on similar users' preferences.")
    
    # User selection
    available_users = list(recommender.name_to_id.keys())
    selected_user = st.selectbox("Select a user:", available_users)
    
    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            recommendations = recommender.collaborative_filtering_recommendation(selected_user, k)
        
        if recommendations:
            st.success(f"Found {len(recommendations)} recommendations for {selected_user}")
            
            # Display recommendations
            for i, (restaurant_id, predicted_rating) in enumerate(recommendations, 1):
                restaurant_info = recommender.restaurants_features[
                    recommender.restaurants_features['restaurant_id'] == restaurant_id
                ]
                
                if not restaurant_info.empty:
                    title = restaurant_info['title'].iloc[0]
                    avg_rating = restaurant_info['stars_mean'].iloc[0]
                    review_count = restaurant_info['reviewCount_actual'].iloc[0]
                    
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{i}. {title}**")
                            st.write(f"📍 Location: {restaurant_info['city'].iloc[0]}")
                        with col2:
                            st.metric("Predicted Rating", f"{predicted_rating:.2f}")
                        with col3:
                            st.metric("Avg Rating", f"{avg_rating:.2f}")
                        
                        # Progress bar for predicted rating
                        st.progress(min(predicted_rating / 5.0, 1.0))
                        st.divider()
        else:
            st.warning("No recommendations found for this user.")

def content_based_page(recommender, k):
    """Content-based filtering page"""
    st.header("📝 Content-Based Recommendations")
    st.write("Find restaurants similar to a selected restaurant based on content analysis.")
    
    # Restaurant selection
    available_restaurants = recommender.restaurants_features[['restaurant_id', 'title', 'city']].drop_duplicates()
    selected_restaurant = st.selectbox(
        "Select a restaurant to find similar ones:",
        available_restaurants['title'].tolist()
    )
    
    if st.button("Find Similar Restaurants"):
        with st.spinner("Finding similar restaurants..."):
            restaurant_id = available_restaurants[
                available_restaurants['title'] == selected_restaurant
            ]['restaurant_id'].iloc[0]
            
            recommendations = recommender.content_based_recommendation(restaurant_id, k)
        
        if recommendations:
            st.success(f"Found {len(recommendations)} similar restaurants to {selected_restaurant}")
            
            # Display recommendations
            for i, (restaurant_id, title, similarity_score) in enumerate(recommendations, 1):
                restaurant_info = recommender.restaurants_features[
                    recommender.restaurants_features['restaurant_id'] == restaurant_id
                ]
                
                if not restaurant_info.empty:
                    avg_rating = restaurant_info['stars_mean'].iloc[0]
                    review_count = restaurant_info['reviewCount_actual'].iloc[0]
                    city = restaurant_info['city'].iloc[0]
                    
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{i}. {title}**")
                            st.write(f"📍 Location: {city}")
                        with col2:
                            st.metric("Similarity", f"{similarity_score:.3f}")
                        with col3:
                            st.metric("Avg Rating", f"{avg_rating:.2f}")
                        
                        # Progress bar for similarity
                        st.progress(similarity_score)
                        st.divider()
        else:
            st.warning("No similar restaurants found.")

def hybrid_page(recommender, k):
    """Hybrid recommendations page"""
    st.header("🔄 Hybrid Recommendations")
    st.write("Combine collaborative and content-based filtering for better recommendations.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User selection
        available_users = list(recommender.name_to_id.keys())
        selected_user = st.selectbox("Select a user:", available_users, key="hybrid_user")
    
    with col2:
        # Restaurant selection for content-based
        available_restaurants = recommender.restaurants_features[['restaurant_id', 'title']].drop_duplicates()
        selected_restaurant = st.selectbox(
            "Select a restaurant for content similarity:",
            available_restaurants['title'].tolist(),
            key="hybrid_restaurant"
        )
    
    # Alpha parameter
    alpha = st.slider("Weight for collaborative filtering (alpha)", 0.0, 1.0, 0.5, 0.1)
    
    if st.button("Get Hybrid Recommendations"):
        with st.spinner("Generating hybrid recommendations..."):
            restaurant_id = available_restaurants[
                available_restaurants['title'] == selected_restaurant
            ]['restaurant_id'].iloc[0]
            
            cf_recs, cb_recs = recommender.hybrid_recommendation(selected_user, restaurant_id, k, alpha)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤝 Collaborative Filtering")
            if cf_recs:
                for i, (restaurant_id, predicted_rating) in enumerate(cf_recs[:k//2], 1):
                    restaurant_info = recommender.restaurants_features[
                        recommender.restaurants_features['restaurant_id'] == restaurant_id
                    ]
                    if not restaurant_info.empty:
                        title = restaurant_info['title'].iloc[0]
                        st.write(f"{i}. {title} (Rating: {predicted_rating:.2f})")
            else:
                st.write("No collaborative recommendations available.")
        
        with col2:
            st.subheader("📝 Content-Based")
            if cb_recs:
                for i, (restaurant_id, title, similarity) in enumerate(cb_recs[:k//2], 1):
                    st.write(f"{i}. {title} (Similarity: {similarity:.3f})")
            else:
                st.write("No content-based recommendations available.")

def popular_restaurants_page(recommender, k):
    """Popular restaurants page"""
    st.header("⭐ Popular Restaurants")
    st.write("Discover the most popular restaurants based on ratings and review counts.")
    
    if st.button("Show Popular Restaurants"):
        with st.spinner("Loading popular restaurants..."):
            popular_restaurants = recommender.get_popular_restaurants(k)
        
        if popular_restaurants:
            st.success(f"Found {len(popular_restaurants)} popular restaurants")
            
            # Create a DataFrame for better display
            df_popular = pd.DataFrame(popular_restaurants)
            
            # Display as a table
            st.dataframe(
                df_popular[['title', 'stars_mean', 'reviewCount_actual']].rename(columns={
                    'title': 'Restaurant',
                    'stars_mean': 'Average Rating',
                    'reviewCount_actual': 'Review Count'
                }),
                use_container_width=True
            )
            
            # Create visualization
            fig = px.bar(
                df_popular,
                x='title',
                y='stars_mean',
                title="Average Ratings of Popular Restaurants",
                labels={'title': 'Restaurant', 'stars_mean': 'Average Rating'},
                color='reviewCount_actual',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No popular restaurants found.")

def analytics_page(recommender):
    """Analytics page"""
    st.header("📊 Analytics Dashboard")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Restaurants", len(recommender.restaurants_features))
    
    with col2:
        st.metric("Total Users", len(recommender.name_to_id))
    
    with col3:
        avg_rating = recommender.restaurants_features['stars_mean'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}")
    
    with col4:
        total_reviews = recommender.restaurants_features['reviewCount_actual'].sum()
        st.metric("Total Reviews", f"{total_reviews:,}")
    
    # Rating distribution
    st.subheader("Rating Distribution")
    fig = px.histogram(
        recommender.restaurants_features,
        x='stars_mean',
        nbins=20,
        title="Distribution of Average Restaurant Ratings"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top cities
    st.subheader("Restaurants by City")
    city_counts = recommender.restaurants_features['city'].value_counts().head(10)
    fig = px.bar(
        x=city_counts.index,
        y=city_counts.values,
        title="Number of Restaurants by City"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 