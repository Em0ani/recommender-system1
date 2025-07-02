import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD, NMF, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
import pickle
import joblib
from scipy.sparse import save_npz, load_npz
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
except:
    pass

class RestaurantRecommender:
    def __init__(self, file_path='restaurant_data.xlsx'):
        self.file_path = file_path
        self.metadata_df = None
        self.customers_df = None
        self.reviews_df = None
        self.full_df = None
        self.user_item_matrix = None
        self.svd_model = None
        self.nmf_model = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosime_sim = None
        self.indices = None
        self.name_to_id = None
        self.restaurants_features = None
        
    def load_data(self):
        """Load data from Excel file"""
        try:
            self.metadata_df = pd.read_excel(self.file_path, sheet_name='Metadata', index_col=0)
            self.customers_df = pd.read_excel(self.file_path, sheet_name='Customers', index_col=0)
            self.reviews_df = pd.read_excel(self.file_path, sheet_name='Reviews', index_col=0)
            print("✅ Data loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data"""
        # Clean text data
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\d+", "", text)
            tokens = text.split()
            tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
            return " ".join(tokens)
        
        # Clean review text
        self.reviews_df["cleaned_text"] = self.reviews_df["textTranslated"].fillna("").apply(clean_text)
        
        # Drop unnecessary columns
        columns_to_drop = ['likesCount', 'reviewContext/Service']
        for col in columns_to_drop:
            if col in self.reviews_df.columns:
                self.reviews_df = self.reviews_df.drop(col, axis=1)
        
        # Create review context average
        rating_columns = [
            'reviewDetailedRating/Service',
            'reviewDetailedRating/Atmosphere', 
            'reviewDetailedRating/Food',
            'reviewDetailedRating/Location'
        ]
        
        available_columns = [col for col in rating_columns if col in self.reviews_df.columns]
        if available_columns:
            self.reviews_df['review_context_avg'] = self.reviews_df[available_columns].mean(axis=1)
            self.reviews_df = self.reviews_df.drop(available_columns, axis=1)
        
        # Merge dataframes
        reviews_customers = self.reviews_df.merge(
            self.customers_df,
            how='left',
            left_on='reviewerId',
            right_on='name'
        )
        
        self.full_df = reviews_customers.merge(
            self.metadata_df.reset_index(),
            how='left',
            on='restaurant_id'
        )
        
        print("✅ Data preprocessing completed")
    
    def create_features(self):
        """Create feature matrices"""
        # TF-IDF for text
        self.tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X_text = self.tfidf.fit_transform(self.reviews_df["cleaned_text"])
        
        # Categorical features
        categorical_features = ["isLocalGuide", "city"]
        self.full_df["isLocalGuide"] = self.full_df["isLocalGuide"].fillna(False).astype(int)
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_cats = encoder.fit_transform(self.full_df[categorical_features].fillna("Unknown"))
        
        # Numerical features
        numerical_features = ["stars", "totalScore", "reviewsCount"]
        scaler = StandardScaler()
        scaled_nums = scaler.fit_transform(self.full_df[numerical_features].fillna(0))
        
        # Combine features
        X = hstack([
            X_text,
            csr_matrix(encoded_cats),
            csr_matrix(scaled_nums)
        ])
        
        # Create user-item matrix
        self.user_item_matrix = (
            self.full_df.groupby(['reviewerId', 'restaurant_id'])['stars']
            .mean()
            .unstack()
            .fillna(0)
        )
        
        print("✅ Features created successfully")
        return X
    
    def train_models(self):
        """Train recommendation models"""
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 5))
        
        # Map names to IDs for Surprise
        self.name_to_id = {name: i for i, name in enumerate(self.full_df['name_x'].unique())}
        
        surprise_df = self.full_df[['name_x', 'restaurant_id', 'stars']].copy()
        surprise_df['user_id'] = surprise_df['name_x'].map(self.name_to_id)
        surprise_df_for_surprise = surprise_df[['user_id', 'restaurant_id', 'stars']].dropna(subset=['user_id'])
        
        # Load dataset
        data = Dataset.load_from_df(surprise_df_for_surprise, reader)
        trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
        
        # Train SVD model
        self.svd_model = SVD(random_state=42)
        self.svd_model.fit(trainset)
        
        # Train NMF model
        try:
            self.nmf_model = NMF(n_factors=15, random_state=42)
            self.nmf_model.fit(trainset)
        except:
            print("⚠️ NMF model training failed, using SVD only")
        
        # Evaluate models
        predictions_svd = self.svd_model.test(testset)
        rmse_svd = accuracy.rmse(predictions_svd)
        mae_svd = accuracy.mae(predictions_svd)
        
        print(f"✅ SVD Model - RMSE: {rmse_svd:.4f}, MAE: {mae_svd:.4f}")
        
        if self.nmf_model:
            predictions_nmf = self.nmf_model.test(testset)
            rmse_nmf = accuracy.rmse(predictions_nmf)
            mae_nmf = accuracy.mae(predictions_nmf)
            print(f"✅ NMF Model - RMSE: {rmse_nmf:.4f}, MAE: {mae_nmf:.4f}")
    
    def create_content_based_features(self):
        """Create content-based features"""
        # Prepare content for TF-IDF
        self.full_df['textTranslated_str'] = self.full_df['textTranslated'].fillna('').astype(str)
        self.full_df['sentiment_str'] = self.full_df['sentiment'].fillna('').astype(str)
        
        self.full_df['content'] = (
            self.full_df['textTranslated_str'] + " " +
            self.full_df['sentiment_str'] + " " +
            self.full_df['title'].fillna('').astype(str) + " " +
            self.full_df['city'].fillna('').astype(str)
        )
        
        # Create restaurant content
        restaurant_content_df = self.full_df.groupby('restaurant_id')['content'].agg(lambda x: ' '.join(x)).reset_index()
        
        # TF-IDF for content
        content_tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = content_tfidf.fit_transform(restaurant_content_df['content'])
        
        # Cosine similarity
        self.cosime_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create indices mapping
        self.indices = pd.Series(restaurant_content_df.index, index=restaurant_content_df['restaurant_id']).drop_duplicates()
        
        print("✅ Content-based features created")
    
    def create_restaurant_features(self):
        """Create restaurant features dataframe"""
        df = self.full_df
        
        # Aggregate scores
        agg_scores = df.groupby('restaurant_id')['stars'].agg(['mean', 'median', 'std']).reset_index()
        agg_scores.rename(columns={
            'mean': 'stars_mean',
            'median': 'stars_median',
            'std': 'stars_std'
        }, inplace=True)
        
        # Review counts
        review_counts = df.groupby('restaurant_id').size().reset_index(name='reviewCount_actual')
        
        # Restaurant features
        restaurants_features = df[['restaurant_id', 'reviewsCount', 'totalScore', 'title',
                                   'location/lat', 'location/lng', 'city']].drop_duplicates()
        
        self.restaurants_features = restaurants_features.merge(agg_scores, on='restaurant_id', how='left')
        self.restaurants_features = self.restaurants_features.merge(review_counts, on='restaurant_id', how='left')
        
        print("✅ Restaurant features created")
    
    def collaborative_filtering_recommendation(self, user_name, k=5):
        """Generate collaborative filtering recommendations"""
        try:
            user_id_internal = self.name_to_id.get(user_name)
            if user_id_internal is None:
                return []
            
            all_restaurant_ids = self.full_df['restaurant_id'].unique()
            rated_restaurants = self.full_df[
                self.full_df['name_x'] == user_name
            ]['restaurant_id'].tolist()
            
            unrated_restaurants = [
                restaurant_id for restaurant_id in all_restaurant_ids
                if restaurant_id not in rated_restaurants
            ]
            
            predictions = []
            for restaurant_id in unrated_restaurants:
                pred = self.svd_model.predict(user_id_internal, restaurant_id)
                predictions.append((restaurant_id, pred.est))
            
            recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:k]
            
            return recommendations
            
        except Exception as e:
            print(f"Error in collaborative filtering: {e}")
            return []
    
    def content_based_recommendation(self, restaurant_id, k=5):
        """Generate content-based recommendations"""
        try:
            if restaurant_id not in self.indices:
                return []
            
            idx = self.indices[restaurant_id]
            sim_scores = list(enumerate(self.cosime_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]
            
            recommendations = []
            for i, score in sim_scores:
                rec_restaurant_id = self.full_df.groupby('restaurant_id')['content'].agg(lambda x: ' '.join(x)).reset_index()['restaurant_id'].iloc[i]
                restaurant_info = self.restaurants_features[self.restaurants_features['restaurant_id'] == rec_restaurant_id]
                restaurant_title = restaurant_info['title'].iloc[0] if not restaurant_info.empty else f"Unknown Title (ID: {rec_restaurant_id})"
                recommendations.append((rec_restaurant_id, restaurant_title, score))
            
            return recommendations
            
        except Exception as e:
            print(f"Error in content-based filtering: {e}")
            return []
    
    def hybrid_recommendation(self, user_name, restaurant_id, k=5, alpha=0.5):
        """Generate hybrid recommendations"""
        cf_recs = self.collaborative_filtering_recommendation(user_name, k)
        cb_recs = self.content_based_recommendation(restaurant_id, k)
        
        return cf_recs, cb_recs
    
    def get_popular_restaurants(self, k=5):
        """Get popular restaurants based on average rating and review count"""
        popular = self.restaurants_features.copy()
        popular['popularity_score'] = popular['stars_mean'] * np.log(popular['reviewCount_actual'] + 1)
        popular = popular.sort_values('popularity_score', ascending=False).head(k)
        
        return popular[['restaurant_id', 'title', 'stars_mean', 'reviewCount_actual']].to_dict('records')
    
    def save_models(self):
        """Save all models and data"""
        try:
            # Save Surprise models
            from surprise import dump
            dump.dump('svd_model.pkl', algo=self.svd_model)
            if self.nmf_model:
                dump.dump('nmf_model.pkl', algo=self.nmf_model)
            
            # Save data
            self.metadata_df.to_parquet('metadata_df.parquet')
            self.restaurants_features.to_parquet('restaurants_features.parquet')
            
            # Save TF-IDF
            joblib.dump(self.tfidf, 'tfidf_vectorizer.joblib')
            save_npz('tfidf_matrix.npz', self.tfidf_matrix)
            save_npz('cosine_sim_matrix.npz', self.cosime_sim)
            
            # Save mappings
            with open('content_indices.pkl', 'wb') as f:
                pickle.dump(self.indices, f)
            with open('name_to_id.pkl', 'wb') as f:
                pickle.dump(self.name_to_id, f)
            
            print("✅ All models and data saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def load_models(self):
        """Load saved models and data"""
        try:
            # Load Surprise models
            from surprise import dump
            _, self.svd_model = dump.load('svd_model.pkl')
            try:
                _, self.nmf_model = dump.load('nmf_model.pkl')
            except:
                self.nmf_model = None
            
            # Load data
            self.metadata_df = pd.read_parquet('metadata_df.parquet')
            self.restaurants_features = pd.read_parquet('restaurants_features.parquet')
            
            # Load TF-IDF
            self.tfidf = joblib.load('tfidf_vectorizer.joblib')
            self.tfidf_matrix = load_npz('tfidf_matrix.npz')
            self.cosime_sim = load_npz('cosine_sim_matrix.npz')
            
            # Load mappings
            with open('content_indices.pkl', 'rb') as f:
                self.indices = pickle.load(f)
            with open('name_to_id.pkl', 'rb') as f:
                self.name_to_id = pickle.load(f)
            
            print("✅ All models and data loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

def main():
    """Main function to train and save the model"""
    recommender = RestaurantRecommender()
    
    if recommender.load_data():
        recommender.preprocess_data()
        recommender.create_features()
        recommender.train_models()
        recommender.create_content_based_features()
        recommender.create_restaurant_features()
        recommender.save_models()
        print("🎉 Model training and saving completed!")
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main() 