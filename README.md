# 🍽️ Restaurant Recommendation System

A comprehensive restaurant recommendation system that combines multiple recommendation algorithms including Collaborative Filtering, Content-Based Filtering, and Hybrid approaches.

## 🚀 Features

- **Collaborative Filtering**: Uses SVD and NMF algorithms to recommend restaurants based on similar users
- **Content-Based Filtering**: Recommends restaurants based on content similarity (reviews, sentiment, location)
- **Hybrid Recommendations**: Combines both approaches for better recommendations
- **Popular Restaurants**: Shows trending restaurants based on ratings and review counts
- **Interactive Web Interface**: Modern Streamlit app with beautiful visualizations
- **Analytics Dashboard**: Insights into restaurant data and user behavior

## 📋 Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd restaurant-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place your `restaurant_data.xlsx` file in the project directory
   - The Excel file should have three sheets: `Metadata`, `Customers`, and `Reviews`

## 🎯 Usage

### Training the Model

1. **Run the training script**
   ```bash
   python restaurant_recommender.py
   ```
   This will:
   - Load and preprocess your data
   - Train SVD and NMF models
   - Create content-based features
   - Save all models and data

### Running the Streamlit App

1. **Start the Streamlit application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:8501`
   - The app will automatically load

## 📊 Data Structure

Your Excel file should contain:

### Metadata Sheet
- `restaurant_id`: Unique identifier for each restaurant
- `title`: Restaurant name
- `city`: City location
- `location/lat`, `location/lng`: GPS coordinates
- `reviewsCount`: Number of reviews
- `totalScore`: Overall score

### Customers Sheet
- `name`: Customer name/ID
- `isLocalGuide`: Boolean indicating if user is a local guide

### Reviews Sheet
- `reviewerId`: Customer ID (should match name in Customers sheet)
- `restaurant_id`: Restaurant ID (should match Metadata sheet)
- `stars`: Rating (1-5)
- `textTranslated`: Review text
- `sentiment`: Sentiment analysis result
- Additional rating columns for different aspects

## 🎨 App Features

### 1. Collaborative Filtering
- Select a user from the dropdown
- Get personalized recommendations based on similar users
- View predicted ratings and restaurant details

### 2. Content-Based Filtering
- Select a restaurant to find similar ones
- Based on review content, sentiment, and location
- View similarity scores

### 3. Hybrid Recommendations
- Combine both collaborative and content-based approaches
- Adjustable weight parameter (alpha)
- Side-by-side comparison of both methods

### 4. Popular Restaurants
- View trending restaurants
- Based on average ratings and review counts
- Interactive visualizations

### 5. Analytics Dashboard
- Overview statistics
- Rating distributions
- Restaurant counts by city
- Interactive charts

## 🔧 Model Details

### Collaborative Filtering
- **SVD (Singular Value Decomposition)**: Matrix factorization approach
- **NMF (Non-negative Matrix Factorization)**: Alternative factorization method
- Uses Surprise library for efficient implementation

### Content-Based Filtering
- **TF-IDF Vectorization**: Text feature extraction
- **Cosine Similarity**: Similarity calculation
- **Sentiment Analysis**: Incorporates review sentiment

### Hybrid Approach
- **Weighted Combination**: Adjustable weights for each method
- **Fallback Mechanisms**: Popular restaurants for new users
- **Cold Start Handling**: Solutions for users with limited data

## 📈 Performance Metrics

The system evaluates models using:
- **RMSE**: Root Mean Square Error for rating prediction
- **MAE**: Mean Absolute Error for rating prediction
- **Precision@k**: Precision at k recommendations
- **Recall@k**: Recall at k recommendations
- **F1-Score@k**: Harmonic mean of precision and recall
- **MAP@k**: Mean Average Precision

## 🚨 Troubleshooting

### Common Issues

1. **Model loading fails**
   - Ensure all model files are present in the directory
   - Re-run the training script if files are missing

2. **Memory issues**
   - Reduce `max_features` in TF-IDF vectorizer
   - Use smaller datasets for testing

3. **Data format issues**
   - Check Excel file structure matches requirements
   - Ensure all required columns are present

4. **Dependency conflicts**
   - Use a virtual environment
   - Install exact versions from requirements.txt

### Performance Tips

- Use GPU acceleration for large datasets
- Implement caching for frequently accessed data
- Consider using sparse matrices for memory efficiency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Surprise library for collaborative filtering algorithms
- Streamlit for the web interface
- Plotly for interactive visualizations
- NLTK for natural language processing

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub

---

**Happy recommending! 🍽️✨** 