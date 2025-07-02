# 🚀 Deployment Guide

This guide covers deploying your Restaurant Recommendation System to various platforms.

## 📋 Prerequisites

- Git repository with your code
- Python 3.8+ environment
- Your `restaurant_data.xlsx` file

## 🐳 Docker Deployment

### Local Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t restaurant-recommender .
docker run -p 8501:8501 restaurant-recommender
```

### Docker Hub
```bash
# Build and push to Docker Hub
docker build -t yourusername/restaurant-recommender .
docker push yourusername/restaurant-recommender

# Run from Docker Hub
docker run -p 8501:8501 yourusername/restaurant-recommender
```

## ☁️ Cloud Deployment

### Heroku
1. **Install Heroku CLI**
2. **Login to Heroku**
   ```bash
   heroku login
   ```
3. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```
4. **Set buildpacks**
   ```bash
   heroku buildpacks:set heroku/python
   ```
5. **Deploy**
   ```bash
   git push heroku main
   ```

### Railway
1. **Connect your GitHub repository**
2. **Add environment variables**
   - `PORT=8501`
3. **Deploy automatically**

### Render
1. **Connect your GitHub repository**
2. **Configure build settings**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
3. **Deploy**

### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/restaurant-recommender
gcloud run deploy --image gcr.io/PROJECT-ID/restaurant-recommender --platform managed
```

### AWS Elastic Beanstalk
1. **Create application**
2. **Upload code or connect Git**
3. **Configure environment**
4. **Deploy**

## 🔧 Environment Variables

Set these environment variables for production:

```bash
PORT=8501
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## 📊 Data Management

### For Production Deployment

1. **Upload your data file** to the deployment platform
2. **Or use cloud storage** (AWS S3, Google Cloud Storage, etc.)
3. **Update the data loading path** in `restaurant_recommender.py`

### Example: AWS S3 Integration
```python
import boto3
import pandas as pd

def load_data_from_s3():
    s3 = boto3.client('s3')
    s3.download_file('your-bucket', 'restaurant_data.xlsx', 'restaurant_data.xlsx')
    return pd.read_excel('restaurant_data.xlsx')
```

## 🔒 Security Considerations

1. **Environment Variables**: Store sensitive data in environment variables
2. **HTTPS**: Enable HTTPS in production
3. **Authentication**: Add user authentication if needed
4. **Rate Limiting**: Implement rate limiting for API endpoints

## 📈 Monitoring

### Health Checks
The app includes health check endpoints:
- `/_stcore/health` - Streamlit health check
- `/health` - Custom health check

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Issues**
   - Ensure `PORT` environment variable is set
   - Check if port is available

2. **Memory Issues**
   - Increase memory allocation
   - Optimize data loading

3. **Dependency Issues**
   - Check `requirements.txt` compatibility
   - Update dependencies

4. **Data Loading Issues**
   - Verify data file path
   - Check file permissions

### Debug Mode
```bash
# Run with debug logging
streamlit run streamlit_app.py --logger.level=debug
```

## 📞 Support

For deployment issues:
1. Check platform-specific documentation
2. Review logs for error messages
3. Test locally first
4. Contact platform support

---

**🎉 Your restaurant recommendation system is now ready for deployment!** 