# 🐙 GitHub Repository Setup

## 📋 Steps to Create GitHub Repository

### 1. Create New Repository on GitHub

1. **Go to GitHub.com** and sign in
2. **Click "New repository"** or the "+" icon
3. **Repository settings:**
   - **Repository name**: `restaurant-recommender`
   - **Description**: `A comprehensive restaurant recommendation system with Streamlit deployment`
   - **Visibility**: Public (or Private if you prefer)
   - **Initialize with**: Don't initialize (we already have files)
4. **Click "Create repository"**

### 2. Connect Local Repository to GitHub

```bash
# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/restaurant-recommender.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 3. Verify Repository

- Go to your GitHub repository URL
- You should see all your files listed
- Check that `.gitignore` is working (no model files should be visible)

## 🔗 Connect to Deployment Platforms

### Heroku
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-restaurant-recommender

# Add Heroku remote
heroku git:remote -a your-restaurant-recommender

# Deploy
git push heroku main
```

### Railway
1. **Go to Railway.app**
2. **Connect GitHub account**
3. **Select your repository**
4. **Deploy automatically**

### Render
1. **Go to Render.com**
2. **Connect GitHub account**
3. **Create new Web Service**
4. **Select your repository**
5. **Configure:**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

### Streamlit Cloud
1. **Go to share.streamlit.io**
2. **Sign in with GitHub**
3. **Select your repository**
4. **Deploy automatically**

## 📁 Repository Structure

Your repository should look like this:

```
restaurant-recommender/
├── .streamlit/
│   └── config.toml
├── restaurant_recommender.py
├── streamlit_app.py
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Procfile
├── runtime.txt
├── setup.sh
├── run_training.py
├── deploy.py
├── test_system.py
├── run.bat
├── run.sh
├── .gitignore
├── README.md
├── QUICK_START.md
├── DEPLOYMENT.md
├── GITHUB_SETUP.md
└── restaurant_data.xlsx (not in repo - add manually)
```

## 🔒 Security Best Practices

### 1. Environment Variables
Never commit sensitive data. Use environment variables:

```bash
# Create .env file (add to .gitignore)
API_KEY=your_secret_key
DATABASE_URL=your_database_url
```

### 2. Data Files
- **Don't commit large data files** to Git
- **Use cloud storage** (AWS S3, Google Drive, etc.)
- **Add data loading instructions** to README

### 3. Secrets Management
For deployment platforms, add secrets in their dashboard:
- API keys
- Database credentials
- External service tokens

## 🚀 Continuous Deployment

### GitHub Actions (Optional)
Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Heroku
      uses: akhileshns/heroku-deploy@v3.12.12
      with:
        heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
        heroku_app_name: "your-app-name"
        heroku_email: "your-email@example.com"
```

## 📊 Repository Insights

### Enable GitHub Features
1. **Issues**: Track bugs and feature requests
2. **Projects**: Organize development tasks
3. **Wiki**: Add detailed documentation
4. **Actions**: Set up CI/CD pipelines

### Repository Settings
1. **Go to Settings** in your repository
2. **Pages**: Enable GitHub Pages for documentation
3. **Branches**: Set up branch protection rules
4. **Collaborators**: Add team members if needed

## 🔄 Updating Your Repository

### Regular Updates
```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push origin main
```

### Version Tags
```bash
# Create a release tag
git tag -a v1.0.0 -m "Version 1.0.0"
git push origin v1.0.0
```

## 📞 Support

### GitHub Issues
- Create issues for bugs
- Use issue templates
- Label issues appropriately

### Documentation
- Keep README.md updated
- Add code comments
- Document API changes

---

**🎉 Your GitHub repository is now ready for deployment!** 