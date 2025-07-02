# 🚀 Quick Start Guide

## For Windows Users

1. **Double-click `run.bat`** - This will automatically:
   - Install all dependencies
   - Train the models
   - Start the Streamlit app

## For Mac/Linux Users

1. **Open terminal in the project directory**
2. **Run:**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

## Manual Setup (All Platforms)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the models:**
   ```bash
   python run_training.py
   ```

3. **Start the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📋 Prerequisites

- Python 3.8 or higher
- Your `restaurant_data.xlsx` file in the project directory
- Internet connection (for downloading dependencies)

## 🎯 What You'll Get

- **Interactive web interface** at `http://localhost:8501`
- **4 recommendation methods:**
  - Collaborative Filtering (SVD/NMF)
  - Content-Based Filtering
  - Hybrid Recommendations
  - Popular Restaurants
- **Analytics dashboard** with visualizations
- **Real-time recommendations** based on your data

## 🚨 Troubleshooting

### Common Issues:

1. **"restaurant_data.xlsx not found"**
   - Place your Excel file in the project directory

2. **"Python not found"**
   - Install Python 3.8+ from python.org

3. **"Dependencies failed to install"**
   - Try: `pip install --upgrade pip`
   - Then: `pip install -r requirements.txt`

4. **"Training failed"**
   - Check your Excel file format
   - Ensure it has Metadata, Customers, and Reviews sheets

### Need Help?

- Check the full `README.md` for detailed documentation
- Run `python test_system.py` to diagnose issues
- Ensure all required columns are present in your data

---

**🎉 Ready to get started? Just run the appropriate script for your platform!** 