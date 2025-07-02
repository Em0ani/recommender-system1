#!/usr/bin/env bash

# Install system dependencies
apt-get update
apt-get install -y python3-dev build-essential

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p .streamlit

# Create Streamlit config
cat > .streamlit/config.toml << EOF
[server]
headless = true
enableCORS = false
port = \$PORT

[browser]
gatherUsageStats = false
EOF 