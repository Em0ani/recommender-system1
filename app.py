#!/usr/bin/env python3
"""
Alternative entry point for deployment platforms
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main Streamlit app
from streamlit_app import main

if __name__ == "__main__":
    main() 