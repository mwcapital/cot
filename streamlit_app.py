"""
CFTC COT Data Dashboard
Main entry point for the Streamlit application
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import main app function - using the one that preserves exact UI
from main import main

if __name__ == "__main__":
    main()