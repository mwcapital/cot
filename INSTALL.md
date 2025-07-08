# Installation Guide

## Requirements

This CFTC COT Data Dashboard requires Python 3.7+ and the following packages:

### Core Dependencies
```bash
pip install streamlit pandas plotly sodapy numpy scipy
```

### Optional Dependencies

For advanced visualizations (Participant Behavior Clusters):
```bash
pip install scikit-learn
```

## Quick Install

1. Clone or download this repository

2. Install all dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run src/legacyF.py
```

## Troubleshooting

### "No module named 'sklearn'" Error

The Participant Behavior Clusters visualization requires scikit-learn. Install it with:
```bash
pip install scikit-learn
```

### API Token Required

You'll need a CFTC API token. Get one free at:
https://publicreporting.cftc.gov/

## Python Environment Setup (Recommended)

Create a virtual environment to avoid conflicts:

```bash
# Create virtual environment
python -m venv cot_env

# Activate it
# On macOS/Linux:
source cot_env/bin/activate
# On Windows:
cot_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```