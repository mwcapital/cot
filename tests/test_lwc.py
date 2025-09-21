import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

st.set_page_config(page_title="Test LWC", layout="wide")

st.title("Test Lightweight Charts")

# Simple test data
chartOptions = {
    "chart": {
        "height": 400,
        "layout": {
            "background": {"type": "solid", "color": "white"},
            "textColor": "black"
        }
    },
    "series": [{
        "type": "Line",
        "data": [
            {"time": "2024-01-01", "value": 100},
            {"time": "2024-01-02", "value": 102},
            {"time": "2024-01-03", "value": 101},
            {"time": "2024-01-04", "value": 105}
        ],
        "options": {
            "color": "blue"
        }
    }]
}

renderLightweightCharts(chartOptions, "test")