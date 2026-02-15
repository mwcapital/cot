"""
Main application entry point - Navigation hub for all report types
"""
import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="CFTC COT Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define pages
legacy_page = st.Page("pages/legacy.py", title="Legacy", default=True)
disaggregated_page = st.Page("pages/disaggregated.py", title="Disaggregated")

# Navigation
pg = st.navigation([legacy_page, disaggregated_page])
pg.run()