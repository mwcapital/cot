"""
Fully modular application file for CFTC COT Data Dashboard
This version doesn't depend on legacyF.py
"""
import streamlit as st
import pandas as pd
from config import PAGE_CONFIG
from data_fetcher import load_instruments_database, fetch_cftc_data
from ui_components import render_sidebar, render_key_metrics, render_data_table
from charts.time_series import create_time_series_chart
from charts.seasonality_charts import create_seasonality_chart
from charts.participation_charts import create_participation_density_dashboard
from charts.share_of_oi import create_share_of_oi_chart
from charts.percentile_charts import create_percentile_chart
from charts.momentum_charts import create_single_variable_momentum_dashboard
from charts.trader_analysis import create_trader_breakdown_charts
from display_functions import (
    display_time_series_chart,
    display_seasonality_chart,
    display_percentile_chart,
    display_momentum_chart,
    display_trader_participation_chart
)


def main():
    """Main application function"""
    # Set page configuration
    st.set_page_config(**PAGE_CONFIG)
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stPlotlyChart {
            background-color: white;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        div[data-testid="metric-container"] {
            background-color: #f0f2f6;
            border: 1px solid #e0e2e6;
            padding: 15px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("📊 CFTC COT Data Dashboard")
    st.markdown("*Analyze Commitment of Traders data with interactive visualizations*")
    
    # Load instruments database
    instruments_db = load_instruments_database()
    if not instruments_db:
        st.error("Failed to load instruments database. Please check the data file.")
        return
    
    # Sidebar - Instrument Selection
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Token input
        api_token = st.text_input(
            "Socrata API Token",
            type="password",
            help="Get your free API token from https://data.cftc.gov/"
        )
        
        if not api_token:
            st.warning("⚠️ Please enter your API token to fetch data")
            return
        
        # Analysis mode selection
        st.subheader("📊 Analysis Mode")
        analysis_mode = st.radio(
            "Select analysis type:",
            ["Single Instrument", "Multi-Instrument Comparison"],
            help="Choose between analyzing one instrument or comparing multiple"
        )
        
        if analysis_mode == "Single Instrument":
            # Chart type selection
            st.subheader("📈 Chart Type")
            chart_type = st.selectbox(
                "Select visualization:",
                [
                    "Time Series",
                    "Share of Open Interest", 
                    "Seasonality",
                    "Percentile",
                    "Momentum",
                    "Trader Participation"
                ]
            )
            
            # Instrument selection
            selected_instrument = render_sidebar(instruments_db)
            
            if selected_instrument and st.button("🚀 Fetch Data", type="primary"):
                with st.spinner("Fetching data..."):
                    df = fetch_cftc_data(selected_instrument, api_token)
                
                if df is not None and not df.empty:
                    st.session_state['fetched_data'] = df
                    st.session_state['selected_instrument'] = selected_instrument
                    st.success(f"✅ Successfully fetched {len(df)} records")
                else:
                    st.error("❌ No data available for the selected instrument")
    
    # Main content area
    if 'fetched_data' in st.session_state and st.session_state['fetched_data'] is not None:
        df = st.session_state['fetched_data']
        instrument_name = st.session_state.get('selected_instrument', 'Unknown')
        
        # Display key metrics
        render_key_metrics(df)
        
        # Display selected chart type
        st.markdown("---")
        
        if chart_type == "Time Series":
            display_time_series_chart(df, instrument_name)
            
        elif chart_type == "Share of Open Interest":
            # Share of OI specific controls
            col1, col2 = st.columns(2)
            with col1:
                calculation_side = st.radio(
                    "Select side for calculation:",
                    ["Long Side", "Short Side"],
                    help="Choose which side to display for share of open interest calculation"
                )
            
            # Create and display the chart
            fig = create_share_of_oi_chart(df, calculation_side, instrument_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Seasonality":
            display_seasonality_chart(df, instrument_name)
            
        elif chart_type == "Percentile":
            display_percentile_chart(df, instrument_name)
            
        elif chart_type == "Momentum":
            display_momentum_chart(df, instrument_name)
            
        elif chart_type == "Trader Participation":
            display_trader_participation_chart(df, instrument_name)
        
        # Data table section
        with st.expander("📋 View Raw Data"):
            render_data_table(df)
    
    else:
        # Welcome message when no data is loaded
        st.info("""
        👋 Welcome to the CFTC COT Data Dashboard!
        
        To get started:
        1. Enter your Socrata API token in the sidebar
        2. Select an analysis mode
        3. Choose a chart type
        4. Search and select an instrument
        5. Click "Fetch Data" to load the data
        
        Need an API token? Get one free at [https://data.cftc.gov/](https://data.cftc.gov/)
        """)


if __name__ == "__main__":
    main()