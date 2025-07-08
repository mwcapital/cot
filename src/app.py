"""
Main application file for CFTC COT Data Dashboard
"""
import streamlit as st
import pandas as pd
from config import PAGE_CONFIG
from data_fetcher import load_instruments_database, fetch_cftc_data
from ui_components import render_sidebar, render_key_metrics, render_data_table, render_chart_selector
from charts.base_charts import create_plotly_chart
from charts.seasonality_charts import create_seasonality_chart
from charts.participation_charts import create_participation_density_dashboard
from charts.share_of_oi import create_share_of_oi_chart
from charts.percentile_charts import create_percentile_chart
from charts.momentum_charts import create_single_variable_momentum_dashboard
from charts.trader_analysis import create_trader_breakdown_charts


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
    selected_instrument = render_sidebar(instruments_db)
    
    if not selected_instrument:
        st.info("👈 Please select an instrument from the sidebar to begin analysis")
        return
    
    # API Token input
    api_token = st.sidebar.text_input(
        "API Token (optional):",
        type="password",
        help="Enter your CFTC API token for higher rate limits"
    )
    
    # Fetch data button
    if st.sidebar.button("🔄 Fetch Data", type="primary"):
        with st.spinner(f"Fetching data for {selected_instrument}..."):
            df = fetch_cftc_data(selected_instrument, api_token or None)
            
            if df is not None and not df.empty:
                st.session_state['fetched_data'] = df
                st.session_state['fetched_instrument'] = selected_instrument
                st.success(f"✅ Successfully fetched {len(df)} records")
            else:
                st.error("❌ No data found for the selected instrument")
    
    # Main content area
    if 'fetched_data' in st.session_state and st.session_state.get('fetched_instrument') == selected_instrument:
        df = st.session_state['fetched_data']
        
        # Display key metrics
        st.markdown("### 📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        render_key_metrics(df, col1, col2, col3, col4)
        
        st.markdown("---")
        
        # Chart section
        chart_type = render_chart_selector()
        
        if chart_type == "Time Series Analysis":
            render_time_series_analysis(df, selected_instrument)
        elif chart_type == "Seasonality Analysis":
            render_seasonality_analysis(df)
        elif chart_type == "Trader Participation Analysis":
            render_trader_participation_analysis(df, selected_instrument)
        elif chart_type == "Positioning & Concentration":
            render_positioning_concentration(df, selected_instrument)
        elif chart_type == "Percentile Analysis":
            render_percentile_analysis(df)
        elif chart_type == "Cross-Asset Comparison":
            st.info("Cross-asset comparison requires multiple instruments. This feature is coming soon!")
        
        # Raw data section
        with st.expander("📊 View Raw Data"):
            render_data_table(df)
    
    else:
        st.info(f"👆 Click 'Fetch Data' to load data for {selected_instrument}")


def render_time_series_analysis(df, instrument_name):
    """Render time series analysis section"""
    st.subheader("📈 Time Series Analysis")
    
    # Add tabs for different views
    tab1, tab2 = st.tabs(["Standard Time Series", "Share of Open Interest"])
    
    with tab1:
        # Column selection
        available_columns = [
            col for col in df.columns 
            if col not in ['report_date_as_yyyy_mm_dd', 'market_and_exchange_names'] 
            and df[col].dtype in ['int64', 'float64']
        ]
        
        selected_columns = st.multiselect(
            "Select data series to plot:",
            available_columns,
            default=['open_interest_all', 'net_noncomm_positions'][:1]  # Default to first available
        )
        
        if selected_columns:
            fig = create_plotly_chart(
                df, 
                selected_columns, 
                f"Time Series Analysis - {instrument_name}"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Share of Open Interest view
        st.markdown("#### Share of Open Interest Over Time")
        
        # Add explanation
        st.info("""
        This chart shows how open interest is distributed among different trader categories as a percentage of total.
        
        **Calculation Method:**
        - **Long Side**: NonComm Long + Spread + Comm Long + NonRep Long = 100%
        - **Short Side**: NonComm Short + Spread + Comm Short + NonRep Short = 100%
        """)
        
        # Calculation side selector
        calculation_side = st.selectbox(
            "Calculate percentages using:",
            ["Long Side", "Short Side"],
            index=0
        )
        
        # Share of OI chart is already imported above
        
        # Create and display the chart
        fig = create_share_of_oi_chart(df, calculation_side, instrument_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_seasonality_analysis(df):
    """Render seasonality analysis section"""
    st.subheader("📅 Seasonality Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        seasonality_column = st.selectbox(
            "Select metric for seasonality:",
            ['net_noncomm_positions', 'open_interest_all', 'traders_tot_all'],
            index=0
        )
    
    with col2:
        lookback_years = st.selectbox(
            "Lookback period:",
            [1, 2, 3, 5, 'all'],
            index=3
        )
    
    with col3:
        zone_type = st.radio(
            "Zone type:",
            ['percentile', 'std'],
            index=0
        )
    
    show_previous = st.checkbox("Show previous year", value=True)
    
    if seasonality_column in df.columns:
        fig = create_seasonality_chart(
            df, 
            seasonality_column, 
            lookback_years, 
            show_previous, 
            zone_type
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_trader_participation_analysis(df, instrument_name):
    """Render trader participation analysis section"""
    st.subheader("👥 Trader Participation Analysis")
    
    tab1, tab2 = st.tabs(["Average Position per Trader", "Trader Breakdown"])
    
    with tab1:
        st.info("""
        Shows average position size per trader alongside total trader count.
        
        **Bar Colors** indicate concentration (% of open interest held by top 4 traders):
        - 🟢 Green: <15% - Well-distributed market
        - 🟡 Yellow: 15-25% - Moderate concentration  
        - 🔴 Red: >25% - High concentration risk
        """)
        
        # Calculate percentile data for avg position per trader
        df['avg_position_per_trader'] = df['open_interest_all'] / df['traders_tot_all']
        
        # Simple percentile calculation
        percentile_data = []
        for i in range(len(df)):
            current_val = df.iloc[i]['avg_position_per_trader']
            percentile = (df['avg_position_per_trader'][:i+1] < current_val).sum() / (i+1) * 100
            percentile_data.append(percentile)
        
        fig = create_participation_density_dashboard(df, instrument_name, percentile_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = create_trader_breakdown_charts(df, instrument_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_positioning_concentration(df, instrument_name):
    """Render positioning and concentration analysis"""
    st.subheader("📊 Positioning & Concentration Analysis")
    
    # Create positioning chart
    positioning_columns = [
        'noncomm_positions_long_all',
        'noncomm_positions_short_all',
        'comm_positions_long_all',
        'comm_positions_short_all'
    ]
    
    available_pos_columns = [col for col in positioning_columns if col in df.columns]
    
    if available_pos_columns:
        fig = create_plotly_chart(
            df,
            available_pos_columns,
            f"Positioning Analysis - {instrument_name}"
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Concentration metrics
    st.markdown("### 🎯 Concentration Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'conc_net_le_4_tdr_long_all' in df.columns:
            fig_conc = create_plotly_chart(
                df,
                ['conc_net_le_4_tdr_long_all', 'conc_net_le_4_tdr_short_all'],
                "Top 4 Traders Net Concentration"
            )
            if fig_conc:
                fig_conc.update_layout(height=400)
                st.plotly_chart(fig_conc, use_container_width=True)
    
    with col2:
        if 'conc_net_le_8_tdr_long_all' in df.columns:
            fig_conc8 = create_plotly_chart(
                df,
                ['conc_net_le_8_tdr_long_all', 'conc_net_le_8_tdr_short_all'],
                "Top 8 Traders Net Concentration"
            )
            if fig_conc8:
                fig_conc8.update_layout(height=400)
                st.plotly_chart(fig_conc8, use_container_width=True)


def render_percentile_analysis(df):
    """Render percentile analysis section"""
    st.subheader("📊 Percentile Analysis")
    
    # Select metric
    numeric_columns = [
        col for col in df.columns 
        if col not in ['report_date_as_yyyy_mm_dd', 'market_and_exchange_names'] 
        and df[col].dtype in ['int64', 'float64']
    ]
    
    selected_metric = st.selectbox(
        "Select metric for percentile analysis:",
        numeric_columns,
        index=numeric_columns.index('net_noncomm_positions') if 'net_noncomm_positions' in numeric_columns else 0
    )
    
    lookback_period = st.selectbox(
        "Historical lookback period:",
        ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "All Time"],
        index=2
    )
    
    # Calculate percentiles
    current_value = df[selected_metric].iloc[-1]
    
    # Map lookback to days
    lookback_map = {
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825,
        "All Time": None
    }
    
    lookback_days = lookback_map[lookback_period]
    
    if lookback_days:
        lookback_date = df['report_date_as_yyyy_mm_dd'].max() - pd.Timedelta(days=lookback_days)
        historical_data = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date][selected_metric]
    else:
        historical_data = df[selected_metric]
    
    percentile = (historical_data < current_value).sum() / len(historical_data) * 100
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Value", f"{current_value:,.0f}")
    
    with col2:
        st.metric("Percentile Rank", f"{percentile:.1f}%")
    
    with col3:
        st.metric("Historical Average", f"{historical_data.mean():,.0f}")
    
    # Create visualization
    st.info(f"Current {selected_metric} is at the {percentile:.1f}th percentile of the {lookback_period} range")


if __name__ == "__main__":
    main()