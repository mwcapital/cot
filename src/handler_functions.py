"""
Handler functions for the restructured app flow
"""

def handle_single_instrument_flow(chart_type, instruments_db, api_token):
    """Handle single instrument selection and analysis"""
    st.header("🎯 Select Instrument")
    
    # Search method selection
    search_method = st.radio(
        "Choose search method:",
        ["Extensive Search", "Search by Commodity Subgroup", "Search by Commodity Type", "Free Text Search"],
        horizontal=True
    )
    
    selected_instrument = None
    
    # [Reuse existing instrument selection logic here]
    # ... (all the instrument selection code from the original)
    
    # After instrument is selected and data is fetched
    if selected_instrument:
        st.markdown("---")
        st.subheader(f"Selected: {selected_instrument}")
        
        if st.button("🚀 Fetch Data", type="primary"):
            with st.spinner(f"Fetching data for {selected_instrument}..."):
                df = fetch_cftc_data(selected_instrument, api_token)
            
            if df is not None and not df.empty:
                st.session_state.fetched_data = df
                st.session_state.fetched_instrument = selected_instrument
                st.success(f"✅ Successfully fetched {len(df)} records")
                
                # Display the appropriate chart based on selection
                st.markdown("---")
                
                if chart_type == "Time Series":
                    display_time_series_analysis(df, selected_instrument)
                    
                elif chart_type == "Seasonality":
                    display_seasonality_analysis(df, selected_instrument)
                    
                elif chart_type == "Percentile":
                    display_percentile_analysis(df, selected_instrument)
                    
                elif chart_type == "Momentum":
                    display_momentum_dashboard(df, selected_instrument)
                    
                elif chart_type == "Trader Participation":
                    display_trader_participation(df, selected_instrument)
                    
            else:
                st.error("❌ No data available for the selected instrument")


def handle_multi_instrument_flow(chart_type, instruments_db, api_token):
    """Handle multi-instrument selection and analysis"""
    st.header("🎯 Select Multiple Instruments")
    
    # Get all instruments
    instruments_list = []
    if instruments_db and 'exchanges' in instruments_db:
        for exchange, groups in instruments_db['exchanges'].items():
            for group, subgroups in groups.items():
                for subgroup, instruments in subgroups.items():
                    instruments_list.extend(instruments)
    
    # Filter to major exchanges
    futures_only = [inst for inst in instruments_list if ' - CHICAGO' in inst or ' - NEW YORK' in inst or ' - ICE' in inst]
    
    selected_instruments = st.multiselect(
        "Choose instruments for comparison:",
        options=sorted(futures_only),
        max_selections=15,
        help="Select up to 15 instruments for comparison"
    )
    
    if selected_instruments and len(selected_instruments) >= 2:
        if st.button("🚀 Fetch Data for All Instruments", type="primary"):
            if chart_type == "Cross-Asset":
                display_cross_asset_comparison(selected_instruments, api_token, instruments_db)
            elif chart_type == "Market Matrix":
                display_market_structure_matrix(selected_instruments, api_token, instruments_db)
    elif selected_instruments:
        st.warning("Please select at least 2 instruments for comparison")
    else:
        st.info("Please select instruments from the list above")


# Display functions for single instrument charts
def display_time_series_analysis(df, instrument_name):
    """Display time series analysis"""
    st.subheader("📈 Time Series Analysis")
    
    # Column selection
    available_columns = [
        "open_interest_all",
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "comm_positions_long_all", 
        "comm_positions_short_all",
        "net_noncomm_positions",
        "net_comm_positions"
    ]
    
    existing_columns = [col for col in available_columns if col in df.columns]
    
    selected_columns = st.multiselect(
        "Select data series to plot:",
        existing_columns,
        default=["open_interest_all", "net_noncomm_positions"][:1]
    )
    
    if selected_columns:
        fig = create_plotly_chart(df, selected_columns, f"Time Series - {instrument_name}")
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def display_seasonality_analysis(df, instrument_name):
    """Display seasonality analysis"""
    st.subheader("📅 Seasonality Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        seasonality_column = st.selectbox(
            "Select metric:",
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
        fig = create_seasonality_chart(df, seasonality_column, lookback_years, show_previous, zone_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def display_percentile_analysis(df, instrument_name):
    """Display percentile analysis"""
    st.subheader("📊 Percentile Analysis")
    
    # Metric selection
    numeric_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'report_date_as_yyyy_mm_dd']
    
    selected_metric = st.selectbox(
        "Select metric for analysis:",
        numeric_columns,
        index=numeric_columns.index('net_noncomm_positions') if 'net_noncomm_positions' in numeric_columns else 0
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        lookback_period = st.selectbox(
            "Historical lookback:",
            ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "All Time"],
            index=2
        )
    
    with col2:
        chart_type = st.radio(
            "Chart type:",
            ["Time Series", "Distribution", "Percentile Curve"],
            index=0
        )
    
    # Create percentile chart
    lookback_map = {
        "3 Months": 3/12,
        "6 Months": 0.5,
        "1 Year": 1,
        "2 Years": 2,
        "5 Years": 5,
        "All Time": 'all'
    }
    
    fig = create_percentile_chart(df, selected_metric, lookback_map[lookback_period], chart_type.lower().replace(' ', '_'))
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def display_momentum_dashboard(df, instrument_name):
    """Display momentum dashboard"""
    st.subheader("🚀 Momentum Dashboard")
    
    # Select variable for momentum analysis
    momentum_vars = ['open_interest_all', 'net_noncomm_positions', 'traders_tot_all']
    available_vars = [var for var in momentum_vars if var in df.columns]
    
    selected_var = st.selectbox(
        "Select variable for momentum analysis:",
        available_vars,
        index=0
    )
    
    # Calculate week-over-week changes
    change_col = f'change_in_{selected_var}'
    df[change_col] = df[selected_var].diff()
    
    # Create momentum dashboard
    fig = create_single_variable_momentum_dashboard(df, selected_var, change_col)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def display_trader_participation(df, instrument_name):
    """Display trader participation analysis"""
    st.subheader("👥 Trader Participation Analysis")
    
    # Percentile lookback selector
    col_lookback, col_empty = st.columns([1, 3])
    with col_lookback:
        lookback_period = st.selectbox(
            "Percentile Lookback:",
            ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "All Time"],
            index=2
        )
    
    # Calculate percentile data (reuse existing logic)
    # ... percentile calculation code ...
    
    # Create participation density dashboard
    density_fig = create_participation_density_dashboard(df, instrument_name, percentile_data=None)
    if density_fig:
        st.plotly_chart(density_fig, use_container_width=True)


# Display functions for multi-instrument charts
def display_cross_asset_comparison(selected_instruments, api_token, instruments_db):
    """Display cross-asset comparison"""
    st.subheader("🔄 Cross-Asset Comparison")
    
    # Trader category selection
    trader_category = st.selectbox(
        "Select trader category:",
        ["Non-Commercial", "Commercial", "Non-Reportable"],
        index=0
    )
    
    # Create cross-asset analysis
    fig = create_cross_asset_analysis(
        selected_instruments, 
        trader_category, 
        api_token,
        lookback_start=pd.Timestamp.now() - pd.DateOffset(years=2),
        show_week_ago=True,
        instruments_db=instruments_db
    )
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def display_market_structure_matrix(selected_instruments, api_token, instruments_db):
    """Display market structure matrix"""
    st.subheader("🎯 Market Structure Matrix")
    
    # Fetch data for all instruments and create matrix
    all_instruments_data = {}
    
    progress_bar = st.progress(0)
    for idx, instrument in enumerate(selected_instruments):
        progress_bar.progress((idx + 1) / len(selected_instruments))
        df = fetch_cftc_data(instrument, api_token)
        if df is not None and not df.empty:
            all_instruments_data[instrument] = df
    
    progress_bar.empty()
    
    if all_instruments_data:
        fig = create_market_structure_matrix(all_instruments_data, selected_instruments)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to fetch data for the selected instruments")