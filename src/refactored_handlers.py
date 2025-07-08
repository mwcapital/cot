# Refactored handler functions to replace the existing ones

def handle_single_instrument_flow(chart_type, instruments_db, api_token):
    """Handle single instrument selection and analysis"""
    # [Keep all the existing instrument selection code from lines 2964-3060]
    # ... existing instrument selection logic ...
    
    # After instrument is selected (around line 3062)
    if selected_instrument:
        st.markdown("---")
        st.subheader(f"Selected: {selected_instrument}")
        
        # Check if we need to fetch new data
        need_new_fetch = (not st.session_state.data_fetched or
                          st.session_state.fetched_instrument != selected_instrument)
        
        if st.button("🚀 Fetch Data", type="primary", use_container_width=False, disabled=not need_new_fetch):
            with st.spinner(f"Fetching data for {selected_instrument}..."):
                df = fetch_cftc_data(selected_instrument, api_token)
            
            if df is not None and not df.empty:
                st.session_state.fetched_data = df
                st.session_state.data_fetched = True
                st.session_state.fetched_instrument = selected_instrument
                st.success(f"✅ Successfully fetched {len(df)} records")
            else:
                st.error("❌ No data available for the selected instrument")
                st.session_state.data_fetched = False
    
    # Display the appropriate chart based on pre-selected chart type
    if st.session_state.data_fetched and st.session_state.fetched_data is not None:
        df = st.session_state.fetched_data
        
        st.markdown("---")
        
        # Display chart based on the chart_type parameter
        if chart_type == "Time Series":
            display_time_series_chart(df, selected_instrument)
            
        elif chart_type == "Seasonality":
            display_seasonality_chart(df, selected_instrument)
            
        elif chart_type == "Percentile":
            display_percentile_chart(df, selected_instrument)
            
        elif chart_type == "Momentum":
            display_momentum_chart(df, selected_instrument)
            
        elif chart_type == "Trader Participation":
            display_trader_participation_chart(df, selected_instrument)


def display_time_series_chart(df, instrument_name):
    """Display time series analysis"""
    st.subheader("📈 Time Series Analysis")
    
    # Date range selection
    min_date = df['report_date_as_yyyy_mm_dd'].min()
    max_date = df['report_date_as_yyyy_mm_dd'].max()
    available_dates = sorted(df['report_date_as_yyyy_mm_dd'].unique())
    
    date_range = st.select_slider(
        "Select Date Range:",
        options=range(len(available_dates)),
        value=(0, len(available_dates) - 1),
        format_func=lambda x: available_dates[x].strftime('%Y-%m-%d')
    )
    
    start_date = available_dates[date_range[0]]
    end_date = available_dates[date_range[1]]
    
    filtered_df = df[
        (df['report_date_as_yyyy_mm_dd'] >= start_date) &
        (df['report_date_as_yyyy_mm_dd'] <= end_date)
    ].copy()
    
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
    
    existing_columns = [col for col in available_columns if col in filtered_df.columns]
    
    selected_columns = st.multiselect(
        "Select data series to plot:",
        existing_columns,
        default=["open_interest_all"] if "open_interest_all" in existing_columns else [existing_columns[0]]
    )
    
    if selected_columns:
        fig = create_plotly_chart(filtered_df, selected_columns, f"Time Series - {instrument_name}")
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def display_seasonality_chart(df, instrument_name):
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


def display_percentile_chart(df, instrument_name):
    """Display percentile analysis"""
    st.subheader("📊 Percentile Analysis")
    
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
            [0.25, 0.5, 1, 2, 5, 'all'],
            format_func=lambda x: {0.25: "3 Months", 0.5: "6 Months", 1: "1 Year", 2: "2 Years", 5: "5 Years", 'all': "All Time"}[x],
            index=2
        )
    
    with col2:
        chart_type = st.radio(
            "Chart type:",
            ["time_series", "distribution", "cumulative"],
            format_func=lambda x: x.replace('_', ' ').title(),
            index=0
        )
    
    fig = create_percentile_chart(df, selected_metric, lookback_period, chart_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def display_momentum_chart(df, instrument_name):
    """Display momentum dashboard"""
    st.subheader("🚀 Momentum Dashboard")
    
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
    
    fig = create_single_variable_momentum_dashboard(df, selected_var, change_col)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def display_trader_participation_chart(df, instrument_name):
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
    
    # Calculate percentile based on selected lookback
    if lookback_days:
        lookback_date = df['report_date_as_yyyy_mm_dd'].max() - pd.Timedelta(days=lookback_days)
        df_lookback = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()
    else:
        df_lookback = df.copy()
    
    # Calculate avg position per trader for all data
    df['avg_position_per_trader'] = df['open_interest_all'] / df['traders_tot_all']
    
    # Calculate percentile for each point based on lookback window
    percentile_data = []
    for i in range(len(df)):
        current_date = df.iloc[i]['report_date_as_yyyy_mm_dd']
        if lookback_days:
            lookback_start = current_date - pd.Timedelta(days=lookback_days)
            window_data = df[(df['report_date_as_yyyy_mm_dd'] >= lookback_start) & 
                           (df['report_date_as_yyyy_mm_dd'] <= current_date)]
        else:
            window_data = df[df['report_date_as_yyyy_mm_dd'] <= current_date]
        
        if len(window_data) > 0:
            current_val = df.iloc[i]['avg_position_per_trader']
            percentile = (window_data['avg_position_per_trader'] < current_val).sum() / len(window_data) * 100
            percentile_data.append(percentile)
        else:
            percentile_data.append(50)
    
    # Update y-axis title for percentile subplot with lookback period
    fig_update_text = f"Avg Position Percentile ({lookback_period})"
    
    # Create participation density chart with percentile data
    density_fig = create_participation_density_dashboard(df, instrument_name, percentile_data)
    
    # Update the percentile y-axis title dynamically
    if density_fig:
        density_fig.update_yaxes(
            title_text=fig_update_text,
            row=2, col=1
        )
        
        st.plotly_chart(density_fig, use_container_width=True)