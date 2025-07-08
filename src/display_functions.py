# Display functions for single instrument charts

import streamlit as st
import pandas as pd
from charts.base_charts import create_plotly_chart
from charts.seasonality_charts import create_seasonality_chart
from charts.percentile_charts import create_percentile_chart
from charts.momentum_charts import create_single_variable_momentum_dashboard
from charts.participation_charts import create_participation_density_dashboard

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
    
    st.info(f"📊 Showing {len(filtered_df)} records from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Column selection
    available_columns = [
        "open_interest_all",
        "noncomm_positions_long_all", 
        "noncomm_positions_short_all",
        "comm_positions_long_all",
        "comm_positions_short_all",
        "net_noncomm_positions",
        "net_comm_positions",
        "net_reportable_positions"
    ]
    
    existing_columns = [col for col in available_columns if col in filtered_df.columns]
    
    selected_columns = st.multiselect(
        "Select data series to plot:",
        existing_columns,
        default=["open_interest_all"] if "open_interest_all" in existing_columns else [existing_columns[0]]
    )
    
    if selected_columns:
        fig = create_plotly_chart(filtered_df, selected_columns, f"{instrument_name} - Time Series Analysis")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one data series to plot")


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
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_metric = st.selectbox(
            "Select metric for analysis:",
            numeric_columns,
            index=numeric_columns.index('net_noncomm_positions') if 'net_noncomm_positions' in numeric_columns else 0
        )
    
    with col2:
        lookback_period = st.selectbox(
            "Historical lookback:",
            ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "All Time"],
            index=2
        )
    
    chart_type = st.radio(
        "Chart type:",
        ["time_series", "distribution", "cumulative"],
        format_func=lambda x: x.replace('_', ' ').title(),
        index=0,
        horizontal=True
    )
    
    # Map lookback to years
    lookback_map = {
        "3 Months": 0.25,
        "6 Months": 0.5,
        "1 Year": 1,
        "2 Years": 2,
        "5 Years": 5,
        "All Time": 'all'
    }
    
    fig = create_percentile_chart(df, selected_metric, lookback_map[lookback_period], chart_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def display_momentum_chart(df, instrument_name):
    """Display momentum dashboard"""
    st.subheader("🚀 Momentum Dashboard")
    
    # Add date range selector for adaptive view
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Define momentum variables with their corresponding API change columns
        momentum_vars = {
            'open_interest_all': {
                'display': 'Open Interest',
                'change_col': 'change_in_open_interest_all'
            },
            'noncomm_positions_long_all': {
                'display': 'Non-Commercial Long',
                'change_col': 'change_in_noncomm_long_all'
            },
            'noncomm_positions_short_all': {
                'display': 'Non-Commercial Short',
                'change_col': 'change_in_noncomm_short_all'
            },
            'comm_positions_long_all': {
                'display': 'Commercial Long',
                'change_col': 'change_in_comm_long_all'
            },
            'comm_positions_short_all': {
                'display': 'Commercial Short',
                'change_col': 'change_in_comm_short_all'
            },
            'tot_rept_positions_long_all': {
                'display': 'Total Reportable Long',
                'change_col': 'change_in_tot_rept_long_all'
            },
            'tot_rept_positions_short': {
                'display': 'Total Reportable Short',
                'change_col': 'change_in_tot_rept_short'
            },
            'nonrept_positions_long_all': {
                'display': 'Non-Reportable Long',
                'change_col': 'change_in_nonrept_long_all'
            },
            'nonrept_positions_short_all': {
                'display': 'Non-Reportable Short',
                'change_col': 'change_in_nonrept_short_all'
            }
        }
        
        # Filter to only available position columns
        available_vars = {k: v for k, v in momentum_vars.items() 
                         if k in df.columns and v['change_col'] in df.columns}
        
        selected_var = st.selectbox(
            "Select variable for momentum analysis:",
            list(available_vars.keys()),
            format_func=lambda x: available_vars[x]['display'],
            index=0
        )
    
    with col2:
        # Date range quick selector
        date_range_option = st.selectbox(
            "Time Period:",
            ["All Time", "5 Years", "2 Years", "1 Year", "6 Months", "3 Months", "Custom"],
            index=0,
            key="momentum_date_range"
        )
    
    # Filter data based on selection
    df_filtered = df.copy()
    
    if date_range_option != "All Time":
        end_date = df['report_date_as_yyyy_mm_dd'].max()
        
        if date_range_option == "5 Years":
            start_date = end_date - pd.DateOffset(years=5)
        elif date_range_option == "2 Years":
            start_date = end_date - pd.DateOffset(years=2)
        elif date_range_option == "1 Year":
            start_date = end_date - pd.DateOffset(years=1)
        elif date_range_option == "6 Months":
            start_date = end_date - pd.DateOffset(months=6)
        elif date_range_option == "3 Months":
            start_date = end_date - pd.DateOffset(months=3)
        elif date_range_option == "Custom":
            with col3:
                # Custom date range
                min_date = df['report_date_as_yyyy_mm_dd'].min()
                max_date = df['report_date_as_yyyy_mm_dd'].max()
                
                date_range = st.date_input(
                    "Select dates:",
                    value=(max_date - pd.DateOffset(years=1), max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="momentum_custom_dates"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = end_date - pd.DateOffset(years=1)
        
        df_filtered = df_filtered[
            (df_filtered['report_date_as_yyyy_mm_dd'] >= pd.Timestamp(start_date)) &
            (df_filtered['report_date_as_yyyy_mm_dd'] <= pd.Timestamp(end_date))
        ]
    
    # Get the corresponding API change column
    change_col = available_vars[selected_var]['change_col']
    
    # Use the position variable for display and API change column for calculations
    display_var = selected_var
    
    # Show data info
    st.info(f"📊 Showing {len(df_filtered)} data points from {df_filtered['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')} to {df_filtered['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')}")
    
    # Import if not already available
    try:
        fig = create_single_variable_momentum_dashboard(df_filtered, display_var, change_col)
    except NameError:
        from charts.momentum_charts import create_single_variable_momentum_dashboard
        fig = create_single_variable_momentum_dashboard(df_filtered, display_var, change_col)
    if fig:
        # Configure plotly to show autoscale buttons
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'momentum_{selected_var}',
                'height': 1000,
                'width': 1400,
                'scale': 1
            }
        }
        st.plotly_chart(fig, use_container_width=True, config=config)


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