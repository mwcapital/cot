# Display functions for single instrument charts - EXACT copy from legacyF.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from charts.base_charts import create_plotly_chart
from charts.seasonality_charts import create_seasonality_chart
from charts.percentile_charts import create_percentile_chart
from charts.momentum_charts import create_single_variable_momentum_dashboard
from charts.participation_charts import create_participation_density_dashboard
from charts.share_of_oi import create_share_of_oi_chart
from charts.trader_participation_analysis import (
    create_concentration_risk_heatmap,
    create_market_structure_quadrant,
    create_concentration_divergence_analysis,
    create_heterogeneity_index,
    create_spreading_activity_analysis
)
from charts.concentration_momentum import create_concentration_momentum_analysis
from charts.participant_behavior_clusters import create_participant_behavior_clusters
from charts.regime_detection import create_regime_detection_dashboard
from charts.market_microstructure import create_market_microstructure_analysis


def calculate_percentiles_for_column(df, column, lookback_days):
    """Calculate rolling percentiles for a specific column - EXACT from legacyF.py"""
    percentiles = []
    for i in range(len(df)):
        current_date = df.iloc[i]['report_date_as_yyyy_mm_dd']
        current_val = df.iloc[i][column]
        
        if pd.isna(current_val):
            percentiles.append(50)
            continue
            
        if lookback_days == "since_2010":
            lookback_start = pd.Timestamp('2010-01-01')
            window_data = df[(df['report_date_as_yyyy_mm_dd'] >= lookback_start) & 
                           (df['report_date_as_yyyy_mm_dd'] <= current_date)]
        elif lookback_days:
            lookback_start = current_date - pd.Timedelta(days=lookback_days)
            window_data = df[(df['report_date_as_yyyy_mm_dd'] >= lookback_start) & 
                           (df['report_date_as_yyyy_mm_dd'] <= current_date)]
        else:
            window_data = df[df['report_date_as_yyyy_mm_dd'] <= current_date]
        
        valid_data = window_data[column].dropna()
        if len(valid_data) > 0:
            percentile = (valid_data < current_val).sum() / len(valid_data) * 100
            percentiles.append(percentile)
        else:
            percentiles.append(50)
    
    return percentiles


def create_participation_density_dashboard_original(df, instrument_name, percentile_data=None, lookback_days=None):
    """Create comprehensive avg position per trader dashboard for all categories - EXACT from legacyF.py"""
    try:
        # Copy and prepare data
        df_plot = df.copy()
        df_plot = df_plot.sort_values('report_date_as_yyyy_mm_dd')
        
        # Calculate average position per trader for overall and each category
        # Overall
        df_plot['avg_pos_per_trader'] = df_plot['open_interest_all'] / df_plot['traders_tot_all']
        
        # Non-Commercial
        df_plot['avg_noncomm_long'] = df_plot['noncomm_positions_long_all'] / df_plot['traders_noncomm_long_all']
        df_plot['avg_noncomm_short'] = df_plot['noncomm_positions_short_all'] / df_plot['traders_noncomm_short_all']
        df_plot['avg_noncomm_spread'] = df_plot['noncomm_postions_spread_all'] / df_plot['traders_noncomm_spread_all']
        
        # Commercial
        df_plot['avg_comm_long'] = df_plot['comm_positions_long_all'] / df_plot['traders_comm_long_all']
        df_plot['avg_comm_short'] = df_plot['comm_positions_short_all'] / df_plot['traders_comm_short_all']
        
        # Total Reportable
        df_plot['avg_rept_long'] = df_plot['tot_rept_positions_long_all'] / df_plot['traders_tot_rept_long_all']
        df_plot['avg_rept_short'] = df_plot['tot_rept_positions_short'] / df_plot['traders_tot_rept_short_all']
        
        # Define categories to plot
        categories = [
            ('avg_pos_per_trader', 'Overall Average', 'traders_tot_all'),
            ('avg_noncomm_long', 'Non-Commercial Long', 'traders_noncomm_long_all'),
            ('avg_noncomm_short', 'Non-Commercial Short', 'traders_noncomm_short_all'),
            ('avg_noncomm_spread', 'Non-Commercial Spread', 'traders_noncomm_spread_all'),
            ('avg_comm_long', 'Commercial Long', 'traders_comm_long_all'),
            ('avg_comm_short', 'Commercial Short', 'traders_comm_short_all'),
            ('avg_rept_long', 'Total Long', 'traders_tot_rept_long_all'),
            ('avg_rept_short', 'Total Short', 'traders_tot_rept_short_all')
        ]
        
        # Calculate percentiles for each category
        percentile_data_dict = {}
        for col, title, trader_col in categories:
            percentile_data_dict[col] = calculate_percentiles_for_column(df_plot, col, lookback_days)
        
        # Create subplot figure - 2 rows per category (value + percentile)
        num_categories = len(categories)
        fig = make_subplots(
            rows=num_categories * 2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.15, 0.08] * num_categories,  # Alternating heights for value and percentile
            specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]],
            subplot_titles=[title if j == 0 else "" for col, title, trader_col in categories for j in range(2)],  # Title only on value charts, not percentile
            horizontal_spacing=0.01
        )
        
        # Plot each category
        for idx, (col, title, trader_col) in enumerate(categories):
            row_value = idx * 2 + 1  # Value chart row
            row_percentile = idx * 2 + 2  # Percentile chart row
            
            # Plot average position bars (green)
            fig.add_trace(
                go.Bar(
                    x=df_plot['report_date_as_yyyy_mm_dd'],
                    y=df_plot[col],
                    name=title,
                    marker_color='#90EE90',  # Light green
                    marker_line_width=0,
                    showlegend=False
                ),
                row=row_value, col=1, secondary_y=False
            )
        
            # Plot trader count on secondary axis (blue)
            fig.add_trace(
                go.Scatter(
                    x=df_plot['report_date_as_yyyy_mm_dd'],
                    y=df_plot[trader_col],
                    name=f'{title} Traders',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ),
                row=row_value, col=1, secondary_y=True
            )
            
            # Plot percentile chart (purple)
            fig.add_trace(
                go.Scatter(
                    x=df_plot['report_date_as_yyyy_mm_dd'],
                    y=percentile_data_dict[col],
                    name=f'{title} Percentile',
                    fill='tozeroy',
                    line=dict(color='purple', width=1),
                    fillcolor='rgba(128, 0, 128, 0.1)',
                    showlegend=False
                ),
                row=row_percentile, col=1
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Avg Pos", row=row_value, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Traders", row=row_value, col=1, secondary_y=True)
            fig.update_yaxes(title_text="%ile", row=row_percentile, col=1, range=[0, 100])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Comprehensive Trader Participation Analysis - {instrument_name}",
                y=0.99,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            height=150 * num_categories * 2,  # Dynamic height based on number of charts
            showlegend=False,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=100, b=50),  # Increased top margin for buttons
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(label="All", step="all")
                    ]),
                    bgcolor='rgba(255,255,255,0.9)',
                    activecolor='lightblue',
                    x=0.01,
                    y=1.02,  # Moved lower to avoid title overlap
                    xanchor='left',
                    yanchor='top',
                    font=dict(size=11)
                ),
                type='date'
            )
        )
        
        # Add range slider only on the last x-axis
        last_row = num_categories * 2
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.02),
            row=last_row, col=1
        )
        
        # Configure all x-axes to be linked
        for i in range(1, last_row + 1):
            fig.update_xaxes(matches='x', row=i, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating participation density dashboard: {str(e)}")
        return None


def display_time_series_chart(df, instrument_name):
    """Display time series analysis - EXACT copy from legacyF.py"""
    st.subheader("📈 Time Series Analysis")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Standard Time Series", "Share of Open Interest", "Seasonality"])
    
    with tab1:
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
        
        # Column selection with checkboxes
        st.markdown("#### Select data series to plot:")
        
        # Define column display names
        column_display_names = {
            "open_interest_all": "Open Interest",
            "noncomm_positions_long_all": "Non-Commercial Long", 
            "noncomm_positions_short_all": "Non-Commercial Short",
            "comm_positions_long_all": "Commercial Long",
            "comm_positions_short_all": "Commercial Short",
            "net_noncomm_positions": "Net Non-Commercial",
            "net_comm_positions": "Net Commercial",
            "net_reportable_positions": "Net Reportable"
        }
        
        # Create columns for checkboxes
        col1, col2, col3 = st.columns(3)
        selected_columns = []
        
        # Get available columns in the data
        available_columns = list(column_display_names.keys())
        existing_columns = [col for col in available_columns if col in filtered_df.columns]
        
        # Distribute checkboxes across columns
        for idx, col in enumerate(existing_columns):
            with [col1, col2, col3][idx % 3]:
                if st.checkbox(column_display_names.get(col, col), 
                              value=(col == "open_interest_all"),
                              key=f"ts_checkbox_{col}"):
                    selected_columns.append(col)
        
        if selected_columns:
            fig = create_plotly_chart(filtered_df, selected_columns, f"{instrument_name} - Time Series Analysis")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one data series to plot")
    
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
            index=0,
            key="share_oi_side_selector"
        )
        
        # Create and display the chart
        fig = create_share_of_oi_chart(df, calculation_side, instrument_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Seasonality Analysis
        st.markdown("#### Seasonality Analysis")
        
        # Create grouped metrics structure
        metric_groups = {
            "📊 Core Metrics": {
                "open_interest_all": "Open Interest"
            },
            "📈 Long Positions": {
                "noncomm_positions_long_all": "Non-Commercial Long",
                "comm_positions_long_all": "Commercial Long",
                "tot_rept_positions_long_all": "Total Reportable Long",
                "nonrept_positions_long_all": "Non-Reportable Long"
            },
            "📉 Short Positions": {
                "noncomm_positions_short_all": "Non-Commercial Short",
                "comm_positions_short_all": "Commercial Short",
                "tot_rept_positions_short": "Total Reportable Short",
                "nonrept_positions_short_all": "Non-Reportable Short"
            },
            "🔄 Spread Positions": {
                "noncomm_postions_spread_all": "Non-Commercial Spread"
            },
            "⚖️ Net Positioning": {
                "net_noncomm_positions": "Net Non-Commercial",
                "net_comm_positions": "Net Commercial",
                "net_reportable_positions": "Net Reportable"
            },
            "👥 Trader Counts": {
                "traders_tot_all": "Total Traders",
                "traders_noncomm_long_all": "Non-Commercial Long Traders",
                "traders_noncomm_short_all": "Non-Commercial Short Traders",
                "traders_noncomm_spread_all": "Non-Commercial Spread Traders",
                "traders_comm_long_all": "Commercial Long Traders",
                "traders_comm_short_all": "Commercial Short Traders",
                "traders_tot_rept_long_all": "Total Reportable Long Traders",
                "traders_tot_rept_short_all": "Total Reportable Short Traders"
            },
            "📊 % of Open Interest": {
                "pct_of_oi_noncomm_long_all": "Non-Commercial Long",
                "pct_of_oi_noncomm_short_all": "Non-Commercial Short",
                "pct_of_oi_noncomm_spread": "Non-Commercial Spread",
                "pct_of_oi_comm_long_all": "Commercial Long",
                "pct_of_oi_comm_short_all": "Commercial Short",
                "pct_of_oi_nonrept_long_all": "Non-Reportable Long",
                "pct_of_oi_nonrept_short_all": "Non-Reportable Short"
            }
        }
        
        # Build options list with available columns only
        options = []
        option_labels = {}
        
        for group_name, group_metrics in metric_groups.items():
            group_options = [(k, v) for k, v in group_metrics.items() if k in df.columns]
            if group_options:
                # Add group header
                for key, label in group_options:
                    options.append(key)
                    option_labels[key] = f"{group_name} → {label}"
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Find default index
            default_key = 'net_noncomm_positions' if 'net_noncomm_positions' in options else (options[0] if options else None)
            default_index = options.index(default_key) if default_key and default_key in options else 0
            
            seasonality_column = st.selectbox(
                "Select metric:",
                options,
                format_func=lambda x: option_labels.get(x, x),
                index=default_index,
                key="seasonality_metric_selector"
            )
        
        with col2:
            lookback_years = st.selectbox(
                "Lookback period:",
                [5, 10, 'all'],
                format_func=lambda x: f"{x} Years" if x != 'all' else "All Time",
                index=0,
                key="seasonality_lookback"
            )
        
        with col3:
            zone_type = st.radio(
                "Zone type:",
                ['percentile', 'std'],
                format_func=lambda x: 'Percentile' if x == 'percentile' else 'Std Dev',
                index=0,
                key="seasonality_zone"
            )
        
        show_previous = st.checkbox("Show previous year", value=True, key="seasonality_prev_year")
        
        if seasonality_column in df.columns:
            fig = create_seasonality_chart(df, seasonality_column, lookback_years, show_previous, zone_type)
            if fig:
                st.plotly_chart(fig, use_container_width=True)


def display_percentile_chart(df, instrument_name):
    """Display percentile analysis - EXACT copy from legacyF.py"""
    st.subheader("📊 Percentile Analysis")
    
    # Create grouped metrics structure (same as seasonality)
    metric_groups = {
        "📊 Core Metrics": {
            "open_interest_all": "Open Interest"
        },
        "📈 Long Positions": {
            "noncomm_positions_long_all": "Non-Commercial Long",
            "comm_positions_long_all": "Commercial Long",
            "tot_rept_positions_long_all": "Total Reportable Long",
            "nonrept_positions_long_all": "Non-Reportable Long"
        },
        "📉 Short Positions": {
            "noncomm_positions_short_all": "Non-Commercial Short",
            "comm_positions_short_all": "Commercial Short",
            "tot_rept_positions_short": "Total Reportable Short",
            "nonrept_positions_short_all": "Non-Reportable Short"
        },
        "🔄 Spread Positions": {
            "noncomm_postions_spread_all": "Non-Commercial Spread"
        },
        "⚖️ Net Positioning": {
            "net_noncomm_positions": "Net Non-Commercial",
            "net_comm_positions": "Net Commercial",
            "net_reportable_positions": "Net Reportable"
        },
        "👥 Trader Counts": {
            "traders_tot_all": "Total Traders",
            "traders_noncomm_long_all": "Non-Commercial Long Traders",
            "traders_noncomm_short_all": "Non-Commercial Short Traders",
            "traders_noncomm_spread_all": "Non-Commercial Spread Traders",
            "traders_comm_long_all": "Commercial Long Traders",
            "traders_comm_short_all": "Commercial Short Traders",
            "traders_tot_rept_long_all": "Total Reportable Long Traders",
            "traders_tot_rept_short_all": "Total Reportable Short Traders"
        },
        "🎯 Concentration (Gross)": {
            "conc_gross_le_4_tdr_long": "Top 4 Long Traders",
            "conc_gross_le_4_tdr_short": "Top 4 Short Traders",
            "conc_gross_le_8_tdr_long": "Top 8 Long Traders",
            "conc_gross_le_8_tdr_short": "Top 8 Short Traders"
        },
        "🎯 Concentration (% of OI)": {
            "conc_net_le_4_tdr_long_all": "Top 4 Long Traders",
            "conc_net_le_4_tdr_short_all": "Top 4 Short Traders",
            "conc_net_le_8_tdr_long_all": "Top 8 Long Traders",
            "conc_net_le_8_tdr_short_all": "Top 8 Short Traders"
        },
        "📊 % of Open Interest": {
            "pct_of_oi_noncomm_long_all": "Non-Commercial Long",
            "pct_of_oi_noncomm_short_all": "Non-Commercial Short",
            "pct_of_oi_noncomm_spread": "Non-Commercial Spread",
            "pct_of_oi_comm_long_all": "Commercial Long",
            "pct_of_oi_comm_short_all": "Commercial Short",
            "pct_of_oi_nonrept_long_all": "Non-Reportable Long",
            "pct_of_oi_nonrept_short_all": "Non-Reportable Short"
        }
    }
    
    # Build options list with available columns only
    options = []
    option_labels = {}
    
    for group_name, group_metrics in metric_groups.items():
        group_options = [(k, v) for k, v in group_metrics.items() if k in df.columns]
        if group_options:
            # Add group header
            for key, label in group_options:
                options.append(key)
                option_labels[key] = f"{group_name} → {label}"
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Find default index
        default_key = 'net_noncomm_positions' if 'net_noncomm_positions' in options else (options[0] if options else None)
        default_index = options.index(default_key) if default_key and default_key in options else 0
        
        selected_metric = st.selectbox(
            "Select metric for analysis:",
            options,
            format_func=lambda x: option_labels.get(x, x),
            index=default_index,
            key="percentile_metric_selector"
        )
    
    with col2:
        lookback_period = st.selectbox(
            "Historical lookback:",
            ["1 Year", "2 Years", "5 Years", "10 Years", "All Time"],
            index=0
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
        "1 Year": 1,
        "2 Years": 2,
        "5 Years": 5,
        "10 Years": 10,
        "All Time": 'all'
    }
    
    fig = create_percentile_chart(df, selected_metric, lookback_map[lookback_period], chart_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def display_momentum_chart(df, instrument_name):
    """Display momentum dashboard - EXACT copy from legacyF.py"""
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
    """Display trader participation analysis - EXACT copy from legacyF.py"""
    st.subheader("👥 Trader Participation Analysis")
    
    # Check if trader count columns exist
    if 'traders_tot_all' not in df.columns:
        st.error("⚠️ Trader count data not available for this instrument.")
        return
    
    # Sub-analysis selection
    analysis_type = st.radio(
        "Select analysis type:",
        ["Participation Density Dashboard", "Concentration Divergence", "Heterogeneity & Regime Analysis", 
         "Concentration Momentum", "Participant Behavior Clusters", "Market Microstructure Analysis"],
        key="trader_analysis_type",
        horizontal=True
    )
    
    if analysis_type == "Participation Density Dashboard":
        st.markdown("#### 📊 Average Position per Trader Analysis")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Enhanced View", "Original View", "Concentration Flow", "Risk Heatmap", "Market Quadrant", "Spreading Activity"])
        
        with tab1:
            # Add explainer
            with st.expander("📖 Understanding This Analysis", expanded=False):
                st.markdown("""
                **What This Dashboard Shows:**
                
                **1. Average Position per Trader (Top Chart)**
                - **Bars**: Show the average position size per trader (Open Interest ÷ Total Traders)
                - **Blue Line**: Total number of traders participating in the market
                - **Bar Colors** (based on historical percentile with your selected lookback):
                    - 🟢 **Green**: Below 33rd percentile - Smaller than usual positions
                    - 🟡 **Yellow**: 33rd-67th percentile - Normal position sizes
                    - 🔴 **Red**: Above 67th percentile - Larger than usual positions
                
                **2. Top 4 Traders Concentration**
                - **Long Concentration**: Percentage of total open interest held by the 4 largest long traders
                - **Short Concentration**: Percentage of total open interest held by the 4 largest short traders
                - Higher values indicate more concentrated/less democratic markets
                
                **3. Percentile Rankings**
                - Shows where current concentration levels rank historically
                - 80th percentile = Higher than 80% of historical values (unusually concentrated)
                - 20th percentile = Lower than 80% of historical values (unusually democratic)
                
                **Why It Matters:**
                - High concentration + Few traders = Market dominated by large players (higher volatility risk)
                - Low concentration + Many traders = More democratic market (typically more stable)
                - Rising concentration often precedes major market moves
                """)
        
            # Add toggle for gross vs net concentration
            col_toggle, col_lookback, col_empty = st.columns([1, 1, 2])
            with col_toggle:
                concentration_type = st.radio(
                "Concentration Type:",
                ["Net", "Gross"],
                index=0,
                horizontal=True,
                help="Net: Position after offsetting long/short | Gross: Total position regardless of direction"
            )
        
            with col_lookback:
                lookback_period = st.selectbox(
                "Percentile Lookback:",
                ["6 Months", "1 Year", "2 Years", "5 Years", "10 Years", "Since 2010", "All Time"],
                index=1
            )
        
            # Map lookback to days
            lookback_map = {
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825,
            "10 Years": 3650,
            "Since 2010": "since_2010",  # Special case
            "All Time": None
        }
            lookback_days = lookback_map[lookback_period]
            
            # Calculate percentile based on selected lookback
            if lookback_days == "since_2010":
                lookback_date = pd.Timestamp('2010-01-01')
                df_lookback = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()
            elif lookback_days:
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
                if lookback_days == "since_2010":
                    lookback_start = pd.Timestamp('2010-01-01')
                    window_data = df[(df['report_date_as_yyyy_mm_dd'] >= lookback_start) & 
                                   (df['report_date_as_yyyy_mm_dd'] <= current_date)]
                elif lookback_days:
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
            
            # Create participation density chart with percentile data
            density_fig = create_participation_density_dashboard(df, instrument_name, percentile_data, concentration_type)
            
            # Update the percentile y-axis title dynamically with lookback period
            if density_fig:
                fig_update_text = f"Percentile ({lookback_period})"
                density_fig.update_yaxes(
                    title_text=fig_update_text,
                    row=2, col=1  # Second subplot is the avg position percentile chart
                )
                
                st.plotly_chart(density_fig, use_container_width=True)
        
        with tab2:
            # Add explainer for original view
            with st.expander("📖 Understanding Original Analysis", expanded=False):
                st.markdown("""
                **Original Comprehensive Participation Analysis**
                
                This shows the original comprehensive analysis from legacyF.py with ALL trader categories:
                
                - **Overall Average**: Average position per trader across all categories
                - **Non-Commercial Long/Short/Spread**: Average positions for each non-commercial category
                - **Commercial Long/Short**: Average positions for commercial traders
                - **Total Reportable Long/Short**: Average positions for all reportable traders
                
                Each category shows:
                - **Green Bars**: Average position size per trader
                - **Blue Line**: Number of traders (on right axis)
                - **Purple Area**: Historical percentile ranking of the average position size (NOT trader count)
                """)
            
            # Percentile lookback selector (same as original)
            col_lookback, col_empty = st.columns([1, 3])
            with col_lookback:
                lookback_period_original = st.selectbox(
                    "Percentile Lookback:",
                    ["6 Months", "1 Year", "2 Years", "5 Years", "10 Years", "Since 2010", "All Time"],
                    index=1,
                    key="original_lookback"
                )
            
            # Map lookback to days
            lookback_map_original = {
                "6 Months": 180,
                "1 Year": 365,
                "2 Years": 730,
                "5 Years": 1825,
                "10 Years": 3650,
                "Since 2010": "since_2010",  # Special case
                "All Time": None
            }
            lookback_days_original = lookback_map_original[lookback_period_original]
            
            # Create the EXACT original chart from legacyF.py
            fig_original = create_participation_density_dashboard_original(
                df, 
                instrument_name, 
                None,  # percentile_data not used in original
                lookback_days_original
            )
            if fig_original:
                st.plotly_chart(fig_original, use_container_width=True)
            else:
                st.error("Unable to create original participation density dashboard")
        
        with tab3:
            st.markdown("#### 🌊 Market Concentration Flow Analysis")
            
            # Explanation expander
            with st.expander("📖 Understanding Market Concentration", expanded=False):
                st.markdown("""
                **What is Market Concentration?**
                
                Market concentration measures whether positions are controlled by a few large traders (concentrated) 
                or distributed among many smaller traders (democratic/dispersed).
                
                **How we measure it:**
                - **Trader Count Percentile (T%)**: Where current trader count ranks vs historical (since 2010)
                - **Average Position Percentile (P%)**: Where average position size ranks vs historical
                
                **Concentration Levels:**
                - 🔴 **High**: Few traders (≤33%ile) with large positions (≥67%ile) - Market dominated by few
                - 🟠 **Medium-High**: Either few traders OR large positions - Somewhat concentrated
                - 🟡 **Medium**: Middle range for both metrics - Balanced market
                - 🟢 **Medium-Low**: Either many traders OR small positions - Somewhat dispersed
                - 🟢 **Low**: Many traders (≥67%ile) with small positions (≤33%ile) - Democratic market
                
                **Why it matters:**
                High concentration suggests potential for larger price moves as few traders control the market.
                Low concentration indicates a more stable, democratized market with diverse participation.
                """)
            
            # Time period selection
            col1, col2 = st.columns([1, 3])
            with col1:
                flow_lookback = st.selectbox(
                    "Compare periods:",
                    ["Week over Week", "Month over Month", "Quarter over Quarter"],
                    index=0
                )
            
            # Map to days
            lookback_map = {"Week over Week": 7, "Month over Month": 30, "Quarter over Quarter": 90}
            lookback_days = lookback_map[flow_lookback]
            
            # Get current and previous period data
            latest_date = df['report_date_as_yyyy_mm_dd'].max()
            previous_date = latest_date - pd.Timedelta(days=lookback_days)
            
            # Find closest available dates
            df['date_diff_prev'] = abs(df['report_date_as_yyyy_mm_dd'] - previous_date)
            prev_idx = df['date_diff_prev'].idxmin()
            prev_data = df.loc[prev_idx]
            
            current_data = df[df['report_date_as_yyyy_mm_dd'] == latest_date].iloc[0]
            
            # Create a grouped bar chart instead of Sankey for better visibility
            categories = [
                ('Non-Comm Long', 'noncomm_positions_long_all', 'traders_noncomm_long_all'),
                ('Non-Comm Short', 'noncomm_positions_short_all', 'traders_noncomm_short_all'),
                ('Commercial Long', 'comm_positions_long_all', 'traders_comm_long_all'),
                ('Commercial Short', 'comm_positions_short_all', 'traders_comm_short_all')
            ]
            
            # Prepare data for visualization
            data_for_plot = []
            for cat_name, pos_col, trader_col in categories:
                # Previous period
                prev_traders = float(prev_data[trader_col]) if pd.notna(prev_data[trader_col]) else 0
                prev_avg = float(prev_data[pos_col]) / prev_traders if prev_traders > 0 else 0
                
                # Current period
                curr_traders = float(current_data[trader_col]) if pd.notna(current_data[trader_col]) else 0
                curr_avg = float(current_data[pos_col]) / curr_traders if curr_traders > 0 else 0
                
                # Classify concentration levels based on historical percentiles
                def get_concentration_level(avg_pos, trader_count, pos_col, trader_col, df):
                    # Calculate historical percentiles for this category
                    # Use all data since 2010 for percentile calculation
                    lookback_date = pd.Timestamp('2010-01-01')
                    hist_data = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()
                    
                    # Calculate average positions for historical data
                    hist_data['avg_pos'] = hist_data[pos_col] / hist_data[trader_col]
                    hist_data = hist_data[hist_data[trader_col] > 0]  # Filter out zero traders
                    
                    # Get percentiles
                    trader_percentile = stats.percentileofscore(hist_data[trader_col], trader_count)
                    avg_pos_percentile = stats.percentileofscore(hist_data['avg_pos'], avg_pos)
                    
                    # High concentration: Few traders (low percentile) with large positions (high percentile)
                    # Low concentration: Many traders (high percentile) with small positions (low percentile)
                    
                    if trader_percentile <= 33 and avg_pos_percentile >= 67:
                        return "High"  # Few traders, large positions
                    elif trader_percentile >= 67 and avg_pos_percentile <= 33:
                        return "Low"   # Many traders, small positions
                    elif trader_percentile <= 33 or avg_pos_percentile >= 67:
                        return "Medium-High"  # Either few traders OR large positions
                    elif trader_percentile >= 67 or avg_pos_percentile <= 33:
                        return "Medium-Low"   # Either many traders OR small positions
                    else:
                        return "Medium"  # Middle range for both
                
                prev_level = get_concentration_level(prev_avg, prev_traders, pos_col, trader_col, df)
                curr_level = get_concentration_level(curr_avg, curr_traders, pos_col, trader_col, df)
                
                # Calculate percentiles for display
                lookback_date = pd.Timestamp('2010-01-01')
                hist_data = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()
                hist_data['avg_pos'] = hist_data[pos_col] / hist_data[trader_col]
                hist_data = hist_data[hist_data[trader_col] > 0]
                
                prev_trader_pct = stats.percentileofscore(hist_data[trader_col], prev_traders)
                prev_pos_pct = stats.percentileofscore(hist_data['avg_pos'], prev_avg)
                curr_trader_pct = stats.percentileofscore(hist_data[trader_col], curr_traders)
                curr_pos_pct = stats.percentileofscore(hist_data['avg_pos'], curr_avg)
                
                data_for_plot.append({
                    'Category': cat_name,
                    'Period': 'Previous',
                    'Concentration': prev_level,
                    'Avg Position': prev_avg,
                    'Trader Count': prev_traders,
                    'Total Position': float(prev_data[pos_col]),
                    'Trader Percentile': prev_trader_pct,
                    'Position Percentile': prev_pos_pct
                })
                
                data_for_plot.append({
                    'Category': cat_name,
                    'Period': 'Current',
                    'Concentration': curr_level,
                    'Avg Position': curr_avg,
                    'Trader Count': curr_traders,
                    'Total Position': float(current_data[pos_col]),
                    'Trader Percentile': curr_trader_pct,
                    'Position Percentile': curr_pos_pct
                })
            
            # Create DataFrame
            plot_df = pd.DataFrame(data_for_plot)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Average Position per Trader', 'Trader Count', 
                              'Concentration Levels (T%=Traders, P%=Position)', 'Total Positions'),
                vertical_spacing=0.18,
                horizontal_spacing=0.12,
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Color mapping
            colors = {'Previous': '#90EE90', 'Current': '#4169E1'}
            concentration_colors = {
                'High': '#DC143C',         # Crimson red - high concentration risk
                'Medium-High': '#FF8C00',  # Dark orange
                'Medium': '#FFD700',       # Gold
                'Medium-Low': '#9ACD32',   # Yellow green
                'Low': '#32CD32'           # Lime green - low concentration (more democratic)
            }
            
            # Plot 1: Average Position per Trader
            for period in ['Previous', 'Current']:
                period_data = plot_df[plot_df['Period'] == period]
                fig.add_trace(
                    go.Bar(
                        x=period_data['Category'],
                        y=period_data['Avg Position'],
                        name=period,
                        marker_color=colors[period],
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Plot 2: Trader Count
            for period in ['Previous', 'Current']:
                period_data = plot_df[plot_df['Period'] == period]
                fig.add_trace(
                    go.Bar(
                        x=period_data['Category'],
                        y=period_data['Trader Count'],
                        name=period,
                        marker_color=colors[period],
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Plot 3: Concentration Level with Percentiles
            for cat in ['Non-Comm Long', 'Non-Comm Short', 'Commercial Long', 'Commercial Short']:
                prev_data_cat = plot_df[(plot_df['Category'] == cat) & (plot_df['Period'] == 'Previous')].iloc[0]
                curr_data_cat = plot_df[(plot_df['Category'] == cat) & (plot_df['Period'] == 'Current')].iloc[0]
                
                # Create text with percentiles
                prev_text = f"T:{prev_data_cat['Trader Percentile']:.0f}%<br>P:{prev_data_cat['Position Percentile']:.0f}%"
                curr_text = f"T:{curr_data_cat['Trader Percentile']:.0f}%<br>P:{curr_data_cat['Position Percentile']:.0f}%"
                
                # Plot previous period
                fig.add_trace(
                    go.Scatter(
                        x=[cat],
                        y=['Previous'],
                        mode='markers+text',
                        marker=dict(
                            size=40,
                            color=concentration_colors[prev_data_cat['Concentration']],
                            line=dict(width=2, color='black')
                        ),
                        text=prev_text,
                        textposition='middle center',
                        textfont=dict(size=10, color='black', family='Arial Black'),
                        showlegend=False,
                        name=prev_data_cat['Concentration']
                    ),
                    row=2, col=1
                )
                
                # Plot current period
                fig.add_trace(
                    go.Scatter(
                        x=[cat],
                        y=['Current'],
                        mode='markers+text',
                        marker=dict(
                            size=40,
                            color=concentration_colors[curr_data_cat['Concentration']],
                            line=dict(width=2, color='black')
                        ),
                        text=curr_text,
                        textposition='middle center',
                        textfont=dict(size=10, color='black', family='Arial Black'),
                        showlegend=False,
                        name=curr_data_cat['Concentration']
                    ),
                    row=2, col=1
                )
            
            # Plot 4: Total Positions
            for period in ['Previous', 'Current']:
                period_data = plot_df[plot_df['Period'] == period]
                fig.add_trace(
                    go.Bar(
                        x=period_data['Category'],
                        y=period_data['Total Position'],
                        name=period,
                        marker_color=colors[period],
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f"Market Concentration Flow Analysis ({flow_lookback})",
                height=800,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(tickangle=-45)
            fig.update_yaxes(title_text="Avg Position", row=1, col=1, tickformat=",.0f")
            fig.update_yaxes(title_text="Trader Count", row=1, col=2, tickformat=",.0f")
            fig.update_yaxes(title_text="Period", row=2, col=1, categoryorder="array", categoryarray=["Previous", "Current"])
            fig.update_xaxes(row=2, col=1, tickangle=-25)
            fig.update_yaxes(title_text="Total Position", row=2, col=2, tickformat=",.0f")
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Previous Date", prev_data['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'))
            with col2:
                st.metric("Current Date", current_data['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'))
            with col3:
                # Calculate total trader change from the plot data
                prev_total = plot_df[plot_df['Period'] == 'Previous']['Trader Count'].sum()
                curr_total = plot_df[plot_df['Period'] == 'Current']['Trader Count'].sum()
                total_trader_change = int(curr_total - prev_total)
                st.metric("Total Trader Change", f"{total_trader_change:+d}")
        
        with tab4:
            st.markdown("#### 🔥 Concentration Risk Heatmap")
            
            # Explanation expander
            with st.expander("📖 Understanding the Concentration Risk Heatmap", expanded=False):
                st.markdown("""
                **What does each square represent?**
                
                Each square shows a **Risk Score (0-100)** for a specific trader category at a specific time period.
                
                **Risk Score Calculation:**
                ```
                Risk Score = (Position Concentration × 70%) + (Inverse Trader Participation × 30%)
                ```
                
                **Important Methodology Note:**
                - The "Top 4/8 Traders" concentration data **does NOT specify which trader groups** are actually in those top positions
                - We estimate concentration risk by comparing trader counts in each category with overall concentration levels
                - **Long Concentration Metrics**: Only analyze Non-Commercial Long & Commercial Long (directionally relevant)
                - **Short Concentration Metrics**: Only analyze Non-Commercial Short & Commercial Short (directionally relevant)
                - **Non-Reportable traders excluded**: These are small traders below reporting thresholds, extremely unlikely to hold positions large enough to be in top 4/8 traders
                - Categories with fewer traders + high concentration = Higher estimated risk of being in the concentrated group
                
                **Color Scale:**
                - 🟢 **Green (0-25)**: Low risk - Many traders, well-distributed positions
                - 🟡 **Gold (25-50)**: Medium risk - Moderate concentration
                - 🟠 **Orange (50-75)**: High risk - Significant concentration
                - 🔴 **Red (75-100)**: Very high risk - Market dominated by few large traders
                """)
            
            st.info("Visualizes concentration risk over time by combining trader participation rates with position concentration metrics")
            
            # Configuration columns
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # Concentration metric selection
                conc_metric = st.selectbox(
                    "Position Concentration Metric:",
                    [
                        ("4 or Less Traders - Gross Long", "conc_gross_le_4_tdr_long"),
                        ("4 or Less Traders - Gross Short", "conc_gross_le_4_tdr_short"),
                        ("8 or Less Traders - Gross Long", "conc_gross_le_8_tdr_long"),
                        ("8 or Less Traders - Gross Short", "conc_gross_le_8_tdr_short"),
                        ("4 or Less Traders - Net Long", "conc_net_le_4_tdr_long_all"),
                        ("4 or Less Traders - Net Short", "conc_net_le_4_tdr_short_all"),
                        ("8 or Less Traders - Net Long", "conc_net_le_8_tdr_long_all"),
                        ("8 or Less Traders - Net Short", "conc_net_le_8_tdr_short_all")
                    ],
                    format_func=lambda x: x[0],
                    index=0,
                    key="risk_heatmap_conc_metric"
                )
            
            with col2:
                # Time aggregation
                time_agg = st.selectbox(
                    "Time Aggregation:",
                    ["Weekly", "Monthly", "Quarterly"],
                    index=0,
                    key="risk_heatmap_time_agg"
                )
            
            with col3:
                # Lookback period
                lookback_years = st.slider(
                    "Years of History:",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                    key="risk_heatmap_lookback"
                )
            
            heatmap_fig = create_concentration_risk_heatmap(df, instrument_name, conc_metric, time_agg, lookback_years)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                st.error("Unable to create concentration risk heatmap")
        
        with tab5:
            st.markdown("#### 🎯 Market Structure Quadrant Analysis")
            
            # Explanation
            with st.expander("📖 Understanding the Market Structure Quadrant", expanded=False):
                st.markdown("""
                **What does this chart show?**
                
                Maps the current market structure based on two key dimensions:
                1. **X-axis**: Trader Participation (Few → Many traders)
                2. **Y-axis**: Position Concentration (Low → High concentration)
                
                **Quadrants:**
                - **Oligopolistic (Red)**: Few traders, high concentration - Highest risk
                - **Crowded Concentration (Orange)**: Many traders, high concentration - Herding risk
                - **Specialized (Yellow)**: Few traders, low concentration - Niche market
                - **Democratic (Green)**: Many traders, low concentration - Most stable
                
                **Visualization Details:**
                - **Bubble size**: Represents total open interest
                - **Bubble opacity**: Current (100%), Past 4 weeks (70%), 2-3 months ago (40%)
                - **Evolution**: Shows past 4 weeks, 2 months ago, and 3 months ago
                - **Arrow**: Shows direction of movement from 1 week ago to current
                """)
            
            # Configuration
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Concentration metric selection
                conc_metric_quad = st.selectbox(
                    "Select Concentration Metric:",
                    [
                        ("Top 4 Net Long", "conc_net_le_4_tdr_long_all"),
                        ("Top 4 Net Short", "conc_net_le_4_tdr_short_all"),
                        ("Top 4 Gross Long", "conc_gross_le_4_tdr_long"),
                        ("Top 4 Gross Short", "conc_gross_le_4_tdr_short")
                    ],
                    format_func=lambda x: x[0],
                    key="quad_conc_metric"
                )
            
            with col2:
                show_evolution = st.checkbox("Show Evolution", value=True, 
                                           help="Shows past 4 weeks, 2 months ago, and 3 months ago",
                                           key="quad_show_evolution")
            
            quad_fig = create_market_structure_quadrant(df, instrument_name, conc_metric_quad, show_evolution)
            if quad_fig:
                st.plotly_chart(quad_fig, use_container_width=True)
                
                # Current position summary
                st.markdown("### Current Market Structure")
                latest = df.iloc[-1]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Latest Date", latest['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'))
                    # Show trader count based on concentration metric direction
                    if 'long' in conc_metric_quad[1].lower():
                        if 'traders_noncomm_long_all' in df.columns:
                            st.metric("Non-Commercial Long Traders", f"{latest['traders_noncomm_long_all']:.0f}")
                    else:  # short
                        if 'traders_noncomm_short_all' in df.columns:
                            st.metric("Non-Commercial Short Traders", f"{latest['traders_noncomm_short_all']:.0f}")
                
                with col2:
                    st.metric("Concentration Level", f"{latest[conc_metric_quad[1]]:.1f}%")
                    # Show trader count based on concentration metric direction
                    if 'long' in conc_metric_quad[1].lower():
                        if 'traders_comm_long_all' in df.columns:
                            st.metric("Commercial Long Traders", f"{latest['traders_comm_long_all']:.0f}")
                    else:  # short
                        if 'traders_comm_short_all' in df.columns:
                            st.metric("Commercial Short Traders", f"{latest['traders_comm_short_all']:.0f}")
        
        with tab6:
            st.markdown("#### 📈 Non-Commercial Spreading Activity")
            
            # Explanation
            with st.expander("📖 Understanding Spreading Activity", expanded=False):
                st.markdown("""
                **What is the Spread/Directional Ratio?**
                
                Measures the relative preference for spread strategies vs directional positions among non-commercial traders:
                
                **Calculation:** Spread Traders ÷ (Long-Only + Short-Only Traders)
                
                **Interpretation:**
                - **Ratio < 0.5**: Strong directional conviction (few spreaders)
                - **Ratio = 1.0**: Equal balance between spread and directional traders
                - **Ratio > 1.5**: High hedging/uncertainty (many spreaders)
                
                **Why This Matters:**
                - Rising ratio often signals increasing market uncertainty
                - Falling ratio suggests growing directional conviction
                - Extreme values may indicate regime changes
                - Helps identify when speculators are hedging vs taking outright positions
                
                **Note:** Only non-commercials can hold spread positions in COT data
                """)
            
            spread_fig = create_spreading_activity_analysis(df, instrument_name)
            if spread_fig:
                st.plotly_chart(spread_fig, use_container_width=True)
                
                # Current status
                latest = df.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Spread Traders", f"{latest['traders_noncomm_spread_all']:.0f}")
                
                with col2:
                    st.metric("Long-Only Traders", f"{latest['traders_noncomm_long_all']:.0f}")
                
                with col3:
                    st.metric("Short-Only Traders", f"{latest['traders_noncomm_short_all']:.0f}")
                
                with col4:
                    ratio = latest['traders_noncomm_spread_all'] / (latest['traders_noncomm_long_all'] + latest['traders_noncomm_short_all'])
                    st.metric("Current Ratio", f"{ratio:.3f}")
    
    
    elif analysis_type == "Concentration Divergence":
        st.markdown("#### 🔄 Concentration Divergence Analysis")
        
        # Explanation
        with st.expander("📖 Understanding Concentration Divergence", expanded=False):
            st.markdown("""
            **What is Concentration Divergence?**
            
            Measures the difference in market concentration between different groups:
            - **Category Divergence**: Commercial vs Non-Commercial concentration
            - **Directional Divergence**: Long vs Short side concentration
            
            **Category Divergence Methodology (Commercial vs Non-Commercial):**
            
            For Long Positions example:
            1. **Position Shares** (% of total reportable long positions):
               - Commercial Share = Commercial Long Positions ÷ Total Reportable Long × 100
               - Non-Commercial Share = (Non-Comm Long + Spread) ÷ Total Reportable Long × 100
            
            2. **Trader Shares** (% of total reportable long traders):
               - Commercial Trader Share = Commercial Long Traders ÷ Total Reportable Long Traders × 100
               - Non-Commercial Trader Share = (Non-Comm Long + Spread Traders) ÷ Total Reportable Long Traders × 100
            
            3. **Concentration Scores** (position dominance relative to participation):
               - Commercial Concentration = Commercial Position Share ÷ Commercial Trader Share
               - Non-Commercial Concentration = Non-Commercial Position Share ÷ Non-Commercial Trader Share
            
            4. **Divergence Score** = (Non-Commercial Concentration - Commercial Concentration) × 10
            
            **Intuition & Rationale:**
            - **Concentration > 1.0**: Group holds MORE than their "fair share" - average trader is larger than typical
            - **Concentration < 1.0**: Group holds LESS than their "fair share" - average trader is smaller than typical
            - **Positive Divergence**: Non-commercials are more concentrated (fewer large traders dominate)
            - **Negative Divergence**: Commercials are more concentrated (fewer large hedgers dominate)
            
            **Example Interpretation:**
            If commercials are 20% of traders but hold 40% of positions → Concentration = 2.0
            If non-commercials are 80% of traders but hold 60% of positions → Concentration = 0.75
            Divergence = (0.75 - 2.0) × 10 = -12.5 (commercials more concentrated)
            
            **Why This Matters:**
            - Identifies which group has larger average position sizes
            - Reveals market structure imbalances
            - High concentration in one group may indicate conviction or risk
            - Changes in divergence can signal shifting market dynamics
            
            **Note on Data:**
            We use reportable positions/traders only (excluding non-reportable) for cleaner institutional analysis.
            Spread positions are grouped with non-commercial as they represent speculative strategies.
            """)
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            divergence_type = st.selectbox(
                "Divergence Type:",
                [
                    "Category Divergence (Commercial vs Non-Commercial)",
                    "Directional Divergence (Long vs Short)"
                ],
                index=0
            )
        
        with col2:
            if divergence_type == "Category Divergence (Commercial vs Non-Commercial)":
                conc_side = st.radio(
                    "Position Side:",
                    ["Long Positions", "Short Positions"],
                    horizontal=True
                )
            elif divergence_type == "Directional Divergence (Long vs Short)":
                trader_cat = st.radio(
                    "Trader Category:",
                    ["All Traders", "Commercial", "Non-Commercial"],
                    horizontal=True
                )
        
        # Determine which parameter to pass based on divergence type
        if divergence_type == "Category Divergence (Commercial vs Non-Commercial)":
            extra_param = conc_side
        else:  # Directional Divergence (Long vs Short)
            extra_param = trader_cat
        
        div_fig = create_concentration_divergence_analysis(df, instrument_name, divergence_type, extra_param)
        if div_fig:
            st.plotly_chart(div_fig, use_container_width=True)
    
    elif analysis_type == "Heterogeneity & Regime Analysis":
        st.markdown("#### 🔀 Market Heterogeneity & Regime Analysis")
        
        # Create tabs for the two analyses
        tab1, tab2 = st.tabs(["Heterogeneity", "Regime Detection"])
        
        with tab1:
            # Explanation
            with st.expander("📖 Understanding Heterogeneity Index - Complete Guide", expanded=False):
                st.markdown("""
            ## **What is the Heterogeneity Index?**
            
            A composite measure (0-100) that quantifies market disagreement between commercial hedgers and non-commercial speculators across four distinct dimensions. Higher values indicate greater divergence in behavior, positioning, and market views between these two key groups.
            
            ## **Why This Matters**
            
            When commercials and non-commercials strongly disagree:
            - **Market turning points** often occur as one group is typically right
            - **Volatility increases** due to opposing forces
            - **Trend changes** become more likely as positions unwind
            - **Risk/reward improves** for following the historically correct group
            
            ---
            
            ## **Component 1: Directional Opposition (0-25 points)**
            
            ### What It Measures
            Whether groups are at different extremes in their NET positioning relative to their own historical patterns.
            
            ### Calculation Details
            1. **Net Position**: `comm_net = comm_long - comm_short` (same for non-comm)
            2. **Z-Score**: How many standard deviations from 52-week average
               - Example: If commercials typically net short -50k but are now -150k, that might be -3 z-score
            3. **Divergence**: `abs(comm_zscore - noncomm_zscore)`
            4. **Scaling**: Divergence × 25, capped at 100, then × 0.25
            
            ### Rationale & Interpretation
            - **Why Z-scores?** Normalizes for different typical position sizes between groups
            - **High values (20-25)**: One group at historical extreme, other is not
            - **Low values (0-5)**: Both groups similarly positioned vs their history
            - **Example**: Commercials at -3σ (very bearish) while non-commercials at +2σ (very bullish) = High divergence
            
            ### Key Questions Answered
            - **Q: Why not just use position differences?** A: Groups have different normal ranges
            - **Q: What if both are extreme but same direction?** A: Low divergence (both at +3σ = similar behavior)
            
            ---
            
            ## **Component 2: Flow Intensity/Urgency (0-25 points)**
            
            ### What It Measures
            Whether one group is making UNUSUALLY LARGE position changes while the other isn't.
            
            ### Calculation Details
            1. **Weekly Flow**: `comm_flow = this_week_net - last_week_net`
            2. **Flow Z-Score**: How unusual is this week's change?
               - Based on 52-week history of flows
            3. **Divergence**: `abs(comm_flow_zscore - noncomm_flow_zscore)`
            4. **Scaling**: Same as Component 1
            
            ### Rationale & Interpretation
            - **Why flows matter**: Captures urgency/panic in positioning
            - **High values (20-25)**: One group scrambling while other is calm
            - **Low values (0-5)**: Both making normal-sized moves
            - **Example**: Commercials suddenly cover shorts (z=3) while non-commercials barely change (z=0.2)
            
            ### Key Questions Answered
            - **Q: How is this different from Component 1?** A: Measures rate of change, not levels
            - **Q: Why use z-scores again?** A: Some weeks naturally have larger flows (expiry, events)
            
            ---
            
            ## **Component 3: Percentile Distance (0-25 points)**
            
            ### What It Measures
            How far apart the groups are in terms of their average position sizes (concentration).
            
            ### Calculation Details
            1. **Average Position**: `nc_avg = noncomm_long_positions / noncomm_long_traders`
            2. **Historical Percentile**: Where does today's average rank vs 52-week history?
               - 95th percentile = larger positions than 95% of past year
            3. **Distance**: `abs(nc_percentile - comm_percentile)`
            4. **Scaling**: Distance × 0.25 (already 0-100)
            
            ### Rationale & Interpretation
            - **Why this matters**: Shows position concentration divergence
            - **High values (20-25)**: One group using huge positions, other using small
            - **Low values (0-5)**: Both groups have similar position concentrations
            - **Example**: Non-comm at 90th percentile (huge positions) vs Comm at 20th (small positions)
            
            ### Key Questions Answered
            - **Q: Why only long positions?** A: Simplicity; could extend to include shorts
            - **Q: What about trader count changes?** A: Normalized by dividing by trader count
            
            ---
            
            ## **Component 4: Cross-Category Positioning (0-25 points)**
            
            ### What It Measures
            Whether groups are betting AGAINST each other (opposite sides of market).
            
            ### Calculation Details
            1. **Position Shares**: Use % of open interest for each position type
            2. **Cross-Alignments**:
               - `NC_long_with_C_short = average(%OI_nc_long, %OI_comm_short)`
               - `NC_short_with_C_long = average(%OI_nc_short, %OI_comm_long)`
            3. **Divergence**: `abs(difference between alignments)`
            4. **Scaling**: Divergence × 2, capped at 100, then × 0.25
            
            ### Rationale & Interpretation
            - **Why this formula?** When groups oppose, one alignment dominates
            - **High values (20-25)**: Groups on opposite sides (classic disagreement)
            - **Low values (0-5)**: Groups on same side (both bullish or bearish)
            - **Example**: NC 35% long + C 30% short vs NC 5% short + C 10% long = Big divergence
            
            ### Key Questions Answered
            - **Q: Why average the cross positions?** A: Captures total "opposition strength"
            - **Q: What's the maximum possible?** A: About 50% difference (very rare)
            
            ---
            
            ## **Final Index Interpretation**
            
            ### Scale Breakdown
            - **0-25**: Groups largely agree → Normal market conditions
            - **25-50**: Some disagreement → Watch for developing divergence
            - **50-75**: Significant disagreement → Potential turning point
            - **75-100**: Extreme disagreement → High probability of major move
            
            ### Using the Dropdown
            View individual components to identify:
            - Which type of divergence is driving the index
            - Whether it's position levels, flows, concentration, or opposition
            - Historical context for each component
            
            ### Trading Implications
            - **Rising index**: Increasing disagreement, volatility likely
            - **Falling index**: Groups aligning, trend continuation likely
            - **Extreme readings**: Often precede significant reversals
            - **Component analysis**: Reveals the nature of disagreement
            
            ---
            
            ## **Real-World Reversal Scenarios**
            
            ### Scenario 1: The Unexpected Commercial Buying
            **Setup**: Prices have been rising steadily, but speculators suddenly stop buying or even start selling, while commercials unexpectedly flip from their typical selling to aggressive buying.
            
            **How Our Index Captures This:**
            - **Component 2 (Flow Intensity)**: Goes to 20-25 as commercials show urgent buying (z=3+) while specs show normal or negative flows
            - **Component 4 (Cross-Category)**: May decrease as groups start aligning (both becoming bullish)
            - **Signal**: This contradiction often foreshadows a major reversal - commercials rarely buy into strength unless they expect higher prices
            
            ### Scenario 2: The Classic Market Top
            **Setup**: Managed-money speculators aggressively ramp up long positions while commercials simultaneously increase their short positions to extreme levels.
            
            **How Our Index Captures This:**
            - **Component 1 (Directional Opposition)**: Maximum score (25) as specs hit +3σ bullish while commercials hit -3σ bearish
            - **Component 2 (Flow Intensity)**: High scores as both groups make urgent moves in opposite directions
            - **Component 4 (Cross-Category)**: Maximum score as NC long % and Comm short % both spike
            - **Total Index**: Often 75-100, screaming "reversal ahead!"
            
            ### Scenario 3: The Quiet Divergence
            **Setup**: While prices drift sideways, commercials quietly accumulate longs using larger position sizes, while speculators maintain many small short positions.
            
            **How Our Index Captures This:**
            - **Component 3 (Percentile Distance)**: High score as commercial average positions hit 90th percentile while spec positions stay at 20th percentile
            - **Component 1**: May be moderate as neither group is at z-score extremes yet
            - **Signal**: Position concentration divergence often leads price divergence
            
            ### Key Insight
                When the index reads above 75, it's often because one of these classic setups is occurring. The beauty of our multi-component approach is that it captures various forms of disagreement - whether it's urgent flows, extreme positions, concentration differences, or direct opposition. Historical analysis shows these patterns consistently precede major market turns.
                """)
            
            # Add component view selector
            col1, col2 = st.columns([1, 3])
            with col1:
                component_view = st.selectbox(
                    "View Component:",
                    ["Full Index", "Directional Opposition", "Flow Intensity", "Percentile Distance", "Cross-Category Positioning"],
                    index=0,
                    key="heterogeneity_component_view"
                )
            
            het_fig = create_heterogeneity_index(df, instrument_name, component_view)
            if het_fig:
                st.plotly_chart(het_fig, use_container_width=True)
        
        with tab2:
            # Explanation
            with st.expander("📖 Understanding Regime Detection - Complete Guide", expanded=False):
                st.markdown("""
            **What is Regime Detection?**
            
            A comprehensive system that identifies distinct market states by analyzing extreme readings across seven key dimensions of trader behavior. This helps identify when market conditions deviate significantly from historical norms.
            
            **Detailed Methodology:**
            
            **Step 1: Calculate Percentile Rankings (52-week rolling window)**
            - Each metric is ranked against its own 52-week history
            - Percentiles range from 0 (lowest in 52 weeks) to 100 (highest in 52 weeks)
            - Minimum 26 weeks of data required to begin calculations
            
            **Step 2: Calculate Individual Metric Components**
            
            1. **Long Concentration Percentile** = `rank(conc_gross_le_4_tdr_long) / count * 100`
               - Measures: How concentrated long positions are among top 4 traders
               - High percentile (>85): Few traders control most long positions
               
            2. **Short Concentration Percentile** = `rank(conc_gross_le_4_tdr_short) / count * 100`
               - Measures: How concentrated short positions are among top 4 traders
               - High percentile (>85): Few traders control most short positions
            
            3. **Commercial Net Percentile** = `rank(comm_long - comm_short) / count * 100`
               - Measures: How extreme commercial net positioning is
               - High percentile (>85): Commercials unusually long
               - Low percentile (<15): Commercials unusually short
            
            4. **Non-Commercial Net Percentile** = `rank(noncomm_long - noncomm_short) / count * 100`
               - Measures: How extreme speculative net positioning is
               - High percentile (>85): Speculators unusually long
               - Low percentile (<15): Speculators unusually short
            
            5. **Flow Intensity Percentile** = `rank(|comm_flow| + |noncomm_flow|) / count * 100`
               - Measures: Magnitude of weekly position changes
               - High percentile (>85): Unusually large position adjustments
            
            6. **Total Traders Percentile** = `rank(traders_tot_all) / count * 100`
               - Measures: Market participation level
               - High percentile (>70): High participation
               - Low percentile (<30): Low participation
            
            7. **Heterogeneity Percentile** = Currently fixed at 50 (placeholder)
               - Would measure: Inter-group behavioral divergence
            
            **Step 3: Calculate Extremity Score (0-100)**
            
            ```
            distance_from_center(x) = |x - 50| × 2
            
            Extremity = 0.25 × max(distance_from_center(long_conc), distance_from_center(short_conc))
                      + 0.25 × max(distance_from_center(comm_net), distance_from_center(noncomm_net))
                      + 0.25 × flow_intensity_percentile
                      + 0.25 × heterogeneity_percentile
            ```
            
            **Step 4: Regime Classification Rules**
            
            The system checks conditions in order and assigns the first matching regime:
            
            1. **🔴 Long Concentration Extreme**
               - Condition: Long concentration >85th AND Short concentration <30th percentile
               - Meaning: Long side dominated by few large traders, short side distributed
               - Risk: Potential long squeeze if large longs liquidate
            
            2. **🔴 Short Concentration Extreme**
               - Condition: Short concentration >85th AND Long concentration <30th percentile
               - Meaning: Short side dominated by few large traders, long side distributed
               - Risk: Potential short squeeze if large shorts cover
            
            3. **🟠 Bilateral Concentration**
               - Condition: Both Long AND Short concentration >85th percentile
               - Meaning: Both sides dominated by large institutional players
               - Risk: Volatile moves when either side adjusts positions
            
            4. **🔴 Speculative Long Extreme**
               - Condition: Non-commercial net >85th AND Commercial net <15th percentile
               - Meaning: Speculators extremely long, commercials extremely short
               - Risk: Classic overbought condition, vulnerable to reversal
            
            5. **🟠 Commercial Long Extreme**
               - Condition: Non-commercial net <15th AND Commercial net >85th percentile
               - Meaning: Commercials extremely long, speculators extremely short
               - Risk: Potential bottom, commercials often early
            
            6. **🟡 High Flow Volatility**
               - Condition: Flow intensity >85th percentile
               - Meaning: Unusually large week-over-week position changes
               - Risk: Market in transition, direction uncertain
            
            7. **🔴 Maximum Divergence**
               - Condition: Heterogeneity >85th percentile
               - Meaning: Trader groups behaving very differently from each other
               - Risk: Fundamental disagreement, potential for large moves
            
            8. **🟢 Balanced Market**
               - Condition: Extremity score <40
               - Meaning: All metrics within normal ranges
               - Risk: Low - normal market conditions
            
            9. **⚪ Transitional**
               - Condition: Some elevation but no specific extreme pattern
               - Meaning: Market between regimes
               - Risk: Moderate - watch for emerging patterns
            
            **Extremity Score Interpretation:**
            - **0-40**: Normal conditions (Green zone)
            - **40-70**: Elevated conditions (Yellow zone)
            - **70-100**: Extreme conditions (Red zone)
            
            **How to Use This Dashboard:**
            
            1. **Check Current Regime**: Identifies the dominant market characteristic
            2. **Monitor Duration**: Longer durations suggest persistent conditions
            3. **Review Extremity Score**: Higher scores = more unusual conditions
            4. **Analyze Spider Chart**: See which specific metrics are extreme
            5. **Study Timeline**: Understand regime persistence and transitions
            
            **Key Insights:**
            - Multiple regimes can flash warnings before major moves
            - Regime changes often precede trend changes
            - Persistent extreme regimes suggest strong trends
                - Rapid regime cycling indicates unstable conditions
                """)
            
            # Calculate regime metrics directly here (matching legacyF.py exactly)
            df_regime = df.copy()
            window = 52
            min_periods = 26
            
            # Step 1: Calculate all percentile metrics
            df_regime['long_conc_pct'] = df_regime['conc_gross_le_4_tdr_long'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
            df_regime['short_conc_pct'] = df_regime['conc_gross_le_4_tdr_short'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
            
            # Net positions
            df_regime['comm_net'] = df_regime['comm_positions_long_all'] - df_regime['comm_positions_short_all']
            df_regime['noncomm_net'] = df_regime['noncomm_positions_long_all'] - df_regime['noncomm_positions_short_all']
            df_regime['comm_net_pct'] = df_regime['comm_net'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
            df_regime['noncomm_net_pct'] = df_regime['noncomm_net'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
            
            # Flow intensity
            df_regime['comm_flow'] = df_regime['comm_net'].diff()
            df_regime['noncomm_flow'] = df_regime['noncomm_net'].diff()
            df_regime['flow_intensity'] = abs(df_regime['comm_flow']) + abs(df_regime['noncomm_flow'])
            df_regime['flow_pct'] = df_regime['flow_intensity'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
            
            # Trader participation
            df_regime['trader_total_pct'] = df_regime['traders_tot_all'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
            
            # Heterogeneity placeholder
            df_regime['heterogeneity_pct'] = 50
            
            # Step 2: Calculate regime extremity score
            def distance_from_center(pct):
                return abs(pct - 50) * 2
            
            df_regime['regime_extremity'] = df_regime.apply(lambda row: 
                max(distance_from_center(row['long_conc_pct']), 
                    distance_from_center(row['short_conc_pct'])) * 0.25 +
                max(distance_from_center(row['comm_net_pct']), 
                    distance_from_center(row['noncomm_net_pct'])) * 0.25 +
                row['flow_pct'] * 0.25 +
                row['heterogeneity_pct'] * 0.25
            , axis=1)
            
            # Step 3: Detect regime
            def detect_regime(row):
                EXTREME_HIGH = 85
                EXTREME_LOW = 15
                MODERATE_HIGH = 70
                MODERATE_LOW = 30
                
                if pd.isna(row['long_conc_pct']):
                    return "Insufficient Data", "gray"
                
                # Check patterns
                if row['long_conc_pct'] > EXTREME_HIGH and row['short_conc_pct'] < MODERATE_LOW:
                    return "Long Concentration Extreme", "red"
                elif row['short_conc_pct'] > EXTREME_HIGH and row['long_conc_pct'] < MODERATE_LOW:
                    return "Short Concentration Extreme", "red"
                elif row['long_conc_pct'] > EXTREME_HIGH and row['short_conc_pct'] > EXTREME_HIGH:
                    return "Bilateral Concentration", "orange"
                elif row['noncomm_net_pct'] > EXTREME_HIGH and row['comm_net_pct'] < EXTREME_LOW:
                    return "Speculative Long Extreme", "red"
                elif row['noncomm_net_pct'] < EXTREME_LOW and row['comm_net_pct'] > EXTREME_HIGH:
                    return "Commercial Long Extreme", "orange"
                elif row['flow_pct'] > EXTREME_HIGH:
                    return "High Flow Volatility", "yellow"
                elif row['heterogeneity_pct'] > EXTREME_HIGH:
                    return "Maximum Divergence", "red"
                elif row['regime_extremity'] < 40:
                    return "Balanced Market", "green"
                else:
                    return "Transitional", "gray"
            
            df_regime[['regime', 'regime_color']] = df_regime.apply(
                lambda row: pd.Series(detect_regime(row)), axis=1
            )
            
            # Create visualization
            latest = df_regime.iloc[-1]
            
            # Main metrics display
            col1, col2, col3 = st.columns([2, 3, 2])
            
            with col1:
                # Gauge-style display for extremity
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = latest['regime_extremity'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Market Extremity"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 85
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with col2:
                # Spider chart of all metrics
                categories = ['Long Conc', 'Short Conc', 'Comm Net', 'NonComm Net', 
                             'Flow', 'Traders', 'Heterogeneity']
                values = [
                    latest['long_conc_pct'],
                    latest['short_conc_pct'],
                    latest['comm_net_pct'],
                    latest['noncomm_net_pct'],
                    latest['flow_pct'],
                    latest['trader_total_pct'],
                    latest['heterogeneity_pct']
                ]
                
                fig_spider = go.Figure()
                fig_spider.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Current',
                    line_color='blue'
                ))
            
                # Add reference circles
                fig_spider.add_trace(go.Scatterpolar(
                    r=[50]*7,
                    theta=categories,
                    name='Normal (50th)',
                    line=dict(color='gray', dash='dash')
                ))
                
                fig_spider.add_trace(go.Scatterpolar(
                    r=[85]*7,
                    theta=categories,
                    name='Extreme (85th)',
                    line=dict(color='red', dash='dot')
                ))
            
                fig_spider.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="Percentile Rankings",
                    height=300
                )
                st.plotly_chart(fig_spider, use_container_width=True)
                
            with col3:
                # Current regime display
                st.markdown("### Current Regime")
                regime_color_map = {
                    'red': '🔴',
                    'orange': '🟠',
                    'yellow': '🟡',
                    'green': '🟢',
                    'gray': '⚪'
                }
                st.markdown(f"## {regime_color_map.get(latest['regime_color'], '⚪')} {latest['regime']}")
                
                # Regime duration
                current_regime = latest['regime']
                regime_duration = 1
                for i in range(2, min(len(df_regime), 20)):
                    if df_regime.iloc[-i]['regime'] == current_regime:
                        regime_duration += 1
                    else:
                        break
            
                st.metric("Duration", f"{regime_duration} weeks")
                st.metric("Extremity Score", f"{latest['regime_extremity']:.1f} / 100")
            
            # Detailed metrics table
            st.markdown("### Current Metrics")
            metrics_data = {
            'Metric': ['Long Concentration', 'Short Concentration', 'Commercial Net', 
                      'Non-Commercial Net', 'Flow Intensity', 'Total Traders'],
            'Value': [
                f"{latest['conc_gross_le_4_tdr_long']:.1f}%",
                f"{latest['conc_gross_le_4_tdr_short']:.1f}%",
                f"{latest['comm_net']:,.0f}",
                f"{latest['noncomm_net']:,.0f}",
                f"{latest['flow_intensity']:,.0f}",
                f"{latest['traders_tot_all']:.0f}"
            ],
            'Percentile': [
                f"{latest['long_conc_pct']:.0f}th",
                f"{latest['short_conc_pct']:.0f}th",
                f"{latest['comm_net_pct']:.0f}th",
                f"{latest['noncomm_net_pct']:.0f}th",
                f"{latest['flow_pct']:.0f}th",
                f"{latest['trader_total_pct']:.0f}th"
            ],
            'Status': [
                '↑↑' if latest['long_conc_pct'] > 85 else '↓↓' if latest['long_conc_pct'] < 15 else '→',
                '↑↑' if latest['short_conc_pct'] > 85 else '↓↓' if latest['short_conc_pct'] < 15 else '→',
                '↑↑' if latest['comm_net_pct'] > 85 else '↓↓' if latest['comm_net_pct'] < 15 else '→',
                '↑↑' if latest['noncomm_net_pct'] > 85 else '↓↓' if latest['noncomm_net_pct'] < 15 else '→',
                '↑↑' if latest['flow_pct'] > 85 else '↓' if latest['flow_pct'] < 15 else '→',
                '↑' if latest['trader_total_pct'] > 70 else '↓' if latest['trader_total_pct'] < 30 else '→'
            ]
        }
        
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Regime Legend
            st.markdown("### Regime Color Legend")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔴 Red Regimes (High Risk)**")
                st.caption("• Long Concentration Extreme")
                st.caption("• Short Concentration Extreme")
                st.caption("• Speculative Long Extreme")
                st.caption("• Maximum Divergence")
            
            with col2:
                st.markdown("**🟠 Orange Regimes (Moderate Risk)**")
                st.caption("• Bilateral Concentration")
                st.caption("• Commercial Long Extreme")
                st.markdown("**🟡 Yellow Regimes**")
                st.caption("• High Flow Volatility")
            
            with col3:
                st.markdown("**🟢 Green Regimes (Low Risk)**")
                st.caption("• Balanced Market")
                st.markdown("**⚪ Gray Regimes**")
                st.caption("• Transitional")
                st.caption("• Insufficient Data")
            
            # Regime timeline
            st.markdown("### Regime History")
            
            # Create regime timeline chart
            fig_timeline = go.Figure()
            
            # Get last 52 weeks of regime data
            timeline_data = df_regime.tail(52).copy()
            
            # Create color mapping
            color_map = {
            'Long Concentration Extreme': 'darkred',
            'Short Concentration Extreme': 'darkred',
            'Bilateral Concentration': 'orange',
            'Speculative Long Extreme': 'red',
            'Commercial Long Extreme': 'darkorange',
            'High Flow Volatility': 'gold',
            'Maximum Divergence': 'darkred',
            'Balanced Market': 'green',
            'Transitional': 'gray',
            'Insufficient Data': 'lightgray'
        }
            
            # Add all possible regimes to ensure complete legend
            all_regimes = [
            'Long Concentration Extreme',
            'Short Concentration Extreme', 
            'Bilateral Concentration',
            'Speculative Long Extreme',
            'Commercial Long Extreme',
            'High Flow Volatility',
            'Maximum Divergence',
            'Balanced Market',
            'Transitional',
            'Insufficient Data'
        ]
            
            # Add regime bars - include all regimes for complete legend
            for regime in all_regimes:
                regime_mask = timeline_data['regime'] == regime
                if regime_mask.sum() > 0:
                    # Regime exists in data
                    fig_timeline.add_trace(go.Bar(
                        x=timeline_data.loc[regime_mask, 'report_date_as_yyyy_mm_dd'],
                        y=[1] * regime_mask.sum(),
                        name=regime,
                        marker_color=color_map.get(regime, 'gray'),
                        hovertemplate='%{x}<br>' + regime + '<extra></extra>',
                        showlegend=True
                    ))
                else:
                    # Add empty trace for legend
                    fig_timeline.add_trace(go.Bar(
                        x=[],
                        y=[],
                        name=regime,
                        marker_color=color_map.get(regime, 'gray'),
                        showlegend=True
                    ))
            
            fig_timeline.update_layout(
                barmode='stack',
                showlegend=True,
                height=200,
                yaxis=dict(showticklabels=False, title=''),
                xaxis=dict(title='Date'),
                title='52-Week Regime Timeline'
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    elif analysis_type == "Concentration Momentum":
        st.markdown("#### 📈 Concentration Momentum Analysis")
        create_concentration_momentum_analysis(df, instrument_name)
    
    elif analysis_type == "Participant Behavior Clusters":
        st.markdown("#### 🎯 Participant Behavior Clusters")
        create_participant_behavior_clusters(df, instrument_name)
    
    elif analysis_type == "Market Microstructure Analysis":
        st.markdown("#### 📊 Market Microstructure Analysis")
        create_market_microstructure_analysis(df, instrument_name)
    
    else:
        st.info(f"Analysis type '{analysis_type}' is not yet implemented.")