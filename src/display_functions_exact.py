# Display functions for single instrument charts - EXACT copy from legacyF.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
from charts.base_charts import create_plotly_chart
from charts.seasonality_charts import create_seasonality_chart
from charts.percentile_charts import create_percentile_chart
from charts.momentum_charts import create_single_variable_momentum_dashboard
from charts.participation_charts import create_participation_density_dashboard
from charts.share_of_oi import create_share_of_oi_chart
from charts.trader_participation_analysis import (
    create_concentration_risk_heatmap,
    create_market_structure_quadrant,
    create_market_structure_timeline,
    create_concentration_divergence_analysis,
    create_heterogeneity_index,
    create_spreading_activity_analysis
)
from charts.concentration_momentum import create_concentration_momentum_analysis
from charts.participant_behavior_clusters import create_participant_behavior_clusters
from charts.regime_detection import create_regime_detection_dashboard
from charts.market_microstructure import create_market_microstructure_analysis
from futures_price_fetcher import FuturesPriceFetcher
from futures_price_viewer_lwc import display_futures_price_chart


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
    """Display time series analysis with futures first approach"""
    from display_functions_futures_first import display_time_series_chart as display_futures_first
    display_futures_first(df, instrument_name)


def display_time_series_chart_old(df, instrument_name):
    """OLD VERSION - DEPRECATED - Use display_functions_futures_first.py instead"""
    st.warning("This function is deprecated. Please use the new futures-first implementation.")
    return

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
        
        # Add custom formula section
        st.markdown("---")
        st.markdown("#### 🧮 Custom Formula (Optional)")
        
        with st.expander("Create custom calculations from metrics", expanded=False):
            st.info("""
            **How to use:**
            - Enter a formula using metric abbreviations
            - Supported operations: +, -, *, /, ()
            - Example formulas:
                - `NC_NET / OI * 100` - Net NC as % of Open Interest
                - `NC_LONG - NC_SHORT` - Calculate net positioning
                - `(NC_LONG + NC_SHORT) / OI` - Total NC activity ratio
            """)
            
            # Metric abbreviations
            st.markdown("**Available metrics:**")
            metric_abbr = {
                "OI": "open_interest_all",
                "NC_LONG": "noncomm_positions_long_all",
                "NC_SHORT": "noncomm_positions_short_all", 
                "NC_SPREAD": "noncomm_postions_spread_all",
                "C_LONG": "comm_positions_long_all",
                "C_SHORT": "comm_positions_short_all",
                "NC_NET": "net_noncomm_positions",
                "C_NET": "net_comm_positions",
                "R_NET": "net_reportable_positions",
                "NR_LONG": "nonrept_positions_long_all",
                "NR_SHORT": "nonrept_positions_short_all",
                "TRADERS": "traders_tot_all"
            }
            
            # Display abbreviations in columns
            abbr_col1, abbr_col2, abbr_col3 = st.columns(3)
            items = list(metric_abbr.items())
            third = len(items) // 3
            
            with abbr_col1:
                for abbr, full in items[:third]:
                    display_name = column_display_names.get(full, full.replace('_', ' ').title())
                    st.text(f"{abbr} = {display_name}")
            with abbr_col2:
                for abbr, full in items[third:2*third]:
                    display_name = column_display_names.get(full, full.replace('_', ' ').title())
                    st.text(f"{abbr} = {display_name}")
            with abbr_col3:
                for abbr, full in items[2*third:]:
                    display_name = column_display_names.get(full, full.replace('_', ' ').title())
                    st.text(f"{abbr} = {display_name}")
            
            # Preset formulas
            st.markdown("**Quick formulas:**")
            preset_formulas = {
                "NC Net % of OI": "NC_NET / OI * 100",
                "Commercial Hedging Pressure": "C_NET / OI * 100",
                "NC Long/Short Ratio": "NC_LONG / NC_SHORT",
                "Total NC Activity": "(NC_LONG + NC_SHORT) / OI * 100",
                "Spec vs Commercial": "NC_NET / C_NET",
                "Average Position per Trader": "OI / TRADERS",
                "NC Spread % of NC Total": "NC_SPREAD / (NC_LONG + NC_SHORT + NC_SPREAD) * 100"
            }
            
            preset_col1, preset_col2 = st.columns([3, 1])
            with preset_col1:
                selected_preset = st.selectbox(
                    "Or select a preset formula:",
                    [""] + list(preset_formulas.keys()),
                    key="preset_formula_select"
                )
            
            # Formula input
            formula_col1, formula_col2 = st.columns([3, 1])
            with formula_col1:
                # Use preset if selected, otherwise allow custom input
                default_formula = preset_formulas.get(selected_preset, "") if selected_preset else ""
                custom_formula = st.text_input(
                    "Enter formula:",
                    value=default_formula,
                    placeholder="e.g., NC_NET / OI * 100",
                    key="custom_formula_input"
                )
            
            with formula_col2:
                default_name = selected_preset if selected_preset else ""
                formula_name = st.text_input(
                    "Formula name:",
                    value=default_name,
                    placeholder="e.g., NC % of OI",
                    key="formula_name_input"
                )
            
            # Process custom formula if provided
            if custom_formula:
                try:
                    # Create a copy of the dataframe for calculations
                    calc_df = filtered_df.copy()
                    
                    # Sort abbreviations by length (longest first) to avoid partial replacements
                    sorted_abbr = sorted(metric_abbr.items(), key=lambda x: len(x[0]), reverse=True)
                    
                    # Replace abbreviations with actual column names
                    formula_expanded = custom_formula.upper()  # Make case-insensitive
                    for abbr, col in sorted_abbr:
                        if col in calc_df.columns:
                            # Use word boundaries to avoid partial replacements
                            import re
                            pattern = r'\b' + re.escape(abbr) + r'\b'
                            formula_expanded = re.sub(pattern, f"calc_df['{col}']", formula_expanded)
                    
                    # Check for any remaining unrecognized variables
                    remaining_vars = re.findall(r'\b[A-Z_]+\b', formula_expanded.replace("calc_df", ""))
                    if remaining_vars:
                        st.warning(f"⚠️ Unrecognized variables: {', '.join(remaining_vars)}")
                    
                    # Handle division by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        calc_df['custom_formula_result'] = eval(formula_expanded)
                        # Replace inf and nan with None for cleaner display
                        calc_df['custom_formula_result'] = calc_df['custom_formula_result'].replace([np.inf, -np.inf], np.nan)
                    
                    # Add to selected columns for plotting
                    if 'custom_formula_result' not in selected_columns:
                        selected_columns.append('custom_formula_result')
                    filtered_df['custom_formula_result'] = calc_df['custom_formula_result']
                    
                    # Update display names
                    column_display_names['custom_formula_result'] = formula_name if formula_name else f"Custom: {custom_formula}"
                    
                    st.success(f"✅ Formula applied: {formula_name if formula_name else custom_formula}")
                    
                    # Show statistics
                    valid_values = calc_df['custom_formula_result'].dropna()
                    if len(valid_values) > 0:
                        st.caption(f"Min: {valid_values.min():.2f} | Max: {valid_values.max():.2f} | Mean: {valid_values.mean():.2f}")
                    
                except SyntaxError as e:
                    st.error(f"❌ Syntax error in formula: {str(e)}")
                    st.info("Check parentheses and operators (+, -, *, /)")
                except KeyError as e:
                    st.error(f"❌ Column not found: {str(e)}")
                    st.info("Make sure all metrics exist in the data")
                except Exception as e:
                    st.error(f"❌ Formula error: {str(e)}")
                    st.info("Please check your formula syntax and ensure all metrics exist in the data")
        
        if selected_columns:
            fig = create_plotly_chart(filtered_df, selected_columns, f"{instrument_name} - Time Series Analysis")
            
            # Update legend labels for custom formula and other columns
            if fig:
                for trace in fig.data:
                    # Check if this is a data trace (not a marker)
                    if hasattr(trace, 'name') and trace.name:
                        # Look for the original column name in our display names
                        for original_name, display_name in column_display_names.items():
                            if trace.name == original_name or trace.name == original_name.replace('_', ' ').title():
                                trace.name = display_name
                                break
                        # Special handling for custom formula result
                        if trace.name == 'Custom Formula Result' and 'custom_formula_result' in column_display_names:
                            trace.name = column_display_names['custom_formula_result']
            
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
        
        **Alternative Calculation:**
        Alternatively, you can calculate Long Side as: NonComm Long + Comm Long + NonRep Long = Total OI Reported - Spreading Total
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

    with tab4:
        # Futures Price tab
        st.markdown("#### Futures Price Chart")

        # Extract COT code from instrument name
        def extract_cot_code(name):
            match = re.search(r'\((\w+)\)', name)
            if match:
                return match.group(1)
            return None

        cot_code = extract_cot_code(instrument_name)

        if not cot_code:
            st.info("No futures price data available for this instrument")
        else:
            # Initialize fetcher
            fetcher = FuturesPriceFetcher()

            # Check if this COT instrument has a matching futures symbol
            futures_symbol = fetcher.get_futures_symbol_for_cot(cot_code)

            if not futures_symbol:
                st.info("No futures price data available for this instrument")
            else:
                # Get symbol info
                symbol_info = fetcher.futures_mapping['futures_symbols'].get(futures_symbol, {})

                # Display symbol info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Symbol", futures_symbol)
                with col2:
                    st.metric("Exchange", symbol_info.get('exchange', 'N/A'))
                with col3:
                    st.metric("Category", symbol_info.get('category', 'N/A'))

                # Options for the chart
                col1, col2 = st.columns([2, 1])

                with col1:
                    show_stats = st.checkbox("Show Price Statistics", value=False, key="futures_show_stats")

                with col2:
                    adjustment_method = st.selectbox(
                        "Adjustment Method",
                        ["REV", "RAD", "NON"],
                        index=0,
                        help="REV: Reverse (Back Adjusted), RAD: Ratio Adjusted, NON: Non-Adjusted"
                    )

                # Date range for price data
                end_date = datetime.now()
                start_date = datetime(2000, 1, 1)

                # Fetch price data
                price_df = fetcher.fetch_weekly_prices(
                    futures_symbol,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    adjustment_method
                )

                if price_df.empty:
                    st.warning("No price data available for the selected parameters")
                else:
                    # Show stats if requested
                    if show_stats:
                        stats = fetcher.get_price_change_stats(price_df)

                        st.markdown("##### Price Statistics")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Latest Close", f"{stats['latest_close']:.2f}")
                            st.metric("52W High", f"{stats['high_52w']:.2f}")

                        with col2:
                            st.metric("Week Change", f"{stats['week_change']:.2f}%")
                            st.metric("52W Low", f"{stats['low_52w']:.2f}")

                        with col3:
                            st.metric("Month Change", f"{stats['month_change']:.2f}%")
                            st.metric("Avg Volume", f"{stats['avg_volume']:,.0f}")

                        with col4:
                            st.metric("Year Change", f"{stats['year_change']:.2f}%")
                            st.metric("Latest OI", f"{stats['latest_oi']:,.0f}")

                    # Call the updated futures price viewer with events support
                    from futures_price_viewer_lwc import (
                        create_lwc_chart,
                        get_category_from_futures_symbol,
                        get_events_for_category
                    )

                    # Add checkbox for showing historical events
                    show_events = st.checkbox(
                        "Show Historical Events",
                        value=False,
                        key=f"show_events_{futures_symbol}_timeseries",
                        help="Display market-moving events relevant to this instrument category"
                    )

                    # Get historical events if checkbox is checked
                    events = None
                    if show_events:
                        category = get_category_from_futures_symbol(futures_symbol)
                        events = get_events_for_category(category)

                    # Create and display chart with events
                    create_lwc_chart(price_df, futures_symbol, events)


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


def display_momentum_chart(df, instrument_name, selected_var=None):
    """Display momentum dashboard - EXACT copy from legacyF.py"""
    if selected_var is None:
        st.subheader("🚀 Momentum Dashboard")

    # Z-score calculation description in grey
    st.markdown('<p style="color: #808080; font-size: 13px;">Z-scores shown below are calculated using a 104-week rolling window (2 years) to measure how many standard deviations the current value is from the rolling mean.</p>', unsafe_allow_html=True)

    # Calculate net positions and their changes if they don't exist
    if 'net_noncomm_positions' not in df.columns and 'noncomm_positions_long_all' in df.columns and 'noncomm_positions_short_all' in df.columns:
        df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
    if 'net_noncomm_positions' in df.columns and 'change_in_net_noncomm' not in df.columns:
        df['change_in_net_noncomm'] = df['net_noncomm_positions'].diff()

    if 'net_comm_positions' not in df.columns and 'comm_positions_long_all' in df.columns and 'comm_positions_short_all' in df.columns:
        df['net_comm_positions'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
    if 'net_comm_positions' in df.columns and 'change_in_net_comm' not in df.columns:
        df['change_in_net_comm'] = df['net_comm_positions'].diff()

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
        },
        'net_noncomm_positions': {
            'display': 'Net Non-Commercial',
            'change_col': 'change_in_net_noncomm'
        },
        'net_comm_positions': {
            'display': 'Net Commercial',
            'change_col': 'change_in_net_comm'
        }
    }

    # Filter to only available position columns
    available_vars = {k: v for k, v in momentum_vars.items()
                     if k in df.columns and v['change_col'] in df.columns}

    if selected_var is None:
        selected_var = st.selectbox(
            "Select variable for momentum analysis:",
            list(available_vars.keys()),
            format_func=lambda x: available_vars[x]['display'],
            index=0,
            key="momentum_var_selector"
        )

    # Use all available data
    df_filtered = df.copy()

    # Get the corresponding API change column
    change_col = available_vars[selected_var]['change_col']

    # Use the position variable for display and API change column for calculations
    display_var = selected_var

    # Don't show data info - removed as requested

    fig = create_single_variable_momentum_dashboard(df_filtered, display_var, change_col)
    if fig:
        # Configure plotly to show autoscale buttons
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'momentum_{selected_var}',
                'height': 1000,
                'width': 1400,
                'scale': 1
            }
        }
        st.plotly_chart(fig, use_container_width=True, config=config)

    return selected_var  # Return the selected variable for use in percentile


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
        st.markdown('<p style="color: #808080; font-size: 13px;">Concentration metrics use Net positions (4 or fewer traders). Percentiles calculated using 2-year rolling lookback. Price chart uses Non-Adjusted method. Bar colors: Green (below 33rd percentile), Yellow (33rd-67th percentile), Red (above 67th percentile).</p>', unsafe_allow_html=True)

        # Fixed price adjustment to Non-Adjusted
        price_adjustment_code = "NON"

        # Fetch price data
        price_df = None
        try:
            import re
            import json
            import os

            # Extract clean instrument name and find futures symbol
            instrument_clean = re.sub(r'\s*\(\d+\)$', '', instrument_name).strip()
            symbol = None
            json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instrument_management', 'futures', 'futures_symbols_enhanced.json')
            with open(json_path, 'r') as f:
                mapping = json.load(f)
                for fut_symbol, info in mapping['futures_symbols'].items():
                    if info['cot_mapping']['matched']:
                        if instrument_clean in info['cot_mapping']['instruments']:
                            symbol = fut_symbol
                            break

            if symbol:
                fetcher = FuturesPriceFetcher()
                start_date = df['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
                end_date = df['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')
                price_df = fetcher.fetch_weekly_prices(symbol, start_date, end_date, price_adjustment_code)
        except Exception as e:
            st.warning(f"Could not fetch price data: {str(e)}")

        # Create tabs for different views
        tab1, tab2 = st.tabs(["Enhanced View", "Spreading Activity"])

        with tab1:
            # Add time range selector buttons
            st.markdown("#### Select Time Range")
            col_buttons = st.columns(5)

            # Initialize session state for range selection if not exists
            if 'participation_range' not in st.session_state:
                st.session_state.participation_range = 'All'

            # Range buttons with highlighting for selected range
            with col_buttons[0]:
                if st.button("1Y", key="part_range_1y", use_container_width=True,
                            type="primary" if st.session_state.participation_range == '1Y' else "secondary"):
                    st.session_state.participation_range = '1Y'
                    st.rerun()
            with col_buttons[1]:
                if st.button("2Y", key="part_range_2y", use_container_width=True,
                            type="primary" if st.session_state.participation_range == '2Y' else "secondary"):
                    st.session_state.participation_range = '2Y'
                    st.rerun()
            with col_buttons[2]:
                if st.button("5Y", key="part_range_5y", use_container_width=True,
                            type="primary" if st.session_state.participation_range == '5Y' else "secondary"):
                    st.session_state.participation_range = '5Y'
                    st.rerun()
            with col_buttons[3]:
                if st.button("10Y", key="part_range_10y", use_container_width=True,
                            type="primary" if st.session_state.participation_range == '10Y' else "secondary"):
                    st.session_state.participation_range = '10Y'
                    st.rerun()
            with col_buttons[4]:
                if st.button("All", key="part_range_all", use_container_width=True,
                            type="primary" if st.session_state.participation_range == 'All' else "secondary"):
                    st.session_state.participation_range = 'All'
                    st.rerun()

            st.markdown("---")

            # Fixed settings: always use Net concentration with 2-year lookback
            concentration_type = 'Net'
            lookback_days = 730  # 2 years

            # Filter data based on selected time range
            df_sorted = df.copy()
            df_sorted = df_sorted.sort_values('report_date_as_yyyy_mm_dd')
            df_sorted['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df_sorted['report_date_as_yyyy_mm_dd'])
            latest_date = df_sorted['report_date_as_yyyy_mm_dd'].max()

            # Apply filtering based on selected range
            if st.session_state.participation_range == '1Y':
                start_date = latest_date - pd.DateOffset(years=1)
                df_filtered = df_sorted[df_sorted['report_date_as_yyyy_mm_dd'] >= start_date].copy()
            elif st.session_state.participation_range == '2Y':
                start_date = latest_date - pd.DateOffset(years=2)
                df_filtered = df_sorted[df_sorted['report_date_as_yyyy_mm_dd'] >= start_date].copy()
            elif st.session_state.participation_range == '5Y':
                start_date = latest_date - pd.DateOffset(years=5)
                df_filtered = df_sorted[df_sorted['report_date_as_yyyy_mm_dd'] >= start_date].copy()
            elif st.session_state.participation_range == '10Y':
                start_date = latest_date - pd.DateOffset(years=10)
                df_filtered = df_sorted[df_sorted['report_date_as_yyyy_mm_dd'] >= start_date].copy()
            else:  # 'All'
                df_filtered = df_sorted.copy()

            df_filtered = df_filtered.reset_index(drop=True)

            # Filter price data to match the same time range
            price_df_filtered = None
            if price_df is not None and not price_df.empty:
                price_df_copy = price_df.copy()
                price_df_copy['date'] = pd.to_datetime(price_df_copy['date'])
                if st.session_state.participation_range != 'All':
                    price_df_filtered = price_df_copy[price_df_copy['date'] >= start_date].copy()
                else:
                    price_df_filtered = price_df_copy.copy()

            # Calculate avg position per trader for all data
            df_filtered['avg_position_per_trader'] = df_filtered['open_interest_all'] / df_filtered['traders_tot_all']

            # Calculate percentile for each point based on 2-year rolling window
            percentile_data = []
            for i in range(len(df_filtered)):
                current_date = df_filtered.iloc[i]['report_date_as_yyyy_mm_dd']
                lookback_start = current_date - pd.Timedelta(days=lookback_days)
                window_data = df_filtered[(df_filtered['report_date_as_yyyy_mm_dd'] >= lookback_start) &
                               (df_filtered['report_date_as_yyyy_mm_dd'] <= current_date)]

                if len(window_data) > 0:
                    current_val = df_filtered.iloc[i]['avg_position_per_trader']
                    percentile = (window_data['avg_position_per_trader'] < current_val).sum() / len(window_data) * 100
                    percentile_data.append(percentile)
                else:
                    percentile_data.append(50)

            # Create participation density chart with filtered price data
            density_fig = create_participation_density_dashboard(df_filtered, instrument_name, percentile_data, concentration_type, price_df_filtered)

            if density_fig:
                st.plotly_chart(density_fig, use_container_width=True)

            # Add individual category charts below
            st.markdown("---")
            st.markdown("#### Individual Trader Categories")

            # Define categories (excluding Overall Average)
            from charts.participation_charts import create_individual_category_chart

            categories = [
                ('noncomm_positions_long_all', 'traders_noncomm_long_all', 'Non-Commercial Long'),
                ('noncomm_positions_short_all', 'traders_noncomm_short_all', 'Non-Commercial Short'),
                ('noncomm_postions_spread_all', 'traders_noncomm_spread_all', 'Non-Commercial Spread'),
                ('comm_positions_long_all', 'traders_comm_long_all', 'Commercial Long'),
                ('comm_positions_short_all', 'traders_comm_short_all', 'Commercial Short'),
                ('tot_rept_positions_long_all', 'traders_tot_rept_long_all', 'Total Reportable Long'),
                ('tot_rept_positions_short', 'traders_tot_rept_short_all', 'Total Reportable Short')
            ]

            # Create and display each category chart
            for position_col, trader_col, title in categories:
                if position_col in df_filtered.columns and trader_col in df_filtered.columns:
                    cat_fig = create_individual_category_chart(df_filtered, position_col, trader_col, title)
                    if cat_fig:
                        st.plotly_chart(cat_fig, use_container_width=True)


        with tab2:
            st.markdown("#### Non-Commercial Spreading Activity")

            # Grey explanatory text
            st.markdown('<p style="color: #808080; font-size: 13px;">Calculation: Spread Traders ÷ (Long-Only + Short-Only Traders). Note: Only non-commercials can hold spread positions in COT data.</p>', unsafe_allow_html=True)

            # Add time range selector buttons
            st.markdown("#### Select Time Range")
            col_buttons = st.columns(5)

            # Initialize session state for spreading range selection if not exists
            if 'spreading_range' not in st.session_state:
                st.session_state.spreading_range = 'All'

            # Range buttons with highlighting for selected range
            with col_buttons[0]:
                if st.button("1Y", key="spread_range_1y", use_container_width=True,
                            type="primary" if st.session_state.spreading_range == '1Y' else "secondary"):
                    st.session_state.spreading_range = '1Y'
                    st.rerun()
            with col_buttons[1]:
                if st.button("2Y", key="spread_range_2y", use_container_width=True,
                            type="primary" if st.session_state.spreading_range == '2Y' else "secondary"):
                    st.session_state.spreading_range = '2Y'
                    st.rerun()
            with col_buttons[2]:
                if st.button("5Y", key="spread_range_5y", use_container_width=True,
                            type="primary" if st.session_state.spreading_range == '5Y' else "secondary"):
                    st.session_state.spreading_range = '5Y'
                    st.rerun()
            with col_buttons[3]:
                if st.button("10Y", key="spread_range_10y", use_container_width=True,
                            type="primary" if st.session_state.spreading_range == '10Y' else "secondary"):
                    st.session_state.spreading_range = '10Y'
                    st.rerun()
            with col_buttons[4]:
                if st.button("All", key="spread_range_all", use_container_width=True,
                            type="primary" if st.session_state.spreading_range == 'All' else "secondary"):
                    st.session_state.spreading_range = 'All'
                    st.rerun()

            st.markdown("---")

            # Filter data based on selected time range
            df_spread_sorted = df.copy()
            df_spread_sorted = df_spread_sorted.sort_values('report_date_as_yyyy_mm_dd')
            df_spread_sorted['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df_spread_sorted['report_date_as_yyyy_mm_dd'])
            latest_date_spread = df_spread_sorted['report_date_as_yyyy_mm_dd'].max()

            # Apply filtering based on selected range
            if st.session_state.spreading_range == '1Y':
                start_date_spread = latest_date_spread - pd.DateOffset(years=1)
                df_spread_filtered = df_spread_sorted[df_spread_sorted['report_date_as_yyyy_mm_dd'] >= start_date_spread].copy()
            elif st.session_state.spreading_range == '2Y':
                start_date_spread = latest_date_spread - pd.DateOffset(years=2)
                df_spread_filtered = df_spread_sorted[df_spread_sorted['report_date_as_yyyy_mm_dd'] >= start_date_spread].copy()
            elif st.session_state.spreading_range == '5Y':
                start_date_spread = latest_date_spread - pd.DateOffset(years=5)
                df_spread_filtered = df_spread_sorted[df_spread_sorted['report_date_as_yyyy_mm_dd'] >= start_date_spread].copy()
            elif st.session_state.spreading_range == '10Y':
                start_date_spread = latest_date_spread - pd.DateOffset(years=10)
                df_spread_filtered = df_spread_sorted[df_spread_sorted['report_date_as_yyyy_mm_dd'] >= start_date_spread].copy()
            else:  # 'All'
                df_spread_filtered = df_spread_sorted.copy()

            df_spread_filtered = df_spread_filtered.reset_index(drop=True)

            # Filter price data to match the same time range
            price_df_spread_filtered = None
            if price_df is not None and not price_df.empty:
                price_df_copy_spread = price_df.copy()
                price_df_copy_spread['date'] = pd.to_datetime(price_df_copy_spread['date'])
                if st.session_state.spreading_range != 'All':
                    price_df_spread_filtered = price_df_copy_spread[price_df_copy_spread['date'] >= start_date_spread].copy()
                else:
                    price_df_spread_filtered = price_df_copy_spread.copy()

            # Display price chart first if available
            if price_df_spread_filtered is not None and not price_df_spread_filtered.empty:
                price_fig = go.Figure()
                price_fig.add_trace(
                    go.Scatter(
                        x=price_df_spread_filtered['date'],
                        y=price_df_spread_filtered['close'],
                        name='Price',
                        line=dict(color='#2E86AB', width=2),
                        mode='lines'
                    )
                )
                price_fig.update_layout(
                    title='Futures Price',
                    height=300,
                    showlegend=False,
                    hovermode='x unified',
                    margin=dict(l=60, r=60, t=40, b=40),
                    yaxis=dict(title="Price"),
                    xaxis=dict(type='date')
                )
                st.plotly_chart(price_fig, use_container_width=True)

            # Create and display spreading activity chart with filtered data
            spread_fig = create_spreading_activity_analysis(df_spread_filtered, instrument_name)
            if spread_fig:
                st.plotly_chart(spread_fig, use_container_width=True)

                # Current status
                latest = df_spread_filtered.iloc[-1]
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
        
        # Configuration - only Category Divergence
        divergence_type = "Category Divergence (Commercial vs Non-Commercial)"

        conc_side = st.radio(
            "Position Side:",
            ["Long Positions", "Short Positions"],
            horizontal=True
        )

        # Add time range selector buttons
        st.markdown("#### Select Time Range")
        col_buttons = st.columns(5)

        # Initialize session state for range selection if not exists
        if 'conc_div_range' not in st.session_state:
            st.session_state.conc_div_range = '2Y'  # Default to 2Y

        # Range buttons with highlighting for selected range
        with col_buttons[0]:
            if st.button("1Y", key="conc_div_range_1y", use_container_width=True,
                        type="primary" if st.session_state.conc_div_range == '1Y' else "secondary"):
                st.session_state.conc_div_range = '1Y'
                st.rerun()
        with col_buttons[1]:
            if st.button("2Y", key="conc_div_range_2y", use_container_width=True,
                        type="primary" if st.session_state.conc_div_range == '2Y' else "secondary"):
                st.session_state.conc_div_range = '2Y'
                st.rerun()
        with col_buttons[2]:
            if st.button("5Y", key="conc_div_range_5y", use_container_width=True,
                        type="primary" if st.session_state.conc_div_range == '5Y' else "secondary"):
                st.session_state.conc_div_range = '5Y'
                st.rerun()
        with col_buttons[3]:
            if st.button("10Y", key="conc_div_range_10y", use_container_width=True,
                        type="primary" if st.session_state.conc_div_range == '10Y' else "secondary"):
                st.session_state.conc_div_range = '10Y'
                st.rerun()
        with col_buttons[4]:
            if st.button("All", key="conc_div_range_all", use_container_width=True,
                        type="primary" if st.session_state.conc_div_range == 'All' else "secondary"):
                st.session_state.conc_div_range = 'All'
                st.rerun()

        st.markdown("---")

        # Filter data based on selected time range
        df_sorted = df.copy()
        df_sorted = df_sorted.sort_values('report_date_as_yyyy_mm_dd')
        df_sorted['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df_sorted['report_date_as_yyyy_mm_dd'])
        latest_date = df_sorted['report_date_as_yyyy_mm_dd'].max()

        # Apply filtering based on selected range
        if st.session_state.conc_div_range == '1Y':
            start_date = latest_date - pd.DateOffset(years=1)
            df_filtered = df_sorted[df_sorted['report_date_as_yyyy_mm_dd'] >= start_date].copy()
            range_label = "1-year"
        elif st.session_state.conc_div_range == '2Y':
            start_date = latest_date - pd.DateOffset(years=2)
            df_filtered = df_sorted[df_sorted['report_date_as_yyyy_mm_dd'] >= start_date].copy()
            range_label = "2-year"
        elif st.session_state.conc_div_range == '5Y':
            start_date = latest_date - pd.DateOffset(years=5)
            df_filtered = df_sorted[df_sorted['report_date_as_yyyy_mm_dd'] >= start_date].copy()
            range_label = "5-year"
        elif st.session_state.conc_div_range == '10Y':
            start_date = latest_date - pd.DateOffset(years=10)
            df_filtered = df_sorted[df_sorted['report_date_as_yyyy_mm_dd'] >= start_date].copy()
            range_label = "10-year"
        else:  # 'All'
            df_filtered = df_sorted.copy()
            range_label = "all-time"

        df_filtered = df_filtered.reset_index(drop=True)

        div_fig = create_concentration_divergence_analysis(df_filtered, instrument_name, divergence_type, conc_side, range_label)
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