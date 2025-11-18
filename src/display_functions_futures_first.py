"""
Display time series with futures prices and OI as the base layer
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from streamlit_lightweight_charts import renderLightweightCharts
from display_synchronized_unified import display_synchronized_charts_unified
from futures_price_fetcher import FuturesPriceFetcher
from futures_price_viewer_lwc import get_category_from_futures_symbol, get_events_for_category
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

def display_time_series_chart(df, instrument_name):
    """Display time series analysis with futures price/OI base layer"""
    st.subheader("📈 Time Series Analysis")

    # Then create tabs for COT analysis (without Futures Price tab)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Standard Time Series",
        "Share of Open Interest",
        "Seasonality",
        "Percentile",
        "Momentum",
        "Extremes & Seasonality",
        "Cycle Composite"
    ])

    with tab1:
        display_cot_time_series_with_price(df, instrument_name)

    with tab2:
        display_share_of_oi(df, instrument_name)

    with tab3:
        display_seasonality(df, instrument_name)

    with tab4:
        display_percentile_tab(df, instrument_name)

    with tab5:
        display_momentum_tab(df, instrument_name)

    with tab6:
        display_extremes_seasonality(df, instrument_name)

    with tab7:
        display_cycle_composite(df, instrument_name)

def display_cot_time_series_with_price(df, instrument_name):
    """Display COT time series data with price chart and synchronized subplots"""

    # Price adjustment selection at the top
    st.markdown("**Price Adjustment Method**")
    price_adjustment = st.radio(
        "Select adjustment:",
        ["Non-Adjusted Method", "Ratio Adjusted Linking Method", "Reverse (Back) Adjusted Method"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
        key="price_adj_main"
    )

    # Map to the API codes
    adjustment_map = {
        "Non-Adjusted Method": "NON",
        "Ratio Adjusted Linking Method": "RAD",
        "Reverse (Back) Adjusted Method": "REV"
    }
    price_adjustment_code = adjustment_map[price_adjustment]

    st.markdown("---")

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

    # Add checkboxes for each column
    items_per_column = len(available_columns) // 3 + (1 if len(available_columns) % 3 else 0)

    for i, col_name in enumerate(available_columns):
        display_name = column_display_names.get(col_name, col_name)

        # Calculate net positions if needed
        if 'net_' in col_name:
            if col_name == 'net_noncomm_positions':
                df[col_name] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
            elif col_name == 'net_comm_positions':
                df[col_name] = df['comm_positions_long_all'] - df['comm_positions_short_all']
            elif col_name == 'net_reportable_positions':
                df[col_name] = (df['noncomm_positions_long_all'] + df['comm_positions_long_all'] -
                               df['noncomm_positions_short_all'] - df['comm_positions_short_all'])

        # Place checkbox in appropriate column
        if i < items_per_column:
            with col1:
                if st.checkbox(display_name, key=f"ts_{col_name}_{instrument_name}"):
                    selected_columns.append((col_name, display_name))
        elif i < 2 * items_per_column:
            with col2:
                if st.checkbox(display_name, key=f"ts_{col_name}_{instrument_name}"):
                    selected_columns.append((col_name, display_name))
        else:
            with col3:
                if st.checkbox(display_name, key=f"ts_{col_name}_{instrument_name}"):
                    selected_columns.append((col_name, display_name))

    # Add separator before formula options
    st.markdown("---")

    # Add third category - Custom Formulas
    st.markdown("#### Select custom formulas to plot:")

    col1_formula, col2_formula = st.columns(2)
    selected_formulas = []

    with col1_formula:
        if st.checkbox("NC_NET / OI × 100", key=f"formula_nc_oi_{instrument_name}"):
            selected_formulas.append(("nc_net_oi_ratio", "NC_NET/OI%"))
        if st.checkbox("NC_LONG / NC_SHORT", key=f"formula_nc_ratio_{instrument_name}"):
            selected_formulas.append(("nc_long_short_ratio", "NC L/S Ratio"))

    with col2_formula:
        if st.checkbox("C_NET / OI × 100", key=f"formula_c_oi_{instrument_name}"):
            selected_formulas.append(("c_net_oi_ratio", "C_NET/OI%"))
        if st.checkbox("NC_NET / C_NET", key=f"formula_nc_c_{instrument_name}"):
            selected_formulas.append(("nc_c_net_ratio", "NC/C NET"))

    # Calculate formula values if needed
    if selected_formulas:
        # Calculate net positions first
        df['net_noncommercial'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
        df['net_commercial'] = df['comm_positions_long_all'] - df['comm_positions_short_all']

        # Calculate each formula
        for formula_col, _ in selected_formulas:
            if formula_col == "nc_net_oi_ratio":
                df[formula_col] = (df['net_noncommercial'] / df['open_interest_all']) * 100
            elif formula_col == "c_net_oi_ratio":
                df[formula_col] = (df['net_commercial'] / df['open_interest_all']) * 100
            elif formula_col == "nc_long_short_ratio":
                df[formula_col] = df['noncomm_positions_long_all'] / df['noncomm_positions_short_all'].replace(0, np.nan)
            elif formula_col == "nc_c_net_ratio":
                df[formula_col] = df['net_noncommercial'] / df['net_commercial'].replace(0, np.nan)

    # Add separator before charts
    st.markdown("---")

    # Display synchronized multi-pane chart
    display_synchronized_charts(df, instrument_name, price_adjustment_code, selected_columns, selected_formulas)

def display_synchronized_charts(df, instrument_name, price_adjustment, selected_columns, selected_formulas=None):
    """Display price chart with synchronized COT data subplots"""

    # Extract instrument name without COT code
    instrument_clean = re.sub(r'\s*\(\d+\)$', '', instrument_name).strip()

    # Find the futures symbol for this COT instrument
    symbol = None
    import os
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instrument_management', 'futures', 'futures_symbols_enhanced.json')
    with open(json_path, 'r') as f:
        mapping = json.load(f)
        for fut_symbol, info in mapping['futures_symbols'].items():
            if info['cot_mapping']['matched']:
                if instrument_clean in info['cot_mapping']['instruments']:
                    symbol = fut_symbol
                    break

    if not symbol:
        st.info("No futures price data available for this instrument")
        # Still show COT data if available
        if selected_columns:
            display_cot_only_charts(df, selected_columns)
        return

    # Display the full instrument name as title above charts
    st.markdown(f"### {instrument_name} - {symbol} ({price_adjustment})")

    # Chart display options
    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        use_unified = st.checkbox("Use unified chart (single container)", value=False, key="use_unified_chart")

    with col_opt2:
        show_events = st.checkbox(
            "Show Historical Events",
            value=False,
            key=f"show_events_{symbol}",
            help="Display market-moving events relevant to this instrument category"
        )

    if use_unified:
        # Call the new unified implementation
        return display_synchronized_charts_unified(df, instrument_name, symbol, selected_columns, selected_formulas, price_adjustment, show_events)

    # Otherwise continue with the panel implementation below...

    # Fetch futures price data
    fetcher = FuturesPriceFetcher()
    start_date = df['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
    end_date = df['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')

    price_df = fetcher.fetch_weekly_prices(symbol, start_date, end_date, price_adjustment)

    if price_df.empty:
        st.warning(f"No price data available for {symbol}")
        if selected_columns:
            display_cot_only_charts(df, selected_columns)
        return

    # Get historical events if checkbox is checked
    events = None
    if show_events:
        category = get_category_from_futures_symbol(symbol)
        events = get_events_for_category(category)

    # Prepare all charts data
    charts = []

    # Chart 1: Price candlesticks with volume
    priceData = []
    volumeData = []
    markers = []

    for _, row in price_df.iterrows():
        date = row['date'].strftime('%Y-%m-%d')
        priceData.append({
            'time': date,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })

        if pd.notna(row['open_interest']) and row['open_interest'] > 0:
            volumeData.append({
                'time': date,
                'value': float(row['open_interest']),
                'color': 'rgba(128, 128, 128, 0.3)'
            })

    # Add event markers if events are available (subtle markers, hover shows text)
    if events:
        data_start = price_df['date'].min()
        data_end = price_df['date'].max()

        for event in events:
            event_date = pd.to_datetime(event['date'])
            # Only add markers for events within the data range
            if data_start <= event_date <= data_end:
                markers.append({
                    'time': event['date'],
                    'position': 'aboveBar',
                    'color': '#808080',  # Gray color for subtle markers
                    'shape': 'circle',   # Small circle instead of arrow
                    'text': event['description'],
                    'size': 0.5  # Very small size
                })

    # Main price chart configuration
    priceChart = {
        "height": 300,
        "layout": {
            "background": {
                "type": 'solid',
                "color": 'white'
            },
            "textColor": '#333',
            "fontSize": 12,
        },
        "grid": {
            "vertLines": {
                "color": 'rgba(197, 203, 206, 0.5)',
            },
            "horzLines": {
                "color": 'rgba(197, 203, 206, 0.5)',
            }
        },
        "crosshair": {
            "mode": 1
        },
        "rightPriceScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "visible": True,
            "scaleMargins": {
                "top": 0.1,
                "bottom": 0.2,
            },
            "width": 90,  # Fixed width for all charts
        },
        "timeScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "timeVisible": True,
            "visible": True,  # Always show time axis
            "secondsVisible": False,
            "rightOffset": 5,
            "barSpacing": 3,  # Reduced spacing for thicker bars
            "fixLeftEdge": True,
            "lockVisibleTimeRangeOnResize": True,
        },
        "watermark": {
            "visible": False,  # Hide watermark since title is shown above
        }
    }

    # Build price series with optional markers
    price_series_config = {
        "type": 'Bar',
        "data": priceData,
        "options": {
            "upColor": 'rgb(38,166,154)',
            "downColor": 'rgb(255,82,82)',
            "wickUpColor": 'rgb(38,166,154)',
            "wickDownColor": 'rgb(255,82,82)',
            "borderVisible": False,
            "thinBars": False,  # Make bars thicker
        }
    }

    # Add markers if events are available
    if markers:
        price_series_config["markers"] = markers

    priceSeries = [price_series_config]

    if volumeData:
        priceSeries.append({
            "type": 'Histogram',
            "data": volumeData,
            "options": {
                "color": '#26a69a',
                "priceFormat": {
                    "type": 'volume',
                },
                "priceScaleId": "",
                "scaleMargins": {
                    "top": 0.8,
                    "bottom": 0,
                }
            }
        })

    charts.append({
        "chart": priceChart,
        "series": priceSeries
    })

    # Add COT data subplots if selected
    if selected_columns:
        # Define colors for different series
        colors = {
            "open_interest_all": "#FFD700",
            "noncomm_positions_long_all": "#00FF00",
            "noncomm_positions_short_all": "#FF0000",
            "comm_positions_long_all": "#FF00FF",
            "comm_positions_short_all": "#FF8C00",
            "net_noncomm_positions": "#00FFFF",
            "net_comm_positions": "#DDA0DD",
            "net_reportable_positions": "#87CEEB"
        }

        # Create a subplot for COT data
        cotData = {}

        for col_name, display_name in selected_columns:
            cotData[col_name] = []

            # Merge COT data with price dates for alignment
            for _, price_row in price_df.iterrows():
                date = price_row['date']
                # Find matching COT data for this date (COT reports are on Tuesdays)
                cot_row = df[df['report_date_as_yyyy_mm_dd'] <= date].iloc[-1] if not df[df['report_date_as_yyyy_mm_dd'] <= date].empty else None

                if cot_row is not None and pd.notna(cot_row[col_name]):
                    cotData[col_name].append({
                        'time': date.strftime('%Y-%m-%d'),
                        'value': float(cot_row[col_name])
                    })

        # Determine if COT chart is the last chart (no formulas selected)
        is_last_chart = not selected_formulas

        # Create COT subplot
        cotChart = {
            "height": 350,
            "layout": {
                "background": {
                    "type": 'solid',
                    "color": 'white'
                },
                "textColor": '#333',
                "fontSize": 12,
            },
            "grid": {
                "vertLines": {
                    "color": 'rgba(197, 203, 206, 0.5)',
                },
                "horzLines": {
                    "color": 'rgba(197, 203, 206, 0.5)',
                }
            },
            "crosshair": {
                "mode": 1
            },
            "rightPriceScale": {
                "borderColor": 'rgba(197, 203, 206, 0.8)',
                "visible": True,
                "scaleMargins": {
                    "top": 0.1,
                    "bottom": 0.1,
                },
                "width": 90,  # Fixed width for all charts
            },
            "timeScale": {
                "borderColor": 'rgba(197, 203, 206, 0.8)',
                "timeVisible": True,
                "visible": True,  # Show time axis on all charts
                "secondsVisible": False,
                "rightOffset": 5,
                "barSpacing": 3,  # Reduced spacing for thicker bars
                "fixLeftEdge": True,
                "lockVisibleTimeRangeOnResize": True,
            }
        }

        cotSeries = []
        for col_name, display_name in selected_columns:
            if cotData[col_name]:
                cotSeries.append({
                    "type": 'Line',
                    "data": cotData[col_name],
                    "options": {
                        "color": colors.get(col_name, '#000000'),
                        "lineWidth": 2,
                        "title": display_name,
                        "priceLineVisible": False,
                        "lastValueVisible": True,
                        "priceFormat": {
                            "type": 'price',
                            "precision": 0,
                            "minMove": 1,
                        }
                    }
                })

        if cotSeries:
            charts.append({
                "chart": cotChart,
                "series": cotSeries
            })

    # Add third subplot for custom formulas if selected
    if selected_formulas:
        formulaData = {}

        for formula_col, display_name in selected_formulas:
            formulaData[formula_col] = []

            # Align formula data with price dates for proper synchronization
            if not price_df.empty:
                # Use price dates and align COT data - same as COT subplot
                for _, price_row in price_df.iterrows():
                    date = price_row['date']
                    # Find matching COT data for this date
                    cot_row = df[df['report_date_as_yyyy_mm_dd'] <= date].iloc[-1] if not df[df['report_date_as_yyyy_mm_dd'] <= date].empty else None

                    if cot_row is not None and pd.notna(cot_row[formula_col]) and not np.isinf(cot_row[formula_col]):
                        formulaData[formula_col].append({
                            'time': date.strftime('%Y-%m-%d'),
                            'value': float(cot_row[formula_col])
                        })
            else:
                # Fallback to COT dates if no price data
                for _, row in df.iterrows():
                    if pd.notna(row[formula_col]) and not np.isinf(row[formula_col]):
                        formulaData[formula_col].append({
                            'time': row['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'),
                            'value': float(row[formula_col])
                        })

        # Define colors for formulas
        formula_colors = {
            "nc_net_oi_ratio": "#FF69B4",
            "c_net_oi_ratio": "#DDA0DD",
            "nc_long_short_ratio": "#DA70D6",
            "nc_c_net_ratio": "#FF1493"
        }

        # Create formula subplot
        formulaChart = {
            "height": 350,
            "layout": {
                "background": {
                    "type": 'solid',
                    "color": 'white'
                },
                "textColor": '#333',
                "fontSize": 12,
            },
            "grid": {
                "vertLines": {
                    "color": 'rgba(197, 203, 206, 0.5)',
                },
                "horzLines": {
                    "color": 'rgba(197, 203, 206, 0.5)',
                }
            },
            "crosshair": {
                "mode": 1
            },
            "rightPriceScale": {
                "borderColor": 'rgba(197, 203, 206, 0.8)',
                "visible": True,
                "scaleMargins": {
                    "top": 0.1,
                    "bottom": 0.1,
                },
                "width": 90,  # Fixed width for all charts
            },
            "timeScale": {
                "borderColor": 'rgba(197, 203, 206, 0.8)',
                "timeVisible": True,
                "secondsVisible": False,
                "visible": True,  # Show time axis on all charts
                "rightOffset": 5,
                "barSpacing": 6,
                "fixLeftEdge": True,
                "lockVisibleTimeRangeOnResize": True,
            }
        }

        formulaSeries = []
        for formula_col, display_name in selected_formulas:
            if formulaData[formula_col]:
                formulaSeries.append({
                    "type": 'Line',
                    "data": formulaData[formula_col],
                    "options": {
                        "color": formula_colors.get(formula_col, '#000000'),
                        "lineWidth": 2,
                        "title": display_name,
                        "priceLineVisible": False,
                        "lastValueVisible": True,
                        "priceFormat": {
                            "type": 'price',
                            "precision": 2,
                            "minMove": 0.01,
                        }
                    }
                })

        if formulaSeries:
            charts.append({
                "chart": formulaChart,
                "series": formulaSeries
            })


    # Render all charts with synchronization
    renderLightweightCharts(charts, 'synchronized_charts')

def display_share_of_oi(df, instrument_name):
    """Display Share of Open Interest chart with futures price data and long/short toggle"""

    # Information box
    st.info("""
    This chart shows how open interest is distributed among different trader categories.

    **Calculation Method:**
    - **Long Side:** NonComm Long + Spread + Comm Long + NonRep Long = Total Open Interest
    - **Short Side:** NonComm Short + Spread + Comm Short + NonRep Short = Total Open Interest
    """)

    # Extract instrument name without COT code
    instrument_clean = re.sub(r'\s*\(\d+\)$', '', instrument_name).strip()

    # Find the futures symbol for this COT instrument
    symbol = None
    import os
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instrument_management', 'futures', 'futures_symbols_enhanced.json')
    with open(json_path, 'r') as f:
        mapping = json.load(f)
        for fut_symbol, info in mapping['futures_symbols'].items():
            if info['cot_mapping']['matched']:
                if instrument_clean in info['cot_mapping']['instruments']:
                    symbol = fut_symbol
                    break

    if not symbol:
        st.warning("No futures price data available for this instrument")
        return

    # Price adjustment options
    col1, col2 = st.columns(2)

    with col1:
        # Radio button for long/short toggle
        side_selection = st.radio(
            "Select Side to Display:",
            ["Long Positions", "Short Positions"],
            key=f"oi_side_{instrument_name}"
        )

    with col2:
        # Price adjustment dropdown (similar to Time Series)
        price_adjustment = st.selectbox(
            "Price Adjustment Method:",
            ["NON - Non-adjusted", "RAD - Ratio Adjusted", "REV - Reverse Adjusted"],
            key=f"price_adj_oi_{instrument_name}"
        )
        price_adjustment_code = price_adjustment.split(" ")[0]

    # Calculate OI raw contract counts
    df_oi = df.copy()

    if side_selection == "Long Positions":
        # Use raw contract counts for long side
        df_oi['noncomm'] = df['noncomm_positions_long_all']
        df_oi['spread'] = df['noncomm_postions_spread_all']  # Note: API has typo in column name
        df_oi['comm'] = df['comm_positions_long_all']
        df_oi['nonrep'] = df['nonrept_positions_long_all']
        df_oi['total_oi'] = df['open_interest_all']

        title_suffix = "Long Positions"
        colors = {
            'noncomm': 'rgba(0, 255, 0, 0.8)',      # Green for non-commercial long
            'spread': 'rgba(255, 215, 0, 0.8)',      # Gold for spread
            'comm': 'rgba(0, 100, 255, 0.8)',        # Blue for commercial long
            'nonrep': 'rgba(128, 128, 128, 0.8)'     # Gray for non-reportable long
        }
    else:
        # Use raw contract counts for short side
        df_oi['noncomm'] = df['noncomm_positions_short_all']
        df_oi['spread'] = df['noncomm_postions_spread_all']  # Note: API has typo in column name
        df_oi['comm'] = df['comm_positions_short_all']
        df_oi['nonrep'] = df['nonrept_positions_short_all']
        df_oi['total_oi'] = df['open_interest_all']

        title_suffix = "Short Positions"
        colors = {
            'noncomm': 'rgba(255, 0, 0, 0.8)',       # Red for non-commercial short
            'spread': 'rgba(255, 215, 0, 0.8)',      # Gold for spread (same)
            'comm': 'rgba(255, 140, 0, 0.8)',        # Orange for commercial short
            'nonrep': 'rgba(64, 64, 64, 0.8)'        # Dark gray for non-reportable short
        }

    # Fetch futures price data
    fetcher = FuturesPriceFetcher()
    start_date = df['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
    end_date = df['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')

    price_df = fetcher.fetch_weekly_prices(symbol, start_date, end_date, price_adjustment_code)

    # Prepare data for lightweight charts
    charts = []

    # Price chart
    if not price_df.empty:
        priceData = []
        for _, row in price_df.iterrows():
            date = row['date'].strftime('%Y-%m-%d')
            priceData.append({
                'time': date,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })

        priceChart = {
            "height": 350,
            "layout": {
                "background": {"type": 'solid', "color": 'white'},
                "textColor": '#333',
                "fontSize": 12,
            },
            "grid": {
                "vertLines": {"color": 'rgba(197, 203, 206, 0.5)'},
                "horzLines": {"color": 'rgba(197, 203, 206, 0.5)'}
            },
            "crosshair": {"mode": 1},
            "rightPriceScale": {
                "borderColor": 'rgba(197, 203, 206, 0.8)',
                "visible": True,
                "scaleMargins": {"top": 0.1, "bottom": 0.1},
                "width": 90,
            },
            "timeScale": {
                "borderColor": 'rgba(197, 203, 206, 0.8)',
                "timeVisible": True,
                "visible": True,
                "secondsVisible": False,
                "rightOffset": 5,
                "barSpacing": 6,
                "fixLeftEdge": True,
                "lockVisibleTimeRangeOnResize": True,
            },
            "watermark": {
                "visible": True,
                "text": f'{symbol} {price_adjustment_code}',
                "fontSize": 24,
                "color": 'rgba(0, 0, 0, 0.1)',
                "horzAlign": 'center',
                "vertAlign": 'center',
            }
        }

        charts.append({
            "chart": priceChart,
            "series": [{
                "type": 'Bar',
                "data": priceData,
                "options": {
                    "upColor": 'rgb(38,166,154)',
                    "downColor": 'rgb(255,82,82)',
                    "wickUpColor": 'rgb(38,166,154)',
                    "wickDownColor": 'rgb(255,82,82)',
                    "borderVisible": False,
                }
            }]
        })

    # OI distribution chart (stacked histogram)
    oiData = []

    # Align OI data with price dates if available
    if not price_df.empty:
        # Create OI data aligned with price dates
        for _, price_row in price_df.iterrows():
            date = price_row['date']
            date_str = date.strftime('%Y-%m-%d')

            # Find the most recent COT data for this date
            cot_rows = df_oi[df_oi['report_date_as_yyyy_mm_dd'] <= date]
            if not cot_rows.empty:
                row = cot_rows.iloc[-1]

                # Calculate cumulative values for stacking
                noncomm_val = float(row['noncomm']) if pd.notna(row['noncomm']) else 0
                spread_val = float(row['spread']) if pd.notna(row['spread']) else 0
                comm_val = float(row['comm']) if pd.notna(row['comm']) else 0
                nonrep_val = float(row['nonrep']) if pd.notna(row['nonrep']) else 0

                oiData.append({
                    'time': date_str,
                    'noncomm': noncomm_val,
                    'spread': noncomm_val + spread_val,
                    'comm': noncomm_val + spread_val + comm_val,
                    'nonrep': noncomm_val + spread_val + comm_val + nonrep_val,
                    'total': float(row['total_oi']) if pd.notna(row['total_oi']) else 0
                })
    else:
        # Fallback to COT dates if no price data
        for _, row in df_oi.iterrows():
            date_str = row['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d')

            # Calculate cumulative values for stacking
            noncomm_val = float(row['noncomm']) if pd.notna(row['noncomm']) else 0
            spread_val = float(row['spread']) if pd.notna(row['spread']) else 0
            comm_val = float(row['comm']) if pd.notna(row['comm']) else 0
            nonrep_val = float(row['nonrep']) if pd.notna(row['nonrep']) else 0

            # Create stacked values
            oiData.append({
                'time': date_str,
                'noncomm': noncomm_val,
                'spread': noncomm_val + spread_val,
                'comm': noncomm_val + spread_val + comm_val,
                'nonrep': noncomm_val + spread_val + comm_val + nonrep_val,
                'total': float(row['total_oi']) if pd.notna(row['total_oi']) else 0
            })

    # Create OI distribution chart
    oiChart = {
        "height": 400,
        "layout": {
            "background": {"type": 'solid', "color": 'white'},
            "textColor": '#333',
            "fontSize": 12,
        },
        "grid": {
            "vertLines": {"color": 'rgba(197, 203, 206, 0.5)'},
            "horzLines": {"color": 'rgba(197, 203, 206, 0.5)'}
        },
        "crosshair": {"mode": 1},
        "rightPriceScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "visible": True,
            "scaleMargins": {"top": 0.1, "bottom": 0.1},
            "width": 90,
        },
        "timeScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "timeVisible": True,
            "visible": True,
            "secondsVisible": False,
            "rightOffset": 5,
            "barSpacing": 6,
            "fixLeftEdge": True,
            "lockVisibleTimeRangeOnResize": True,
        },
        "watermark": {
            "visible": True,
            "text": f'Share of Open Interest - {title_suffix}',
            "fontSize": 24,
            "color": 'rgba(0, 0, 0, 0.1)',
            "horzAlign": 'center',
            "vertAlign": 'center',
        }
    }

    # Create histogram series for stacked bars
    oiSeries = []

    # Add Non-Reportable (top layer)
    nonrepData = [{'time': d['time'], 'value': d['nonrep']} for d in oiData]
    oiSeries.append({
        "type": 'Histogram',
        "data": nonrepData,
        "options": {
            "color": colors['nonrep'],
            "priceLineVisible": False,
            "lastValueVisible": True,
            "title": "Non-Reportable"
        }
    })

    # Add Commercial
    commData = [{'time': d['time'], 'value': d['comm']} for d in oiData]
    oiSeries.append({
        "type": 'Histogram',
        "data": commData,
        "options": {
            "color": colors['comm'],
            "priceLineVisible": False,
            "lastValueVisible": False,
            "title": "Commercial"
        }
    })

    # Add Spread
    spreadData = [{'time': d['time'], 'value': d['spread']} for d in oiData]
    oiSeries.append({
        "type": 'Histogram',
        "data": spreadData,
        "options": {
            "color": colors['spread'],
            "priceLineVisible": False,
            "lastValueVisible": False,
            "title": "Spread"
        }
    })

    # Add Non-Commercial (bottom layer)
    noncommData = [{'time': d['time'], 'value': d['noncomm']} for d in oiData]
    oiSeries.append({
        "type": 'Histogram',
        "data": noncommData,
        "options": {
            "color": colors['noncomm'],
            "priceLineVisible": False,
            "lastValueVisible": False,
            "title": "Non-Commercial"
        }
    })

    # Add total open interest line for reference
    totalOiData = [{'time': d['time'], 'value': d['total']} for d in oiData]
    oiSeries.append({
        "type": 'Line',
        "data": totalOiData,
        "options": {
            "color": 'rgba(0, 0, 0, 0.5)',
            "lineWidth": 2,
            "lineStyle": 2,  # Dashed
            "priceLineVisible": False,
            "lastValueVisible": True,
            "title": "Total OI"
        }
    })

    # Add the OI chart to the charts array for synchronized rendering
    charts.append({
        "chart": oiChart,
        "series": oiSeries
    })

    # Render all charts with synchronization
    renderLightweightCharts(charts, 'share_of_oi_charts')


def display_seasonality(df, instrument_name):
    """Display seasonality analysis for the instrument"""
    st.markdown("### Seasonality Analysis")

    # Check if data is available
    if df is None or df.empty:
        st.info("📊 Please fetch data first using the 'Fetch Data' button above to view seasonality analysis.")
        return

    # Create columns for controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        # Select column to analyze with cleaner naming
        column_display_names = {
            "net_noncomm_positions": "Net Positioning → Net Non-Commercial",
            "net_comm_positions": "Net Positioning → Net Commercial",
            "open_interest_all": "Open Interest",
            "noncomm_positions_long_all": "Non-Commercial Long",
            "noncomm_positions_short_all": "Non-Commercial Short",
            "comm_positions_long_all": "Commercial Long",
            "comm_positions_short_all": "Commercial Short",
            "nonrept_positions_long_all": "Non-Reportable Long",
            "nonrept_positions_short_all": "Non-Reportable Short",
            "net_reportable_positions": "Net Reportable",
            "traders_tot_all": "Total Traders"
        }

        numeric_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64']
                          and col not in ['report_date_as_yyyy_mm_dd', 'day_of_year', 'year']]

        # Create display options
        display_options = []
        for col in numeric_columns:
            if col in column_display_names:
                display_options.append(column_display_names[col])
            else:
                display_options.append(col.replace('_', ' ').title())

        # Default to Net Non-Commercial if available
        default_idx = 0
        if 'net_noncomm_positions' in numeric_columns:
            default_idx = numeric_columns.index('net_noncomm_positions')
        elif 'open_interest_all' in numeric_columns:
            default_idx = numeric_columns.index('open_interest_all')

        selected_display = st.selectbox(
            "Select metric:",
            options=display_options,
            index=default_idx,
            key=f"seasonality_column_{instrument_name}",
            label_visibility="visible"
        )

        # Map back to actual column name
        selected_column = numeric_columns[display_options.index(selected_display)]

    with col2:
        # Lookback period selection with simpler options
        lookback_options = {
            "5 Years": 5,
            "10 Years": 10,
            "15 Years": 15,
            "All Data": 'all'
        }

        lookback_label = st.selectbox(
            "Lookback period:",
            options=list(lookback_options.keys()),
            index=0,
            key=f"seasonality_lookback_{instrument_name}",
            label_visibility="visible"
        )
        lookback_years = lookback_options[lookback_label]

    with col3:
        # Zone type as radio buttons
        zone_type = st.radio(
            "Zone type:",
            options=['Percentile', 'Std Dev'],
            index=0,
            key=f"seasonality_zone_{instrument_name}",
            label_visibility="visible"
        )
        zone_type_value = 'percentile' if zone_type == 'Percentile' else 'std'

    # Show previous year checkbox below
    show_previous_year = st.checkbox(
        "Show previous year",
        value=True,
        key=f"seasonality_prev_year_{instrument_name}"
    )

    # Create and display the seasonality chart
    fig = create_seasonality_chart(
        df,
        selected_column,
        lookback_years,
        show_previous_year,
        zone_type_value
    )

    if fig:
        st.plotly_chart(fig, use_container_width=True)


def create_seasonality_chart(df, column, lookback_years=5, show_previous_year=True, zone_type='percentile'):
    """Create seasonality chart showing historical patterns with smooth zones"""
    try:
        # Add day of year column
        df_season = df.copy()
        df_season['day_of_year'] = df_season['report_date_as_yyyy_mm_dd'].dt.dayofyear
        df_season['year'] = df_season['report_date_as_yyyy_mm_dd'].dt.year

        # Determine lookback period
        if lookback_years == 'all':
            start_year = df_season['year'].min()
        else:
            start_year = df_season['year'].max() - lookback_years + 1

        # Filter for lookback period
        df_lookback = df_season[df_season['year'] >= start_year]

        # Create figure
        fig = go.Figure()

        # If lookback is 5 years and not 'all', just plot the raw data for each year
        if lookback_years == 5:
            # Create a reference year for x-axis (using current year for display)
            current_year = pd.Timestamp.now().year

            # Plot each year's data overlaid on the same annual timeline
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for idx, year in enumerate(sorted(df_lookback['year'].unique(), reverse=True)):
                df_year = df_lookback[df_lookback['year'] == year].copy()
                if not df_year.empty:
                    # Create display dates using a common year for overlay
                    df_year['display_date'] = pd.to_datetime(
                        df_year['day_of_year'].astype(str) + f'-{current_year}',
                        format='%j-%Y'
                    )

                    # Use different styling for current year
                    is_current_year = (year == df_lookback['year'].max())

                    fig.add_trace(go.Scatter(
                        x=df_year['display_date'],
                        y=df_year[column],
                        mode='lines+markers' if is_current_year else 'lines',
                        name=str(year),
                        line=dict(
                            width=3 if is_current_year else 2,
                            color=colors[idx % len(colors)]
                        ),
                        marker=dict(size=4) if is_current_year else None
                    ))

            # Update layout for seasonality view
            fig.update_layout(
                title=f'{column.replace("_", " ").title()} - Last {lookback_years} Years Seasonal Comparison',
                xaxis_title='Month',
                yaxis_title=column.replace('_', ' ').title(),
                hovermode='x unified',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                )
            )

            # Format x-axis to show months
            fig.update_xaxes(
                tickformat="%b",
                dtick="M1",
                ticklabelmode="period",
                rangeslider_visible=True,
                rangeslider_thickness=0.05
            )

            return fig

        # Otherwise, continue with the original statistical analysis
        # Instead of grouping by exact day, use all historical data within a rolling window
        # This gives us more data points for percentile calculation
        all_days = range(1, 367)  # Include leap year day
        daily_stats = []

        for day in all_days:
            # Get data within a 15-day window centered on this day
            window_start = day - 15
            window_end = day + 15

            # Handle year boundary wrapping
            if window_start < 1:
                mask = (df_lookback['day_of_year'] >= (365 + window_start)) | (df_lookback['day_of_year'] <= window_end)
            elif window_end > 365:
                mask = (df_lookback['day_of_year'] >= window_start) | (df_lookback['day_of_year'] <= (window_end - 365))
            else:
                mask = (df_lookback['day_of_year'] >= window_start) & (df_lookback['day_of_year'] <= window_end)

            window_data = df_lookback[mask][column].dropna()

            if len(window_data) > 0:
                daily_stats.append({
                    'day_of_year': day,
                    'mean': window_data.mean(),
                    'std': window_data.std(),
                    'count': len(window_data),
                    'q10': window_data.quantile(0.1),
                    'q25': window_data.quantile(0.25),
                    'q50': window_data.quantile(0.5),
                    'q75': window_data.quantile(0.75),
                    'q90': window_data.quantile(0.9)
                })

        daily_stats = pd.DataFrame(daily_stats)

        # Sort by day of year
        daily_stats = daily_stats.sort_values('day_of_year')

        # Apply smoothing to make cleaner zones
        from scipy.signal import savgol_filter
        window_length = min(31, len(daily_stats) // 2 * 2 - 1)  # Must be odd
        if window_length >= 5:  # Only smooth if we have enough data
            for col in ['mean', 'q10', 'q25', 'q50', 'q75', 'q90']:
                if col in daily_stats.columns:
                    daily_stats[f'{col}_smooth'] = savgol_filter(
                        daily_stats[col].ffill().bfill(),
                        window_length=window_length,
                        polyorder=3
                    )
        else:
            # If not enough data for smoothing, use original values
            for col in ['mean', 'q10', 'q25', 'q50', 'q75', 'q90']:
                if col in daily_stats.columns:
                    daily_stats[f'{col}_smooth'] = daily_stats[col]

        # Determine which smoothed columns to use based on zone_type
        if zone_type == 'percentile':
            upper_col = 'q90_smooth'
            lower_col = 'q10_smooth'
            upper_mid_col = 'q75_smooth'
            lower_mid_col = 'q25_smooth'
            zone_label = 'Percentile Zones'
        else:  # standard deviation
            daily_stats['upper_2std'] = daily_stats['mean'] + 2 * daily_stats['std']
            daily_stats['lower_2std'] = daily_stats['mean'] - 2 * daily_stats['std']
            daily_stats['upper_1std'] = daily_stats['mean'] + daily_stats['std']
            daily_stats['lower_1std'] = daily_stats['mean'] - daily_stats['std']

            # Smooth std-based zones
            if window_length >= 5:
                for col in ['upper_2std', 'lower_2std', 'upper_1std', 'lower_1std']:
                    daily_stats[f'{col}_smooth'] = savgol_filter(
                        daily_stats[col].ffill().bfill(),
                        window_length=window_length,
                        polyorder=3
                    )
            else:
                for col in ['upper_2std', 'lower_2std', 'upper_1std', 'lower_1std']:
                    daily_stats[f'{col}_smooth'] = daily_stats[col]

            upper_col = 'upper_2std_smooth'
            lower_col = 'lower_2std_smooth'
            upper_mid_col = 'upper_1std_smooth'
            lower_mid_col = 'lower_1std_smooth'
            zone_label = 'Standard Deviation Zones'

        # Create x-axis with dates for better display
        current_year = pd.Timestamp.now().year
        daily_stats['display_date'] = pd.to_datetime(
            daily_stats['day_of_year'].astype(str) + f'-{current_year}',
            format='%j-%Y'
        )

        # Add shaded zones
        # Outer zone (90-10 percentiles or ±2 std)
        fig.add_trace(go.Scatter(
            x=daily_stats['display_date'],
            y=daily_stats[upper_col],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=daily_stats['display_date'],
            y=daily_stats[lower_col],
            mode='lines',
            line=dict(width=0),
            name=zone_label,
            fill='tonexty',
            fillcolor='rgba(128, 128, 255, 0.1)',
            hoverinfo='skip'
        ))

        # Inner zone (75-25 percentiles or ±1 std)
        fig.add_trace(go.Scatter(
            x=daily_stats['display_date'],
            y=daily_stats[upper_mid_col],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=daily_stats['display_date'],
            y=daily_stats[lower_mid_col],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            fill='tonexty',
            fillcolor='rgba(128, 128, 255, 0.2)',
            hoverinfo='skip'
        ))

        # Add mean/median line
        center_col = 'q50_smooth' if zone_type == 'percentile' else 'mean_smooth'
        fig.add_trace(go.Scatter(
            x=daily_stats['display_date'],
            y=daily_stats[center_col],
            mode='lines',
            name='Historical Average' if zone_type == 'std' else 'Historical Median',
            line=dict(color='blue', width=2, dash='dash')
        ))

        # Get current year data
        current_year_actual = df_season['year'].max()
        df_current = df_season[df_season['year'] == current_year_actual].copy()

        if not df_current.empty:
            # Create display dates for current year
            df_current['display_date'] = pd.to_datetime(
                df_current['day_of_year'].astype(str) + f'-{current_year}',
                format='%j-%Y'
            )

            # Add current year line
            fig.add_trace(go.Scatter(
                x=df_current['display_date'],
                y=df_current[column],
                mode='lines+markers',
                name=f'Current Year ({current_year_actual})',
                line=dict(color='red', width=3),
                marker=dict(size=4)
            ))

        # Add previous year if requested
        if show_previous_year and current_year_actual > start_year:
            df_previous = df_season[df_season['year'] == current_year_actual - 1].copy()
            if not df_previous.empty:
                df_previous['display_date'] = pd.to_datetime(
                    df_previous['day_of_year'].astype(str) + f'-{current_year}',
                    format='%j-%Y'
                )

                fig.add_trace(go.Scatter(
                    x=df_previous['display_date'],
                    y=df_previous[column],
                    mode='lines',
                    name=f'Previous Year ({current_year_actual - 1})',
                    line=dict(color='orange', width=2, dash='dot'),
                    opacity=0.7
                ))

        # Update layout
        lookback_text = "All Years" if lookback_years == 'all' else f"{lookback_years} Year"
        fig.update_layout(
            title=f'Seasonality Analysis: {column.replace("_", " ").title()} ({lookback_text} Lookback)',
            xaxis_title='Month',
            yaxis_title=column.replace('_', ' ').title(),
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Format x-axis to show months
        fig.update_xaxes(
            tickformat="%b",
            dtick="M1",
            ticklabelmode="period"
        )

        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05
        )

        return fig

    except Exception as e:
        import traceback
        st.error(f"Error creating seasonality chart: {str(e)}")
        st.text(f"Full error trace:\n{traceback.format_exc()}")
        return None


def display_cot_only_charts(df, selected_columns):
    """Display only COT data when no price data is available"""

    # Define colors for different series
    colors = {
        "open_interest_all": "#FFD700",
        "noncomm_positions_long_all": "#00FF00",
        "noncomm_positions_short_all": "#FF0000",
        "comm_positions_long_all": "#FF00FF",
        "comm_positions_short_all": "#FF8C00",
        "net_noncomm_positions": "#00FFFF",
        "net_comm_positions": "#DDA0DD",
        "net_reportable_positions": "#87CEEB"
    }

    cotData = {}
    for col_name, display_name in selected_columns:
        cotData[col_name] = []
        for _, row in df.iterrows():
            if pd.notna(row[col_name]):
                cotData[col_name].append({
                    'time': row['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'),
                    'value': float(row[col_name])
                })

    cotChart = {
        "height": 400,
        "layout": {
            "background": {
                "type": 'solid',
                "color": 'white'
            },
            "textColor": '#333',
        },
        "grid": {
            "vertLines": {
                "color": 'rgba(197, 203, 206, 0.5)',
            },
            "horzLines": {
                "color": 'rgba(197, 203, 206, 0.5)',
            }
        },
        "crosshair": {
            "mode": 1
        },
        "rightPriceScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "visible": True,
        },
        "timeScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "timeVisible": True,
            "secondsVisible": False,
        }
    }

    cotSeries = []
    for col_name, display_name in selected_columns:
        if cotData[col_name]:
            cotSeries.append({
                "type": 'Line',
                "data": cotData[col_name],
                "options": {
                    "color": colors.get(col_name, '#000000'),
                    "lineWidth": 2,
                    "title": display_name,
                    "priceLineVisible": False,
                    "lastValueVisible": True,
                }
            })

    if cotSeries:
        renderLightweightCharts([{
            "chart": cotChart,
            "series": cotSeries
        }], 'cot_only_chart')



def display_oi_split_chart(df, price_df, instrument_name):
    """Display Open Interest Split chart using lightweight charts"""

    st.markdown("### Open Interest Split Analysis")

    # Calculate percentages of open interest
    df_oi = df.copy()
    df_oi['noncomm_long_pct'] = (df_oi['noncomm_positions_long_all'] / df_oi['open_interest_all']) * 100
    df_oi['noncomm_short_pct'] = (df_oi['noncomm_positions_short_all'] / df_oi['open_interest_all']) * 100
    df_oi['comm_long_pct'] = (df_oi['comm_positions_long_all'] / df_oi['open_interest_all']) * 100
    df_oi['comm_short_pct'] = (df_oi['comm_positions_short_all'] / df_oi['open_interest_all']) * 100

    # Calculate non-reportable (small traders)
    df_oi['nonrep_long_pct'] = ((df_oi['nonrep_positions_long_all'] if 'nonrep_positions_long_all' in df_oi.columns
                                  else (df_oi['open_interest_all'] - df_oi['noncomm_positions_long_all'] - df_oi['comm_positions_long_all']))
                                 / df_oi['open_interest_all']) * 100
    df_oi['nonrep_short_pct'] = ((df_oi['nonrep_positions_short_all'] if 'nonrep_positions_short_all' in df_oi.columns
                                   else (df_oi['open_interest_all'] - df_oi['noncomm_positions_short_all'] - df_oi['comm_positions_short_all']))
                                  / df_oi['open_interest_all']) * 100

    # Prepare data for lightweight charts - align with price dates
    oiSplitData = {
        'noncomm_long': [],
        'noncomm_short': [],
        'comm_long': [],
        'comm_short': [],
        'nonrep_long': [],
        'nonrep_short': []
    }

    if not price_df.empty:
        # Align with price dates for consistency
        for _, price_row in price_df.iterrows():
            date = price_row['date']
            # Find matching COT data
            cot_row = df_oi[df_oi['report_date_as_yyyy_mm_dd'] <= date].iloc[-1] if not df_oi[df_oi['report_date_as_yyyy_mm_dd'] <= date].empty else None

            if cot_row is not None:
                date_str = date.strftime('%Y-%m-%d')
                oiSplitData['noncomm_long'].append({'time': date_str, 'value': float(cot_row['noncomm_long_pct']) if pd.notna(cot_row['noncomm_long_pct']) else 0})
                oiSplitData['noncomm_short'].append({'time': date_str, 'value': -float(cot_row['noncomm_short_pct']) if pd.notna(cot_row['noncomm_short_pct']) else 0})
                oiSplitData['comm_long'].append({'time': date_str, 'value': float(cot_row['comm_long_pct']) if pd.notna(cot_row['comm_long_pct']) else 0})
                oiSplitData['comm_short'].append({'time': date_str, 'value': -float(cot_row['comm_short_pct']) if pd.notna(cot_row['comm_short_pct']) else 0})
                oiSplitData['nonrep_long'].append({'time': date_str, 'value': float(cot_row['nonrep_long_pct']) if pd.notna(cot_row['nonrep_long_pct']) else 0})
                oiSplitData['nonrep_short'].append({'time': date_str, 'value': -float(cot_row['nonrep_short_pct']) if pd.notna(cot_row['nonrep_short_pct']) else 0})
    else:
        # Use COT dates directly
        for _, row in df_oi.iterrows():
            date_str = row['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d')
            oiSplitData['noncomm_long'].append({'time': date_str, 'value': float(row['noncomm_long_pct']) if pd.notna(row['noncomm_long_pct']) else 0})
            oiSplitData['noncomm_short'].append({'time': date_str, 'value': -float(row['noncomm_short_pct']) if pd.notna(row['noncomm_short_pct']) else 0})
            oiSplitData['comm_long'].append({'time': date_str, 'value': float(row['comm_long_pct']) if pd.notna(row['comm_long_pct']) else 0})
            oiSplitData['comm_short'].append({'time': date_str, 'value': -float(row['comm_short_pct']) if pd.notna(row['comm_short_pct']) else 0})
            oiSplitData['nonrep_long'].append({'time': date_str, 'value': float(row['nonrep_long_pct']) if pd.notna(row['nonrep_long_pct']) else 0})
            oiSplitData['nonrep_short'].append({'time': date_str, 'value': -float(row['nonrep_short_pct']) if pd.notna(row['nonrep_short_pct']) else 0})

    # Create chart configuration
    oiChart = {
        "height": 400,
        "layout": {
            "background": {
                "type": 'solid',
                "color": 'white'
            },
            "textColor": '#333',
            "fontSize": 12,
        },
        "grid": {
            "vertLines": {
                "color": 'rgba(197, 203, 206, 0.5)',
            },
            "horzLines": {
                "color": 'rgba(197, 203, 206, 0.5)',
            }
        },
        "crosshair": {
            "mode": 1
        },
        "rightPriceScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "visible": True,
            "scaleMargins": {
                "top": 0.1,
                "bottom": 0.1,
            },
            "width": 90,
        },
        "timeScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "timeVisible": True,
            "visible": True,
            "secondsVisible": False,
            "rightOffset": 5,
            "barSpacing": 6,
            "fixLeftEdge": True,
            "lockVisibleTimeRangeOnResize": True,
        },
        "watermark": {
            "visible": False,
        }
    }

    # Create series
    oiSeries = []

    # Add long positions (positive values)
    oiSeries.append({
        "type": 'Area',
        "data": oiSplitData['noncomm_long'],
        "options": {
            "topColor": 'rgba(0, 255, 0, 0.4)',
            "bottomColor": 'rgba(0, 255, 0, 0.05)',
            "lineColor": 'rgba(0, 255, 0, 1)',
            "lineWidth": 2,
            "title": "Non-Comm Long %",
            "priceLineVisible": False,
        }
    })

    oiSeries.append({
        "type": 'Area',
        "data": oiSplitData['comm_long'],
        "options": {
            "topColor": 'rgba(255, 0, 255, 0.4)',
            "bottomColor": 'rgba(255, 0, 255, 0.05)',
            "lineColor": 'rgba(255, 0, 255, 1)',
            "lineWidth": 2,
            "title": "Commercial Long %",
            "priceLineVisible": False,
        }
    })

    oiSeries.append({
        "type": 'Area',
        "data": oiSplitData['nonrep_long'],
        "options": {
            "topColor": 'rgba(128, 128, 128, 0.4)',
            "bottomColor": 'rgba(128, 128, 128, 0.05)',
            "lineColor": 'rgba(128, 128, 128, 1)',
            "lineWidth": 2,
            "title": "Non-Rep Long %",
            "priceLineVisible": False,
        }
    })

    # Add short positions (negative values)
    oiSeries.append({
        "type": 'Area',
        "data": oiSplitData['noncomm_short'],
        "options": {
            "topColor": 'rgba(255, 0, 0, 0.05)',
            "bottomColor": 'rgba(255, 0, 0, 0.4)',
            "lineColor": 'rgba(255, 0, 0, 1)',
            "lineWidth": 2,
            "title": "Non-Comm Short %",
            "priceLineVisible": False,
        }
    })

    oiSeries.append({
        "type": 'Area',
        "data": oiSplitData['comm_short'],
        "options": {
            "topColor": 'rgba(255, 140, 0, 0.05)',
            "bottomColor": 'rgba(255, 140, 0, 0.4)',
            "lineColor": 'rgba(255, 140, 0, 1)',
            "lineWidth": 2,
            "title": "Commercial Short %",
            "priceLineVisible": False,
        }
    })

    oiSeries.append({
        "type": 'Area',
        "data": oiSplitData['nonrep_short'],
        "options": {
            "topColor": 'rgba(64, 64, 64, 0.05)',
            "bottomColor": 'rgba(64, 64, 64, 0.4)',
            "lineColor": 'rgba(64, 64, 64, 1)',
            "lineWidth": 2,
            "title": "Non-Rep Short %",
            "priceLineVisible": False,
        }
    })

    # Add zero line
    if oiSplitData['noncomm_long']:
        zero_line = [{'time': point['time'], 'value': 0} for point in oiSplitData['noncomm_long']]
        oiSeries.append({
            "type": 'Line',
            "data": zero_line,
            "options": {
                "color": 'rgba(0, 0, 0, 0.5)',
                "lineWidth": 1,
                "lineStyle": 2,  # Dashed
                "priceLineVisible": False,
                "lastValueVisible": False,
            }
        })

    # Render chart
    renderLightweightCharts([{
        "chart": oiChart,
        "series": oiSeries
    }], 'oi_split_chart')


def display_percentile_tab(df, instrument_name):
    """Display percentile analysis within Time Series tab"""
    from display_functions_exact import display_percentile_chart
    display_percentile_chart(df, instrument_name)


def display_momentum_tab(df, instrument_name):
    """Display momentum analysis within Time Series tab"""
    from display_functions_exact import display_momentum_chart
    display_momentum_chart(df, instrument_name)


def display_extremes_seasonality(df, instrument_name):
    """Tab 6: Extremes & Seasonality - Are we at a turning point?
    Combines percentile bands with seasonal patterns
    """
    st.markdown("### 🎯 Extremes & Seasonality - Are we at a turning point?")
    st.markdown("Historical extremes aligned with seasonal patterns")

    # Create combined view
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],
        subplot_titles=[
            "Percentile Bands & Historical Extremes",
            "Seasonal Pattern Overlay"
        ],
        vertical_spacing=0.15,
        shared_xaxes=True
    )

    # Calculate percentiles
    if 'net_noncomm_positions' not in df.columns:
        df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']

    # Rolling percentiles
    window = 52
    df['pct_90'] = df['net_noncomm_positions'].rolling(window=window, min_periods=20).quantile(0.9)
    df['pct_75'] = df['net_noncomm_positions'].rolling(window=window, min_periods=20).quantile(0.75)
    df['pct_50'] = df['net_noncomm_positions'].rolling(window=window, min_periods=20).quantile(0.5)
    df['pct_25'] = df['net_noncomm_positions'].rolling(window=window, min_periods=20).quantile(0.25)
    df['pct_10'] = df['net_noncomm_positions'].rolling(window=window, min_periods=20).quantile(0.1)

    # 1. Percentile Bands Chart
    fig.add_trace(
        go.Scatter(
            x=df['report_date_as_yyyy_mm_dd'],
            y=df['pct_90'],
            name='90th Percentile',
            line=dict(color='red', dash='dash', width=1),
            showlegend=True
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['report_date_as_yyyy_mm_dd'],
            y=df['pct_10'],
            name='10th Percentile',
            line=dict(color='green', dash='dash', width=1),
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.2)',
            showlegend=True
        ),
        row=1, col=1
    )

    # Add actual position
    fig.add_trace(
        go.Scatter(
            x=df['report_date_as_yyyy_mm_dd'],
            y=df['net_noncomm_positions'],
            name='Net Position',
            line=dict(color='blue', width=2),
            showlegend=True
        ),
        row=1, col=1
    )

    # Mark extremes
    extremes_high = df[df['net_noncomm_positions'] >= df['pct_90']]
    extremes_low = df[df['net_noncomm_positions'] <= df['pct_10']]

    if not extremes_high.empty:
        fig.add_trace(
            go.Scatter(
                x=extremes_high['report_date_as_yyyy_mm_dd'],
                y=extremes_high['net_noncomm_positions'],
                mode='markers',
                name='High Extreme',
                marker=dict(color='red', size=8, symbol='triangle-up'),
                showlegend=True
            ),
            row=1, col=1
        )

    if not extremes_low.empty:
        fig.add_trace(
            go.Scatter(
                x=extremes_low['report_date_as_yyyy_mm_dd'],
                y=extremes_low['net_noncomm_positions'],
                mode='markers',
                name='Low Extreme',
                marker=dict(color='green', size=8, symbol='triangle-down'),
                showlegend=True
            ),
            row=1, col=1
        )

    # 2. Seasonal Pattern (simplified version)
    # Group by week of year and calculate averages
    df['week_of_year'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd']).dt.isocalendar().week
    df['year'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd']).dt.year

    # Calculate seasonal average for last 5 years
    recent_years = df[df['year'] >= df['year'].max() - 5]
    seasonal_avg = recent_years.groupby('week_of_year')['net_noncomm_positions'].mean()

    # Current year data
    current_year = df[df['year'] == df['year'].max()]

    # Create week mapping for current year
    weeks = list(range(1, 54))
    seasonal_line = []
    for week in weeks:
        if week in seasonal_avg.index:
            seasonal_line.append(seasonal_avg[week])
        else:
            seasonal_line.append(None)

    # Plot seasonal average
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=seasonal_line,
            name='5Y Seasonal Average',
            line=dict(color='orange', width=2),
            showlegend=True
        ),
        row=2, col=1
    )

    # Plot current year
    fig.add_trace(
        go.Scatter(
            x=current_year['week_of_year'],
            y=current_year['net_noncomm_positions'],
            name='Current Year',
            line=dict(color='blue', width=2),
            mode='lines+markers',
            showlegend=True
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        title=f"{instrument_name} - Extremes & Seasonality"
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Week of Year", row=2, col=1)
    fig.update_yaxes(title_text="Net Position", row=1, col=1)
    fig.update_yaxes(title_text="Net Position", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def display_cycle_composite(df, instrument_name):
    """Tab 7: Cycle Composite - Putting it all together
    Synthesis view with cycle phase indicator
    """
    st.markdown("### 🔄 Cycle Composite - Complete Market Cycle Analysis")
    st.markdown("Multi-factor synthesis identifying cycle phase and historical analogues")

    # Calculate all indicators
    if 'net_noncomm_positions' not in df.columns:
        df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']

    # Calculate cycle indicators
    lookback = min(52, len(df))
    df['percentile_rank'] = df['net_noncomm_positions'].rolling(window=lookback, min_periods=20).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 0 else 50
    )

    df['z_score'] = (df['net_noncomm_positions'] -
                     df['net_noncomm_positions'].rolling(window=lookback, min_periods=20).mean()) / \
                    df['net_noncomm_positions'].rolling(window=lookback, min_periods=20).std()

    df['momentum_4w'] = df['net_noncomm_positions'].pct_change(4) * 100
    df['momentum_13w'] = df['net_noncomm_positions'].pct_change(13) * 100

    # Determine cycle phase
    def determine_cycle_phase(row):
        if pd.isna(row['percentile_rank']) or pd.isna(row['momentum_4w']):
            return "Insufficient Data"

        percentile = row['percentile_rank']
        momentum = row['momentum_4w']

        if percentile > 80:
            if momentum > 0:
                return "Distribution (Late Cycle)"
            else:
                return "Top Reversal"
        elif percentile < 20:
            if momentum < 0:
                return "Accumulation (Early Cycle)"
            else:
                return "Bottom Reversal"
        else:
            if abs(momentum) > 10:
                return "Trending" if momentum > 0 else "Correcting"
            else:
                return "Consolidation"

    df['cycle_phase'] = df.apply(determine_cycle_phase, axis=1)

    # Current status
    if len(df) > 0:
        current_phase = df['cycle_phase'].iloc[-1]
        current_percentile = df['percentile_rank'].iloc[-1]
        current_z_score = df['z_score'].iloc[-1]
        current_momentum = df['momentum_4w'].iloc[-1]

        # Display current cycle phase with color coding
        phase_colors = {
            "Distribution (Late Cycle)": "🔴",
            "Top Reversal": "🟠",
            "Accumulation (Early Cycle)": "🟢",
            "Bottom Reversal": "🟡",
            "Trending": "🔵",
            "Correcting": "🟣",
            "Consolidation": "⚪",
            "Insufficient Data": "⚫"
        }

        # Summary box
        st.info(f"""
        ### {phase_colors.get(current_phase, "⚪")} Current Cycle Phase: **{current_phase}**

        **Key Indicators:**
        - Percentile Rank: {current_percentile:.1f}%
        - Z-Score: {current_z_score:.2f}
        - 4-Week Momentum: {current_momentum:+.1f}%

        **Interpretation:**
        {get_cycle_interpretation(current_phase, current_percentile, current_momentum)}
        """)

    # Multi-factor scoring chart
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=[
            "Composite Cycle Score",
            "Individual Factor Contributions",
            "Historical Cycle Phases"
        ],
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Calculate composite score
    df['composite_score'] = (
        (df['percentile_rank'] - 50) / 50 * 0.4 +  # Percentile contribution
        df['z_score'].clip(-3, 3) / 3 * 0.3 +       # Z-score contribution
        df['momentum_4w'].clip(-50, 50) / 50 * 0.3  # Momentum contribution
    )

    # 1. Composite Score
    fig.add_trace(
        go.Scatter(
            x=df['report_date_as_yyyy_mm_dd'],
            y=df['composite_score'],
            name='Composite Score',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ),
        row=1, col=1
    )

    # Add threshold lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=1, col=1)

    # 2. Individual Factors
    factors = ['percentile_rank', 'z_score', 'momentum_4w']
    colors = ['purple', 'orange', 'green']

    for factor, color in zip(factors, colors):
        if factor == 'percentile_rank':
            normalized = (df[factor] - 50) / 50
        elif factor == 'z_score':
            normalized = df[factor].clip(-3, 3) / 3
        else:  # momentum
            normalized = df[factor].clip(-50, 50) / 50

        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=normalized,
                name=factor.replace('_', ' ').title(),
                line=dict(color=color, width=1.5)
            ),
            row=2, col=1
        )

    # 3. Cycle Phase Timeline
    # Create color mapping for phases
    phase_color_map = {
        "Distribution (Late Cycle)": "red",
        "Top Reversal": "orange",
        "Accumulation (Early Cycle)": "green",
        "Bottom Reversal": "yellow",
        "Trending": "blue",
        "Correcting": "purple",
        "Consolidation": "gray",
        "Insufficient Data": "black"
    }

    # Create numeric mapping for phases
    phase_numeric = {phase: i for i, phase in enumerate(phase_color_map.keys())}
    df['phase_numeric'] = df['cycle_phase'].map(phase_numeric)

    fig.add_trace(
        go.Scatter(
            x=df['report_date_as_yyyy_mm_dd'],
            y=df['phase_numeric'],
            mode='markers',
            name='Cycle Phase',
            marker=dict(
                color=[phase_color_map[phase] for phase in df['cycle_phase']],
                size=6
            ),
            text=df['cycle_phase'],
            hovertemplate='%{text}<extra></extra>'
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        title=f"{instrument_name} - Cycle Composite Analysis"
    )

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Composite Score", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Value", row=2, col=1)
    fig.update_yaxes(title_text="Phase", ticktext=list(phase_color_map.keys()),
                     tickvals=list(range(len(phase_color_map))), row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Historical Analogues Section
    st.markdown("---")
    st.markdown("### 📚 Historical Analogues")

    # Find similar historical setups
    if len(df) > 52:
        current_percentile = df['percentile_rank'].iloc[-1]
        current_momentum = df['momentum_4w'].iloc[-1]

        # Find similar conditions
        similar_conditions = df[
            (abs(df['percentile_rank'] - current_percentile) < 10) &
            (abs(df['momentum_4w'] - current_momentum) < 5) &
            (df.index < len(df) - 1)  # Exclude current
        ]

        if len(similar_conditions) > 0:
            st.write(f"Found {len(similar_conditions)} similar historical setups:")

            # Show what happened next
            outcomes = []
            for idx in similar_conditions.index:
                if idx + 13 < len(df):  # Look 13 weeks ahead
                    future_return = ((df['net_noncomm_positions'].iloc[idx + 13] -
                                    df['net_noncomm_positions'].iloc[idx]) /
                                   abs(df['net_noncomm_positions'].iloc[idx]) * 100)
                    outcomes.append({
                        'Date': df['report_date_as_yyyy_mm_dd'].iloc[idx],
                        'Percentile': df['percentile_rank'].iloc[idx],
                        'Momentum': df['momentum_4w'].iloc[idx],
                        '13W Forward Return': future_return
                    })

            if outcomes:
                outcomes_df = pd.DataFrame(outcomes)
                st.dataframe(outcomes_df.style.format({
                    'Percentile': '{:.1f}%',
                    'Momentum': '{:+.1f}%',
                    '13W Forward Return': '{:+.1f}%'
                }))

                avg_outcome = outcomes_df['13W Forward Return'].mean()
                st.write(f"**Average 13-week forward return in similar setups: {avg_outcome:+.1f}%**")


def get_cycle_interpretation(phase, percentile, momentum):
    """Provide interpretation for each cycle phase"""
    interpretations = {
        "Distribution (Late Cycle)": "Market positioning is extremely bullish and still building. This often precedes tops but can persist longer than expected.",
        "Top Reversal": "Bullish positioning is extreme and starting to reverse. Watch for confirmation of trend change.",
        "Accumulation (Early Cycle)": "Market positioning is extremely bearish and still building. This often precedes bottoms.",
        "Bottom Reversal": "Bearish positioning is extreme and starting to reverse. Potential opportunity developing.",
        "Trending": f"Market is in a strong {'uptrend' if momentum > 0 else 'downtrend'} with momentum. Trend following conditions.",
        "Correcting": "Market is pulling back within the larger trend. Monitor for continuation or reversal signals.",
        "Consolidation": "Market is range-bound with low momentum. Wait for breakout direction.",
        "Insufficient Data": "Not enough historical data to determine cycle phase."
    }

    return interpretations.get(phase, "Unable to determine current market conditions.")