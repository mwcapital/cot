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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

def display_time_series_chart(df, instrument_name):
    """Display time series analysis with futures price/OI base layer"""
    st.subheader("📈 Time Series Analysis")

    # Then create tabs for COT analysis (without Futures Price tab)
    tab1, tab2, tab3 = st.tabs(["Standard Time Series", "Share of Open Interest", "Seasonality"])

    with tab1:
        display_cot_time_series_with_price(df, instrument_name)

    with tab2:
        display_share_of_oi(df, instrument_name)

    with tab3:
        display_seasonality(df, instrument_name)

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
    with open('/Users/makson/Desktop/COT-Analysis/instrument_management/futures_symbols_enhanced.json', 'r') as f:
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

    # Chart display option
    use_unified = st.checkbox("Use unified chart (single container)", value=False, key="use_unified_chart")

    if use_unified:
        # Call the new unified implementation
        return display_synchronized_charts_unified(df, instrument_name, symbol, selected_columns, selected_formulas, price_adjustment)

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

    # Prepare all charts data
    charts = []

    # Chart 1: Price candlesticks with volume
    priceData = []
    volumeData = []

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
            "barSpacing": 6,
            "fixLeftEdge": True,
            "lockVisibleTimeRangeOnResize": True,
        },
        "watermark": {
            "visible": False,  # Hide watermark since title is shown above
        }
    }

    priceSeries = [
        {
            "type": 'Bar',
            "data": priceData,
            "options": {
                "upColor": 'rgb(38,166,154)',
                "downColor": 'rgb(255,82,82)',
                "wickUpColor": 'rgb(38,166,154)',
                "wickDownColor": 'rgb(255,82,82)',
                "borderVisible": False,
            }
        }
    ]

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
                "barSpacing": 6,
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
    with open('/Users/makson/Desktop/COT-Analysis/instrument_management/futures_symbols_enhanced.json', 'r') as f:
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