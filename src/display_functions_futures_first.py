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
            "type": 'Candlestick',
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
    This chart shows how open interest is distributed among different trader categories as a percentage of total.

    **Calculation Method:**
    - **Long Side:** NonComm Long % + Spread % + Comm Long % + NonRep Long % = 100%
    - **Short Side:** NonComm Short % + Spread % + Comm Short % + NonRep Short % = 100%

    The spread positions are counted on both long and short sides.
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

    # Calculate OI percentages
    df_oi = df.copy()

    if side_selection == "Long Positions":
        # Calculate long side percentages
        df_oi['noncomm_pct'] = (df['noncomm_positions_long_all'] / df['open_interest_all']) * 100
        df_oi['spread_pct'] = (df['noncomm_postions_spread_all'] / df['open_interest_all']) * 100
        df_oi['comm_pct'] = (df['comm_positions_long_all'] / df['open_interest_all']) * 100
        df_oi['nonrep_pct'] = (df['nonrep_positions_long_all'] / df['open_interest_all']) * 100

        title_suffix = "Long Positions"
        colors = {
            'noncomm': 'rgba(0, 255, 0, 0.8)',      # Green for non-commercial long
            'spread': 'rgba(255, 215, 0, 0.8)',      # Gold for spread
            'comm': 'rgba(0, 100, 255, 0.8)',        # Blue for commercial long
            'nonrep': 'rgba(128, 128, 128, 0.8)'     # Gray for non-reportable long
        }
    else:
        # Calculate short side percentages
        df_oi['noncomm_pct'] = (df['noncomm_positions_short_all'] / df['open_interest_all']) * 100
        df_oi['spread_pct'] = (df['noncomm_postions_spread_all'] / df['open_interest_all']) * 100
        df_oi['comm_pct'] = (df['comm_positions_short_all'] / df['open_interest_all']) * 100
        df_oi['nonrep_pct'] = (df['nonrep_positions_short_all'] / df['open_interest_all']) * 100

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
                "type": 'Candlestick',
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

    # OI distribution chart (stacked area)
    oiData = {
        'noncomm': [],
        'spread': [],
        'comm': [],
        'nonrep': []
    }

    # Align with price dates if available, otherwise use COT dates
    if not price_df.empty:
        for _, price_row in price_df.iterrows():
            date = price_row['date']
            # Find matching COT data
            cot_row = df_oi[df_oi['report_date_as_yyyy_mm_dd'] <= date].iloc[-1] if not df_oi[df_oi['report_date_as_yyyy_mm_dd'] <= date].empty else None

            if cot_row is not None:
                date_str = date.strftime('%Y-%m-%d')
                oiData['noncomm'].append({'time': date_str, 'value': float(cot_row['noncomm_pct']) if pd.notna(cot_row['noncomm_pct']) else 0})
                oiData['spread'].append({'time': date_str, 'value': float(cot_row['spread_pct']) if pd.notna(cot_row['spread_pct']) else 0})
                oiData['comm'].append({'time': date_str, 'value': float(cot_row['comm_pct']) if pd.notna(cot_row['comm_pct']) else 0})
                oiData['nonrep'].append({'time': date_str, 'value': float(cot_row['nonrep_pct']) if pd.notna(cot_row['nonrep_pct']) else 0})
    else:
        for _, row in df_oi.iterrows():
            date_str = row['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d')
            oiData['noncomm'].append({'time': date_str, 'value': float(row['noncomm_pct']) if pd.notna(row['noncomm_pct']) else 0})
            oiData['spread'].append({'time': date_str, 'value': float(row['spread_pct']) if pd.notna(row['spread_pct']) else 0})
            oiData['comm'].append({'time': date_str, 'value': float(row['comm_pct']) if pd.notna(row['comm_pct']) else 0})
            oiData['nonrep'].append({'time': date_str, 'value': float(row['nonrep_pct']) if pd.notna(row['nonrep_pct']) else 0})

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
            "text": f'OI Distribution - {title_suffix}',
            "fontSize": 24,
            "color": 'rgba(0, 0, 0, 0.1)',
            "horzAlign": 'center',
            "vertAlign": 'center',
        }
    }

    # Add OI series as stacked areas
    oiSeries = []

    # Add in order so they stack properly
    cumulative = []
    categories = [
        ('noncomm', 'Non-Commercial' if side_selection == "Long Positions" else 'Non-Commercial'),
        ('spread', 'Spread'),
        ('comm', 'Commercial' if side_selection == "Long Positions" else 'Commercial'),
        ('nonrep', 'Non-Reportable' if side_selection == "Long Positions" else 'Non-Reportable')
    ]

    for cat_key, cat_name in categories:
        if oiData[cat_key]:
            # Calculate cumulative values for stacking
            if not cumulative:
                cumulative = [{**d} for d in oiData[cat_key]]
            else:
                for i in range(len(cumulative)):
                    cumulative[i]['value'] += oiData[cat_key][i]['value']

            oiSeries.append({
                "type": 'Area',
                "data": [{**d} for d in cumulative],
                "options": {
                    "topColor": colors[cat_key],
                    "bottomColor": 'rgba(255, 255, 255, 0)',
                    "lineColor": colors[cat_key],
                    "lineWidth": 2,
                    "title": f"{cat_name} %",
                    "priceLineVisible": False,
                    "lastValueVisible": True,
                }
            })

    # Reverse the series so they display correctly
    oiSeries.reverse()

    charts.append({
        "chart": oiChart,
        "series": oiSeries
    })

    # Add 100% reference line
    if oiData['noncomm']:
        hundred_line = [{'time': point['time'], 'value': 100} for point in oiData['noncomm']]
        charts[-1]['series'].append({
            "type": 'Line',
            "data": hundred_line,
            "options": {
                "color": 'rgba(0, 0, 0, 0.3)',
                "lineWidth": 1,
                "lineStyle": 2,  # Dashed
                "priceLineVisible": False,
                "lastValueVisible": False,
                "title": "100%"
            }
        })

    # Render all charts with synchronization
    renderLightweightCharts(charts, 'share_of_oi_charts')


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


def display_seasonality(df, instrument_name):
    """Display Seasonality analysis"""
    from charts.seasonality_charts import create_seasonality_chart
    import plotly.graph_objects as go
    fig = create_seasonality_chart(df, instrument_name)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

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