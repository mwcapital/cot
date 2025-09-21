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
                if st.checkbox(display_name, value=(col_name == "open_interest_all"), key=f"ts_{col_name}"):
                    selected_columns.append((col_name, display_name))
        elif i < 2 * items_per_column:
            with col2:
                if st.checkbox(display_name, key=f"ts_{col_name}"):
                    selected_columns.append((col_name, display_name))
        else:
            with col3:
                if st.checkbox(display_name, key=f"ts_{col_name}"):
                    selected_columns.append((col_name, display_name))

    # Add separator before formula options
    st.markdown("---")

    # Add third category - Custom Formulas
    st.markdown("#### Select custom formulas to plot:")

    col1_formula, col2_formula = st.columns(2)
    selected_formulas = []

    with col1_formula:
        if st.checkbox("NC_NET / OI × 100", key="formula_nc_oi"):
            selected_formulas.append(("nc_net_oi_ratio", "NC_NET/OI%"))
        if st.checkbox("NC_LONG / NC_SHORT", key="formula_nc_ratio"):
            selected_formulas.append(("nc_long_short_ratio", "NC L/S Ratio"))

    with col2_formula:
        if st.checkbox("C_NET / OI × 100", key="formula_c_oi"):
            selected_formulas.append(("c_net_oi_ratio", "C_NET/OI%"))
        if st.checkbox("NC_NET / C_NET", key="formula_nc_c"):
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
        },
        "timeScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "timeVisible": True,
            "secondsVisible": False,
        },
        "watermark": {
            "color": 'rgba(0, 0, 0, 0.1)',
            "visible": True,
            "text": f'{symbol} {price_adjustment}',
            "fontSize": 24,
            "horzAlign": 'center',
            "vertAlign": 'center',
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

        # Create COT subplot
        cotChart = {
            "height": 200,
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
                "scaleMargins": {
                    "top": 0.1,
                    "bottom": 0.1,
                },
            },
            "timeScale": {
                "borderColor": 'rgba(197, 203, 206, 0.8)',
                "timeVisible": False,
                "visible": True,
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
            charts.append({
                "chart": cotChart,
                "series": cotSeries
            })

    # Add third subplot for custom formulas if selected
    if selected_formulas:
        formulaData = {}

        for formula_col, display_name in selected_formulas:
            formulaData[formula_col] = []

            # Use COT data dates for formulas
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
            "height": 200,
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
                "timeVisible": False,
                "visible": True,
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
                    }
                })

        if formulaSeries:
            charts.append({
                "chart": formulaChart,
                "series": formulaSeries
            })

    # Render all charts with synchronization
    renderLightweightCharts(charts, 'synchronized_charts')

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

def display_share_of_oi(df, instrument_name):
    """Display Share of Open Interest analysis"""
    try:
        from charts.share_of_oi import create_share_of_oi_chart
        import plotly.graph_objects as go
        # The function requires a chart_title parameter
        chart_title = f"{instrument_name} - Share of Open Interest"
        fig = create_share_of_oi_chart(df, instrument_name, chart_title)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying Share of OI chart: {str(e)}")

def display_seasonality(df, instrument_name):
    """Display Seasonality analysis"""
    from charts.seasonality_charts import create_seasonality_chart
    import plotly.graph_objects as go
    fig = create_seasonality_chart(df, instrument_name)
    if fig:
        st.plotly_chart(fig, use_container_width=True)