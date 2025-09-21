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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_time_series_chart(df, instrument_name):
    """Display time series analysis with futures price/OI base layer"""
    st.subheader("📈 Time Series Analysis")

    # Then create tabs for COT analysis (without Futures Price tab)
    tab1, tab2, tab3 = st.tabs(["Standard Time Series", "Share of Open Interest", "Seasonality"])

    with tab1:
        display_cot_time_series(df, instrument_name)

    with tab2:
        display_share_of_oi(df, instrument_name)

    with tab3:
        display_seasonality(df, instrument_name)

def display_futures_base_layer(df, instrument_name, price_adjustment):
    """Display futures price and OI using lightweight charts"""

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
        return

    # Fetch futures price data
    fetcher = FuturesPriceFetcher()
    start_date = df['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
    end_date = df['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')

    price_df = fetcher.fetch_weekly_prices(symbol, start_date, end_date, price_adjustment)

    if price_df.empty:
        st.warning(f"No price data available for {symbol}")
        return

    # Prepare chart data
    priceData = []
    volumeData = []

    for _, row in price_df.iterrows():
        # Format date properly for lightweight charts
        date = row['date'].strftime('%Y-%m-%d')

        # Candlestick data
        priceData.append({
            'time': date,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })

        # Volume/OI data
        if pd.notna(row['open_interest']) and row['open_interest'] > 0:
            volumeData.append({
                'time': date,
                'value': float(row['open_interest']),
                'color': 'rgba(128, 128, 128, 0.3)'
            })

    # Configure chart options
    chartOptions = {
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
            "mode": 1  # Normal crosshair
        },
        "rightPriceScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "visible": True,
        },
        "leftPriceScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "visible": True,
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

    # Configure series
    seriesList = [
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

    # Add volume if we have data
    if volumeData:
        seriesList.append({
            "type": 'Histogram',
            "data": volumeData,
            "options": {
                "color": '#26a69a',
                "priceFormat": {
                    "type": 'volume',
                },
                "priceScaleId": "",  # Use overlay
                "scaleMargins": {
                    "top": 0.8,
                    "bottom": 0,
                }
            }
        })

    # Create chart dict
    renderLightweightCharts([{
        "chart": chartOptions,
        "series": seriesList
    }], 'futures')

def display_cot_time_series(df, instrument_name):
    """Display COT time series data with selectable columns"""

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
                    selected_columns.append(col_name)
        elif i < 2 * items_per_column:
            with col2:
                if st.checkbox(display_name, key=f"ts_{col_name}"):
                    selected_columns.append(col_name)
        else:
            with col3:
                if st.checkbox(display_name, key=f"ts_{col_name}"):
                    selected_columns.append(col_name)

    # Add separator and price adjustment selection
    st.markdown("---")
    st.markdown("#### Futures Price Chart")

    # Price adjustment selection
    st.markdown("**Price Adjustment Method**")
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        price_adjustment = st.radio(
            "Select adjustment:",
            ["NON", "RAD", "REV"],
            index=1,
            label_visibility="collapsed",
            horizontal=True,
            help="NON = No adjustment | RAD = Ratio Adjusted | REV = Reverse Adjusted",
            key="price_adj_main"
        )

    # Display futures prices chart
    display_futures_base_layer(df, instrument_name, price_adjustment)

    # Plot selected columns
    if selected_columns:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=[f"{instrument_name} - Time Series"],
            specs=[[{"secondary_y": True}]]
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for idx, col_name in enumerate(selected_columns):
            if col_name in df.columns:
                display_name = column_display_names.get(col_name, col_name)
                use_secondary = 'net_' in col_name  # Use secondary axis for net positions

                fig.add_trace(
                    go.Scatter(
                        x=df['report_date_as_yyyy_mm_dd'],
                        y=df[col_name],
                        mode='lines',
                        name=display_name,
                        line=dict(color=colors[idx % len(colors)], width=2)
                    ),
                    secondary_y=use_secondary
                )

        # Update layout
        fig.update_xaxes(title_text="Date", showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Contracts", secondary_y=False, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Net Positions", secondary_y=True, showgrid=False)

        fig.update_layout(
            height=600,
            template="plotly_dark",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Select at least one data series to display the chart")

def display_share_of_oi(df, instrument_name):
    """Display Share of Open Interest analysis"""
    from charts.share_of_oi import create_share_of_oi_chart
    fig = create_share_of_oi_chart(df, instrument_name)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

def display_seasonality(df, instrument_name):
    """Display Seasonality analysis"""
    from charts.seasonality_charts import create_seasonality_chart
    fig = create_seasonality_chart(df, instrument_name)
    if fig:
        st.plotly_chart(fig, use_container_width=True)