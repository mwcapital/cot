"""
Futures Price Viewer with Lightweight Charts
Weekly OHLC bars visualization for historical futures data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
import re
from futures_price_fetcher import FuturesPriceFetcher


def extract_cot_code(instrument_name):
    """Extract COT code from instrument name string"""
    # Pattern: "INSTRUMENT NAME - EXCHANGE (CODE)"
    match = re.search(r'\((\w+)\)', instrument_name)
    if match:
        return match.group(1)
    return None


def create_lwc_chart(df, symbol_name=""):
    """Create a lightweight-charts with vintage Bloomberg terminal styling"""

    # Prepare OHLC and Open Interest data
    ohlc_data = []
    oi_data = []

    for _, row in df.iterrows():
        time_str = row['date'].strftime('%Y-%m-%d')
        ohlc_data.append({
            'time': time_str,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })
        oi_data.append({
            'time': time_str,
            'value': float(row['open_interest'])
        })

    # Use Bar series for OHLC bars (not candlesticks)
    price_series = {
        "type": 'Bar',
        "data": ohlc_data,
        "options": {
            "upColor": '#FFA500',      # Amber/orange like Bloomberg
            "downColor": '#FFA500',    # Same color for consistency
            "openVisible": True,       # Show open tick
            "thinBars": False          # Make bars visible
        }
    }

    # Open Interest as histogram at bottom
    oi_series = {
        "type": 'Histogram',
        "data": oi_data,
        "options": {
            "color": '#FFA500',
            "priceScaleId": 'volume',  # Use separate scale
            "priceFormat": {
                "type": 'volume'
            }
        },
        "priceScale": {
            "scaleMargins": {
                "top": 0.7,    # Push to bottom 30% of chart
                "bottom": 0
            }
        }
    }

    # Vintage Bloomberg terminal styling
    chart_options = {
        "layout": {
            "background": {
                "type": 'solid',
                "color": '#000000'     # Pure black like Bloomberg
            },
            "textColor": '#FFA500',    # Amber text
            "fontFamily": "'Courier New', monospace"
        },
        "grid": {
            "vertLines": {
                "color": '#1a1a1a',    # Very subtle grid
                "style": 1             # Dotted
            },
            "horzLines": {
                "color": '#1a1a1a',
                "style": 1
            }
        },
        "crosshair": {
            "mode": 1,                 # Magnet mode
            "vertLine": {
                "color": '#FFA500',
                "width": 1,
                "style": 2             # Dashed
            },
            "horzLine": {
                "color": '#FFA500',
                "width": 1,
                "style": 2
            }
        },
        "priceScale": {
            "borderColor": '#FFA500'
        },
        "timeScale": {
            "borderColor": '#FFA500',
            "timeVisible": True
        }
    }

    # Create a unique key for this chart
    chart_key = f'lwc_chart_{symbol_name}'

    # Create the chart configuration with both price and OI
    chart_config = {
        "chart": chart_options,
        "series": [price_series, oi_series]
    }

    # Render the chart
    renderLightweightCharts([chart_config], key=chart_key)


def display_futures_price_chart(instrument_name, cot_df=None):
    """
    Display futures price chart for a COT instrument using lightweight charts

    Args:
        instrument_name: Full COT instrument name with code
        cot_df: Optional COT data DataFrame (not used in this version)
    """

    # Check if instrument_name is valid
    if not instrument_name:
        return None

    # Extract COT code
    cot_code = extract_cot_code(instrument_name)
    if not cot_code:
        return None

    # Initialize fetcher
    fetcher = FuturesPriceFetcher()

    # Check if this COT instrument has a matching futures symbol
    futures_symbol = fetcher.get_futures_symbol_for_cot(cot_code)

    if not futures_symbol:
        return None

    # Get symbol info
    symbol_info = fetcher.futures_mapping['futures_symbols'].get(futures_symbol, {})

    # Default to Reverse (Back Adjusted)
    adjustment_method = "REV"

    # Use all available data
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
        return None

    # Create and display lightweight chart
    create_lwc_chart(price_df, futures_symbol)

    return price_df