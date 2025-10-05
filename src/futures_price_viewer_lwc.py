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
import json
from futures_price_fetcher import FuturesPriceFetcher


def extract_cot_code(instrument_name):
    """Extract COT code from instrument name string"""
    # Pattern: "INSTRUMENT NAME - EXCHANGE (CODE)"
    match = re.search(r'\((\w+)\)', instrument_name)
    if match:
        return match.group(1)
    return None


@st.cache_data(ttl=3600)
def load_historical_events():
    """Load historical events from JSON file"""
    try:
        events_path = 'instrument_management/futures/events.json'
        with open(events_path, 'r') as f:
            events_data = json.load(f)
        return events_data
    except Exception as e:
        st.warning(f"Could not load historical events: {e}")
        return {}


def get_category_from_futures_symbol(futures_symbol):
    """Get category for a futures symbol from futures_symbols_enhanced.json"""
    try:
        mapping_path = 'instrument_management/futures/futures_symbols_enhanced.json'
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)

        symbol_data = mapping_data.get('futures_symbols', {}).get(futures_symbol, {})
        return symbol_data.get('category', 'All')
    except Exception:
        return 'All'


def get_events_for_category(category):
    """Get events for a specific category, including 'All' category events"""
    events_data = load_historical_events()

    # Get category-specific events
    category_events = events_data.get(category, [])

    # Get 'All' category events (cross-cutting macro events)
    all_events = events_data.get('All', [])

    # Combine both lists
    combined_events = category_events + all_events

    # Remove duplicates based on date and description
    seen = set()
    unique_events = []
    for event in combined_events:
        key = (event['date'], event['description'])
        if key not in seen:
            seen.add(key)
            unique_events.append(event)

    # Sort by date
    unique_events.sort(key=lambda x: x['date'])

    return unique_events


def create_lwc_chart(df, symbol_name="", events=None):
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

    # Prepare markers for historical events
    markers = []
    if events:
        # Create a date range from the data
        data_start = df['date'].min()
        data_end = df['date'].max()

        for event in events:
            event_date = pd.to_datetime(event['date'])
            # Only add markers for events within the data range
            if data_start <= event_date <= data_end:
                markers.append({
                    'time': event['date'],
                    'position': 'aboveBar',
                    'color': '#26a69a',  # Teal color for event markers (visible on black)
                    'shape': 'arrowDown',
                    'text': event['description'],
                    'size': 1
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

    # Add markers to price series if events exist
    if markers:
        price_series["markers"] = markers

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

    # Add checkbox for showing historical events
    show_events = st.checkbox(
        "Show Historical Events",
        value=False,
        key=f"show_events_{futures_symbol}",
        help="Display market-moving events relevant to this instrument category"
    )

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

    # Get historical events if checkbox is checked
    events = None
    if show_events:
        category = get_category_from_futures_symbol(futures_symbol)
        events = get_events_for_category(category)

    # Create and display lightweight chart with events
    create_lwc_chart(price_df, futures_symbol, events)

    return price_df