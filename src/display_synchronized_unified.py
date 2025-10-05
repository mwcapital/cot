"""
Unified chart implementation with multiple price scales for perfect alignment
"""
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_lightweight_charts import renderLightweightCharts
from futures_price_fetcher import FuturesPriceFetcher
from futures_price_viewer_lwc import get_category_from_futures_symbol, get_events_for_category

def display_synchronized_charts_unified(df, instrument_name, symbol, selected_columns, selected_formulas, price_adjustment='NON', show_events=False):
    """Display synchronized charts in a single container with multiple price scales"""

    if df.empty:
        st.warning("No data to display")
        return

    # Fetch futures price data
    fetcher = FuturesPriceFetcher()
    start_date = df['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
    end_date = df['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')

    price_df = fetcher.fetch_weekly_prices(symbol, start_date, end_date, price_adjustment)

    if price_df.empty:
        st.warning(f"No price data available for {symbol}")
        return

    # Get historical events if checkbox is checked
    events = None
    if show_events:
        category = get_category_from_futures_symbol(symbol)
        events = get_events_for_category(category)

    # Prepare price data
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

    # Prepare COT data aligned with price dates
    cotData = {}
    if selected_columns:
        for col_name, display_name in selected_columns:
            cotData[col_name] = []

            for _, price_row in price_df.iterrows():
                date = price_row['date']
                cot_row = df[df['report_date_as_yyyy_mm_dd'] <= date].iloc[-1] if not df[df['report_date_as_yyyy_mm_dd'] <= date].empty else None

                if cot_row is not None and pd.notna(cot_row[col_name]):
                    cotData[col_name].append({
                        'time': date.strftime('%Y-%m-%d'),
                        'value': float(cot_row[col_name])
                    })

    # Prepare formula data aligned with price dates
    formulaData = {}
    if selected_formulas:
        for formula_col, display_name in selected_formulas:
            formulaData[formula_col] = []

            for _, price_row in price_df.iterrows():
                date = price_row['date']
                cot_row = df[df['report_date_as_yyyy_mm_dd'] <= date].iloc[-1] if not df[df['report_date_as_yyyy_mm_dd'] <= date].empty else None

                if cot_row is not None and pd.notna(cot_row[formula_col]) and not np.isinf(cot_row[formula_col]):
                    formulaData[formula_col].append({
                        'time': date.strftime('%Y-%m-%d'),
                        'value': float(cot_row[formula_col])
                    })

    # Calculate total height based on what's being displayed
    total_height = 400  # Base height for price
    if selected_columns:
        total_height += 250  # Additional for COT
    if selected_formulas:
        total_height += 250  # Additional for formulas

    # Create single unified chart configuration
    chart = {
        "height": total_height,
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
            "mode": 0,  # Normal mode for price
        },
        "leftPriceScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "visible": bool(selected_columns),  # Show if COT data exists
            "mode": 0,
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

    # Build series array
    series = []

    # Add candlestick data (default right scale) with optional markers
    price_series = {
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

    # Add markers if events are available
    if markers:
        price_series["markers"] = markers

    series.append(price_series)

    # Add volume as histogram on same scale
    if volumeData:
        series.append({
            "type": 'Histogram',
            "data": volumeData,
            "options": {
                "color": '#26a69a',
                "priceFormat": {
                    "type": 'volume',
                },
                "priceScaleId": "",  # Same scale as price
                "scaleMargins": {
                    "top": 0.8,
                    "bottom": 0,
                }
            }
        })

    # Define colors for COT series
    cot_colors = {
        "open_interest_all": "#FFD700",
        "noncomm_positions_long_all": "#00FF00",
        "noncomm_positions_short_all": "#FF0000",
        "comm_positions_long_all": "#FF00FF",
        "comm_positions_short_all": "#FF8C00",
        "net_noncomm_positions": "#00FFFF",
        "net_comm_positions": "#DDA0DD",
        "net_reportable_positions": "#87CEEB"
    }

    # Add COT data on left scale
    if selected_columns:
        for col_name, display_name in selected_columns:
            if cotData.get(col_name):
                series.append({
                    "type": 'Line',
                    "data": cotData[col_name],
                    "options": {
                        "color": cot_colors.get(col_name, '#000000'),
                        "lineWidth": 2,
                        "title": display_name,
                        "priceScaleId": 'left',  # Use left scale
                        "priceLineVisible": False,
                        "lastValueVisible": True,
                    }
                })

    # Define colors for formula series
    formula_colors = {
        "nc_net_oi_ratio": "#FF69B4",
        "c_net_oi_ratio": "#DDA0DD",
        "nc_long_short_ratio": "#DA70D6",
        "nc_c_net_ratio": "#FF1493"
    }

    # Add formula data on overlay scale
    if selected_formulas:
        for formula_col, display_name in selected_formulas:
            if formulaData.get(formula_col):
                series.append({
                    "type": 'Line',
                    "data": formulaData[formula_col],
                    "options": {
                        "color": formula_colors.get(formula_col, '#000000'),
                        "lineWidth": 2,
                        "title": display_name,
                        "priceScaleId": 'formula',  # Create new scale
                        "priceLineVisible": False,
                        "lastValueVisible": True,
                        "scaleMargins": {
                            "top": 0.7,  # Push to bottom area
                            "bottom": 0,
                        }
                    }
                })

    # Render the unified chart
    renderLightweightCharts([{
        "chart": chart,
        "series": series
    }], 'unified_chart')