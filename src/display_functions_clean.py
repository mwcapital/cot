"""
Clean implementation - Price and Exchange OI are ALWAYS shown as base layer
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_lightweight_charts import renderLightweightCharts
try:
    from futures_price_fetcher import FuturesPriceFetcher
except ImportError:
    from .futures_price_fetcher import FuturesPriceFetcher

def display_time_series_chart_lwc(df, instrument_name, custom_formulas=None):
    """Display charts with Price and Exchange OI ALWAYS visible as base layer"""

    st.header("📈 Time Series Analysis - Unified View")

    # Price adjustment selection
    st.markdown("**Price Adjustment Method**")
    price_adjustment = st.radio(
        "Select adjustment:",
        ["NON", "RAD", "REV"],
        index=1,
        label_visibility="collapsed",
        horizontal=True,
        help="NON = No adjustment | RAD = Ratio Adjusted | REV = Reverse Adjusted"
    )

    # FETCH PRICE DATA FIRST - THIS IS ALWAYS SHOWN
    fetcher = FuturesPriceFetcher()
    # Get the instrument name without the COT code
    # e.g. "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE (067411)" -> "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE"
    import re
    instrument_clean = re.sub(r'\s*\(\d+\)$', '', instrument_name).strip()

    # Find the futures symbol for this COT instrument
    symbol = None
    with open('/Users/makson/Desktop/COT-Analysis/instrument_management/futures_symbols_enhanced.json', 'r') as f:
        import json
        mapping = json.load(f)
        for fut_symbol, info in mapping['futures_symbols'].items():
            if info['cot_mapping']['matched']:
                if instrument_clean in info['cot_mapping']['instruments']:
                    symbol = fut_symbol
                    break

    price_df = pd.DataFrame()
    if symbol:
        start_date = df['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
        end_date = df['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')
        price_df = fetcher.fetch_weekly_prices(symbol, start_date, end_date, price_adjustment)

    # BASE CHART - ALWAYS SHOWN
    chart_series = []

    if not price_df.empty:
        # PRICE CANDLESTICKS - ALWAYS SHOWN
        price_data = []
        for _, row in price_df.iterrows():
            price_data.append({
                'time': row['date'].strftime('%Y-%m-%d'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })

        chart_series.append({
            'type': 'Candlestick',
            'data': price_data,
            'options': {
                'upColor': '#26a69a',
                'downColor': '#ef5350',
                'borderVisible': False,
                'wickUpColor': '#26a69a',
                'wickDownColor': '#ef5350',
                'priceScaleId': 'right',
                'title': f'{symbol} Price'
            }
        })

        # EXCHANGE OPEN INTEREST - ALWAYS SHOWN
        oi_data = []
        for _, row in price_df.iterrows():
            if pd.notna(row['open_interest']) and row['open_interest'] > 0:
                oi_data.append({
                    'time': row['date'].strftime('%Y-%m-%d'),
                    'value': float(row['open_interest']),
                    'color': 'rgba(128, 128, 128, 0.3)'
                })

        if oi_data:
            chart_series.append({
                'type': 'Histogram',
                'data': oi_data,
                'options': {
                    'priceFormat': {'type': 'volume'},
                    'priceScaleId': 'left',
                    'title': 'Exchange OI'
                }
            })

    # COT data selection - OPTIONAL OVERLAYS
    st.markdown("---")
    st.markdown("**Optional COT Overlays (select to add):**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Non-Commercial")
        show_nc_long = st.checkbox("Non-Commercial Long")
        show_nc_short = st.checkbox("Non-Commercial Short")
        show_nc_spreading = st.checkbox("Non-Commercial Spreading")
        show_nc_net = st.checkbox("Non-Commercial Net")

    with col2:
        st.markdown("##### Commercial")
        show_c_long = st.checkbox("Commercial Long")
        show_c_short = st.checkbox("Commercial Short")
        show_c_net = st.checkbox("Commercial Net")

    with col3:
        st.markdown("##### Other")
        show_nr_long = st.checkbox("Non-Reportable Long")
        show_nr_short = st.checkbox("Non-Reportable Short")
        show_nr_net = st.checkbox("Non-Reportable Net")
        show_oi_cot = st.checkbox("Open Interest (COT)")

    # Custom formulas section
    st.markdown("**Custom Formula Overlays**")
    cols = st.columns(4)
    with cols[0]:
        show_nc_net_oi = st.checkbox("NC_NET/OI×100")
    with cols[1]:
        show_c_net_oi = st.checkbox("C_NET/OI×100")
    with cols[2]:
        show_nc_ratio = st.checkbox("NC Long/Short Ratio")
    with cols[3]:
        show_nc_c_ratio = st.checkbox("NC_NET/C_NET")

    # Calculate net positions if needed
    if show_nc_net or show_c_net or show_nr_net or show_nc_net_oi or show_c_net_oi or show_nc_c_ratio:
        df['net_commercial'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
        df['net_noncommercial'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
        df['net_nonreportable'] = df['nonrept_positions_long_all'] - df['nonrept_positions_short_all']

    # Calculate custom formulas if needed
    if show_nc_net_oi or show_c_net_oi:
        if 'open_interest_all' in df.columns:
            df['nc_net_oi'] = (df['net_noncommercial'] / df['open_interest_all']) * 100
            df['c_net_oi'] = (df['net_commercial'] / df['open_interest_all']) * 100

    if show_nc_ratio:
        df['nc_ratio'] = df['noncomm_positions_long_all'] / df['noncomm_positions_short_all'].replace(0, np.nan)

    if show_nc_c_ratio:
        df['nc_c_ratio'] = df['net_noncommercial'] / df['net_commercial'].replace(0, np.nan)

    # Add selected COT overlays (using separate price scale)
    cot_configs = [
        (show_nc_long, 'noncomm_positions_long_all', 'NC Long', '#00FF00'),
        (show_nc_short, 'noncomm_positions_short_all', 'NC Short', '#FF0000'),
        (show_nc_spreading, 'noncomm_positions_spread_all', 'NC Spread', '#DDA0DD'),
        (show_nc_net, 'net_noncommercial', 'NC Net', '#00FFFF'),
        (show_c_long, 'comm_positions_long_all', 'C Long', '#FF00FF'),
        (show_c_short, 'comm_positions_short_all', 'C Short', '#FF8C00'),
        (show_c_net, 'net_commercial', 'C Net', '#FFD700'),
        (show_nr_long, 'nonrept_positions_long_all', 'NR Long', '#87CEEB'),
        (show_nr_short, 'nonrept_positions_short_all', 'NR Short', '#FA8072'),
        (show_nr_net, 'net_nonreportable', 'NR Net', '#B0C4DE'),
        (show_oi_cot, 'open_interest_all', 'COT OI', '#C0C0C0'),
        (show_nc_net_oi, 'nc_net_oi', 'NC/OI%', '#FF69B4'),
        (show_c_net_oi, 'c_net_oi', 'C/OI%', '#DDA0DD'),
        (show_nc_ratio, 'nc_ratio', 'NC L/S', '#DA70D6'),
        (show_nc_c_ratio, 'nc_c_ratio', 'NC/C', '#FF1493')
    ]

    # Add COT overlays on a third scale
    for show, col, title, color in cot_configs:
        if show and col in df.columns:
            data = []
            for _, row in df.iterrows():
                if pd.notna(row[col]) and not (isinstance(row[col], float) and np.isinf(row[col])):
                    data.append({
                        'time': row['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'),
                        'value': float(row[col])
                    })

            if data:
                chart_series.append({
                    'type': 'Line',
                    'data': data,
                    'options': {
                        'color': color,
                        'lineWidth': 2,
                        'priceScaleId': 'overlay',
                        'title': title
                    }
                })

    # Render the chart with all series
    if chart_series:
        # Debug: Print what we're trying to render
        st.write(f"Debug: Found {len(chart_series)} series to render")
        st.write(f"Debug: Symbol = {symbol}")

        chart_config = {
            'chart': {
                'height': 600,
                'layout': {
                    'background': {'color': 'transparent'},
                    'textColor': '#DDD'
                },
                'grid': {
                    'vertLines': {'color': '#444'},
                    'horzLines': {'color': '#444'}
                },
                'crosshair': {
                    'mode': 0
                },
                'rightPriceScale': {
                    'borderColor': '#71649C',
                    'visible': True
                },
                'leftPriceScale': {
                    'borderColor': '#71649C',
                    'visible': True
                },
                'timeScale': {
                    'borderColor': '#71649C',
                    'timeVisible': True
                }
            },
            'series': chart_series
        }

        # Use a simpler key for the chart
        import hashlib
        chart_key = hashlib.md5(instrument_name.encode()).hexdigest()[:10]

        renderLightweightCharts(
            chart_config,
            f'chart_{chart_key}'
        )
    else:
        st.warning("No price data available for this instrument")