"""
Synchronized multi-chart implementation using standard lightweight-charts
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
    """Display synchronized multi-pane charts with proper alignment"""

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

    # COT data selection - ORIGINAL LAYOUT
    st.markdown("**Select data series to plot:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Non-Commercial")
        show_nc_long = st.checkbox("Non-Commercial Long", value=True)
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
    st.markdown("**Custom Formula**")
    cols = st.columns(4)
    with cols[0]:
        show_nc_net_oi = st.checkbox("NC_NET/OI×100", value=True if custom_formulas else False)
    with cols[1]:
        show_c_net_oi = st.checkbox("C_NET/OI×100")
    with cols[2]:
        show_nc_ratio = st.checkbox("NC Long/Short Ratio")
    with cols[3]:
        show_nc_c_ratio = st.checkbox("NC_NET/C_NET")

    # Calculate net positions
    df['net_commercial'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
    df['net_noncommercial'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
    df['net_nonreportable'] = df['nonrept_positions_long_all'] - df['nonrept_positions_short_all']

    # Calculate custom formulas
    if 'open_interest_all' in df.columns:
        df['nc_net_oi'] = (df['net_noncommercial'] / df['open_interest_all']) * 100
        df['c_net_oi'] = (df['net_commercial'] / df['open_interest_all']) * 100

    df['nc_ratio'] = df['noncomm_positions_long_all'] / df['noncomm_positions_short_all'].replace(0, np.nan)
    df['nc_c_ratio'] = df['net_noncommercial'] / df['net_commercial'].replace(0, np.nan)

    # Fetch price data
    fetcher = FuturesPriceFetcher()
    cot_code = instrument_name.split(' - ')[0]
    symbol = fetcher.get_futures_symbol_for_cot(cot_code)

    if symbol:
        start_date = df['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
        end_date = df['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')
        price_df = fetcher.fetch_weekly_prices(symbol, start_date, end_date, price_adjustment)
    else:
        price_df = pd.DataFrame()

    # Prepare three separate charts
    charts_data = []

    # CHART 1: Price and Supabase OI
    price_series = []
    if not price_df.empty:
        # Price candlesticks
        price_data = []
        for _, row in price_df.iterrows():
            price_data.append({
                'time': row['date'].strftime('%Y-%m-%d'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })

        price_series.append({
            'type': 'Candlestick',
            'data': price_data,
            'options': {
                'upColor': '#26a69a',
                'downColor': '#ef5350',
                'borderVisible': False,
                'wickUpColor': '#26a69a',
                'wickDownColor': '#ef5350'
            },
            'priceScaleId': 'right'
        })

        # Supabase OI as histogram
        oi_data = []
        for _, row in price_df.iterrows():
            if pd.notna(row['open_interest']) and row['open_interest'] > 0:
                oi_data.append({
                    'time': row['date'].strftime('%Y-%m-%d'),
                    'value': float(row['open_interest']),
                    'color': 'rgba(128, 128, 128, 0.3)'
                })

        if oi_data:
            price_series.append({
                'type': 'Histogram',
                'data': oi_data,
                'options': {
                    'priceFormat': {
                        'type': 'volume'
                    }
                },
                'priceScaleId': 'left'
            })

    if price_series:
        charts_data.append({
            'chart': {
                'height': 300,
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
                    'borderColor': '#71649C'
                },
                'leftPriceScale': {
                    'visible': True,
                    'borderColor': '#71649C'
                },
                'timeScale': {
                    'borderColor': '#71649C',
                    'timeVisible': True
                }
            },
            'series': price_series
        })

    # CHART 2: COT Metrics
    cot_series = []

    series_config = [
        (show_nc_long, 'noncomm_positions_long_all', 'Noncomm Long', '#00FF00'),
        (show_nc_short, 'noncomm_positions_short_all', 'Noncomm Short', '#FF0000'),
        (show_nc_spreading, 'noncomm_positions_spread_all', 'NC Spreading', '#DDA0DD'),
        (show_nc_net, 'net_noncommercial', 'NC Net', '#00FFFF'),
        (show_c_long, 'comm_positions_long_all', 'Comm Long', '#FF00FF'),
        (show_c_short, 'comm_positions_short_all', 'Comm Short', '#FF8C00'),
        (show_c_net, 'net_commercial', 'Comm Net', '#FFD700'),
        (show_nr_long, 'nonrept_positions_long_all', 'NR Long', '#87CEEB'),
        (show_nr_short, 'nonrept_positions_short_all', 'NR Short', '#FA8072'),
        (show_nr_net, 'net_nonreportable', 'NR Net', '#B0C4DE'),
        (show_oi_cot, 'open_interest_all', 'Open Interest', '#C0C0C0')
    ]

    for show, col, title, color in series_config:
        if show and col in df.columns:
            data = []
            for _, row in df.iterrows():
                if pd.notna(row[col]):
                    data.append({
                        'time': row['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'),
                        'value': float(row[col])
                    })

            if data:
                cot_series.append({
                    'type': 'Line',
                    'data': data,
                    'options': {
                        'color': color,
                        'lineWidth': 2
                    }
                })

    if cot_series:
        charts_data.append({
            'chart': {
                'height': 250,
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
                    'borderColor': '#71649C'
                },
                'timeScale': {
                    'borderColor': '#71649C',
                    'visible': False  # Hide for middle chart
                }
            },
            'series': cot_series
        })

    # CHART 3: Custom Formulas
    formula_series = []

    formula_config = [
        (show_nc_net_oi, 'nc_net_oi', 'NC_NET/OI×100', '#FF69B4'),
        (show_c_net_oi, 'c_net_oi', 'C_NET/OI×100', '#DDA0DD'),
        (show_nc_ratio, 'nc_ratio', 'NC Long/Short', '#DA70D6'),
        (show_nc_c_ratio, 'nc_c_ratio', 'NC_NET/C_NET', '#FF1493')
    ]

    for show, col, title, color in formula_config:
        if show and col in df.columns:
            data = []
            for _, row in df.iterrows():
                if pd.notna(row[col]) and not np.isinf(row[col]):
                    data.append({
                        'time': row['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'),
                        'value': float(row[col])
                    })

            if data:
                formula_series.append({
                    'type': 'Line',
                    'data': data,
                    'options': {
                        'color': color,
                        'lineWidth': 2
                    }
                })

    if formula_series:
        charts_data.append({
            'chart': {
                'height': 200,
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
                    'borderColor': '#71649C'
                },
                'timeScale': {
                    'borderColor': '#71649C',
                    'timeVisible': True
                }
            },
            'series': formula_series
        })

    # Render all charts with synchronization
    if charts_data:
        # Add unique keys and sync options
        for i, chart_data in enumerate(charts_data):
            # Create container with specific styling
            with st.container():
                renderLightweightCharts(
                    chart_data,
                    f'chart_{i}_{instrument_name}'
                )

        # JavaScript for synchronization
        st.markdown("""
        <script>
        // Synchronize chart crosshairs and time scales
        document.addEventListener('DOMContentLoaded', function() {
            const charts = window.tvCharts || [];
            if (charts.length > 1) {
                let isSyncing = false;

                charts.forEach((chart, index) => {
                    // Sync crosshair position
                    chart.subscribeCrosshairMove((param) => {
                        if (!isSyncing) {
                            isSyncing = true;
                            charts.forEach((otherChart, otherIndex) => {
                                if (otherIndex !== index) {
                                    otherChart.setCrosshairPosition(param.point, param.time);
                                }
                            });
                            isSyncing = false;
                        }
                    });

                    // Sync visible range
                    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
                        if (!isSyncing) {
                            isSyncing = true;
                            charts.forEach((otherChart, otherIndex) => {
                                if (otherIndex !== index) {
                                    otherChart.timeScale().setVisibleLogicalRange(range);
                                }
                            });
                            isSyncing = false;
                        }
                    });
                });
            }
        });
        </script>
        """, unsafe_allow_html=True)