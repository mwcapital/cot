"""
Display functions for Disaggregated COT single-instrument time series analysis.
Adapted from display_functions_futures_first.py for 4 trader categories:
Producer/Merchant, Swap Dealers, Managed Money, Other Reportables.
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from streamlit_lightweight_charts import renderLightweightCharts
from futures_price_fetcher import FuturesPriceFetcher
from futures_price_viewer_lwc import get_category_from_futures_symbol, get_events_for_category


def display_disagg_time_series_chart(df, instrument_name):
    """Display time series analysis with tabs for disaggregated data"""
    st.subheader("Time Series Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Standard Time Series",
        "Seasonality",
        "Momentum & Percentile"
    ])

    with tab1:
        display_disagg_regime_detection(df, instrument_name)

    with tab2:
        _display_disagg_seasonality(df, instrument_name)

    with tab3:
        _display_disagg_momentum_percentile(df, instrument_name)


def display_disagg_regime_detection(df, instrument_name):
    """Market Regime Detection adapted for disaggregated 4-category data.

    Extremity score (0-100) = average of 3 components:
      (1) Concentration: max distance-from-50 of top-4 long/short concentration percentiles
      (2) Net Positioning: max distance-from-50 across all 4 category net percentiles
      (3) Flow Intensity: percentile rank of summed absolute weekly flows across all 4 categories

    Option A: max-of-4 for net positioning (slightly higher scores than Legacy max-of-2).
    Balanced Market threshold raised to 50 (from 40) to compensate.
    """
    st.markdown("#### Market Regime Detection")
    st.markdown(
        '<p style="color: #808080; font-size: 13px;">'
        'Market Extremity score (0-100) is the average of three components, each weighted by a third: '
        '<b>(1) Concentration</b> - how extreme is top-4 trader dominance vs past 52 weeks, '
        '<b>(2) Net Positioning</b> - how extreme are net positions across all 4 trader categories '
        '(MM, Prod/Merc, Swap, Other), '
        '<b>(3) Flow Intensity</b> - how much repositioning is happening across all categories. '
        'Higher score = further from normal market conditions.</p>',
        unsafe_allow_html=True
    )

    df_regime = df.copy()
    window = 52
    min_periods = 26

    # --- Step 1: Calculate all percentile metrics ---

    # Concentration (same as Legacy)
    df_regime['long_conc_pct'] = (
        df_regime['conc_gross_le_4_tdr_long']
        .rolling(window, min_periods=min_periods).rank(pct=True) * 100
    )
    df_regime['short_conc_pct'] = (
        df_regime['conc_gross_le_4_tdr_short']
        .rolling(window, min_periods=min_periods).rank(pct=True) * 100
    )

    # Net positions for all 4 categories
    df_regime['mm_net'] = df_regime['m_money_positions_long_all'] - df_regime['m_money_positions_short_all']
    df_regime['pm_net'] = df_regime['prod_merc_positions_long'] - df_regime['prod_merc_positions_short']
    df_regime['swap_net'] = df_regime['swap_positions_long_all'] - df_regime['swap__positions_short_all']
    df_regime['other_net'] = df_regime['other_rept_positions_long'] - df_regime['other_rept_positions_short']

    df_regime['mm_net_pct'] = df_regime['mm_net'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
    df_regime['pm_net_pct'] = df_regime['pm_net'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
    df_regime['swap_net_pct'] = df_regime['swap_net'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
    df_regime['other_net_pct'] = df_regime['other_net'].rolling(window, min_periods=min_periods).rank(pct=True) * 100

    # Flow intensity: sum of absolute weekly changes across all 4 categories
    df_regime['mm_flow'] = df_regime['mm_net'].diff()
    df_regime['pm_flow'] = df_regime['pm_net'].diff()
    df_regime['swap_flow'] = df_regime['swap_net'].diff()
    df_regime['other_flow'] = df_regime['other_net'].diff()
    df_regime['flow_intensity'] = (
        abs(df_regime['mm_flow']) + abs(df_regime['pm_flow']) +
        abs(df_regime['swap_flow']) + abs(df_regime['other_flow'])
    )
    df_regime['flow_pct'] = (
        df_regime['flow_intensity']
        .rolling(window, min_periods=min_periods).rank(pct=True) * 100
    )

    # Trader participation
    df_regime['trader_total_pct'] = (
        df_regime['traders_tot_all']
        .rolling(window, min_periods=min_periods).rank(pct=True) * 100
    )

    # --- Step 2: Calculate regime extremity score ---
    def distance_from_center(pct):
        return abs(pct - 50) * 2

    df_regime['regime_extremity'] = df_regime.apply(lambda row:
        max(distance_from_center(row['long_conc_pct']),
            distance_from_center(row['short_conc_pct'])) / 3 +
        max(distance_from_center(row['mm_net_pct']),
            distance_from_center(row['pm_net_pct']),
            distance_from_center(row['swap_net_pct']),
            distance_from_center(row['other_net_pct'])) / 3 +
        row['flow_pct'] / 3
    , axis=1)

    # --- Step 3: Detect regime ---
    def detect_regime(row):
        EXTREME_HIGH = 80
        EXTREME_LOW = 20

        if pd.isna(row['long_conc_pct']):
            return "Insufficient Data", "gray"

        # 1. Bilateral concentration
        if row['long_conc_pct'] > EXTREME_HIGH and row['short_conc_pct'] > EXTREME_HIGH:
            return "Bilateral Concentration", "orange"

        # 2. Single-sided concentration
        if row['long_conc_pct'] > EXTREME_HIGH:
            return "Long Concentration Extreme", "red"
        if row['short_conc_pct'] > EXTREME_HIGH:
            return "Short Concentration Extreme", "red"

        # 3. Money Manager extremes (replaces Legacy "Speculative")
        if row['mm_net_pct'] > EXTREME_HIGH:
            return "MM Long Extreme", "red"
        if row['mm_net_pct'] < EXTREME_LOW:
            return "MM Short Extreme", "red"

        # 4. Prod/Merc extremes (replaces Legacy "Commercial")
        if row['pm_net_pct'] > EXTREME_HIGH:
            return "Prod/Merc Long Extreme", "red"
        if row['pm_net_pct'] < EXTREME_LOW:
            return "Prod/Merc Short Extreme", "red"

        # 5. Swap Dealer extremes
        if row['swap_net_pct'] > EXTREME_HIGH:
            return "Swap Long Extreme", "orange"
        if row['swap_net_pct'] < EXTREME_LOW:
            return "Swap Short Extreme", "orange"

        # 6. Other Reportables extremes
        if row['other_net_pct'] > EXTREME_HIGH:
            return "Other Rept Long Extreme", "orange"
        if row['other_net_pct'] < EXTREME_LOW:
            return "Other Rept Short Extreme", "orange"

        # 7. High flow volatility
        if row['flow_pct'] > EXTREME_HIGH:
            return "High Flow Volatility", "yellow"

        # 8. Balanced market (threshold raised to 50 for max-of-4 compensation)
        if row['regime_extremity'] < 50:
            return "Balanced Market", "green"

        # 9. Default
        return "Transitional", "gray"

    df_regime[['regime', 'regime_color']] = df_regime.apply(
        lambda row: pd.Series(detect_regime(row)), axis=1
    )

    # --- Visualization ---
    latest = df_regime.iloc[-1]
    prev_week = df_regime.iloc[-2] if len(df_regime) >= 2 else None

    col1, col2 = st.columns([2, 3])

    # Column 1: Gauge
    with col1:
        prev_value = prev_week['regime_extremity'] if prev_week is not None and pd.notna(prev_week['regime_extremity']) else None

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest['regime_extremity'],
            delta={
                'reference': prev_value,
                'relative': False,
                'valueformat': '.1f',
                'increasing': {'color': 'red'},
                'decreasing': {'color': 'green'},
            } if prev_value is not None else None,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Market Extremity"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': prev_value if prev_value is not None else 0
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Column 2: Radar chart (8 axes)
    with col2:
        categories = [
            'Long Conc', 'Short Conc',
            'MM Net', 'PM Net', 'Swap Net', 'Other Net',
            'Flow', 'Traders'
        ]
        values = [
            latest['long_conc_pct'],
            latest['short_conc_pct'],
            latest['mm_net_pct'],
            latest['pm_net_pct'],
            latest['swap_net_pct'],
            latest['other_net_pct'],
            latest['flow_pct'],
            latest['trader_total_pct']
        ]

        fig_spider = go.Figure()
        fig_spider.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current',
            line_color='blue'
        ))
        fig_spider.add_trace(go.Scatterpolar(
            r=[50] * 8,
            theta=categories,
            name='Normal (50th)',
            line=dict(color='gray', dash='dash')
        ))
        fig_spider.add_trace(go.Scatterpolar(
            r=[85] * 8,
            theta=categories,
            name='Extreme (85th)',
            line=dict(color='red', dash='dot')
        ))
        fig_spider.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Percentile Rankings",
            height=300
        )
        st.plotly_chart(fig_spider, use_container_width=True)

    # Regime Color Legend
    st.markdown("### Regime Color Legend")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Red Regimes (High Risk)**")
        st.caption("Long Concentration Extreme")
        st.caption("Short Concentration Extreme")
        st.caption("MM Long Extreme")
        st.caption("MM Short Extreme")
        st.caption("Prod/Merc Long Extreme")
        st.caption("Prod/Merc Short Extreme")

    with col2:
        st.markdown("**Orange Regimes (Moderate Risk)**")
        st.caption("Bilateral Concentration")
        st.caption("Swap Long Extreme")
        st.caption("Swap Short Extreme")
        st.caption("Other Rept Long Extreme")
        st.caption("Other Rept Short Extreme")
        st.markdown("**Yellow Regimes**")
        st.caption("High Flow Volatility")

    with col3:
        st.markdown("**Green Regimes (Low Risk)**")
        st.caption("Balanced Market")
        st.markdown("**Gray Regimes**")
        st.caption("Transitional")
        st.caption("Insufficient Data")

    # 52-Week Regime Timeline
    st.markdown("### Regime History (52 Weeks)")

    fig_timeline = go.Figure()
    timeline_data = df_regime.tail(52).copy()

    regime_color_map = {
        'Long Concentration Extreme': 'darkred',
        'Short Concentration Extreme': 'darkred',
        'Bilateral Concentration': 'orange',
        'MM Long Extreme': 'red',
        'MM Short Extreme': 'red',
        'Prod/Merc Long Extreme': 'firebrick',
        'Prod/Merc Short Extreme': 'firebrick',
        'Swap Long Extreme': 'darkorange',
        'Swap Short Extreme': 'darkorange',
        'Other Rept Long Extreme': 'sandybrown',
        'Other Rept Short Extreme': 'sandybrown',
        'High Flow Volatility': 'gold',
        'Balanced Market': 'green',
        'Transitional': 'gray',
        'Insufficient Data': 'lightgray'
    }

    all_regimes = list(regime_color_map.keys())

    for regime in all_regimes:
        regime_mask = timeline_data['regime'] == regime
        if regime_mask.sum() > 0:
            fig_timeline.add_trace(go.Bar(
                x=timeline_data.loc[regime_mask, 'report_date_as_yyyy_mm_dd'],
                y=[1] * regime_mask.sum(),
                name=regime,
                marker_color=regime_color_map[regime],
                hovertemplate='%{x}<br>' + regime + '<extra></extra>',
                showlegend=True
            ))
        else:
            fig_timeline.add_trace(go.Bar(
                x=[], y=[],
                name=regime,
                marker_color=regime_color_map[regime],
                showlegend=True
            ))

    fig_timeline.update_layout(
        barmode='stack',
        showlegend=True,
        height=200,
        yaxis=dict(showticklabels=False, title=''),
        xaxis=dict(title='Date'),
        title='52-Week Regime Timeline'
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("---")

    # --- Price Adjustment & Data Series Selection ---
    _display_disagg_variable_charts(df, instrument_name)


def _display_disagg_variable_charts(df, instrument_name):
    """Price adjustment selector, data series checkboxes, custom formulas, and synchronized charts."""

    # Price adjustment selection
    st.markdown("**Price Adjustment Method**")
    price_adjustment = st.radio(
        "Select adjustment:",
        ["Non-Adjusted Method", "Ratio Adjusted Linking Method", "Reverse (Back) Adjusted Method"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
        key="disagg_price_adj_main"
    )
    adjustment_map = {
        "Non-Adjusted Method": "NON",
        "Ratio Adjusted Linking Method": "RAD",
        "Reverse (Back) Adjusted Method": "REV"
    }
    price_adjustment_code = adjustment_map[price_adjustment]

    st.markdown("---")

    # --- Data series checkboxes ---
    st.markdown("#### Select data series to plot:")

    column_display_names = {
        "open_interest_all": "Open Interest",
        "m_money_positions_long_all": "MM Long",
        "m_money_positions_short_all": "MM Short",
        "net_mm_positions": "Net MM",
        "prod_merc_positions_long": "Prod/Merc Long",
        "prod_merc_positions_short": "Prod/Merc Short",
        "net_pm_positions": "Net Prod/Merc",
        "swap_positions_long_all": "Swap Long",
        "swap__positions_short_all": "Swap Short",
        "net_swap_positions": "Net Swap",
        "other_rept_positions_long": "Other Rept Long",
        "other_rept_positions_short": "Other Rept Short",
        "net_other_positions": "Net Other Rept",
        "nonrept_positions_long_all": "Non-Reportable Long",
        "nonrept_positions_short_all": "Non-Reportable Short",
        "net_nonrept_positions": "Net Non-Reportable",
    }

    col1, col2, col3 = st.columns(3)
    selected_columns = []
    available_columns = list(column_display_names.keys())
    items_per_column = len(available_columns) // 3 + (1 if len(available_columns) % 3 else 0)

    # Ensure net positions are calculated
    if 'net_mm_positions' not in df.columns:
        df['net_mm_positions'] = df['m_money_positions_long_all'] - df['m_money_positions_short_all']
    if 'net_pm_positions' not in df.columns:
        df['net_pm_positions'] = df['prod_merc_positions_long'] - df['prod_merc_positions_short']
    if 'net_swap_positions' not in df.columns:
        df['net_swap_positions'] = df['swap_positions_long_all'] - df['swap__positions_short_all']
    if 'net_other_positions' not in df.columns:
        df['net_other_positions'] = df['other_rept_positions_long'] - df['other_rept_positions_short']
    if 'net_nonrept_positions' not in df.columns:
        df['net_nonrept_positions'] = df['nonrept_positions_long_all'] - df['nonrept_positions_short_all']

    for i, col_name in enumerate(available_columns):
        display_name = column_display_names[col_name]
        if i < items_per_column:
            target_col = col1
        elif i < 2 * items_per_column:
            target_col = col2
        else:
            target_col = col3
        with target_col:
            if st.checkbox(display_name, key=f"disagg_ts_{col_name}_{instrument_name}"):
                selected_columns.append((col_name, display_name))

    st.markdown("---")

    # --- Custom formulas ---
    st.markdown("#### Select custom formulas to plot:")

    col1_f, col2_f = st.columns(2)
    selected_formulas = []

    with col1_f:
        if st.checkbox("MM_NET / OI x 100", key=f"disagg_formula_mm_oi_{instrument_name}"):
            selected_formulas.append(("mm_net_oi_ratio", "MM_NET/OI%"))
        if st.checkbox("MM_LONG / MM_SHORT", key=f"disagg_formula_mm_ratio_{instrument_name}"):
            selected_formulas.append(("mm_long_short_ratio", "MM L/S Ratio"))

    with col2_f:
        if st.checkbox("PM_NET / OI x 100", key=f"disagg_formula_pm_oi_{instrument_name}"):
            selected_formulas.append(("pm_net_oi_ratio", "PM_NET/OI%"))
        if st.checkbox("MM_NET / PM_NET", key=f"disagg_formula_mm_pm_{instrument_name}"):
            selected_formulas.append(("mm_pm_net_ratio", "MM/PM NET"))

    # Calculate formula values
    if selected_formulas:
        for formula_col, _ in selected_formulas:
            if formula_col == "mm_net_oi_ratio":
                df[formula_col] = (df['net_mm_positions'] / df['open_interest_all']) * 100
            elif formula_col == "pm_net_oi_ratio":
                df[formula_col] = (df['net_pm_positions'] / df['open_interest_all']) * 100
            elif formula_col == "mm_long_short_ratio":
                df[formula_col] = df['m_money_positions_long_all'] / df['m_money_positions_short_all'].replace(0, np.nan)
            elif formula_col == "mm_pm_net_ratio":
                df[formula_col] = df['net_mm_positions'] / df['net_pm_positions'].replace(0, np.nan)

    st.markdown("---")

    # --- Display synchronized charts ---
    _display_disagg_synchronized_charts(df, instrument_name, price_adjustment_code, selected_columns, selected_formulas)

    # --- Market Concentration ---
    st.markdown("---")
    _display_disagg_market_concentration(df)


def _display_disagg_synchronized_charts(df, instrument_name, price_adjustment, selected_columns, selected_formulas=None):
    """Display price chart with synchronized COT data subplots for disaggregated data."""

    instrument_clean = re.sub(r'\s*\(\d+\)$', '', instrument_name).strip()

    # Find the futures symbol for this COT instrument
    symbol = None
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'instrument_management', 'futures', 'futures_symbols_enhanced.json'
    )
    try:
        with open(json_path, 'r') as f:
            mapping = json.load(f)
            for fut_symbol, info in mapping['futures_symbols'].items():
                if info['cot_mapping']['matched']:
                    if instrument_clean in info['cot_mapping']['instruments']:
                        symbol = fut_symbol
                        break
    except Exception:
        pass

    if not symbol:
        st.info("No futures price data available for this instrument")
        if selected_columns:
            _display_cot_only_charts(df, selected_columns)
        return

    st.markdown(f"### {instrument_name} - {symbol} ({price_adjustment})")

    # Chart display options
    col_opt1, col_opt2 = st.columns(2)
    with col_opt2:
        show_events = st.checkbox(
            "Show Historical Events",
            value=False,
            key=f"disagg_show_events_{symbol}",
            help="Display market-moving events relevant to this instrument category"
        )

    # Fetch futures price data
    fetcher = FuturesPriceFetcher()
    start_date = df['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
    end_date = df['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')

    price_df = fetcher.fetch_weekly_prices(symbol, start_date, end_date, price_adjustment)

    if price_df.empty:
        st.warning(f"No price data available for {symbol}")
        if selected_columns:
            _display_cot_only_charts(df, selected_columns)
        return

    # Get historical events
    events = None
    if show_events:
        category = get_category_from_futures_symbol(symbol)
        events = get_events_for_category(category)

    # --- Build charts ---
    charts = []

    # Chart 1: Price candlesticks with OI overlay
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

    if events:
        data_start = price_df['date'].min()
        data_end = price_df['date'].max()
        for event in events:
            event_date = pd.to_datetime(event['date'])
            if data_start <= event_date <= data_end:
                markers.append({
                    'time': event['date'],
                    'position': 'aboveBar',
                    'color': '#808080',
                    'shape': 'circle',
                    'text': event['description'],
                    'size': 0.5
                })

    chart_layout_base = {
        "background": {"type": 'solid', "color": 'white'},
        "textColor": '#333',
        "fontSize": 12,
    }
    grid_config = {
        "vertLines": {"color": 'rgba(197, 203, 206, 0.5)'},
        "horzLines": {"color": 'rgba(197, 203, 206, 0.5)'}
    }
    time_scale_config = {
        "borderColor": 'rgba(197, 203, 206, 0.8)',
        "timeVisible": True,
        "visible": True,
        "secondsVisible": False,
        "rightOffset": 5,
        "barSpacing": 3,
        "fixLeftEdge": True,
        "lockVisibleTimeRangeOnResize": True,
    }

    priceChart = {
        "height": 300,
        "layout": chart_layout_base,
        "grid": grid_config,
        "crosshair": {"mode": 1},
        "rightPriceScale": {
            "borderColor": 'rgba(197, 203, 206, 0.8)',
            "visible": True,
            "scaleMargins": {"top": 0.1, "bottom": 0.2},
            "width": 90,
        },
        "timeScale": time_scale_config,
        "watermark": {"visible": False}
    }

    price_series_config = {
        "type": 'Bar',
        "data": priceData,
        "options": {
            "upColor": 'rgb(38,166,154)',
            "downColor": 'rgb(255,82,82)',
            "wickUpColor": 'rgb(38,166,154)',
            "wickDownColor": 'rgb(255,82,82)',
            "borderVisible": False,
            "thinBars": False,
        }
    }
    if markers:
        price_series_config["markers"] = markers

    priceSeries = [price_series_config]
    if volumeData:
        priceSeries.append({
            "type": 'Histogram',
            "data": volumeData,
            "options": {
                "color": '#26a69a',
                "priceFormat": {"type": 'volume'},
                "priceScaleId": "",
                "scaleMargins": {"top": 0.8, "bottom": 0}
            }
        })

    charts.append({"chart": priceChart, "series": priceSeries})

    # --- COT data subplot ---
    colors = {
        "open_interest_all": "#FFD700",
        "m_money_positions_long_all": "#00CC00",
        "m_money_positions_short_all": "#FF0000",
        "net_mm_positions": "#00FFFF",
        "prod_merc_positions_long": "#FF00FF",
        "prod_merc_positions_short": "#FF8C00",
        "net_pm_positions": "#DDA0DD",
        "swap_positions_long_all": "#4169E1",
        "swap__positions_short_all": "#DC143C",
        "net_swap_positions": "#87CEEB",
        "other_rept_positions_long": "#32CD32",
        "other_rept_positions_short": "#B22222",
        "net_other_positions": "#DAA520",
        "nonrept_positions_long_all": "#808080",
        "nonrept_positions_short_all": "#A0A0A0",
        "net_nonrept_positions": "#C0C0C0",
    }

    if selected_columns:
        cotData = {}
        for col_name, display_name in selected_columns:
            cotData[col_name] = []
            for _, price_row in price_df.iterrows():
                date = price_row['date']
                cot_match = df[df['report_date_as_yyyy_mm_dd'] <= date]
                if not cot_match.empty:
                    cot_row = cot_match.iloc[-1]
                    if pd.notna(cot_row[col_name]):
                        cotData[col_name].append({
                            'time': date.strftime('%Y-%m-%d'),
                            'value': float(cot_row[col_name])
                        })

        is_last_chart = not selected_formulas

        cotChart = {
            "height": 350,
            "layout": chart_layout_base,
            "grid": grid_config,
            "crosshair": {"mode": 1},
            "rightPriceScale": {
                "borderColor": 'rgba(197, 203, 206, 0.8)',
                "visible": True,
                "scaleMargins": {"top": 0.1, "bottom": 0.1},
                "width": 90,
            },
            "timeScale": time_scale_config,
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
                        "priceFormat": {"type": 'price', "precision": 0, "minMove": 1}
                    }
                })

        if cotSeries:
            charts.append({"chart": cotChart, "series": cotSeries})

    # --- Formula subplot ---
    if selected_formulas:
        formulaData = {}
        for formula_col, display_name in selected_formulas:
            formulaData[formula_col] = []
            for _, price_row in price_df.iterrows():
                date = price_row['date']
                cot_match = df[df['report_date_as_yyyy_mm_dd'] <= date]
                if not cot_match.empty:
                    cot_row = cot_match.iloc[-1]
                    if formula_col in cot_row.index and pd.notna(cot_row[formula_col]):
                        formulaData[formula_col].append({
                            'time': date.strftime('%Y-%m-%d'),
                            'value': float(cot_row[formula_col])
                        })

        formula_colors = ['#E91E63', '#9C27B0', '#2196F3', '#FF9800']

        formulaChart = {
            "height": 250,
            "layout": chart_layout_base,
            "grid": grid_config,
            "crosshair": {"mode": 1},
            "rightPriceScale": {
                "borderColor": 'rgba(197, 203, 206, 0.8)',
                "visible": True,
                "scaleMargins": {"top": 0.1, "bottom": 0.1},
                "width": 90,
            },
            "timeScale": time_scale_config,
        }

        formulaSeries = []
        for idx, (formula_col, display_name) in enumerate(selected_formulas):
            if formulaData[formula_col]:
                formulaSeries.append({
                    "type": 'Line',
                    "data": formulaData[formula_col],
                    "options": {
                        "color": formula_colors[idx % len(formula_colors)],
                        "lineWidth": 2,
                        "title": display_name,
                        "priceLineVisible": False,
                        "lastValueVisible": True,
                        "priceFormat": {"type": 'price', "precision": 2, "minMove": 0.01}
                    }
                })

        if formulaSeries:
            charts.append({"chart": formulaChart, "series": formulaSeries})

    # Render all charts
    if charts:
        renderLightweightCharts(charts, 'disagg_time_series')


def _display_cot_only_charts(df, selected_columns):
    """Fallback: display COT data without price overlay when no futures symbol is matched."""
    fig = go.Figure()
    for col_name, display_name in selected_columns:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df[col_name],
                mode='lines',
                name=display_name
            ))
    fig.update_layout(height=500, xaxis_title='Date', yaxis_title='Contracts')
    st.plotly_chart(fig, use_container_width=True)


def _display_disagg_market_concentration(df):
    """Market Concentration cards for all 8 disaggregated category legs.

    8 cards in 2 rows of 4:
      Row 1: MM Long, MM Short, Prod/Merc Long, Prod/Merc Short
      Row 2: Swap Long, Swap Short, Other Rept Long, Other Rept Short
    """
    st.markdown("#### Market Concentration")
    st.markdown(
        '<p style="color: #808080; font-size: 13px;">'
        'Trader Pctl = where current trader count ranks vs historical since 2010. '
        'Position Pctl = where average position size per trader ranks vs historical. '
        'Concentration level: High (red) = few traders with large positions '
        '(T% &le; 33 and P% &ge; 67), Low (green) = many traders with small positions '
        '(T% &ge; 67 and P% &le; 33). MoM deltas show month-over-month change in '
        'trader count and average position.</p>',
        unsafe_allow_html=True
    )

    # Category definitions: (display_name, position_col, trader_col)
    categories = [
        # Row 1
        ('MM Long', 'm_money_positions_long_all', 'traders_m_money_long_all'),
        ('MM Short', 'm_money_positions_short_all', 'traders_m_money_short_all'),
        ('Prod/Merc Long', 'prod_merc_positions_long', 'traders_prod_merc_long_all'),
        ('Prod/Merc Short', 'prod_merc_positions_short', 'traders_prod_merc_short_all'),
        # Row 2
        ('Swap Long', 'swap_positions_long_all', 'traders_swap_long_all'),
        ('Swap Short', 'swap__positions_short_all', 'traders_swap_short_all'),
        ('Other Rept Long', 'other_rept_positions_long', 'traders_other_rept_long_all'),
        ('Other Rept Short', 'other_rept_positions_short', 'traders_other_rept_short'),
    ]

    # MoM comparison
    lookback_days = 30
    latest_date = df['report_date_as_yyyy_mm_dd'].max()
    previous_date = latest_date - pd.Timedelta(days=lookback_days)

    df_temp = df.copy()
    df_temp['date_diff_prev'] = abs(df_temp['report_date_as_yyyy_mm_dd'] - previous_date)
    prev_idx = df_temp['date_diff_prev'].idxmin()
    prev_data = df_temp.loc[prev_idx]
    current_data = df_temp[df_temp['report_date_as_yyyy_mm_dd'] == latest_date].iloc[0]

    # Historical data for percentiles (since 2010)
    lookback_start = pd.Timestamp('2010-01-01')
    hist_data_full = df[df['report_date_as_yyyy_mm_dd'] >= lookback_start].copy()

    def get_concentration_level(trader_pct, pos_pct):
        if trader_pct <= 33 and pos_pct >= 67:
            return "High"
        elif trader_pct >= 67 and pos_pct <= 33:
            return "Low"
        elif trader_pct <= 33 or pos_pct >= 67:
            return "Medium-High"
        elif trader_pct >= 67 or pos_pct <= 33:
            return "Medium-Low"
        return "Medium"

    concentration_border = {
        'High': '#DC143C', 'Medium-High': '#FF8C00', 'Medium': '#FFD700',
        'Medium-Low': '#9ACD32', 'Low': '#32CD32'
    }
    concentration_bg = {
        'High': 'rgba(220, 20, 60, 0.15)', 'Medium-High': 'rgba(255, 140, 0, 0.12)',
        'Medium': 'rgba(255, 215, 0, 0.12)', 'Medium-Low': 'rgba(154, 205, 50, 0.12)',
        'Low': 'rgba(50, 205, 50, 0.12)'
    }

    def render_card(cat_name, pos_col, trader_col, container):
        """Render a single concentration card."""
        # Check columns exist
        if pos_col not in df.columns or trader_col not in df.columns:
            with container:
                st.caption(f"{cat_name}: data not available")
            return

        prev_traders = float(prev_data[trader_col]) if pd.notna(prev_data[trader_col]) else 0
        prev_avg = float(prev_data[pos_col]) / prev_traders if prev_traders > 0 else 0
        curr_traders = float(current_data[trader_col]) if pd.notna(current_data[trader_col]) else 0
        curr_avg = float(current_data[pos_col]) / curr_traders if curr_traders > 0 else 0

        # Historical percentiles
        h = hist_data_full.copy()
        h['avg_pos'] = h[pos_col] / h[trader_col]
        h = h[h[trader_col] > 0]

        if len(h) < 10:
            with container:
                st.caption(f"{cat_name}: insufficient history")
            return

        t_pct = stats.percentileofscore(h[trader_col], curr_traders)
        p_pct = stats.percentileofscore(h['avg_pos'], curr_avg)

        level = get_concentration_level(t_pct, p_pct)
        border = concentration_border[level]
        bg = concentration_bg[level]

        trader_delta = int(curr_traders - prev_traders)
        avg_delta = curr_avg - prev_avg
        t_sign = "+" if trader_delta > 0 else ""
        a_sign = "+" if avg_delta > 0 else ""

        with container:
            st.markdown(f"""<div style="background:{bg}; border-left:4px solid {border}; padding:12px; border-radius:4px;">
<div style="font-weight:bold; font-size:14px;">{cat_name}</div>
<div style="color:{border}; font-size:12px; font-weight:bold; margin-bottom:10px;">{level}</div>
<div><span style="font-size:11px; color:#808080;">Traders</span><br>
<span style="font-size:20px; font-weight:bold;">{int(curr_traders)}</span>
<span style="font-size:12px; color:#808080;"> ({t_sign}{trader_delta})</span></div>
<div style="margin-top:6px;"><span style="font-size:11px; color:#808080;">Avg Position</span><br>
<span style="font-size:20px; font-weight:bold;">{curr_avg:,.0f}</span>
<span style="font-size:12px; color:#808080;"> ({a_sign}{avg_delta:,.0f})</span></div>
<div style="margin-top:10px;">
<div style="display:flex; justify-content:space-between; font-size:11px; color:#808080; margin-bottom:2px;">
<span>Trader Pctl</span><span style="font-weight:bold; color:#333;">{t_pct:.0f}%</span></div>
<div style="background:#e0e0e0; border-radius:3px; height:8px; width:100%;">
<div style="background:#4169E1; border-radius:3px; height:8px; width:{t_pct:.0f}%;"></div></div>
<div style="display:flex; justify-content:space-between; font-size:11px; color:#808080; margin-top:6px; margin-bottom:2px;">
<span>Position Pctl</span><span style="font-weight:bold; color:#333;">{p_pct:.0f}%</span></div>
<div style="background:#e0e0e0; border-radius:3px; height:8px; width:100%;">
<div style="background:#FF8C00; border-radius:3px; height:8px; width:{p_pct:.0f}%;"></div></div>
</div>
</div>""", unsafe_allow_html=True)

    # Row 1: MM + Prod/Merc
    cols_row1 = st.columns(4)
    for i in range(4):
        cat_name, pos_col, trader_col = categories[i]
        render_card(cat_name, pos_col, trader_col, cols_row1[i])

    st.markdown("")  # spacer

    # Row 2: Swap + Other Rept
    cols_row2 = st.columns(4)
    for i in range(4):
        cat_name, pos_col, trader_col = categories[i + 4]
        render_card(cat_name, pos_col, trader_col, cols_row2[i])


def _display_disagg_seasonality(df, instrument_name):
    """Seasonality analysis for disaggregated data - reuses Legacy charting functions."""
    from display_functions_futures_first import create_seasonality_chart, create_simple_seasonal_overlay

    st.markdown("### Seasonality Analysis")

    if df is None or df.empty:
        st.info("Please fetch data first to view seasonality analysis.")
        return

    # Ensure net positions are calculated
    if 'net_mm_positions' not in df.columns:
        df['net_mm_positions'] = df['m_money_positions_long_all'] - df['m_money_positions_short_all']
    if 'net_pm_positions' not in df.columns:
        df['net_pm_positions'] = df['prod_merc_positions_long'] - df['prod_merc_positions_short']
    if 'net_swap_positions' not in df.columns:
        df['net_swap_positions'] = df['swap_positions_long_all'] - df['swap__positions_short_all']
    if 'net_other_positions' not in df.columns:
        df['net_other_positions'] = df['other_rept_positions_long'] - df['other_rept_positions_short']
    if 'net_nonrept_positions' not in df.columns:
        df['net_nonrept_positions'] = df['nonrept_positions_long_all'] - df['nonrept_positions_short_all']

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        # Disaggregated metric options
        column_display_names = {
            "net_mm_positions": "Net Positioning -> Net Money Managers",
            "net_pm_positions": "Net Positioning -> Net Prod/Merc",
            "net_swap_positions": "Net Positioning -> Net Swap Dealers",
            "net_other_positions": "Net Positioning -> Net Other Reportables",
            "open_interest_all": "Open Interest",
            "m_money_positions_long_all": "MM Long",
            "m_money_positions_short_all": "MM Short",
            "prod_merc_positions_long": "Prod/Merc Long",
            "prod_merc_positions_short": "Prod/Merc Short",
            "swap_positions_long_all": "Swap Long",
            "swap__positions_short_all": "Swap Short",
            "other_rept_positions_long": "Other Rept Long",
            "other_rept_positions_short": "Other Rept Short",
            "nonrept_positions_long_all": "Non-Reportable Long",
            "nonrept_positions_short_all": "Non-Reportable Short",
            "net_nonrept_positions": "Net Non-Reportable",
            "traders_tot_all": "Total Traders",
        }

        # Filter to columns actually in the data
        available = [(col, name) for col, name in column_display_names.items() if col in df.columns]
        display_options = [name for _, name in available]
        column_keys = [col for col, _ in available]

        # Default to Net MM
        default_idx = 0
        if 'net_mm_positions' in column_keys:
            default_idx = column_keys.index('net_mm_positions')

        selected_display = st.selectbox(
            "Select metric:",
            options=display_options,
            index=default_idx,
            key=f"disagg_seasonality_column_{instrument_name}",
        )

        selected_column = column_keys[display_options.index(selected_display)]

    with col2:
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
            key=f"disagg_seasonality_lookback_{instrument_name}",
        )
        lookback_years = lookback_options[lookback_label]

    with col3:
        zone_type = st.radio(
            "Zone type:",
            options=['Percentile', 'Std Dev'],
            index=0,
            key=f"disagg_seasonality_zone_{instrument_name}",
        )
        zone_type_value = 'percentile' if zone_type == 'Percentile' else 'std'

    show_previous_year = True
    if lookback_years == 'all' or (isinstance(lookback_years, int) and lookback_years >= 10):
        show_previous_year = st.checkbox(
            "Show previous year",
            value=True,
            key=f"disagg_seasonality_prev_year_{instrument_name}"
        )

    show_simple_average = False
    if lookback_years == 5:
        show_simple_average = st.checkbox(
            "Show as simple average (current year vs 5-year average only)",
            value=False,
            key=f"disagg_seasonality_simple_{instrument_name}"
        )

    # Create chart using Legacy charting functions
    if lookback_years == 5 and show_simple_average:
        fig = create_simple_seasonal_overlay(df, selected_column)
    else:
        fig = create_seasonality_chart(
            df, selected_column, lookback_years,
            show_previous_year, zone_type_value
        )

    if fig:
        st.plotly_chart(fig, use_container_width=True)


def _display_disagg_momentum_percentile(df, instrument_name):
    """Momentum & Percentile analysis for disaggregated data."""
    from charts.momentum_charts import create_momentum_analysis_charts
    from charts.percentile_charts import create_percentile_chart

    st.markdown("### Momentum & Percentile Analysis")
    st.markdown(
        '<p style="color: #808080; font-size: 13px;">'
        'Percentile ranks, distribution analysis, and z-scores of changes are '
        'calculated using a 2-year (104-week) rolling lookback period.</p>',
        unsafe_allow_html=True
    )

    # Ensure net positions exist
    if 'net_mm_positions' not in df.columns:
        df['net_mm_positions'] = df['m_money_positions_long_all'] - df['m_money_positions_short_all']
    if 'net_pm_positions' not in df.columns:
        df['net_pm_positions'] = df['prod_merc_positions_long'] - df['prod_merc_positions_short']
    if 'net_swap_positions' not in df.columns:
        df['net_swap_positions'] = df['swap_positions_long_all'] - df['swap__positions_short_all']
    if 'net_other_positions' not in df.columns:
        df['net_other_positions'] = df['other_rept_positions_long'] - df['other_rept_positions_short']
    if 'net_nonrept_positions' not in df.columns:
        df['net_nonrept_positions'] = df['nonrept_positions_long_all'] - df['nonrept_positions_short_all']

    variable_names = {
        "open_interest_all": "Open Interest",
        "m_money_positions_long_all": "MM Long",
        "m_money_positions_short_all": "MM Short",
        "prod_merc_positions_long": "Prod/Merc Long",
        "prod_merc_positions_short": "Prod/Merc Short",
        "swap_positions_long_all": "Swap Long",
        "swap__positions_short_all": "Swap Short",
        "other_rept_positions_long": "Other Rept Long",
        "other_rept_positions_short": "Other Rept Short",
        "nonrept_positions_long_all": "Non-Reportable Long",
        "nonrept_positions_short_all": "Non-Reportable Short",
    }

    available_vars = [var for var in variable_names.keys() if var in df.columns]

    default_idx = 0
    if 'open_interest_all' in available_vars:
        default_idx = available_vars.index('open_interest_all')

    selected_var = st.selectbox(
        "Select variable for analysis:",
        options=available_vars,
        format_func=lambda x: variable_names.get(x, x),
        index=default_idx,
        key=f"disagg_momentum_var_{instrument_name}"
    )

    if not selected_var:
        st.error("No variable selected")
        return

    # 1. Actual Values with range selector
    st.markdown("---")
    st.markdown("#### Actual Values")

    col_buttons = st.columns(5)

    if 'disagg_momentum_range' not in st.session_state:
        st.session_state.disagg_momentum_range = 'All'

    with col_buttons[0]:
        if st.button("1Y", key="disagg_range_1y", use_container_width=True,
                     type="primary" if st.session_state.disagg_momentum_range == '1Y' else "secondary"):
            st.session_state.disagg_momentum_range = '1Y'
            st.rerun()
    with col_buttons[1]:
        if st.button("2Y", key="disagg_range_2y", use_container_width=True,
                     type="primary" if st.session_state.disagg_momentum_range == '2Y' else "secondary"):
            st.session_state.disagg_momentum_range = '2Y'
            st.rerun()
    with col_buttons[2]:
        if st.button("5Y", key="disagg_range_5y", use_container_width=True,
                     type="primary" if st.session_state.disagg_momentum_range == '5Y' else "secondary"):
            st.session_state.disagg_momentum_range = '5Y'
            st.rerun()
    with col_buttons[3]:
        if st.button("10Y", key="disagg_range_10y", use_container_width=True,
                     type="primary" if st.session_state.disagg_momentum_range == '10Y' else "secondary"):
            st.session_state.disagg_momentum_range = '10Y'
            st.rerun()
    with col_buttons[4]:
        if st.button("All", key="disagg_range_all", use_container_width=True,
                     type="primary" if st.session_state.disagg_momentum_range == 'All' else "secondary"):
            st.session_state.disagg_momentum_range = 'All'
            st.rerun()

    # Filter data based on selected range
    df_sorted = df.copy().sort_values('report_date_as_yyyy_mm_dd')
    if df_sorted.empty:
        st.warning("No data available.")
        return

    df_sorted['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df_sorted['report_date_as_yyyy_mm_dd'])
    latest_date = df_sorted['report_date_as_yyyy_mm_dd'].max()

    range_map = {'1Y': 1, '2Y': 2, '5Y': 5, '10Y': 10}
    current_range = st.session_state.disagg_momentum_range
    if current_range in range_map:
        start_date = latest_date - pd.DateOffset(years=range_map[current_range])
        df_filtered = df_sorted[df_sorted['report_date_as_yyyy_mm_dd'] >= start_date].copy()
    else:
        df_filtered = df_sorted.copy()
        start_date = df_sorted['report_date_as_yyyy_mm_dd'].min()

    df_filtered = df_filtered.reset_index(drop=True)

    if selected_var not in df_filtered.columns:
        st.error(f"Variable '{selected_var}' not found in the data.")
        return

    # Clean data
    dates = df_filtered['report_date_as_yyyy_mm_dd'].tolist()
    values = df_filtered[selected_var].tolist()
    clean_dates = []
    clean_values = []
    for d, v in zip(dates, values):
        if pd.notna(v):
            clean_dates.append(d)
            clean_values.append(float(v))

    if not clean_values:
        st.error(f"No valid data for {selected_var} in the selected time range")
        return

    # Actual values chart
    fig_actual = go.Figure()
    fig_actual.add_trace(go.Scatter(
        x=clean_dates,
        y=clean_values,
        mode='lines',
        name=variable_names.get(selected_var, selected_var),
        line=dict(color='#1f77b4', width=3),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Value: %{y:,.0f}<extra></extra>',
    ))
    fig_actual.update_layout(
        height=350,
        showlegend=False,
        hovermode='x unified',
        margin=dict(t=20, l=80, r=80, b=50),
        xaxis=dict(title='Date', showgrid=True, gridcolor='rgba(200,200,200,0.3)', type='date'),
        yaxis=dict(title='Value', showgrid=True, gridcolor='rgba(200,200,200,0.3)',
                   autorange=True, zeroline=True, zerolinecolor='rgba(128,128,128,0.3)')
    )

    date_range_str = f"{clean_dates[0].strftime('%Y-%m-%d')} to {clean_dates[-1].strftime('%Y-%m-%d')}"
    st.caption(f"Showing: {current_range} | {len(clean_values)} data points | {date_range_str}")
    st.plotly_chart(fig_actual, use_container_width=True)

    # 2. Percentile Analysis
    st.markdown("#### Percentile Analysis")

    fig_time_series = create_percentile_chart(df, selected_var, 2, 'time_series')
    if fig_time_series:
        fig_time_series.update_xaxes(range=[start_date, latest_date])
        st.plotly_chart(fig_time_series, use_container_width=True, key=f'disagg_pctl_ts_{selected_var}')

    fig_distribution = create_percentile_chart(df, selected_var, 2, 'distribution')
    if fig_distribution:
        st.plotly_chart(fig_distribution, use_container_width=True, key=f'disagg_pctl_dist_{selected_var}')

    # 3. Momentum Analysis
    st.markdown("#### Momentum Analysis")
    fig_momentum = create_momentum_analysis_charts(df, selected_var)
    if fig_momentum:
        fig_momentum.update_xaxes(range=[start_date, latest_date])
        st.plotly_chart(fig_momentum, use_container_width=True, key=f'disagg_momentum_{selected_var}')
