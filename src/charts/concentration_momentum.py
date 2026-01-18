"""
Concentration Momentum Analysis for CFTC COT Dashboard
Complete implementation from legacyF.py
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import re
import json
import os

def create_concentration_momentum_analysis(df, instrument_name):
    """Create concentration momentum analysis dashboard"""

    # Time range selector buttons
    st.markdown("#### Select Time Range")
    col_buttons = st.columns(5)

    if 'conc_momentum_range' not in st.session_state:
        st.session_state.conc_momentum_range = '2Y'

    with col_buttons[0]:
        if st.button("1Y", key="conc_mom_range_1y", use_container_width=True,
                    type="primary" if st.session_state.conc_momentum_range == '1Y' else "secondary"):
            st.session_state.conc_momentum_range = '1Y'
            st.rerun()
    with col_buttons[1]:
        if st.button("2Y", key="conc_mom_range_2y", use_container_width=True,
                    type="primary" if st.session_state.conc_momentum_range == '2Y' else "secondary"):
            st.session_state.conc_momentum_range = '2Y'
            st.rerun()
    with col_buttons[2]:
        if st.button("5Y", key="conc_mom_range_5y", use_container_width=True,
                    type="primary" if st.session_state.conc_momentum_range == '5Y' else "secondary"):
            st.session_state.conc_momentum_range = '5Y'
            st.rerun()
    with col_buttons[3]:
        if st.button("10Y", key="conc_mom_range_10y", use_container_width=True,
                    type="primary" if st.session_state.conc_momentum_range == '10Y' else "secondary"):
            st.session_state.conc_momentum_range = '10Y'
            st.rerun()
    with col_buttons[4]:
        if st.button("All", key="conc_mom_range_all", use_container_width=True,
                    type="primary" if st.session_state.conc_momentum_range == 'All' else "secondary"):
            st.session_state.conc_momentum_range = 'All'
            st.rerun()

    st.markdown("---")

    # Fetch price data
    price_df = None
    try:
        from futures_price_fetcher import FuturesPriceFetcher

        instrument_clean = re.sub(r'\s*\(\d+\)$', '', instrument_name).strip()
        symbol = None
        json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instrument_management', 'futures', 'futures_symbols_enhanced.json')
        with open(json_path, 'r') as f:
            mapping = json.load(f)
            for fut_symbol, info in mapping['futures_symbols'].items():
                if info['cot_mapping']['matched']:
                    if instrument_clean in info['cot_mapping']['instruments']:
                        symbol = fut_symbol
                        break

        if symbol:
            fetcher = FuturesPriceFetcher()
            start_date = df['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
            end_date = df['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')
            price_df = fetcher.fetch_weekly_prices(symbol, start_date, end_date, 'NON')
    except Exception as e:
        pass  # Price data optional

    # Explanation
    with st.expander("📖 Understanding Concentration Momentum", expanded=False):
        st.markdown("""
        **What is Concentration Momentum?**

        Tracks the rate of change in market concentration to identify when control is shifting between many traders (democratic) and few traders (oligopolistic).

        **Key Concepts:**
        - **Concentration Level**: Current % of market controlled by top traders
        - **Momentum**: Rate of change in concentration (4-week z-score)
        
        **Concentration Types:**
        - **Gross**: Total long OR short positions (shows raw market power)
        - **Net**: Long minus short positions (shows directional commitment)
        - **Top 4**: Most concentrated measure (oligopoly indicator)
        - **Top 8**: Broader institutional control measure
        
        **Regime Types:**
        - **Bilateral Concentration Building**: Both sides concentrating (institutional battle forming)
        - **Market Democratizing**: Both sides distributing (more participants entering)
        - **Asymmetric Shift**: One side concentrating while other distributes (squeeze setup)
        - **Power Balance Shifting**: Large change in concentration spread
        - **Stable Structure**: Little change in concentration
        
        **Warning Signs:**
        - High absolute concentration (>40% for top 4, >60% for top 8)
        - Rapid concentration changes (high momentum z-scores)
        - Large imbalances between long/short concentration
        """)
    
    # Concentration type selector
    col1, col2 = st.columns([2, 3])
    
    with col1:
        concentration_type = st.selectbox(
            "Select Concentration Metric:",
            ["Top 4 Gross", "Top 4 Net", "Top 8 Gross", "Top 8 Net"],
            help="""
            • Gross: Total positions (shows market power)
            • Net: Long-short (shows directional bias)
            • Top 4: Most concentrated (oligopoly risk)
            • Top 8: Broader institutional view
            """
        )
    
    with col2:
        # Show comparison checkbox
        show_comparison = st.checkbox("Show all concentration metrics comparison", value=False)
    
    # Map selection to data columns
    concentration_map = {
        "Top 4 Gross": ("conc_gross_le_4_tdr_long", "conc_gross_le_4_tdr_short"),
        "Top 4 Net": ("conc_net_le_4_tdr_long_all", "conc_net_le_4_tdr_short_all"),
        "Top 8 Gross": ("conc_gross_le_8_tdr_long", "conc_gross_le_8_tdr_short"),
        "Top 8 Net": ("conc_net_le_8_tdr_long_all", "conc_net_le_8_tdr_short_all")
    }
    
    long_conc_col, short_conc_col = concentration_map[concentration_type]
    
    # Calculate momentum metrics
    df_momentum = df.copy()
    window = 52
    min_periods = 26
    
    # Core concentration metrics
    df_momentum['long_concentration'] = df_momentum[long_conc_col]
    df_momentum['short_concentration'] = df_momentum[short_conc_col]
    df_momentum['concentration_spread'] = abs(df_momentum['long_concentration'] - df_momentum['short_concentration'])
    df_momentum['max_concentration'] = df_momentum[['long_concentration', 'short_concentration']].max(axis=1)
    
    # Momentum calculations (4-week)
    df_momentum['long_momentum_4w'] = df_momentum['long_concentration'].diff(4)
    df_momentum['short_momentum_4w'] = df_momentum['short_concentration'].diff(4)
    df_momentum['spread_momentum'] = df_momentum['concentration_spread'].diff(4)
    
    # Calculate z-scores for momentum
    df_momentum['long_momentum_mean'] = df_momentum['long_momentum_4w'].rolling(window, min_periods=min_periods).mean()
    df_momentum['long_momentum_std'] = df_momentum['long_momentum_4w'].rolling(window, min_periods=min_periods).std()
    df_momentum['long_momentum_zscore'] = np.where(
        df_momentum['long_momentum_std'] > 0,
        (df_momentum['long_momentum_4w'] - df_momentum['long_momentum_mean']) / df_momentum['long_momentum_std'],
        0
    )
    
    df_momentum['short_momentum_mean'] = df_momentum['short_momentum_4w'].rolling(window, min_periods=min_periods).mean()
    df_momentum['short_momentum_std'] = df_momentum['short_momentum_4w'].rolling(window, min_periods=min_periods).std()
    df_momentum['short_momentum_zscore'] = np.where(
        df_momentum['short_momentum_std'] > 0,
        (df_momentum['short_momentum_4w'] - df_momentum['short_momentum_mean']) / df_momentum['short_momentum_std'],
        0
    )
    
    # Spread momentum z-score
    df_momentum['spread_momentum_mean'] = df_momentum['spread_momentum'].rolling(window, min_periods=min_periods).mean()
    df_momentum['spread_momentum_std'] = df_momentum['spread_momentum'].rolling(window, min_periods=min_periods).std()
    df_momentum['spread_momentum_zscore'] = np.where(
        df_momentum['spread_momentum_std'] > 0,
        (df_momentum['spread_momentum'] - df_momentum['spread_momentum_mean']) / df_momentum['spread_momentum_std'],
        0
    )
    
    
    # Regime detection
    def detect_momentum_regime(row):
        long_z = row['long_momentum_zscore']
        short_z = row['short_momentum_zscore']
        spread_z = row['spread_momentum_zscore']
        
        # Adjust thresholds based on concentration type
        if "Net" in concentration_type:
            high_threshold = 2.0
            low_threshold = -2.0
        else:
            high_threshold = 1.5
            low_threshold = -1.5
        
        if pd.isna(long_z) or pd.isna(short_z):
            return "Insufficient Data", "gray"
        
        if long_z > high_threshold and short_z > high_threshold:
            return "Bilateral Concentration Building", "red"
        elif long_z < low_threshold and short_z < low_threshold:
            return "Market Democratizing", "green"
        elif long_z > high_threshold and short_z < low_threshold:
            return "Long Concentration / Short Distribution", "orange"
        elif long_z < low_threshold and short_z > high_threshold:
            return "Short Concentration / Long Distribution", "orange"
        elif abs(spread_z) > 2.0:
            return "Power Balance Shifting", "yellow"
        elif max(abs(long_z), abs(short_z)) < 0.5:
            return "Stable Structure", "lightgreen"
        else:
            return "Transitional", "gray"
    
    df_momentum[['momentum_regime', 'regime_color']] = df_momentum.apply(
        lambda row: pd.Series(detect_momentum_regime(row)), axis=1
    )
    
    # Calculate regime duration
    current_regime = df_momentum['momentum_regime'].iloc[-1]
    regime_duration = 1
    for i in range(2, min(len(df_momentum), 20)):
        if df_momentum.iloc[-i]['momentum_regime'] == current_regime:
            regime_duration += 1
        else:
            break
    
    # Get latest values for display
    latest = df_momentum.iloc[-1]

    # Apply time range filter for visualization
    df_momentum['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df_momentum['report_date_as_yyyy_mm_dd'])
    latest_date = df_momentum['report_date_as_yyyy_mm_dd'].max()

    if st.session_state.conc_momentum_range == '1Y':
        start_date = latest_date - pd.DateOffset(years=1)
    elif st.session_state.conc_momentum_range == '2Y':
        start_date = latest_date - pd.DateOffset(years=2)
    elif st.session_state.conc_momentum_range == '5Y':
        start_date = latest_date - pd.DateOffset(years=5)
    elif st.session_state.conc_momentum_range == '10Y':
        start_date = latest_date - pd.DateOffset(years=10)
    else:  # All
        start_date = df_momentum['report_date_as_yyyy_mm_dd'].min()

    df_plot = df_momentum[df_momentum['report_date_as_yyyy_mm_dd'] >= start_date].copy()

    # Filter price data to match
    if price_df is not None and len(price_df) > 0:
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df_filtered = price_df[price_df['date'] >= start_date].copy()
    else:
        price_df_filtered = None

    has_price = price_df_filtered is not None and len(price_df_filtered) > 0

    # Create visualizations
    if show_comparison:
        # Comparison of all concentration types
        fig_compare = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Top 4 Gross', 'Top 4 Net', 'Top 8 Gross', 'Top 8 Net'],
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        for idx, (metric_name, (long_col, short_col)) in enumerate(concentration_map.items()):
            row = idx // 2 + 1
            col = idx % 2 + 1

            if long_col in df_plot.columns and short_col in df_plot.columns:
                fig_compare.add_trace(
                    go.Scatter(
                        x=df_plot['report_date_as_yyyy_mm_dd'],
                        y=df_plot[long_col],
                        name=f'Long',
                        line=dict(color='green', width=2),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )

                fig_compare.add_trace(
                    go.Scatter(
                        x=df_plot['report_date_as_yyyy_mm_dd'],
                        y=df_plot[short_col],
                        name=f'Short',
                        line=dict(color='red', width=2),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )

        fig_compare.update_layout(
            height=600,
            title="Concentration Metrics Comparison",
            showlegend=True,
            hovermode='x unified'
        )

        st.plotly_chart(fig_compare, use_container_width=True)
    
    # Main momentum visualization
    if has_price:
        fig_main = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.25, 0.30, 0.30, 0.15],
            subplot_titles=[
                'Price',
                f'{concentration_type} Concentration Levels',
                'Momentum Z-Scores (4-week change)',
                'Momentum Regime'
            ]
        )
        price_row, conc_row, mom_row, regime_row = 1, 2, 3, 4
    else:
        fig_main = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.40, 0.40, 0.20],
            subplot_titles=[
                f'{concentration_type} Concentration Levels',
                'Momentum Z-Scores (4-week change)',
                'Momentum Regime'
            ]
        )
        conc_row, mom_row, regime_row = 1, 2, 3

    # Plot price if available
    if has_price:
        fig_main.add_trace(
            go.Scatter(
                x=price_df_filtered['date'],
                y=price_df_filtered['close'],
                name='Price',
                line=dict(color='blue', width=1.5),
                showlegend=False
            ),
            row=price_row, col=1
        )

    # Concentration levels
    fig_main.add_trace(
        go.Scatter(
            x=df_plot['report_date_as_yyyy_mm_dd'],
            y=df_plot['long_concentration'],
            name='Long Concentration',
            line=dict(color='green', width=2)
        ),
        row=conc_row, col=1
    )

    fig_main.add_trace(
        go.Scatter(
            x=df_plot['report_date_as_yyyy_mm_dd'],
            y=df_plot['short_concentration'],
            name='Short Concentration',
            line=dict(color='red', width=2)
        ),
        row=conc_row, col=1
    )

    # Momentum z-scores
    fig_main.add_trace(
        go.Scatter(
            x=df_plot['report_date_as_yyyy_mm_dd'],
            y=df_plot['long_momentum_zscore'],
            name='Long Momentum Z',
            line=dict(color='green', width=2)
        ),
        row=mom_row, col=1
    )

    fig_main.add_trace(
        go.Scatter(
            x=df_plot['report_date_as_yyyy_mm_dd'],
            y=df_plot['short_momentum_zscore'],
            name='Short Momentum Z',
            line=dict(color='red', width=2)
        ),
        row=mom_row, col=1
    )

    # Add threshold lines
    fig_main.add_hline(y=1.5, row=mom_row, col=1, line_dash="dash", line_color="orange", opacity=0.5)
    fig_main.add_hline(y=-1.5, row=mom_row, col=1, line_dash="dash", line_color="orange", opacity=0.5)
    fig_main.add_hline(y=0, row=mom_row, col=1, line_dash="solid", line_color="gray", opacity=0.3)

    # Regime timeline
    for regime in df_plot['momentum_regime'].unique():
        regime_mask = df_plot['momentum_regime'] == regime
        color = df_plot.loc[regime_mask, 'regime_color'].iloc[0] if regime_mask.sum() > 0 else 'gray'

        fig_main.add_trace(
            go.Bar(
                x=df_plot.loc[regime_mask, 'report_date_as_yyyy_mm_dd'],
                y=[1] * regime_mask.sum(),
                name=regime,
                marker_color=color,
                showlegend=False,
                hovertemplate='%{x}<br>' + regime + '<extra></extra>'
            ),
            row=regime_row, col=1
        )

    # Update layout
    if has_price:
        fig_main.update_yaxes(title_text="Price", row=price_row, col=1)
    fig_main.update_yaxes(title_text="Concentration %", row=conc_row, col=1)
    fig_main.update_yaxes(title_text="Z-Score", row=mom_row, col=1)
    fig_main.update_yaxes(showticklabels=False, row=regime_row, col=1)
    fig_main.update_xaxes(title_text="Date", row=regime_row, col=1)

    fig_main.update_layout(
        height=800 if has_price else 650,
        title=f"Concentration Momentum Analysis - {instrument_name}",
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Current metrics dashboard
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Regime", current_regime)
    with col2:
        st.metric("Duration", f"{regime_duration} weeks")
    with col3:
        st.metric("Long Conc.", f"{latest['long_concentration']:.1f}%",
                 f"{latest['long_momentum_4w']:+.1f}% (4w)")
    with col4:
        st.metric("Short Conc.", f"{latest['short_concentration']:.1f}%",
                 f"{latest['short_momentum_4w']:+.1f}% (4w)")
    with col5:
        st.metric("Spread", f"{latest['concentration_spread']:.1f}%",
                 f"{latest['spread_momentum']:+.1f}% (4w)")

    # Generate signals based on z-scores and regime
    signals = []

    if latest['long_momentum_zscore'] > 2:
        signals.append(("⚠️ Long Concentration Rising Fast", "Long side concentration momentum is unusually high"))

    if latest['short_momentum_zscore'] > 2:
        signals.append(("⚠️ Short Concentration Rising Fast", "Short side concentration momentum is unusually high"))

    if abs(latest['long_momentum_zscore'] - latest['short_momentum_zscore']) > 3:
        signals.append(("🔄 Extreme Asymmetry", "Large divergence in concentration momentum between long and short"))

    if regime_duration > 8 and "Building" in current_regime:
        signals.append(("⏰ Persistent Concentration", f"{current_regime} has persisted for {regime_duration} weeks"))

    if signals:
        st.markdown("### Current Signals")
        for signal, description in signals:
            st.warning(f"{signal}: {description}")