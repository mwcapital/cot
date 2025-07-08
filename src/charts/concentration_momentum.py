"""
Concentration Momentum Analysis for CFTC COT Dashboard
Complete implementation from legacyF.py
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_concentration_momentum_analysis(df, instrument_name):
    """Create concentration momentum analysis dashboard"""
    
    # Explanation
    with st.expander("📖 Understanding Concentration Momentum", expanded=False):
        st.markdown("""
        **What is Concentration Momentum?**
        
        Tracks the rate of change (velocity) and acceleration of market concentration to identify when control is shifting between many traders (democratic) and few traders (oligopolistic).
        
        **Key Concepts:**
        - **Concentration Level**: Current % of market controlled by top traders
        - **Momentum (Velocity)**: Rate of change in concentration
        - **Acceleration**: Change in momentum (speeding up/slowing down)
        
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
        
        **Risk Factors:**
        1. High absolute concentration (>40% for top 4, >60% for top 8)
        2. Rapid concentration changes (high momentum)
        3. Accelerating concentration (momentum increasing)
        4. Large imbalances between long/short concentration
        
        **Composite Risk Score (0-100):**
        
        The risk score combines three components:
        
        1. **Concentration Risk (40% weight)**: How concentrated is the market?
           - Takes the WORST (max) of long/short concentration percentiles
           - 90th percentile concentration = 90% risk component
           - Captures: "Are we at dangerous concentration levels?"
        
        2. **Momentum Risk (40% weight)**: How fast is concentration changing?
           - Measures 4-week change in concentration percentiles
           - Extreme momentum in EITHER direction = high risk
           - 50th percentile (no unusual change) = 0% risk
           - 0th or 100th percentile (extreme change) = 100% risk
           - Captures: "Is market structure changing rapidly?"
        
        3. **Imbalance Risk (20% weight)**: How lopsided is the market?
           - Measures the SPREAD between long and short concentration
           - Large spread = one side dominates = squeeze potential
           - 80th percentile spread = 80% risk component
           - Captures: "Is there an asymmetric setup?"
        
        **Example:**
        - Long concentration: 42% (90th percentile) → Concentration risk uses 90%
        - Short concentration: 25% (45th percentile)
        - Spread: 17% (80th percentile) → Imbalance risk = 80%
        - Long momentum: +3.5% in 4 weeks (85th percentile) → Momentum risk = 70%
        - Composite = (0.90×0.4 + 0.70×0.4 + 0.80×0.2) × 100 = 80% HIGH RISK
        
        **Key Insight**: Concentration and Imbalance risks use the same values differently:
        - Concentration: MAX(long%, short%) → "Is either side extreme?"
        - Imbalance: |long% - short%| → "How different are they?"
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
    
    # Acceleration (1-week momentum change)
    df_momentum['long_momentum_1w'] = df_momentum['long_concentration'].diff(1)
    df_momentum['short_momentum_1w'] = df_momentum['short_concentration'].diff(1)
    df_momentum['long_acceleration'] = df_momentum['long_momentum_1w'].diff(1)
    df_momentum['short_acceleration'] = df_momentum['short_momentum_1w'].diff(1)
    
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
    
    # Risk calculations
    latest = df_momentum.iloc[-1]
    
    # Percentile-based risk metrics for concentration
    long_pct = df_momentum['long_concentration'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-1] * 100
    short_pct = df_momentum['short_concentration'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-1] * 100
    spread_pct = df_momentum['concentration_spread'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-1] * 100
    
    # NEW: Percentile-based momentum risk calculation
    # Calculate momentum percentiles directly (more adaptive to each asset)
    long_momentum_pct = df_momentum['long_momentum_4w'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-1]
    short_momentum_pct = df_momentum['short_momentum_4w'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-1]
    
    # For risk, we care about extremes in either direction
    # Maps: 0th percentile → 1.0 risk, 50th percentile → 0.0 risk, 100th percentile → 1.0 risk
    long_momentum_risk = 2 * abs(long_momentum_pct - 0.5)
    short_momentum_risk = 2 * abs(short_momentum_pct - 0.5)
    
    # Take the maximum momentum risk from either side
    momentum_risk = max(long_momentum_risk, short_momentum_risk)
    
    # Concentration risk remains the same (already percentile-based)
    concentration_risk = max(long_pct, short_pct) / 100
    
    # Imbalance risk remains the same (already percentile-based)
    imbalance_risk = spread_pct / 100
    
    # Composite risk calculation with same weights
    composite_risk = (concentration_risk * 0.4 + momentum_risk * 0.4 + imbalance_risk * 0.2) * 100
    
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
            
            if long_col in df.columns and short_col in df.columns:
                fig_compare.add_trace(
                    go.Scatter(
                        x=df['report_date_as_yyyy_mm_dd'],
                        y=df[long_col],
                        name=f'Long',
                        line=dict(color='green', width=2),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
                
                fig_compare.add_trace(
                    go.Scatter(
                        x=df['report_date_as_yyyy_mm_dd'],
                        y=df[short_col],
                        name=f'Short',
                        line=dict(color='red', width=2),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
                
                # Add title annotations
                fig_compare.add_annotation(
                    text=metric_name,
                    xref=f"x{idx+1} domain",
                    yref=f"y{idx+1} domain",
                    x=0.5, y=1.05,
                    showarrow=False,
                    font=dict(size=12, color='black')
                )
        
        fig_compare.update_layout(
            height=600,
            title="Concentration Metrics Comparison",
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # Main momentum visualization
    fig_main = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.35, 0.25, 0.25, 0.15],
        subplot_titles=[
            f'{concentration_type} Concentration Levels',
            'Momentum Z-Scores (4-week change)',
            'Acceleration (Rate of change)',
            'Momentum Regime'
        ]
    )
    
    # Plot 1: Concentration levels
    fig_main.add_trace(
        go.Scatter(
            x=df_momentum['report_date_as_yyyy_mm_dd'],
            y=df_momentum['long_concentration'],
            name='Long Concentration',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    fig_main.add_trace(
        go.Scatter(
            x=df_momentum['report_date_as_yyyy_mm_dd'],
            y=df_momentum['short_concentration'],
            name='Short Concentration',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Add spread as area
    fig_main.add_trace(
        go.Scatter(
            x=df_momentum['report_date_as_yyyy_mm_dd'],
            y=df_momentum['concentration_spread'],
            name='Spread',
            fill='tozeroy',
            fillcolor='rgba(128, 128, 128, 0.2)',
            line=dict(color='gray', width=1)
        ),
        row=1, col=1
    )
    
    # Plot 2: Momentum z-scores
    fig_main.add_trace(
        go.Scatter(
            x=df_momentum['report_date_as_yyyy_mm_dd'],
            y=df_momentum['long_momentum_zscore'],
            name='Long Momentum Z',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    fig_main.add_trace(
        go.Scatter(
            x=df_momentum['report_date_as_yyyy_mm_dd'],
            y=df_momentum['short_momentum_zscore'],
            name='Short Momentum Z',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Add threshold lines
    fig_main.add_hline(y=1.5, row=2, col=1, line_dash="dash", line_color="orange", opacity=0.5)
    fig_main.add_hline(y=-1.5, row=2, col=1, line_dash="dash", line_color="orange", opacity=0.5)
    fig_main.add_hline(y=0, row=2, col=1, line_dash="solid", line_color="gray", opacity=0.3)
    
    # Plot 3: Acceleration
    fig_main.add_trace(
        go.Bar(
            x=df_momentum['report_date_as_yyyy_mm_dd'],
            y=df_momentum['long_acceleration'],
            name='Long Acceleration',
            marker_color='green',
            opacity=0.6
        ),
        row=3, col=1
    )
    
    fig_main.add_trace(
        go.Bar(
            x=df_momentum['report_date_as_yyyy_mm_dd'],
            y=df_momentum['short_acceleration'],
            name='Short Acceleration',
            marker_color='red',
            opacity=0.6
        ),
        row=3, col=1
    )
    
    # Plot 4: Regime timeline
    for regime in df_momentum['momentum_regime'].unique():
        regime_mask = df_momentum['momentum_regime'] == regime
        color = df_momentum.loc[regime_mask, 'regime_color'].iloc[0] if regime_mask.sum() > 0 else 'gray'
        
        fig_main.add_trace(
            go.Bar(
                x=df_momentum.loc[regime_mask, 'report_date_as_yyyy_mm_dd'],
                y=[1] * regime_mask.sum(),
                name=regime,
                marker_color=color,
                showlegend=False,
                hovertemplate='%{x}<br>' + regime + '<extra></extra>'
            ),
            row=4, col=1
        )
    
    # Update layout
    fig_main.update_yaxes(title_text="Concentration %", row=1, col=1)
    fig_main.update_yaxes(title_text="Z-Score", row=2, col=1)
    fig_main.update_yaxes(title_text="% Change", row=3, col=1)
    fig_main.update_yaxes(showticklabels=False, row=4, col=1)
    fig_main.update_xaxes(title_text="Date", row=4, col=1)
    
    fig_main.update_layout(
        height=900,
        title=f"Concentration Momentum Analysis - {instrument_name}",
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Risk and regime dashboard
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Calculate last week's risk score for reference
        if len(df_momentum) > 1:
            # Get last week's values
            last_week = df_momentum.iloc[-2]
            
            # Recalculate last week's risk components
            lw_long_pct = df_momentum['long_concentration'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-2] * 100
            lw_short_pct = df_momentum['short_concentration'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-2] * 100
            lw_spread_pct = df_momentum['concentration_spread'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-2] * 100
            
            lw_long_momentum_pct = df_momentum['long_momentum_4w'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-2]
            lw_short_momentum_pct = df_momentum['short_momentum_4w'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-2]
            
            lw_long_momentum_risk = 2 * abs(lw_long_momentum_pct - 0.5)
            lw_short_momentum_risk = 2 * abs(lw_short_momentum_pct - 0.5)
            
            lw_momentum_risk = max(lw_long_momentum_risk, lw_short_momentum_risk)
            lw_concentration_risk = max(lw_long_pct, lw_short_pct) / 100
            lw_imbalance_risk = lw_spread_pct / 100
            
            last_week_risk = (lw_concentration_risk * 0.4 + lw_momentum_risk * 0.4 + lw_imbalance_risk * 0.2) * 100
        else:
            last_week_risk = None
        
        # Risk gauge with last week reference
        fig_risk = go.Figure()
        
        # Add the main gauge
        fig_risk.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = composite_risk,
            delta = {'reference': last_week_risk} if last_week_risk is not None else None,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Composite Risk Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        # Add a reference marker for last week's value
        if last_week_risk is not None:
            # Add a thin line/marker within the gauge arc to show last week's position
            # We'll add it as a very thin bar segment at the last week's position
            fig_risk.add_trace(go.Indicator(
                mode = "gauge",
                value = last_week_risk,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100], 'visible': False},
                    'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0},  # Invisible main bar
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'steps': [],
                    'threshold': {
                        'line': {'color': "white", 'width': 8},
                        'thickness': 0.8,
                        'value': last_week_risk
                    }
                }
            ))
            
            # Add annotation
            fig_risk.add_annotation(
                x=0.5, y=-0.1,
                text=f"◆ Last Week: {last_week_risk:.1f}",
                showarrow=False,
                font=dict(size=10, color="gray")
            )
        
        fig_risk.update_layout(height=250)
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.metric("Current Regime", current_regime)
        st.metric("Regime Duration", f"{regime_duration} weeks")
        
        # Add risk breakdown
        with st.expander("Risk Score Breakdown", expanded=False):
            st.markdown(f"""
            **Concentration Risk:** {concentration_risk:.1%}
            - Long: {long_pct:.0f}th percentile
            - Short: {short_pct:.0f}th percentile
            
            **Momentum Risk:** {momentum_risk:.1%}
            - Long: {long_momentum_pct:.1%} → {long_momentum_risk:.1%} risk
            - Short: {short_momentum_pct:.1%} → {short_momentum_risk:.1%} risk
            
            **Imbalance Risk:** {imbalance_risk:.1%}
            - Spread: {spread_pct:.0f}th percentile
            
            **Calculation:**
            - Composite = 40% × {concentration_risk:.1%} + 40% × {momentum_risk:.1%} + 20% × {imbalance_risk:.1%}
            - Composite = {composite_risk:.1f}%
            """)
    
    with col2:
        # Momentum quadrant chart
        fig_quad = go.Figure()
        
        # Add quadrant backgrounds
        fig_quad.add_shape(
            type="rect", x0=-3, x1=0, y0=0, y1=3,
            fillcolor="lightgreen", opacity=0.2,
            line=dict(width=0)
        )
        fig_quad.add_shape(
            type="rect", x0=0, x1=3, y0=0, y1=3,
            fillcolor="lightcoral", opacity=0.2,
            line=dict(width=0)
        )
        fig_quad.add_shape(
            type="rect", x0=-3, x1=0, y0=-3, y1=0,
            fillcolor="lightblue", opacity=0.2,
            line=dict(width=0)
        )
        fig_quad.add_shape(
            type="rect", x0=0, x1=3, y0=-3, y1=0,
            fillcolor="lightyellow", opacity=0.2,
            line=dict(width=0)
        )
        
        # Add current position
        fig_quad.add_trace(go.Scatter(
            x=[latest['long_momentum_zscore']],
            y=[latest['short_momentum_zscore']],
            mode='markers+text',
            marker=dict(size=20, color='darkblue'),
            text=['Current'],
            textposition='top center',
            showlegend=False
        ))
        
        # Add historical trail
        trail_data = df_momentum.tail(12)
        fig_quad.add_trace(go.Scatter(
            x=trail_data['long_momentum_zscore'],
            y=trail_data['short_momentum_zscore'],
            mode='lines+markers',
            line=dict(color='blue', width=1),
            marker=dict(size=5),
            opacity=0.5,
            name='12-week trail'
        ))
        
        fig_quad.update_xaxes(
            title="Long Momentum Z-Score",
            range=[-3, 3],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        )
        fig_quad.update_yaxes(
            title="Short Momentum Z-Score",
            range=[-3, 3],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        )
        
        fig_quad.update_layout(
            title="Momentum Quadrant Analysis",
            height=300,
            annotations=[
                dict(x=-1.5, y=2.5, text="Short Building<br>Long Unwinding", showarrow=False),
                dict(x=1.5, y=2.5, text="Both Building<br>(Bilateral)", showarrow=False),
                dict(x=-1.5, y=-2.5, text="Both Unwinding<br>(Democratizing)", showarrow=False),
                dict(x=1.5, y=-2.5, text="Long Building<br>Short Unwinding", showarrow=False)
            ]
        )
        
        st.plotly_chart(fig_quad, use_container_width=True)
    
    with col3:
        # Current metrics
        st.markdown("### Current Metrics")
        st.metric("Long Concentration", f"{latest['long_concentration']:.1f}%", 
                 f"{latest['long_momentum_4w']:+.1f}% (4w)")
        st.metric("Short Concentration", f"{latest['short_concentration']:.1f}%",
                 f"{latest['short_momentum_4w']:+.1f}% (4w)")
        st.metric("Concentration Spread", f"{latest['concentration_spread']:.1f}%",
                 f"{latest['spread_momentum']:+.1f}% (4w)")
    
    # Type-specific insights
    st.markdown("### Concentration Type Insights")
    
    if "Gross" in concentration_type:
        st.info(f"""
        **{concentration_type}** measures total positions held by top traders.
        - Current Long: {latest['long_concentration']:.1f}% ({long_pct:.0f}th percentile)
        - Current Short: {latest['short_concentration']:.1f}% ({short_pct:.0f}th percentile)
        - Shows raw market power regardless of hedging
        """)
    else:  # Net
        st.info(f"""
        **{concentration_type}** measures net exposure (long - short) of top traders.
        - Current Long: {latest['long_concentration']:.1f}% ({long_pct:.0f}th percentile)
        - Current Short: {latest['short_concentration']:.1f}% ({short_pct:.0f}th percentile)
        - Shows directional commitment after hedging
        """)
    
    # Generate signals
    signals = []
    
    if long_pct > 90 and latest['long_momentum_zscore'] > 1:
        signals.append(("⚠️ Long Concentration Critical", "Long side dangerously concentrated and still rising"))
    
    if short_pct > 90 and latest['short_momentum_zscore'] > 1:
        signals.append(("⚠️ Short Concentration Critical", "Short side dangerously concentrated and still rising"))
    
    if long_pct > 85 and latest['long_momentum_zscore'] < -1:
        signals.append(("📉 Long Distribution Beginning", "High long concentration starting to unwind"))
    
    if abs(latest['long_momentum_zscore'] - latest['short_momentum_zscore']) > 3:
        signals.append(("🔄 Extreme Asymmetry", "Unprecedented divergence in concentration momentum"))
    
    if regime_duration > 8 and "Building" in current_regime:
        signals.append(("⏰ Persistent Concentration", f"{current_regime} has persisted for {regime_duration} weeks"))
    
    if signals:
        st.markdown("### Current Signals")
        for signal, description in signals:
            st.warning(f"{signal}: {description}")