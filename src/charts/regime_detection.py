"""
Regime Detection Dashboard for CFTC COT Dashboard
Complete implementation from legacyF.py
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
try:
    from config import PARTICIPATION_CHART_HEIGHT
except ImportError:
    PARTICIPATION_CHART_HEIGHT = 600


def create_regime_detection_dashboard(df, instrument_name):
    """Create comprehensive regime detection analysis"""
    try:
        # Calculate regime metrics
        df_regime = df.copy()
        window = 52
        min_periods = 26
        
        # Step 1: Calculate all percentile metrics
        df_regime['long_conc_pct'] = df_regime['conc_gross_le_4_tdr_long'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        df_regime['short_conc_pct'] = df_regime['conc_gross_le_4_tdr_short'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # Net positions
        df_regime['comm_net'] = df_regime['comm_positions_long_all'] - df_regime['comm_positions_short_all']
        df_regime['noncomm_net'] = df_regime['noncomm_positions_long_all'] - df_regime['noncomm_positions_short_all']
        df_regime['comm_net_pct'] = df_regime['comm_net'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        df_regime['noncomm_net_pct'] = df_regime['noncomm_net'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # Flow intensity
        df_regime['comm_flow'] = df_regime['comm_net'].diff()
        df_regime['noncomm_flow'] = df_regime['noncomm_net'].diff()
        df_regime['flow_intensity'] = abs(df_regime['comm_flow']) + abs(df_regime['noncomm_flow'])
        df_regime['flow_pct'] = df_regime['flow_intensity'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # Trader participation
        df_regime['trader_total_pct'] = df_regime['traders_tot_all'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # Placeholder for heterogeneity
        df_regime['heterogeneity_pct'] = 50  # Would use actual heterogeneity index
        
        # Step 2: Calculate regime extremity score
        def distance_from_center(pct):
            return abs(pct - 50) * 2
        
        df_regime['regime_extremity'] = df_regime.apply(lambda row: 
            max(distance_from_center(row['long_conc_pct']), 
                distance_from_center(row['short_conc_pct'])) * 0.25 +
            max(distance_from_center(row['comm_net_pct']), 
                distance_from_center(row['noncomm_net_pct'])) * 0.25 +
            row['flow_pct'] * 0.25 +
            row['heterogeneity_pct'] * 0.25
        , axis=1)
        
        # Step 3: Detect regime
        def detect_regime(row):
            EXTREME_HIGH = 85
            EXTREME_LOW = 15
            MODERATE_HIGH = 70
            MODERATE_LOW = 30
            
            if pd.isna(row['long_conc_pct']):
                return "Insufficient Data", "gray"
            
            # Check patterns
            if row['long_conc_pct'] > EXTREME_HIGH and row['short_conc_pct'] < MODERATE_LOW:
                return "Long Concentration Extreme", "red"
            elif row['short_conc_pct'] > EXTREME_HIGH and row['long_conc_pct'] < MODERATE_LOW:
                return "Short Concentration Extreme", "red"
            elif row['long_conc_pct'] > EXTREME_HIGH and row['short_conc_pct'] > EXTREME_HIGH:
                return "Bilateral Concentration", "orange"
            elif row['noncomm_net_pct'] > EXTREME_HIGH and row['comm_net_pct'] < EXTREME_LOW:
                return "Speculative Long Extreme", "red"
            elif row['noncomm_net_pct'] < EXTREME_LOW and row['comm_net_pct'] > EXTREME_HIGH:
                return "Commercial Long Extreme", "orange"
            elif row['flow_pct'] > EXTREME_HIGH:
                return "High Flow Volatility", "yellow"
            elif row['heterogeneity_pct'] > EXTREME_HIGH:
                return "Maximum Divergence", "red"
            elif row['regime_extremity'] < 40:
                return "Balanced Market", "green"
            else:
                return "Transitional", "gray"
        
        df_regime[['regime', 'regime_color']] = df_regime.apply(
            lambda row: pd.Series(detect_regime(row)), axis=1
        )
        
        # Get latest values
        latest = df_regime.iloc[-1]
        
        # Create main visualization
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "polar"}, {"type": "scatter"}],
                [{"colspan": 3}, None, None]
            ],
            row_heights=[0.4, 0.6],
            subplot_titles=["Market Extremity", "Percentile Rankings", "Current Regime",
                           "Regime Timeline (Past 52 Weeks)"]
        )
        
        # 1. Gauge chart for extremity
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = latest['regime_extremity'],
            title = {'text': "Extremity Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ), row=1, col=1)
        
        # 2. Spider chart of metrics
        categories = ['Long Conc', 'Short Conc', 'Comm Net', 'NonComm Net', 
                     'Flow', 'Traders', 'Heterogeneity']
        values = [
            latest['long_conc_pct'],
            latest['short_conc_pct'],
            latest['comm_net_pct'],
            latest['noncomm_net_pct'],
            latest['flow_pct'],
            latest['trader_total_pct'],
            latest['heterogeneity_pct']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current',
            line_color='blue'
        ), row=1, col=2)
        
        # Add reference circles
        fig.add_trace(go.Scatterpolar(
            r=[50]*7,
            theta=categories,
            name='Normal (50th)',
            line=dict(color='gray', dash='dash')
        ), row=1, col=2)
        
        fig.add_trace(go.Scatterpolar(
            r=[85]*7,
            theta=categories,
            name='Extreme (85th)',
            line=dict(color='red', dash='dot')
        ), row=1, col=2)
        
        # 3. Current regime display (using annotations instead of scatter)
        regime_color_map = {
            'red': '#FF0000',
            'orange': '#FFA500',
            'yellow': '#FFD700',
            'green': '#00FF00',
            'gray': '#808080'
        }
        
        fig.add_annotation(
            text=f"<b>{latest['regime']}</b>",
            xref="x3", yref="y3",
            x=0.5, y=0.7,
            showarrow=False,
            font=dict(size=18, color=regime_color_map.get(latest['regime_color'], 'black')),
            row=1, col=3
        )
        
        # Calculate regime duration
        current_regime = latest['regime']
        regime_duration = 1
        for i in range(2, min(len(df_regime), 20)):
            if df_regime.iloc[-i]['regime'] == current_regime:
                regime_duration += 1
            else:
                break
        
        fig.add_annotation(
            text=f"Duration: {regime_duration} weeks<br>Score: {latest['regime_extremity']:.1f}/100",
            xref="x3", yref="y3",
            x=0.5, y=0.3,
            showarrow=False,
            font=dict(size=12),
            row=1, col=3
        )
        
        # 4. Regime timeline
        timeline_data = df_regime.tail(52).copy()
        
        # Create color mapping
        color_map = {
            'Long Concentration Extreme': 'darkred',
            'Short Concentration Extreme': 'darkred',
            'Bilateral Concentration': 'orange',
            'Speculative Long Extreme': 'red',
            'Commercial Long Extreme': 'darkorange',
            'High Flow Volatility': 'gold',
            'Maximum Divergence': 'darkred',
            'Balanced Market': 'green',
            'Transitional': 'gray',
            'Insufficient Data': 'lightgray'
        }
        
        # Plot regime timeline
        for regime, color in color_map.items():
            mask = timeline_data['regime'] == regime
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=timeline_data.loc[mask, 'report_date_as_yyyy_mm_dd'],
                    y=timeline_data.loc[mask, 'regime_extremity'],
                    mode='markers',
                    marker=dict(color=color, size=10),
                    name=regime,
                    legendgroup=regime,
                    showlegend=True
                ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Market Regime Detection - {instrument_name}",
                font=dict(size=18),
                x=0.5,
                xanchor='center'
            ),
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Extremity Score", range=[0, 100], row=2, col=1)
        
        # Update polar axis
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            ),
            row=1, col=2
        )
        
        # Hide axes for regime display
        fig.update_xaxes(visible=False, row=1, col=3)
        fig.update_yaxes(visible=False, row=1, col=3)
        
        return fig, df_regime, latest
        
    except Exception as e:
        st.error(f"Error creating regime detection dashboard: {str(e)}")
        return None, None, None