"""
Time series analysis charts for CFTC COT data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_time_series_chart(df, selected_columns, chart_title):
    """Create time series chart with multiple data series"""
    try:
        if df.empty or not selected_columns:
            return None
            
        # Create figure
        fig = go.Figure()
        
        # Color palette
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Add traces for each selected column
        for i, col in enumerate(selected_columns):
            if col in df.columns:
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter(
                    x=df['report_date_as_yyyy_mm_dd'],
                    y=df[col],
                    mode='lines',
                    name=col.replace('_', ' ').title(),
                    line=dict(width=2, color=color),
                    hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating time series chart: {str(e)}")
        return None