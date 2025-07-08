"""
Base chart creation functions for CFTC COT Dashboard
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from config import CHART_HEIGHT


def create_plotly_chart(df, selected_columns, chart_title):
    """Create interactive Plotly chart with dual y-axis for open interest"""
    try:
        if df.empty or not selected_columns:
            return None

        # Separate open interest from other columns
        open_interest_cols = [col for col in selected_columns if 'open_interest' in col.lower()]
        other_cols = [col for col in selected_columns if 'open_interest' not in col.lower()]

        # Create subplot with secondary y-axis if open interest is included
        if open_interest_cols and other_cols:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add open interest on secondary y-axis
            for col in open_interest_cols:
                if col in df.columns and not df[col].isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[col],
                            name=col.replace('_', ' ').title(),
                            line=dict(width=3),
                            opacity=0.8
                        ),
                        secondary_y=True
                    )

            # Add other columns on primary y-axis
            for col in other_cols:
                if col in df.columns and not df[col].isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[col],
                            name=col.replace('_', ' ').title(),
                            line=dict(width=2)
                        ),
                        secondary_y=False
                    )

            # Update y-axis labels
            fig.update_yaxes(title_text="Positions (Contracts)", secondary_y=False)
            fig.update_yaxes(title_text="Open Interest (Contracts)", secondary_y=True)

        else:
            # Single y-axis chart
            fig = go.Figure()

            for col in selected_columns:
                if col in df.columns and not df[col].isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[col],
                            name=col.replace('_', ' ').title(),
                            line=dict(width=2),
                            mode='lines'
                        )
                    )

            fig.update_yaxes(title_text="Positions (Contracts)")

        # Update layout
        fig.update_layout(
            title=chart_title,
            xaxis_title="Date",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=CHART_HEIGHT,
            showlegend=True
        )

        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

        # Enable dynamic y-axis scaling
        fig.update_yaxes(
            autorange=True,
            fixedrange=False,
            scaleanchor=None,
            constraintoward='middle'
        )
        
        # For secondary y-axis if it exists
        if open_interest_cols and other_cols:
            fig.update_yaxes(
                autorange=True,
                fixedrange=False,
                scaleanchor=None,
                constraintoward='middle',
                secondary_y=True
            )
        
        # Configuration for better zoom behavior
        config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': chart_title.replace(' ', '_'),
                'height': 600,
                'width': 1200,
                'scale': 2
            },
            'responsive': True
        }
        
        fig._config = config
        return fig

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None