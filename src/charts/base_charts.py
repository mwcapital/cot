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
    """Create interactive Plotly chart with dual y-axis for open interest and latest value indicators"""
    try:
        if df.empty or not selected_columns:
            return None

        # Get the latest date and values
        latest_date = df['report_date_as_yyyy_mm_dd'].max()
        latest_row = df[df['report_date_as_yyyy_mm_dd'] == latest_date].iloc[0]
        
        # Define colors for consistency
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        color_map = {}
        
        # Separate open interest from other columns
        open_interest_cols = [col for col in selected_columns if 'open_interest' in col.lower()]
        other_cols = [col for col in selected_columns if 'open_interest' not in col.lower()]

        # Create subplot with secondary y-axis if open interest is included
        if open_interest_cols and other_cols:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add open interest on secondary y-axis
            color_idx = 0
            for col in open_interest_cols:
                if col in df.columns and not df[col].isna().all():
                    color = colors[color_idx % len(colors)]
                    color_map[col] = color
                    color_idx += 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[col],
                            name=col.replace('_', ' ').title(),
                            line=dict(width=3, color=color),
                            opacity=0.8
                        ),
                        secondary_y=True
                    )
                    
                    # Add latest point marker
                    latest_val = latest_row[col] if col in latest_row.index else None
                    if latest_val is not None and not pd.isna(latest_val):
                        fig.add_trace(
                            go.Scatter(
                                x=[latest_date],
                                y=[latest_val],
                                mode='markers+text',
                                marker=dict(size=15, color=color, symbol='circle', 
                                          line=dict(color='white', width=2)),
                                text=[f'{latest_val:,.0f}'],
                                textposition='top center',
                                textfont=dict(size=12, color=color),
                                showlegend=False,
                                hovertemplate=f'{col.replace("_", " ").title()}<br>Latest: %{{y:,.0f}}<br>Date: {latest_date.strftime("%Y-%m-%d")}<extra></extra>'
                            ),
                            secondary_y=True
                        )

            # Add other columns on primary y-axis
            for col in other_cols:
                if col in df.columns and not df[col].isna().all():
                    color = colors[color_idx % len(colors)]
                    color_map[col] = color
                    color_idx += 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[col],
                            name=col.replace('_', ' ').title(),
                            line=dict(width=2, color=color)
                        ),
                        secondary_y=False
                    )
                    
                    # Add latest point marker
                    latest_val = latest_row[col] if col in latest_row.index else None
                    if latest_val is not None and not pd.isna(latest_val):
                        fig.add_trace(
                            go.Scatter(
                                x=[latest_date],
                                y=[latest_val],
                                mode='markers+text',
                                marker=dict(size=15, color=color, symbol='circle',
                                          line=dict(color='white', width=2)),
                                text=[f'{latest_val:,.0f}'],
                                textposition='top center',
                                textfont=dict(size=12, color=color),
                                showlegend=False,
                                hovertemplate=f'{col.replace("_", " ").title()}<br>Latest: %{{y:,.0f}}<br>Date: {latest_date.strftime("%Y-%m-%d")}<extra></extra>'
                            ),
                            secondary_y=False
                        )

            # Update y-axis labels
            fig.update_yaxes(title_text="Positions (Contracts)", secondary_y=False)
            fig.update_yaxes(title_text="Open Interest (Contracts)", secondary_y=True)

        else:
            # Single y-axis chart
            fig = go.Figure()

            color_idx = 0
            for col in selected_columns:
                if col in df.columns and not df[col].isna().all():
                    color = colors[color_idx % len(colors)]
                    color_map[col] = color
                    color_idx += 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[col],
                            name=col.replace('_', ' ').title(),
                            line=dict(width=2, color=color),
                            mode='lines'
                        )
                    )
                    
                    # Add latest point marker with value
                    latest_val = latest_row[col] if col in latest_row.index else None
                    if latest_val is not None and not pd.isna(latest_val):
                        fig.add_trace(
                            go.Scatter(
                                x=[latest_date],
                                y=[latest_val],
                                mode='markers+text',
                                marker=dict(size=15, color=color, symbol='circle',
                                          line=dict(color='white', width=2)),
                                text=[f'{latest_val:,.0f}'],
                                textposition='top center',
                                textfont=dict(size=12, color=color),
                                showlegend=False,
                                hovertemplate=f'{col.replace("_", " ").title()}<br>Latest: %{{y:,.0f}}<br>Date: {latest_date.strftime("%Y-%m-%d")}<extra></extra>'
                            )
                        )

            fig.update_yaxes(title_text="Positions (Contracts)")
        
        # Add dashed lines from latest points to axes
        for col in selected_columns:
            if col in df.columns and col in latest_row.index:
                latest_val = latest_row[col]
                if latest_val is not None and not pd.isna(latest_val):
                    # Determine which y-axis to use
                    use_secondary = col in open_interest_cols and open_interest_cols and other_cols
                    line_color = color_map.get(col, 'gray')
                    
                    # Vertical line from point to x-axis
                    fig.add_shape(
                        type="line",
                        x0=latest_date, x1=latest_date,
                        y0=latest_val, y1=0,
                        line=dict(color=line_color, width=1, dash="dot"),
                        opacity=0.3,
                        yref="y2" if use_secondary else "y",
                        layer="below"
                    )
                    
                    # Horizontal line from point to y-axis
                    fig.add_shape(
                        type="line",
                        x0=latest_date, x1=df['report_date_as_yyyy_mm_dd'].min(),
                        y0=latest_val, y1=latest_val,
                        line=dict(color=line_color, width=1, dash="dot"),
                        opacity=0.3,
                        yref="y2" if use_secondary else "y",
                        layer="below"
                    )

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