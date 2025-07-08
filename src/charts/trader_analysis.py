"""
Trader analysis charts for CFTC COT data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_trader_breakdown_charts(df, trader_category):
    """Create side-by-side charts showing trader positions breakdown"""
    try:
        # Define column mappings for each trader category
        category_columns = {
            "Non-Commercial": {
                "long": "noncomm_positions_long_all",
                "short": "noncomm_positions_short_all",
                "spread": "noncomm_postions_spread_all",
                "net": "net_noncomm_positions"
            },
            "Commercial": {
                "long": "comm_positions_long_all",
                "short": "comm_positions_short_all",
                "net": "net_comm_positions"
            },
            "Non-Reportable": {
                "long": "nonrept_positions_long_all",
                "short": "nonrept_positions_short_all"
            }
        }
        
        # Get columns for selected category
        cols = category_columns[trader_category]
        
        # Calculate net position if not already in dataframe
        if trader_category == "Non-Reportable" and "net" not in cols:
            df['net_nonrept_positions'] = df[cols['long']] - df[cols['short']]
            cols['net'] = 'net_nonrept_positions'
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.5, 0.5],
            horizontal_spacing=0.12,
            subplot_titles=(
                f'{trader_category} Positions (Contracts)',
                f'{trader_category} as % of Open Interest'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # Left chart - Absolute positions
        # Add area for long positions
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df[cols['long']],
                name='Long',
                fill='tozeroy',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(0, 0, 139, 0.7)',  # Dark blue
                showlegend=True,
                hovertemplate='Long: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add area for short positions (negative)
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=-df[cols['short']],
                name='Short',
                fill='tozeroy',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(30, 144, 255, 0.7)',  # Lighter blue
                showlegend=True,
                hovertemplate='Short: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add net position line
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df[cols['net']],
                name='Net',
                mode='lines',
                line=dict(color='yellow', width=3),
                showlegend=True,
                hovertemplate='Net: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, row=1, col=1)
        
        # Right chart - Percentage of open interest
        # Calculate percentages
        if 'open_interest_all' in df.columns:
            net_pct = (df[cols['net']] / df['open_interest_all'] * 100).fillna(0)
            
            # Add open interest area (background) on primary y-axis
            fig.add_trace(
                go.Scatter(
                    x=df['report_date_as_yyyy_mm_dd'],
                    y=df['open_interest_all'],
                    name='Open Interest (LHS)',
                    fill='tozeroy',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(100, 100, 255, 0.3)',  # Semi-transparent blue
                    showlegend=True,
                    hovertemplate='OI: %{y:,.0f}<extra></extra>'
                ),
                row=1, col=2,
                secondary_y=False
            )
            
            # Add net percentage line on secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=df['report_date_as_yyyy_mm_dd'],
                    y=net_pct,
                    name='% of Open Interest (RHS)',
                    mode='lines',
                    line=dict(color='yellow', width=3),
                    showlegend=True,
                    hovertemplate='Net %: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=2,
                secondary_y=True
            )
            
            # Add zero line for percentage
            fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{trader_category} Trader Analysis",
                y=0.98,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            height=600,
            template='plotly_dark',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.07,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, l=50, r=50, b=50),
            # Configure x-axis with range selector for right subplot
            xaxis2=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=False),
                type="date"
            )
        )
        
        # Configure axes for left subplot
        fig.update_yaxes(
            title_text="Contracts ('000 lots)",
            tickformat=',.0f',
            row=1, col=1
        )
        
        # Configure axes for right subplot
        # Primary y-axis (left side) for Open Interest
        fig.update_yaxes(
            title_text="Open Interest",
            tickformat=',.0f',
            showgrid=True,
            row=1, col=2,
            secondary_y=False
        )
        
        # Secondary y-axis (right side) for Percentage
        fig.update_yaxes(
            title_text="% of Open Interest",
            tickformat='.1f',
            ticksuffix='%',
            showgrid=False,
            zeroline=True,
            row=1, col=2,
            secondary_y=True
        )
        
        
        # Update axes properties
        fig.update_xaxes(matches='x', row=1, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating trader breakdown charts: {str(e)}")
        return None