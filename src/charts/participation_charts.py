"""
Participation and trader analysis charts for CFTC COT Dashboard
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import PARTICIPATION_CHART_HEIGHT, CONCENTRATION_COLORS


def create_participation_density_dashboard(df, instrument_name, percentile_data=None, concentration_type='Net'):
    """Create avg position per trader dashboard with concentration analysis"""
    try:
        # Copy and prepare data
        df_plot = df.copy()
        df_plot = df_plot.sort_values('report_date_as_yyyy_mm_dd')
        
        # Calculate average position per trader
        df_plot['avg_pos_per_trader'] = df_plot['open_interest_all'] / df_plot['traders_tot_all']
        
        # Calculate concentration
        if 'conc_net_le_4_tdr_long_all' in df_plot.columns and 'conc_net_le_4_tdr_short_all' in df_plot.columns:
            df_plot['concentration_4'] = (df_plot['conc_net_le_4_tdr_long_all'] + df_plot['conc_net_le_4_tdr_short_all']) / 2
        else:
            df_plot['concentration_4'] = 20  # Default value
        
        # Handle percentile data
        if percentile_data is None:
            percentile_data = [50] * len(df_plot)
        
        # Create fresh subplot figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}]
            ],
            subplot_titles=(
                'Average Position per Trader',
                'Top 4 Traders Concentration',
                'Percentile Rank'
            ),
            horizontal_spacing=0.01
        )
        
        # Chart 1: Average Position per Trader (bars) with Total Traders (line)
        # Create color array based on concentration levels
        bar_colors = []
        for conc in df_plot['concentration_4']:
            if conc < 15:
                bar_colors.append(CONCENTRATION_COLORS['low'])
            elif conc < 25:
                bar_colors.append(CONCENTRATION_COLORS['medium'])
            else:
                bar_colors.append(CONCENTRATION_COLORS['high'])
        
        fig.add_trace(
            go.Bar(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=df_plot['avg_pos_per_trader'],
                name='Avg Position',
                marker_color=bar_colors,
                marker_line_width=0
            ),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=df_plot['traders_tot_all'],
                name='Total Traders',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Chart 2: Concentration
        fig.add_trace(
            go.Scatter(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=df_plot['concentration_4'],
                name='Top 4 Concentration',
                fill='tozeroy',
                line=dict(color='red'),
                fillcolor='rgba(255, 0, 0, 0.1)'
            ),
            row=2, col=1
        )
        
        # Chart 3: Percentile
        fig.add_trace(
            go.Scatter(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=percentile_data,
                name='Percentile',
                fill='tozeroy',
                line=dict(color='purple'),
                fillcolor='rgba(128, 0, 128, 0.1)'
            ),
            row=3, col=1
        )
        
        # Add reference lines
        fig.add_hline(y=15, row=2, col=1, line_dash="dash", line_color="green", annotation_text="Low")
        fig.add_hline(y=25, row=2, col=1, line_dash="dash", line_color="orange", annotation_text="Medium")
        fig.add_hline(y=35, row=2, col=1, line_dash="dash", line_color="red", annotation_text="High")
        
        fig.add_hline(y=20, row=3, col=1, line_dash="dot", line_color="green", annotation_text="20th")
        fig.add_hline(y=50, row=3, col=1, line_dash="dot", line_color="gray", annotation_text="50th")
        fig.add_hline(y=80, row=3, col=1, line_dash="dot", line_color="red", annotation_text="80th")
        
        # Update axes
        fig.update_yaxes(title_text="Avg Position", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Traders", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Concentration %", row=2, col=1)
        fig.update_yaxes(title_text="Percentile", row=3, col=1, range=[0, 100])
        
        # Update layout
        fig.update_layout(
            title=f"Avg Position per Trader Analysis - {instrument_name}",
            height=PARTICIPATION_CHART_HEIGHT,
            showlegend=False,
            hovermode='x unified',
            autosize=True,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Force proper x-axis range for all subplots
        for row in range(1, 4):
            fig.update_xaxes(
                type='date',
                autorange=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                row=row, 
                col=1
            )
        
        # Add range selector to bottom x-axis only
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(label="All", step="all")
                ]),
                xanchor='left',
                x=0
            ),
            row=3, col=1
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating participation density dashboard: {str(e)}")
        return None


def create_trader_breakdown_charts(df, instrument_name):
    """Create trader breakdown analysis charts"""
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Commercial vs Non-Commercial Traders',
                'Long vs Short Traders',
                'Trader Participation Rate',
                'Average Position Size by Category'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Chart 1: Commercial vs Non-Commercial
        if 'traders_comm_long_all' in df.columns and 'traders_noncomm_long_all' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['report_date_as_yyyy_mm_dd'],
                    y=df['traders_comm_long_all'],
                    name='Commercial',
                    line=dict(color='green', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['report_date_as_yyyy_mm_dd'],
                    y=df['traders_noncomm_long_all'],
                    name='Non-Commercial',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Chart 2: Long vs Short
        if 'traders_tot_rept_long_all' in df.columns and 'traders_tot_rept_short_all' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['report_date_as_yyyy_mm_dd'],
                    y=df['traders_tot_rept_long_all'],
                    name='Long',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['report_date_as_yyyy_mm_dd'],
                    y=df['traders_tot_rept_short_all'],
                    name='Short',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
        
        # Chart 3: Participation Rate
        if 'traders_tot_all' in df.columns and 'open_interest_all' in df.columns:
            # Calculate participation rate (traders per 1000 contracts)
            df['participation_rate'] = (df['traders_tot_all'] / df['open_interest_all']) * 1000
            
            fig.add_trace(
                go.Scatter(
                    x=df['report_date_as_yyyy_mm_dd'],
                    y=df['participation_rate'],
                    name='Traders per 1000 contracts',
                    line=dict(color='purple', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(128, 0, 128, 0.1)'
                ),
                row=2, col=1
            )
        
        # Chart 4: Average Position Size by Category
        categories = []
        avg_positions = []
        
        if 'comm_positions_long_all' in df.columns and 'traders_comm_long_all' in df.columns:
            df['avg_comm_position'] = df['comm_positions_long_all'] / df['traders_comm_long_all']
            categories.append('Commercial')
            avg_positions.append(df['avg_comm_position'].iloc[-1])
        
        if 'noncomm_positions_long_all' in df.columns and 'traders_noncomm_long_all' in df.columns:
            df['avg_noncomm_position'] = df['noncomm_positions_long_all'] / df['traders_noncomm_long_all']
            categories.append('Non-Commercial')
            avg_positions.append(df['avg_noncomm_position'].iloc[-1])
        
        if categories:
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=avg_positions,
                    marker_color=['green', 'blue'][:len(categories)]
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Trader Breakdown Analysis - {instrument_name}",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Category", row=2, col=2)
        
        fig.update_yaxes(title_text="Number of Traders", row=1, col=1)
        fig.update_yaxes(title_text="Number of Traders", row=1, col=2)
        fig.update_yaxes(title_text="Traders per 1000 Contracts", row=2, col=1)
        fig.update_yaxes(title_text="Average Position Size", row=2, col=2)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating trader breakdown charts: {str(e)}")
        return None