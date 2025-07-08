"""
Improved participation and trader analysis charts for CFTC COT Dashboard
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import PARTICIPATION_CHART_HEIGHT, CONCENTRATION_COLORS


def create_participation_density_dashboard(df, instrument_name, percentile_data=None, concentration_type='Net'):
    """Create improved avg position per trader dashboard with concentration analysis"""
    try:
        # Copy and prepare data
        df_plot = df.copy()
        df_plot = df_plot.sort_values('report_date_as_yyyy_mm_dd')
        
        # Calculate average position per trader
        df_plot['avg_pos_per_trader'] = df_plot['open_interest_all'] / df_plot['traders_tot_all']
        
        # Calculate concentrations for both long and short based on type
        if concentration_type == 'Gross':
            # Use gross concentration columns
            if 'conc_gross_le_4_tdr_long' in df_plot.columns and 'conc_gross_le_4_tdr_short' in df_plot.columns:
                df_plot['concentration_4_long'] = df_plot['conc_gross_le_4_tdr_long']
                df_plot['concentration_4_short'] = df_plot['conc_gross_le_4_tdr_short']
                df_plot['concentration_4_avg'] = (df_plot['conc_gross_le_4_tdr_long'] + df_plot['conc_gross_le_4_tdr_short']) / 2
            else:
                df_plot['concentration_4_long'] = 20  # Default value
                df_plot['concentration_4_short'] = 20
                df_plot['concentration_4_avg'] = 20
        else:
            # Use net concentration columns (default)
            if 'conc_net_le_4_tdr_long_all' in df_plot.columns and 'conc_net_le_4_tdr_short_all' in df_plot.columns:
                df_plot['concentration_4_long'] = df_plot['conc_net_le_4_tdr_long_all']
                df_plot['concentration_4_short'] = df_plot['conc_net_le_4_tdr_short_all']
                df_plot['concentration_4_avg'] = (df_plot['conc_net_le_4_tdr_long_all'] + df_plot['conc_net_le_4_tdr_short_all']) / 2
            else:
                df_plot['concentration_4_long'] = 20  # Default value
                df_plot['concentration_4_short'] = 20
                df_plot['concentration_4_avg'] = 20
        
        # Handle percentile data
        if percentile_data is None:
            percentile_data = [50] * len(df_plot)
        
        # Calculate percentiles for concentrations
        percentile_long = []
        percentile_short = []
        for i in range(len(df_plot)):
            # Calculate percentile for long concentration
            current_long = df_plot.iloc[i]['concentration_4_long']
            pct_long = (df_plot['concentration_4_long'].iloc[:i+1] < current_long).sum() / (i+1) * 100
            percentile_long.append(pct_long)
            
            # Calculate percentile for short concentration
            current_short = df_plot.iloc[i]['concentration_4_short']
            pct_short = (df_plot['concentration_4_short'].iloc[:i+1] < current_short).sum() / (i+1) * 100
            percentile_short.append(pct_short)
        
        # Create subplot figure with 6 rows
        fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}]
            ],
            subplot_titles=(
                'Average Position per Trader (Color based on percentile rank)',
                'Average Position Percentile Rank',
                f'Top 4 Traders {concentration_type} Long Concentration',
                'Top 4 Long Concentration Percentile',
                f'Top 4 Traders {concentration_type} Short Concentration',
                'Top 4 Short Concentration Percentile'
            ),
            horizontal_spacing=0.01
        )
        
        # Chart 1: Average Position per Trader (bars) with Total Traders (line)
        # Create color array based on percentile of average position
        bar_colors = []
        for i, pct in enumerate(percentile_data):
            if pct < 33:
                bar_colors.append('green')  # Low percentile (small positions)
            elif pct < 67:
                bar_colors.append('yellow')  # Medium percentile
            else:
                bar_colors.append('red')  # High percentile (large positions)
        
        fig.add_trace(
            go.Bar(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=df_plot['avg_pos_per_trader'],
                name='Avg Position',
                marker_color=bar_colors,
                marker_line_width=0,
                hovertemplate='Date: %{x}<br>Avg Position: %{y:,.0f}<br>Color shows concentration level<extra></extra>'
            ),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=df_plot['traders_tot_all'],
                name='Total Traders',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Total Traders: %{y}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Chart 2: Average Position Percentile
        fig.add_trace(
            go.Scatter(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=percentile_data,
                name='Avg Position Percentile',
                fill='tozeroy',
                line=dict(color='purple'),
                fillcolor='rgba(128, 0, 128, 0.1)',
                hovertemplate='Date: %{x}<br>Percentile: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Chart 3: Long Concentration
        fig.add_trace(
            go.Scatter(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=df_plot['concentration_4_long'],
                name='Top 4 Long',
                fill='tozeroy',
                line=dict(color='darkgreen'),
                fillcolor='rgba(0, 128, 0, 0.1)',
                hovertemplate='Date: %{x}<br>Top 4 Long: %{y:.1f}%<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Chart 4: Long Percentile
        fig.add_trace(
            go.Scatter(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=percentile_long,
                name='Long Percentile',
                fill='tozeroy',
                line=dict(color='green'),
                fillcolor='rgba(0, 255, 0, 0.1)',
                hovertemplate='Date: %{x}<br>Percentile: %{y:.1f}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # Chart 5: Short Concentration
        fig.add_trace(
            go.Scatter(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=df_plot['concentration_4_short'],
                name='Top 4 Short',
                fill='tozeroy',
                line=dict(color='darkred'),
                fillcolor='rgba(139, 0, 0, 0.1)',
                hovertemplate='Date: %{x}<br>Top 4 Short: %{y:.1f}%<extra></extra>'
            ),
            row=5, col=1
        )
        
        # Chart 6: Short Percentile
        fig.add_trace(
            go.Scatter(
                x=df_plot['report_date_as_yyyy_mm_dd'],
                y=percentile_short,
                name='Short Percentile',
                fill='tozeroy',
                line=dict(color='red'),
                fillcolor='rgba(255, 0, 0, 0.1)',
                hovertemplate='Date: %{x}<br>Percentile: %{y:.1f}<extra></extra>'
            ),
            row=6, col=1
        )
        
        # Add reference lines for avg position percentile
        fig.add_hline(y=33, row=2, col=1, line_dash="dot", line_color="green", 
                     annotation_text="33rd", annotation_position="right")
        fig.add_hline(y=50, row=2, col=1, line_dash="dot", line_color="gray", 
                     annotation_text="50th", annotation_position="right")
        fig.add_hline(y=67, row=2, col=1, line_dash="dot", line_color="red", 
                     annotation_text="67th", annotation_position="right")
        
        # No reference lines for concentration charts per user request
        
        # Add reference lines for percentiles
        for row in [4, 6]:  # Percentile rows
            fig.add_hline(y=20, row=row, col=1, line_dash="dot", line_color="green", 
                         annotation_text="20th", annotation_position="right")
            fig.add_hline(y=50, row=row, col=1, line_dash="dot", line_color="gray", 
                         annotation_text="50th", annotation_position="right")
            fig.add_hline(y=80, row=row, col=1, line_dash="dot", line_color="red", 
                         annotation_text="80th", annotation_position="right")
        
        # Update axes
        fig.update_yaxes(title_text="Avg Position", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Traders", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Avg Position Percentile (1 Year)", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="% of OI", row=3, col=1)
        fig.update_yaxes(title_text="Percentile", row=4, col=1, range=[0, 100])
        fig.update_yaxes(title_text="% of OI", row=5, col=1)
        fig.update_yaxes(title_text="Percentile", row=6, col=1, range=[0, 100])
        
        # Update layout with range selector at the top
        fig.update_layout(
            title=dict(
                text=f"Trader Participation & Concentration Analysis - {instrument_name}",
                y=0.99,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            height=1400,  # Increased height for 6 subplots
            showlegend=False,
            hovermode='x unified',
            autosize=True,
            margin=dict(l=80, r=120, t=180, b=80),  # Increased top margin for buttons and titles
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(label="All", step="all")
                    ]),
                    bgcolor='rgba(255,255,255,0.9)',
                    activecolor='lightblue',
                    x=0.01,
                    y=1.08,  # Moved down to avoid title overlap
                    xanchor='left',
                    yanchor='top',
                    font=dict(size=11)
                ),
                type='date'
            )
        )
        
        # Ensure titles don't overlap - increase spacing
        fig.update_annotations(font=dict(size=12))
        
        # Force proper x-axis range for all subplots
        for row in range(1, 7):  # Updated for 6 rows
            fig.update_xaxes(
                type='date',
                autorange=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                row=row, 
                col=1
            )
        
        # Add range slider only to bottom chart
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.03,
            row=6, col=1  # Updated for 6th row
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating participation density dashboard: {str(e)}")
        return None