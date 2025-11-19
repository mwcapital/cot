"""
Momentum analysis charts for CFTC COT data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_single_variable_momentum_dashboard(df, variable_name, change_col):
    """Create focused momentum dashboard for a single variable"""
    try:
        # Prepare data
        df_momentum = df.copy()
        df_momentum = df_momentum.sort_values('report_date_as_yyyy_mm_dd')
        
        # Get the actual values and changes
        actual_values = df_momentum[variable_name].fillna(0)
        change_values = df_momentum[change_col].fillna(0)
        
        # Calculate z-scores using 104-week rolling window (2 years)
        rolling_mean = change_values.rolling(window=104, min_periods=1).mean()
        rolling_std = change_values.rolling(window=104, min_periods=1).std()

        # Calculate z-score: (current - rolling mean) / rolling std
        z_scores = pd.Series(index=change_values.index, dtype=float)
        for i in range(len(change_values)):
            if i == 0:
                z_scores.iloc[i] = 0
            else:
                # Use up to 104 weeks of history (2 years)
                lookback_start = max(0, i - 103)
                historical_values = change_values.iloc[lookback_start:i+1]

                if len(historical_values) > 1 and historical_values.std() > 0:
                    mean = historical_values[:-1].mean()  # Exclude current value from mean
                    std = historical_values[:-1].std()    # Exclude current value from std
                    z_scores.iloc[i] = (change_values.iloc[i] - mean) / std
                else:
                    z_scores.iloc[i] = 0
        
        # Create figure with subplots
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=[0.3, 0.25, 0.25, 0.2],
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(
                f'{variable_name.replace("_", " ").title()} - Actual Values',
                'Week-over-Week Changes',
                'Change Magnitude (Absolute)',
                'Z-Score of Changes'
            )
        )
        
        # 1. Actual values with trend
        fig.add_trace(
            go.Scatter(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=actual_values,
                mode='lines',
                name='Actual Value',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        
        # 2. Changes as bars with color coding
        colors = ['green' if x > 0 else 'red' for x in change_values]
        
        fig.add_trace(
            go.Bar(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=change_values,
                name='Weekly Change',
                marker=dict(color=colors),
                hovertemplate='Date: %{x}<br>Change: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)
        
        # 3. Absolute magnitude of changes (to see volatility)
        abs_changes = abs(change_values)
        percentile_90 = np.percentile(abs_changes[abs_changes > 0], 90) if len(abs_changes[abs_changes > 0]) > 0 else 0
        
        fig.add_trace(
            go.Bar(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=abs_changes,
                name='Absolute Change',
                marker=dict(
                    color=abs_changes,
                    colorscale='Viridis',
                    cmin=0,
                    cmax=percentile_90 * 1.2,
                    colorbar=dict(
                        title="Magnitude",
                        x=1.02,
                        len=0.2,
                        y=0.4
                    )
                ),
                hovertemplate='Date: %{x}<br>|Change|: %{y:,.0f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add percentile lines
        if len(abs_changes[abs_changes > 0]) > 5:
            p50 = np.percentile(abs_changes[abs_changes > 0], 50)
            p90 = np.percentile(abs_changes[abs_changes > 0], 90)
            
            fig.add_hline(y=p50, line_dash="dash", line_color="gray", line_width=1, 
                         annotation_text="Median", annotation_position="right", row=3, col=1)
            fig.add_hline(y=p90, line_dash="dash", line_color="red", line_width=2,
                         annotation_text="90th %ile", annotation_position="right", row=3, col=1)
        
        # 4. Z-Score time series
        fig.add_trace(
            go.Scatter(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=z_scores,
                mode='lines',
                fill='tozeroy',
                name='Z-Score',
                line=dict(color='purple', width=2),
                fillcolor='rgba(128, 0, 128, 0.2)',
                hovertemplate='Date: %{x}<br>Z-Score: %{y:.2f}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # Add z-score reference lines
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=4, col=1)
        fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1, 
                     annotation_text="+2σ", annotation_position="right", row=4, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1,
                     annotation_text="-2σ", annotation_position="right", row=4, col=1)
        
        # Shade extreme zones
        fig.add_hrect(y0=2, y1=4, fillcolor="red", opacity=0.1, line_width=0, row=4, col=1)
        fig.add_hrect(y0=-4, y1=-2, fillcolor="red", opacity=0.1, line_width=0, row=4, col=1)
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=False,
            hovermode='x unified',
            margin=dict(t=40, l=80, r=80, b=40),  # Consistent margins
            xaxis=dict(
                rangeslider=dict(visible=False),
                autorange=True
            ),
            xaxis2=dict(
                rangeslider=dict(visible=False),
                autorange=True
            ),
            xaxis3=dict(
                rangeslider=dict(visible=False),
                autorange=True
            ),
            xaxis4=dict(
                rangeslider=dict(visible=False),
                title='Date',
                autorange=True
            ),
            yaxis=dict(
                title="Value",
                autorange=True,
                fixedrange=False,
                rangemode='tozero'
            ),
            yaxis2=dict(
                title="Change",
                autorange=True,
                fixedrange=False
            ),
            yaxis3=dict(
                title="Absolute Change",
                autorange=True,
                fixedrange=False,
                rangemode='tozero'
            ),
            yaxis4=dict(
                title="Z-Score",
                zeroline=True,
                autorange=True,
                fixedrange=False,
                range=[-4, 4]  # Initial range for z-score
            )
        )
        
        # Configure all x-axes to be linked and enable auto-ranging
        for i in range(1, 5):
            fig['layout'][f'xaxis{i}']['autorange'] = True
            fig['layout'][f'xaxis{i}']['matches'] = 'x4'
            
        # Set up relay out zoom for auto y-axis scaling
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating single variable momentum dashboard: {str(e)}")
        return None


def create_momentum_dashboard(df, lookback_period=52):
    """
    Create comprehensive momentum dashboard with heat strips and advanced analytics
    
    Args:
        df: DataFrame with CFTC data
        lookback_period: Number of weeks for z-score calculation (default: 52)
    """
    try:
        # Implementation would go here - this is a placeholder
        # The full implementation is quite complex and would need to be extracted from legacyF.py
        pass
    except Exception as e:
        st.error(f"Error creating momentum dashboard: {str(e)}")
        return None


def create_actual_values_chart(df, variable_name):
    """Create chart showing only the actual values with trend"""
    try:
        # Prepare data
        df_momentum = df.copy()
        df_momentum = df_momentum.sort_values('report_date_as_yyyy_mm_dd')

        # Get the actual values
        actual_values = df_momentum[variable_name].fillna(0)

        # Create simple figure with single plot
        fig = go.Figure()

        # Add actual values line
        fig.add_trace(
            go.Scatter(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=actual_values,
                mode='lines',
                name='Actual Value',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
            )
        )

        # Calculate date ranges for buttons
        latest_date = df_momentum['report_date_as_yyyy_mm_dd'].max()
        one_year_ago = latest_date - pd.DateOffset(years=1)
        two_years_ago = latest_date - pd.DateOffset(years=2)
        five_years_ago = latest_date - pd.DateOffset(years=5)
        ten_years_ago = latest_date - pd.DateOffset(years=10)

        # Helper function to calculate y-range for a given date range
        def get_y_range(start_date, end_date):
            mask = (df_momentum['report_date_as_yyyy_mm_dd'] >= start_date) & \
                   (df_momentum['report_date_as_yyyy_mm_dd'] <= end_date)
            if mask.any():
                y_data = actual_values[mask]
                y_min = y_data.min()
                y_max = y_data.max()
                # Add 5% padding
                padding = (y_max - y_min) * 0.05
                return [y_min - padding, y_max + padding]
            return None

        # Pre-calculate y-ranges for each button
        y_ranges = {
            "1Y": get_y_range(one_year_ago, latest_date),
            "2Y": get_y_range(two_years_ago, latest_date),
            "5Y": get_y_range(five_years_ago, latest_date),
            "10Y": get_y_range(ten_years_ago, latest_date),
            "All": None  # Will use autorange
        }

        # Update layout with enhanced range selector
        fig.update_layout(
            height=400,  # Match other charts
            showlegend=False,
            hovermode='x unified',
            margin=dict(t=60, l=80, r=80, b=40),
            xaxis=dict(
                title='Date',
                rangeslider=dict(visible=False),
                rangeselector=dict(
                    buttons=list([
                        dict(
                            count=1,
                            label="1Y",
                            step="year",
                            stepmode="backward"
                        ),
                        dict(
                            count=2,
                            label="2Y",
                            step="year",
                            stepmode="backward"
                        ),
                        dict(
                            count=5,
                            label="5Y",
                            step="year",
                            stepmode="backward"
                        ),
                        dict(
                            count=10,
                            label="10Y",
                            step="year",
                            stepmode="backward"
                        ),
                        dict(
                            step="all",
                            label="All"
                        )
                    ]),
                    bgcolor='rgba(255,255,255,0.9)',
                    activecolor='lightblue',
                    x=0.01,
                    y=1.15,
                    yanchor='top'
                ),
                type="date"
            ),
            yaxis=dict(
                title="Value",
                autorange=True,
                fixedrange=False
            ),
            # This helps with auto-ranging
            datarevision=0
        )

        # Force y-axis to be responsive
        fig.update_yaxes(autorange=True, autorangeoptions=dict(clipmin=0.0, clipmax=1.0))

        return fig

    except Exception as e:
        st.error(f"Error creating actual values chart: {str(e)}")
        return None


def create_momentum_analysis_charts(df, variable_name):
    """Create the 3 momentum analysis charts (WoW changes, magnitude, z-score)"""
    try:
        # Prepare data
        df_momentum = df.copy()
        df_momentum = df_momentum.sort_values('report_date_as_yyyy_mm_dd')

        # Map variable to its change column
        change_col_mapping = {
            'open_interest_all': 'change_in_open_interest_all',
            'noncomm_positions_long_all': 'change_in_noncomm_long_all',
            'noncomm_positions_short_all': 'change_in_noncomm_short_all',
            'comm_positions_long_all': 'change_in_comm_long_all',
            'comm_positions_short_all': 'change_in_comm_short_all',
            'tot_rept_positions_long_all': 'change_in_tot_rept_long_all',
            'tot_rept_positions_short': 'change_in_tot_rept_short',
            'nonrept_positions_long_all': 'change_in_nonrept_long_all',
            'nonrept_positions_short_all': 'change_in_nonrept_short_all',
            'net_noncomm_positions': 'change_in_net_noncomm',
            'net_comm_positions': 'change_in_net_comm'
        }

        # Get the change column
        change_col = change_col_mapping.get(variable_name, '')
        if not change_col or change_col not in df_momentum.columns:
            # For net positions, calculate the change if not available
            if 'net_' in variable_name:
                df_momentum[change_col] = df_momentum[variable_name].diff()
            else:
                return None

        # Get change values
        change_values = df_momentum[change_col].fillna(0)

        # Calculate z-scores using 104-week rolling window (2 years)
        z_scores = pd.Series(index=change_values.index, dtype=float)
        for i in range(len(change_values)):
            if i == 0:
                z_scores.iloc[i] = 0
            else:
                # Use up to 104 weeks of history (2 years)
                lookback_start = max(0, i - 103)
                historical_values = change_values.iloc[lookback_start:i+1]

                if len(historical_values) > 1 and historical_values.std() > 0:
                    mean = historical_values[:-1].mean()  # Exclude current value from mean
                    std = historical_values[:-1].std()    # Exclude current value from std
                    z_scores.iloc[i] = (change_values.iloc[i] - mean) / std
                else:
                    z_scores.iloc[i] = 0

        # Create figure with 3 subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.33, 0.33, 0.34],
            shared_xaxes=True,
            vertical_spacing=0.08,  # Good spacing for 550px height
            subplot_titles=(
                'Week-over-Week Changes',
                'Change Magnitude (Absolute)',
                'Z-Score of Changes'
            )
        )

        # 1. Changes as bars with color coding
        colors = ['green' if x > 0 else 'red' for x in change_values]

        fig.add_trace(
            go.Bar(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=change_values,
                name='Weekly Change',
                marker=dict(color=colors),
                hovertemplate='Date: %{x}<br>Change: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=1, col=1)

        # 2. Absolute magnitude of changes (to see volatility)
        abs_changes = abs(change_values)
        percentile_90 = np.percentile(abs_changes[abs_changes > 0], 90) if len(abs_changes[abs_changes > 0]) > 0 else 0

        fig.add_trace(
            go.Bar(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=abs_changes,
                name='Absolute Change',
                marker=dict(
                    color=abs_changes,
                    colorscale='Viridis',
                    cmin=0,
                    cmax=percentile_90 * 1.2,
                    colorbar=dict(
                        title="Magnitude",
                        x=1.02,
                        len=0.3,
                        y=0.5
                    )
                ),
                hovertemplate='Date: %{x}<br>|Change|: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add percentile lines
        if len(abs_changes[abs_changes > 0]) > 5:
            p50 = np.percentile(abs_changes[abs_changes > 0], 50)
            p90 = np.percentile(abs_changes[abs_changes > 0], 90)

            fig.add_hline(y=p50, line_dash="dash", line_color="gray", line_width=1,
                         annotation_text="Median", annotation_position="right", row=2, col=1)
            fig.add_hline(y=p90, line_dash="dash", line_color="red", line_width=2,
                         annotation_text="90th %ile", annotation_position="right", row=2, col=1)

        # 3. Z-Score time series
        fig.add_trace(
            go.Scatter(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=z_scores,
                mode='lines',
                fill='tozeroy',
                name='Z-Score',
                line=dict(color='purple', width=2),
                fillcolor='rgba(128, 0, 128, 0.2)',
                hovertemplate='Date: %{x}<br>Z-Score: %{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )

        # Add z-score reference lines
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=3, col=1)
        fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1,
                     annotation_text="+2σ", annotation_position="right", row=3, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1,
                     annotation_text="-2σ", annotation_position="right", row=3, col=1)

        # Shade extreme zones
        fig.add_hrect(y0=2, y1=4, fillcolor="red", opacity=0.1, line_width=0, row=3, col=1)
        fig.add_hrect(y0=-4, y1=-2, fillcolor="red", opacity=0.1, line_width=0, row=3, col=1)

        # Update layout
        fig.update_layout(
            height=900,  # Height for 3 subplots with good visibility
            showlegend=False,
            hovermode='x unified',
            margin=dict(t=40, l=80, r=80, b=40),  # Adjusted margins for better space utilization
            xaxis=dict(
                rangeslider=dict(visible=False),
                autorange=True
            ),
            xaxis2=dict(
                rangeslider=dict(visible=False),
                autorange=True
            ),
            xaxis3=dict(
                rangeslider=dict(visible=False),
                title='Date',
                autorange=True
            ),
            yaxis=dict(
                title="Change",
                autorange=True,
                fixedrange=False
            ),
            yaxis2=dict(
                title="Absolute Change",
                autorange=True,
                fixedrange=False,
                rangemode='tozero'
            ),
            yaxis3=dict(
                title="Z-Score",
                zeroline=True,
                autorange=True,
                fixedrange=False,
                range=[-4, 4]
            )
        )

        # Configure all x-axes to be linked
        for i in range(1, 4):
            fig['layout'][f'xaxis{i}']['autorange'] = True
            fig['layout'][f'xaxis{i}']['matches'] = 'x3'

        # Set up auto y-axis scaling
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)

        return fig

    except Exception as e:
        st.error(f"Error creating momentum analysis charts: {str(e)}")
        return None