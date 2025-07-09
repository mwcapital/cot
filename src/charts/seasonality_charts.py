"""
Seasonality analysis charts for CFTC COT Dashboard
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats


def create_seasonality_chart(df, column, lookback_years=5, show_previous_year=True, zone_type='percentile'):
    """Create seasonality chart showing historical patterns with smooth zones"""
    try:
        # Add day of year column
        df_season = df.copy()
        df_season['day_of_year'] = df_season['report_date_as_yyyy_mm_dd'].dt.dayofyear
        df_season['year'] = df_season['report_date_as_yyyy_mm_dd'].dt.year

        # Determine lookback period
        if lookback_years == 'all':
            start_year = df_season['year'].min()
        else:
            start_year = df_season['year'].max() - lookback_years + 1

        # Filter for lookback period
        df_lookback = df_season[df_season['year'] >= start_year]

        # Create figure
        fig = go.Figure()

        # If lookback is 5 years and not 'all', just plot the raw data for each year
        if lookback_years == 5:
            # Create a reference year for x-axis (using current year for display)
            current_year = pd.Timestamp.now().year
            
            # Plot each year's data overlaid on the same annual timeline
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for idx, year in enumerate(sorted(df_lookback['year'].unique(), reverse=True)):
                df_year = df_lookback[df_lookback['year'] == year].copy()
                if not df_year.empty:
                    # Create display dates using a common year for overlay
                    df_year['display_date'] = pd.to_datetime(
                        df_year['day_of_year'].astype(str) + f'-{current_year}',
                        format='%j-%Y'
                    )
                    
                    # Use different styling for current year
                    is_current_year = (year == df_lookback['year'].max())
                    
                    fig.add_trace(go.Scatter(
                        x=df_year['display_date'],
                        y=df_year[column],
                        mode='lines+markers' if is_current_year else 'lines',
                        name=str(year),
                        line=dict(
                            width=3 if is_current_year else 2,
                            color=colors[idx % len(colors)]
                        ),
                        marker=dict(size=4) if is_current_year else None
                    ))
            
            # Update layout for seasonality view
            fig.update_layout(
                title=f'{column.replace("_", " ").title()} - Last {lookback_years} Years Seasonal Comparison',
                xaxis_title='Month',
                yaxis_title=column.replace('_', ' ').title(),
                hovermode='x unified',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                )
            )
            
            # Format x-axis to show months
            fig.update_xaxes(
                tickformat="%b",
                dtick="M1",
                ticklabelmode="period",
                rangeslider_visible=True,
                rangeslider_thickness=0.05
            )
            
            return fig

        # Otherwise, continue with the original statistical analysis
        # Calculate statistics for each day of year
        daily_stats = df_lookback.groupby('day_of_year')[column].agg([
            'mean', 'std', 'count',
            ('q10', lambda x: x.quantile(0.1)),
            ('q25', lambda x: x.quantile(0.25)),
            ('q50', lambda x: x.quantile(0.5)),
            ('q75', lambda x: x.quantile(0.75)),
            ('q90', lambda x: x.quantile(0.9))
        ]).reset_index()

        # Sort by day of year
        daily_stats = daily_stats.sort_values('day_of_year')

        # Apply smoothing to make cleaner zones
        from scipy.signal import savgol_filter
        window_length = min(31, len(daily_stats) // 2 * 2 - 1)  # Must be odd
        if window_length >= 5:  # Only smooth if we have enough data
            for col in ['mean', 'q10', 'q25', 'q50', 'q75', 'q90']:
                if col in daily_stats.columns:
                    daily_stats[f'{col}_smooth'] = savgol_filter(
                        daily_stats[col].fillna(method='ffill').fillna(method='bfill'),
                        window_length=window_length,
                        polyorder=3
                    )
        else:
            # If not enough data for smoothing, use original values
            for col in ['mean', 'q10', 'q25', 'q50', 'q75', 'q90']:
                if col in daily_stats.columns:
                    daily_stats[f'{col}_smooth'] = daily_stats[col]

        # Create figure
        fig = go.Figure()

        # Determine which smoothed columns to use based on zone_type
        if zone_type == 'percentile':
            upper_col = 'q90_smooth'
            lower_col = 'q10_smooth'
            upper_mid_col = 'q75_smooth'
            lower_mid_col = 'q25_smooth'
            zone_label = 'Percentile Zones'
        else:  # standard deviation
            daily_stats['upper_2std'] = daily_stats['mean'] + 2 * daily_stats['std']
            daily_stats['lower_2std'] = daily_stats['mean'] - 2 * daily_stats['std']
            daily_stats['upper_1std'] = daily_stats['mean'] + daily_stats['std']
            daily_stats['lower_1std'] = daily_stats['mean'] - daily_stats['std']
            
            # Smooth std-based zones
            if window_length >= 5:
                for col in ['upper_2std', 'lower_2std', 'upper_1std', 'lower_1std']:
                    daily_stats[f'{col}_smooth'] = savgol_filter(
                        daily_stats[col].fillna(method='ffill').fillna(method='bfill'),
                        window_length=window_length,
                        polyorder=3
                    )
            else:
                for col in ['upper_2std', 'lower_2std', 'upper_1std', 'lower_1std']:
                    daily_stats[f'{col}_smooth'] = daily_stats[col]
            
            upper_col = 'upper_2std_smooth'
            lower_col = 'lower_2std_smooth'
            upper_mid_col = 'upper_1std_smooth'
            lower_mid_col = 'lower_1std_smooth'
            zone_label = 'Standard Deviation Zones'

        # Create x-axis with dates for better display
        current_year = pd.Timestamp.now().year
        daily_stats['display_date'] = pd.to_datetime(
            daily_stats['day_of_year'].astype(str) + f'-{current_year}',
            format='%j-%Y'
        )

        # Add shaded zones
        # Outer zone (90-10 percentiles or ±2 std)
        fig.add_trace(go.Scatter(
            x=daily_stats['display_date'],
            y=daily_stats[upper_col],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=daily_stats['display_date'],
            y=daily_stats[lower_col],
            mode='lines',
            line=dict(width=0),
            name=zone_label,
            fill='tonexty',
            fillcolor='rgba(128, 128, 255, 0.1)',
            hoverinfo='skip'
        ))

        # Inner zone (75-25 percentiles or ±1 std)
        fig.add_trace(go.Scatter(
            x=daily_stats['display_date'],
            y=daily_stats[upper_mid_col],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=daily_stats['display_date'],
            y=daily_stats[lower_mid_col],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            fill='tonexty',
            fillcolor='rgba(128, 128, 255, 0.2)',
            hoverinfo='skip'
        ))

        # Add mean/median line
        center_col = 'q50_smooth' if zone_type == 'percentile' else 'mean_smooth'
        fig.add_trace(go.Scatter(
            x=daily_stats['display_date'],
            y=daily_stats[center_col],
            mode='lines',
            name='Historical Average' if zone_type == 'std' else 'Historical Median',
            line=dict(color='blue', width=2, dash='dash')
        ))

        # Get current year data
        current_year_actual = df_season['year'].max()
        df_current = df_season[df_season['year'] == current_year_actual].copy()
        
        if not df_current.empty:
            # Create display dates for current year
            df_current['display_date'] = pd.to_datetime(
                df_current['day_of_year'].astype(str) + f'-{current_year}',
                format='%j-%Y'
            )
            
            # Add current year line
            fig.add_trace(go.Scatter(
                x=df_current['display_date'],
                y=df_current[column],
                mode='lines+markers',
                name=f'Current Year ({current_year_actual})',
                line=dict(color='red', width=3),
                marker=dict(size=4)
            ))

        # Add previous year if requested
        if show_previous_year and current_year_actual > start_year:
            df_previous = df_season[df_season['year'] == current_year_actual - 1].copy()
            if not df_previous.empty:
                df_previous['display_date'] = pd.to_datetime(
                    df_previous['day_of_year'].astype(str) + f'-{current_year}',
                    format='%j-%Y'
                )
                
                fig.add_trace(go.Scatter(
                    x=df_previous['display_date'],
                    y=df_previous[column],
                    mode='lines',
                    name=f'Previous Year ({current_year_actual - 1})',
                    line=dict(color='orange', width=2, dash='dot'),
                    opacity=0.7
                ))

        # Update layout
        lookback_text = "All Years" if lookback_years == 'all' else f"{lookback_years} Year"
        fig.update_layout(
            title=f'Seasonality Analysis: {column.replace("_", " ").title()} ({lookback_text} Lookback)',
            xaxis_title='Month',
            yaxis_title=column.replace('_', ' ').title(),
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Format x-axis to show months
        fig.update_xaxes(
            tickformat="%b",
            dtick="M1",
            ticklabelmode="period"
        )

        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05
        )

        return fig

    except Exception as e:
        st.error(f"Error creating seasonality chart: {str(e)}")
        return None