"""
Percentile analysis charts for CFTC COT data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def create_percentile_chart(df, column, lookback_years=5, chart_type='time_series'):
    """Create percentile chart - time series, distribution, or cumulative curve"""
    try:
        df_pct = df.copy()
        df_pct['year'] = df_pct['report_date_as_yyyy_mm_dd'].dt.year
        current_year = df_pct['year'].max()

        # Always show all data for display
        # The time range buttons in the chart will handle filtering
        df_display = df_pct.copy()

        if chart_type == 'time_series':
            # Calculate rolling percentile rank
            # Use the specified lookback period for the rolling window
            if lookback_years == 'all':
                window_days = None  # Use all available data
            else:
                window_days = 365 * lookback_years  # Use calendar days to match distribution view
            
            df_display['percentile_rank'] = np.nan
            df_display['actual_value'] = df_display[column]
            
            # Calculate percentile for each point
            for idx in range(len(df_display)):
                current_date = df_display.iloc[idx]['report_date_as_yyyy_mm_dd']
                current_value = df_display.iloc[idx][column]
                
                # Define the lookback window
                if window_days is None:
                    # Use all available data up to current date
                    mask = df_pct['report_date_as_yyyy_mm_dd'] <= current_date
                else:
                    # Use fixed window matching distribution view calculation
                    window_start = current_date - pd.DateOffset(years=lookback_years)
                    mask = (df_pct['report_date_as_yyyy_mm_dd'] >= window_start) & \
                           (df_pct['report_date_as_yyyy_mm_dd'] <= current_date)
                
                window_values = df_pct.loc[mask, column].values
                
                if len(window_values) > 1:
                    percentile = (window_values < current_value).sum() / len(window_values) * 100
                    df_display.iloc[idx, df_display.columns.get_loc('percentile_rank')] = percentile

            # Remove NaN values
            df_display = df_display.dropna(subset=['percentile_rank'])

            # Create subplot with mini-map
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05,
                subplot_titles=(
                    f'Percentile Rank (Rolling {lookback_years if lookback_years != "all" else "All Time"} Window)', 
                    'Full Period Navigator'
                )
            )

            # Determine bar width based on data density
            # Use bars only for recent data or sparse data
            days_between_points = (df_display['report_date_as_yyyy_mm_dd'].max() - 
                                  df_display['report_date_as_yyyy_mm_dd'].min()).days / len(df_display)
            
            if days_between_points > 3 or len(df_display) < 200:  # Weekly data or less than 200 points
                # Use bar chart
                fig.add_trace(
                    go.Bar(
                        x=df_display['report_date_as_yyyy_mm_dd'],
                        y=df_display['percentile_rank'],
                        name='Percentile Rank',
                        marker=dict(
                            color=df_display['percentile_rank'],
                            colorscale=[
                                [0, 'darkgreen'],
                                [0.1, 'green'],
                                [0.25, 'lightgreen'],
                                [0.5, 'yellow'],
                                [0.75, 'orange'],
                                [0.9, 'red'],
                                [1, 'darkred']
                            ],
                            cmin=0,
                            cmax=100,
                            colorbar=dict(
                                title="Percentile",
                                tickmode='array',
                                tickvals=[0, 10, 25, 50, 75, 90, 100],
                                ticktext=['0%', '10%', '25%', '50%', '75%', '90%', '100%'],
                                x=1.02
                            )
                        ),
                        text=df_display['actual_value'].round(0).astype(str),
                        textposition='none',
                        hovertemplate='Date: %{x|%Y-%m-%d}<br>Percentile: %{y:.1f}%<br>Value: %{text}<extra></extra>'
                    ),
                    row=1, col=1
                )
            else:
                # Use area chart for dense data
                fig.add_trace(
                    go.Scatter(
                        x=df_display['report_date_as_yyyy_mm_dd'],
                        y=df_display['percentile_rank'],
                        mode='lines',
                        fill='tozeroy',
                        name='Percentile Rank',
                        line=dict(color='blue', width=2),
                        fillcolor='rgba(0, 100, 255, 0.3)',
                        hovertemplate='Date: %{x|%Y-%m-%d}<br>Percentile: %{y:.1f}%<br>Value: ' + 
                                     df_display['actual_value'].round(0).astype(str) + '<extra></extra>'
                    ),
                    row=1, col=1
                )

            # Add reference lines
            fig.add_hline(y=50, line_dash="dash", line_color="black", line_width=1, 
                         annotation_text="50th", annotation_position="right", row=1, col=1)
            fig.add_hline(y=90, line_dash="dot", line_color="red", line_width=2,
                         annotation_text="90th", annotation_position="right", row=1, col=1)
            fig.add_hline(y=10, line_dash="dot", line_color="green", line_width=2,
                         annotation_text="10th", annotation_position="right", row=1, col=1)

            # Add subtle shaded zones
            fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor="red", opacity=0.05, row=1, col=1)
            fig.add_hrect(y0=0, y1=25, line_width=0, fillcolor="green", opacity=0.05, row=1, col=1)

            # Add mini-map navigator showing full data
            fig.add_trace(
                go.Scatter(
                    x=df_display['report_date_as_yyyy_mm_dd'],
                    y=df_display['percentile_rank'],
                    mode='lines',
                    name='',
                    line=dict(color='rgba(0,0,255,0.5)', width=1),
                    fill='tozeroy',
                    fillcolor='rgba(0,0,255,0.1)',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"Historical Percentile: {column.replace('_', ' ').title()}",
                    y=0.95,
                    x=0.5,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=18)
                ),
                hovermode='x unified',
                height=700,
                showlegend=False,
                dragmode='zoom',
                margin=dict(t=120, l=80, r=80, b=80),  # Increase top margin for buttons
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(count=2, label="2Y", step="year", stepmode="backward"),
                            dict(count=5, label="5Y", step="year", stepmode="backward"),
                            dict(count=10, label="10Y", step="year", stepmode="backward"),
                            dict(label="All", step="all")
                        ]),
                        bgcolor='rgba(255,255,255,0.9)',
                        x=0,
                        xanchor='left',
                        y=1.15,
                        yanchor='top',
                        font=dict(size=11)
                    ),
                    type='date'
                ),
                yaxis=dict(
                    title="Percentile Rank (%)",
                    range=[-5, 105],
                    tickmode='array',
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=['0%', '25%', '50%', '75%', '100%']
                ),
                xaxis2=dict(
                    type='date',
                    rangeslider=dict(
                        visible=True,
                        thickness=0.1,
                        bgcolor='rgba(0,0,0,0.05)'
                    )
                ),
                yaxis2=dict(
                    title="",
                    range=[0, 100],
                    visible=False
                )
            )

            # Configure axes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)

        elif chart_type == 'distribution':
            return create_distribution_chart(df_pct, column, lookback_years)

        elif chart_type == 'cumulative':
            return create_cumulative_percentile_chart(df_pct, df, column, lookback_years)

        return fig

    except Exception as e:
        st.error(f"Error creating percentile chart: {str(e)}")
        return None


def create_distribution_chart(df_pct, column, lookback_years):
    """Create distribution chart with adaptive binning"""
    try:
        # Get historical data with dates for time-based coloring
        if lookback_years == 'all':
            df_hist = df_pct[[column, 'report_date_as_yyyy_mm_dd']].dropna()
        else:
            lookback_start = df_pct['report_date_as_yyyy_mm_dd'].max() - pd.DateOffset(years=lookback_years)
            df_hist = df_pct[df_pct['report_date_as_yyyy_mm_dd'] >= lookback_start][[column, 'report_date_as_yyyy_mm_dd']].dropna()
        
        historical_values = df_hist[column].values
        historical_dates = df_hist['report_date_as_yyyy_mm_dd']
        current_value = df_pct[column].iloc[-1] if not df_pct.empty else np.nan
        current_date = df_pct['report_date_as_yyyy_mm_dd'].max()
        
        # Add toggle for histogram type
        col1_toggle, col2_toggle = st.columns([1, 3])
        with col1_toggle:
            use_density = st.toggle("Probability Density", value=True, key="density_toggle")
        
        # Add explainer
        with col2_toggle:
            with st.expander("ℹ️ How to read this chart"):
                if use_density:
                    st.markdown("""
                    **Probability Density View (Adaptive Binning):**
                    
                    The height alone doesn't tell you probability - you need to consider the width of each bar too.
                    
                    For example:
                    - Your tallest bar shows about 15μ (0.000015) density
                    - If each bar covers about 10,000 units on the x-axis
                    - Then probability = 0.000015 × 10,000 = 0.15 = 15% of observations fall in that bar
                    
                    Quick visual interpretation:
                    - Taller bars = More common values (higher concentration)
                    - Shorter bars = Less common values
                    - Multiple peaks = Multiple common value ranges
                    
                    **Freedman-Diaconis Method:**
                    - Bin width = 2 × IQR × n^(-1/3)
                    - IQR (Interquartile Range) is robust to outliers
                    - Automatically balances detail vs noise
                    - More data → narrower bins → more detail
                    - Reveals true distribution shape that fixed bins might hide
                    
                    Key features:
                    - Narrower bins where data is dense (better resolution)
                    - Wider bins where data is sparse (less noise)
                    - Total area under all bars = 1 (100% probability)
                    - Optimal for financial data with outliers
                    """)
                else:
                    st.markdown("""
                    **Percentage View:**
                    
                    The height directly shows what percentage of observations fall in each bar.
                    
                    For example:
                    - A bar with height 5 means 5% of all historical values are in that range
                    - All bars together sum to 100%
                    
                    Quick visual interpretation:
                    - Taller bars = More frequent values in history
                    - Shorter bars = Rare values
                    - You can directly read percentages without calculation
                    """)

        # Create histogram with distribution curve
        fig = go.Figure()

        # Get values from recent weeks
        weeks_data = []
        week_colors = {
            1: 'orange',
            2: 'yellow', 
            3: 'lightgreen',
            4: 'lightcoral'
        }
        
        for weeks_back in [1, 2, 3, 4]:
            target_date = current_date - pd.DateOffset(weeks=weeks_back)
            date_diffs = abs(df_hist['report_date_as_yyyy_mm_dd'] - target_date)
            if len(date_diffs) > 0:
                closest_idx = date_diffs.idxmin()
                value = df_hist.loc[closest_idx, column]
                weeks_data.append((weeks_back, value, week_colors[weeks_back]))
        
        if use_density:
            # Use adaptive binning for probability density
            # Use Freedman-Diaconis rule for bin width
            q75, q25 = np.percentile(historical_values, [75, 25])
            iqr = q75 - q25
            n = len(historical_values)
            
            # Freedman-Diaconis bin width
            if iqr > 0 and n > 0:
                bin_width = 2 * iqr * (n ** (-1/3))
                n_bins = int((historical_values.max() - historical_values.min()) / bin_width)
                # Limit bins to reasonable range
                n_bins = max(10, min(n_bins, 100))
            else:
                n_bins = 30  # fallback
            
            # Calculate histogram to get bin edges
            counts, bin_edges = np.histogram(historical_values, bins=n_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Create color array - highlight bins containing recent week values
            colors = ['lightblue'] * len(counts)
            for weeks_back, value, color in weeks_data:
                # Find which bin contains this week's value
                for i in range(len(bin_edges)-1):
                    if bin_edges[i] <= value <= bin_edges[i+1]:
                        # If already colored, keep the most recent week's color
                        if colors[i] == 'lightblue':
                            colors[i] = color
                        break
            
            # Create histogram with custom colors
            fig.add_trace(go.Bar(
                x=bin_centers,
                y=counts / (np.sum(counts) * np.diff(bin_edges)),  # Convert to density
                width=np.diff(bin_edges),
                name='Historical Distribution',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='Range: %{x}<br>Density: %{y}<extra></extra>'
            ))
        else:
            # Use fixed bins for percentage view
            # Calculate histogram to get bin edges
            counts, bin_edges = np.histogram(historical_values, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Create color array - highlight bins containing recent week values
            colors = ['lightblue'] * len(counts)
            for weeks_back, value, color in weeks_data:
                # Find which bin contains this week's value
                for i in range(len(bin_edges)-1):
                    if bin_edges[i] <= value <= bin_edges[i+1]:
                        # If already colored, keep the most recent week's color
                        if colors[i] == 'lightblue':
                            colors[i] = color
                        break
            
            # Create histogram with custom colors
            fig.add_trace(go.Bar(
                x=bin_centers,
                y=counts / len(historical_values) * 100,  # Convert to percentage
                width=np.diff(bin_edges),
                name='Historical Distribution',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='Range: %{x}<br>Percentage: %{y:.1f}%<extra></extra>'
            ))

        # Calculate statistical measures
        mean_val = np.mean(historical_values)
        median_val = np.median(historical_values)
        std_val = np.std(historical_values)
        
        # Calculate and add KDE curve
        kde = stats.gaussian_kde(historical_values)
        x_range = np.linspace(historical_values.min(), historical_values.max(), 200)
        kde_values = kde(x_range)
        
        # Scale KDE values to match the histogram scale
        if use_density:
            # KDE is already in density scale
            kde_y_values = kde_values
        else:
            # Convert KDE to percentage scale
            # Approximate by scaling to match histogram
            hist_counts, bin_edges = np.histogram(historical_values, bins=50)
            hist_pct = hist_counts / len(historical_values) * 100
            max_hist_pct = max(hist_pct)
            max_kde = max(kde_values)
            kde_y_values = kde_values * (max_hist_pct / max_kde) if max_kde > 0 else kde_values
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_y_values,
            mode='lines',
            name='Distribution Shape',
            line=dict(color='blue', width=2)
        ))
        
        # Add mean line
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="green",
            line_width=2,
            name="Mean"
        )
        
        # Add a dummy trace for mean legend entry
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Mean',
            showlegend=True
        ))
        
        # Add median line
        if abs(median_val - mean_val) > 0.01 * abs(mean_val):  # Only show if meaningfully different
            fig.add_vline(
                x=median_val,
                line_dash="dashdot",
                line_color="purple",
                line_width=2,
                name="Median"
            )
            
            # Add a dummy trace for median legend entry
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color='purple', width=2, dash='dashdot'),
                name='Median',
                showlegend=True
            ))
        
        # Add standard deviation bands
        for i, multiplier in enumerate([1, 2]):
            fig.add_vrect(
                x0=mean_val - multiplier * std_val,
                x1=mean_val + multiplier * std_val,
                fillcolor="gray",
                opacity=0.1 - i * 0.05,
                line_width=0
            )

        # Add current value marker
        if not np.isnan(current_value):
            fig.add_vline(
                x=current_value,
                line_dash="solid",
                line_color="red",
                line_width=3,
                annotation_text=f"Current: {current_value:,.0f}"
            )
        
        # Add legend entries for recent weeks
        for weeks_back, value, color in weeks_data:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=15, color=color, symbol='square'),
                name=f'{weeks_back} Week{"s" if weeks_back > 1 else ""} Ago',
                showlegend=True
            ))

            # Calculate percentile
            percentile = (historical_values < current_value).sum() / len(historical_values) * 100

            # Add percentile annotation
            fig.add_annotation(
                x=current_value,
                y=0,
                text=f"{percentile:.1f}%ile",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                ax=0,
                ay=-40,
                bgcolor="white",
                bordercolor="red",
                borderwidth=2,
                yref="paper"
            )

        # Add percentile markers (only show key ones to avoid clutter)
        key_percentiles = [10, 90]
        for p in key_percentiles:
            value = np.percentile(historical_values, p)
            fig.add_vline(
                x=value,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                annotation_text=f"{p}%"
            )
        
        # Calculate skewness
        skewness = stats.skew(historical_values)
        
        # Add text annotation for stats
        stats_text = f"Skewness: {skewness:.2f}<br>Std Dev: {std_val:,.0f}"
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )

        fig.update_layout(
            title=f"Distribution Analysis: {column.replace('_', ' ').title()} ({lookback_years if lookback_years != 'all' else 'All'} {'Years' if lookback_years != 'all' else 'Time'})",
            xaxis_title="Value",
            yaxis_title="Density" if use_density else "Percentage (%)",
            showlegend=True,
            height=500,
            barmode='overlay',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )

        return fig

    except Exception as e:
        st.error(f"Error creating distribution chart: {str(e)}")
        return None


def create_cumulative_percentile_chart(df_pct, df, column, lookback_years):
    """Create cumulative percentile curve"""
    try:
        # Get historical values based on lookback period
        if lookback_years == 'all':
            historical_values = df_pct[column].dropna().values
        else:
            lookback_start = df_pct['report_date_as_yyyy_mm_dd'].max() - pd.DateOffset(years=lookback_years)
            historical_values = df_pct[df_pct['report_date_as_yyyy_mm_dd'] >= lookback_start][column].dropna().values
        sorted_values = np.sort(historical_values)
        current_value = df[column].iloc[-1] if not df.empty else np.nan

        # Calculate cumulative percentiles
        percentiles = np.arange(0, 100, 100 / len(sorted_values))

        fig = go.Figure()

        # Add the main percentile curve
        fig.add_trace(go.Scatter(
            x=sorted_values,
            y=percentiles,
            mode='lines',
            name='Percentile Curve',
            line=dict(color='blue', width=3),
            hovertemplate='Value: %{x:,.0f}<br>Percentile: %{y:.1f}%<extra></extra>'
        ))

        # Add current value
        if not np.isnan(current_value):
            current_percentile = (historical_values < current_value).sum() / len(historical_values) * 100

            # Add current value point
            fig.add_trace(go.Scatter(
                x=[current_value],
                y=[current_percentile],
                mode='markers',
                name=f'Current Value',
                marker=dict(size=12, color='red', symbol='circle'),
                hovertemplate=f'Current Value: {current_value:,.0f}<br>Percentile: {current_percentile:.1f}%<extra></extra>'
            ))

            # Add dotted lines from axes to current point
            fig.add_shape(
                type="line",
                x0=sorted_values.min(),
                y0=current_percentile,
                x1=current_value,
                y1=current_percentile,
                line=dict(color="red", width=1, dash="dot"),
            )
            fig.add_shape(
                type="line",
                x0=current_value,
                y0=0,
                x1=current_value,
                y1=current_percentile,
                line=dict(color="red", width=1, dash="dot"),
            )

        # Add reference lines for key percentiles
        for p in [10, 25, 50, 75, 90]:
            value_at_p = np.percentile(sorted_values, p)
            fig.add_hline(
                y=p,
                line_dash="dot",
                line_color="lightgray",
                opacity=0.5,
                annotation_text=f"{p}th %ile",
                annotation_position="right"
            )

        # Add shaded zones
        fig.add_hrect(y0=0, y1=10, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=90, y1=100, fillcolor="red", opacity=0.1, line_width=0)

        fig.update_layout(
            title=f"Cumulative Percentile Curve: {column.replace('_', ' ').title()} ({lookback_years if lookback_years != 'all' else 'All'} {'Years' if lookback_years != 'all' else 'Time'})",
            xaxis_title=f"{column.replace('_', ' ').title()} Value",
            yaxis_title="Percentile (%)",
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            height=400,
            hovermode='closest'
        )

        return fig

    except Exception as e:
        st.error(f"Error creating cumulative percentile chart: {str(e)}")
        return None