import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from scipy import stats  # Add this line


@st.cache_data
def plot_historical_range(data, selected_column, time_range='all'):
    """
    Create a historical range plot for a selected column showing:
    - Historical range (min to max)
    - Historical average
    - Current year data
    - Previous year data

    Parameters:
    data (pd.DataFrame): The CFTC data, must contain 'date' column
    selected_column (str): The column to analyze
    time_range (str): 'all', '5y', or '10y' to filter historical data
    """
    if 'date' not in data.columns or selected_column not in data.columns:
        st.error(f"Required columns missing from data: need 'date' and '{selected_column}'")
        return

    # Ensure data is sorted by date
    data = data.sort_values('date')

    # Make a copy to avoid modifications to original data
    df = data.copy()

    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Add year and month columns for aggregation
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Get current year and previous year
    current_year = df['year'].max()
    prev_year = current_year - 1

    # Filter data based on time range
    historical_df = df.copy()

    if time_range == '5y':
        # Keep only last 5 years (current, prev, and 3 more)
        min_year = current_year - 4
        range_label = f"Last 5 Years ({min_year}-{current_year})"
        historical_df = historical_df[historical_df['year'] >= min_year]
    elif time_range == '10y':
        # Keep only last 10 years
        min_year = current_year - 9
        range_label = f"Last 10 Years ({min_year}-{current_year})"
        historical_df = historical_df[historical_df['year'] >= min_year]
    else:  # 'all'
        min_year = historical_df['year'].min()
        range_label = f"All Time ({min_year}-{current_year})"

    # Get the range of years for historical data (excluding current and previous)
    historical_years = sorted(historical_df['year'].unique())
    if current_year in historical_years:
        historical_years.remove(current_year)
    if prev_year in historical_years:
        historical_years.remove(prev_year)

    if len(historical_years) < 1:
        st.warning(
            f"Not enough historical data for {selected_column} with selected time range. Need data from at least 3 different years.")
        return

    # Get the earliest year for the range label
    earliest_year = min(historical_years)

    # Create monthly aggregations for range, current year, and previous year
    monthly_data = {}

    # For each month, get min, max, and average from historical years
    for month in range(1, 13):
        month_data = historical_df[(historical_df['month'] == month) & historical_df['year'].isin(historical_years)][
            selected_column]
        if not month_data.empty:
            monthly_data[month] = {
                'min': month_data.min(),
                'max': month_data.max(),
                'avg': month_data.mean()
            }
        else:
            # Use the previous month's data if this month is missing
            if month > 1 and month - 1 in monthly_data:
                monthly_data[month] = monthly_data[month - 1]
            else:
                # Use default values if no previous month
                monthly_data[month] = {'min': np.nan, 'max': np.nan, 'avg': np.nan}

    # Extract data for current year and previous year
    current_year_data = df[df['year'] == current_year].sort_values('month')
    prev_year_data = df[df['year'] == prev_year].sort_values('month')

    # Prepare x-axis labels (month names)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Prepare data for plotting
    x_vals = months
    min_vals = [monthly_data[m + 1]['min'] if m + 1 in monthly_data else np.nan for m in range(12)]
    max_vals = [monthly_data[m + 1]['max'] if m + 1 in monthly_data else np.nan for m in range(12)]
    avg_vals = [monthly_data[m + 1]['avg'] if m + 1 in monthly_data else np.nan for m in range(12)]

    # Create arrays for current and previous year data, filling missing months with NaN
    current_year_months = current_year_data['month'].values if not current_year_data.empty else []
    current_year_vals = [np.nan] * 12
    for i, month in enumerate(current_year_months):
        if i < len(current_year_data):
            month_idx = month - 1  # Convert 1-based month to 0-based index
            if 0 <= month_idx < 12:  # Ensure index is valid
                current_year_vals[month_idx] = current_year_data.iloc[i][selected_column]

    prev_year_months = prev_year_data['month'].values if not prev_year_data.empty else []
    prev_year_vals = [np.nan] * 12
    for i, month in enumerate(prev_year_months):
        if i < len(prev_year_data):
            month_idx = month - 1  # Convert 1-based month to 0-based index
            if 0 <= month_idx < 12:  # Ensure index is valid
                prev_year_vals[month_idx] = prev_year_data.iloc[i][selected_column]

    # Create figure
    fig = go.Figure()

    # Add range area (min to max)
    fig.add_trace(
        go.Scatter(
            x=x_vals + x_vals[::-1],  # x values for forward and backward path
            y=max_vals + min_vals[::-1],  # y values for forward and backward path
            fill='toself',
            fillcolor='rgba(173, 216, 230, 0.5)',  # Light blue with transparency
            line=dict(color='rgba(173, 216, 230, 0)'),  # Transparent line
            name=f'Range {range_label}',
            showlegend=True
        )
    )

    # Add average line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=avg_vals,
            mode='lines',
            line=dict(color='rgba(70, 130, 180, 0.8)', width=2),  # Steel blue
            name=f'Average {range_label}'
        )
    )

    # Add previous year line
    if not all(np.isnan(prev_year_vals)):
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=prev_year_vals,
                mode='lines',
                line=dict(color='rgba(148, 0, 211, 0.8)', width=2),  # Purple
                name=f'{prev_year}'
            )
        )

    # Add current year line
    if not all(np.isnan(current_year_vals)):
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=current_year_vals,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.8)', width=3),  # Red
                name=f'{current_year}'
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{selected_column} - Monthly Seasonal Patterns ({range_label})",
        xaxis_title="Month",
        yaxis_title=selected_column,
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template="plotly_dark" if st.session_state.get('theme', '') == 'dark' else "plotly_white"
    )

    return fig


@st.cache_data
def plot_percentile_curve(data, selected_column, time_range='all'):
    """
    Create a percentile curve plot showing the distribution of values
    across the historical dataset, with current value highlighted.

    Parameters:
    data (pd.DataFrame): The CFTC data
    selected_column (str): The column to analyze
    time_range (str): 'all', '5y', or '10y' to filter historical data
    """
    if selected_column not in data.columns:
        return None

    # Make a copy to avoid modifications to original data
    df = data.copy()

    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Add year column for filtering
    df['year'] = df['date'].dt.year

    # Get current year for filtering and highlighting
    current_year = df['year'].max()

    # Filter data based on time range
    if time_range == '5y':
        # Keep only last 5 years
        min_year = current_year - 4
        range_label = f"Last 5 Years ({min_year}-{current_year})"
        df = df[df['year'] >= min_year]
    elif time_range == '10y':
        # Keep only last 10 years
        min_year = current_year - 9
        range_label = f"Last 10 Years ({min_year}-{current_year})"
        df = df[df['year'] >= min_year]
    else:  # 'all'
        min_year = df['year'].min()
        range_label = f"All Time ({min_year}-{current_year})"

    # Extract the values for the selected column
    values = df[selected_column].dropna()

    if len(values) < 5:
        return None

    # Sort values for percentile calculation
    sorted_values = sorted(values)

    # Calculate percentiles (0 to 100)
    percentiles = np.arange(0, 101, 1)
    percentile_values = np.percentile(sorted_values, percentiles)

    # Get the most recent value for highlighting
    most_recent = df.loc[df['date'].idxmax(), selected_column]

    # Calculate the percentile rank of the most recent value
    most_recent_percentile = stats.percentileofscore(sorted_values, most_recent)

    # Create the figure
    fig = go.Figure()

    # Add the percentile curve
    fig.add_trace(
        go.Scatter(
            x=percentile_values,
            y=percentiles,
            mode='lines',
            line=dict(color='royalblue', width=3),
            name='Percentile Curve'
        )
    )

    # Add the most recent value
    fig.add_trace(
        go.Scatter(
            x=[most_recent],
            y=[most_recent_percentile],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                line=dict(color='black', width=2)
            ),
            name=f'Current Value ({most_recent:.0f})'
        )
    )

    # Add horizontal lines for key percentiles
    for p in [10, 25, 50, 75, 90]:
        fig.add_shape(
            type="line",
            x0=min(sorted_values),
            x1=max(sorted_values),
            y0=p,
            y1=p,
            line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"),
        )

    # Update layout
    fig.update_layout(
        title=f"{selected_column} - Percentile Distribution ({range_label})",
        xaxis_title=f"{selected_column} Value",
        yaxis_title="Percentile (%)",
        height=400,
        hovermode="closest",
        template="plotly_dark" if st.session_state.get('theme', '') == 'dark' else "plotly_white",
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 10, 25, 50, 75, 90, 100],
            ticktext=['0%', '10%', '25%', '50%', '75%', '90%', '100%']
        )
    )

    # Add annotations for key percentiles
    fig.add_annotation(
        x=percentile_values[50],
        y=50,
        text="Median",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=-40
    )

    return fig


def display_historical_analysis(data, columns):
    """
    Display the historical range analysis for a selected column

    Parameters:
    data (pd.DataFrame): The CFTC data
    columns (list): List of columns to choose from
    """
    st.subheader("Seasonal Patterns Analysis")
    # Filter columns to only include numerical ones that make sense to plot
    if 'date' in columns:
        columns = [col for col in columns if col not in ['None', 'contract_code', 'type', 'date']]

    if not columns:
        st.info("No suitable columns found for seasonal analysis.")
        return

    # Create a row with column selection and time range options
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create a selectbox for choosing which column to analyze
        analysis_col = st.selectbox(
            "Select a column for seasonal analysis:",
            options=columns
        )

    with col2:
        # Add time range selection
        time_range = st.radio(
            "Historical Range:",
            options=["Last 5 Years", "Last 10 Years", "All Time"],
            horizontal=True,
            index=1  # Default to Last 10 Years
        )

        # Convert selection to parameter value
        range_param = {
            "Last 5 Years": "5y",
            "Last 10 Years": "10y",
            "All Time": "all"
        }[time_range]

    # Plot the historical range for the selected column
    if analysis_col:
        with st.spinner(f"Analyzing patterns for {analysis_col}..."):
            # Create a container for both plots
            plot_container = st.container()

            with plot_container:
                # Create two columns for the plots
                fig_col1, fig_col2 = st.columns([2, 1])

                with fig_col1:
                    # Display the seasonal patterns plot
                    st.subheader("Seasonal Patterns")
                    fig_seasonal = plot_historical_range(data, analysis_col, range_param)
                    if fig_seasonal is not None:
                        st.plotly_chart(fig_seasonal, use_container_width=True)

                        # Add a brief description of the chart
                        st.write(f"""
                        This chart shows seasonal patterns:
                        - Blue band: historical range (min to max) by month
                        - Blue line: historical average by month
                        - Purple line: last year's values
                        - Red line: current year values
                        """)

                with fig_col2:
                    # Display the percentile distribution plot
                    st.subheader("Percentile Distribution")
                    fig_percentile = plot_percentile_curve(data, analysis_col, range_param)
                    if fig_percentile is not None:
                        st.plotly_chart(fig_percentile, use_container_width=True)

                        # Add a brief description of the chart
                        st.write(f"""
                        This chart shows the percentile distribution:
                        - Y-axis: percentile rank (0-100%)
                        - X-axis: actual values of {analysis_col}
                        - Red dot: current value and its percentile
                        - Use this to see where current values rank historically
                        """)
                    else:
                        st.info("Not enough data to create percentile distribution plot.")