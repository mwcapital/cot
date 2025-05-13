import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime


@st.cache_data
def plot_historical_range(data, selected_column):
    """
    Create a historical range plot for a selected column showing:
    - Historical range (min to max)
    - Historical average
    - Current year data
    - Previous year data

    Parameters:
    data (pd.DataFrame): The CFTC data, must contain 'date' column
    selected_column (str): The column to analyze
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

    # Get the range of years for historical data (excluding current and previous)
    historical_years = sorted(df['year'].unique())
    if current_year in historical_years:
        historical_years.remove(current_year)
    if prev_year in historical_years:
        historical_years.remove(prev_year)

    if len(historical_years) < 1:
        st.warning(f"Not enough historical data for {selected_column}. Need data from at least 3 different years.")
        return

    # Get the earliest year for the range label
    earliest_year = min(historical_years)

    # Create monthly aggregations for range, current year, and previous year
    monthly_data = {}

    # For each month, get min, max, and average from historical years
    for month in range(1, 13):
        month_data = df[(df['month'] == month) & df['year'].isin(historical_years)][selected_column]
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
            name=f'Range {earliest_year}-{current_year - 1}',
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
            name=f'Average {earliest_year}-{current_year - 1}'
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
        title=f"{selected_column} - Monthly Seasonal Patterns",
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

    # Create a selectbox for choosing which column to analyze
    analysis_col = st.selectbox(
        "Select a column for seasonal analysis:",
        options=columns
    )

    # Plot the historical range for the selected column
    if analysis_col:
        with st.spinner(f"Analyzing seasonal patterns for {analysis_col}..."):
            fig = plot_historical_range(data, analysis_col)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

                # Add a brief description of the chart
                st.write("""
                This chart shows seasonal patterns for the selected data series:
                - The blue band shows the historical range (min to max) for each month
                - The blue line shows the historical average for each month
                - The purple line shows last year's values
                - The red line shows current year values
                """)