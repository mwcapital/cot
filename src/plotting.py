import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime


@st.cache_data
def calculate_correct_pct_change(series):
    pct_changes = []
    for i in range(1, len(series)):
        prev_val = series.iloc[i - 1]
        current_val = series.iloc[i]

        # If both values have the same sign
        if (prev_val >= 0 and current_val >= 0) or (prev_val <= 0 and current_val <= 0):
            # For negative values, a more negative value means decrease
            if prev_val < 0:
                # Calculate the absolute change
                abs_change = abs(current_val) - abs(prev_val)
                # A positive abs_change means the value became more negative
                pct_change = -(abs_change / abs(prev_val) * 100) if prev_val != 0 else 0
            else:
                # Normal calculation for positive values
                pct_change = ((current_val - prev_val) / abs(prev_val) * 100) if prev_val != 0 else 0
        else:
            # When values cross zero, use absolute difference
            pct_change = ((current_val - prev_val) / abs(prev_val) * 100) if prev_val != 0 else 0

        pct_changes.append(pct_change)

    # Add NaN for the first row where there's no previous value
    pct_changes = [float('nan')] + pct_changes
    return pd.Series(pct_changes, index=series.index).round(1)


@st.cache_data
def create_hover_text(row, col):
    date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
    value_str = f"{row[col]:,.0f}"

    # Change values
    change = row[f'{col}_change']
    change_pct = row[f'{col}_change_pct']
    change_str = f"Change: {change:+,.0f} ({change_pct:+.1f}%)" if not pd.isna(change) else "Change: N/A"

    # Percentile ranks
    pct_ytd = row[f'{col}_pct_ytd']
    pct_1yr = row[f'{col}_pct_1yr']
    pct_2yr = row[f'{col}_pct_2yr']
    pct_5yr = row[f'{col}_pct_5yr']
    pct_10yr = row[f'{col}_pct_10yr'] if f'{col}_pct_10yr' in row else np.nan  # Add 10Y percentile

    pct_ytd_str = f"YTD Percentile: {pct_ytd:.1f}%" if not pd.isna(pct_ytd) else "YTD Percentile: N/A"
    pct_1yr_str = f"1Y Percentile: {pct_1yr:.1f}%" if not pd.isna(pct_1yr) else "1Y Percentile: N/A"
    pct_2yr_str = f"2Y Percentile: {pct_2yr:.1f}%" if not pd.isna(pct_2yr) else "2Y Percentile: N/A"
    pct_5yr_str = f"5Y Percentile: {pct_5yr:.1f}%" if not pd.isna(pct_5yr) else "5Y Percentile: N/A"
    pct_10yr_str = f"10Y Percentile: {pct_10yr:.1f}%" if not pd.isna(pct_10yr) else "10Y Percentile: N/A"  # Add this line

    hover_text = f"<b>{date_str}</b><br>{col}: {value_str}<br>{change_str}<br>{pct_ytd_str}<br>{pct_1yr_str}<br>{pct_2yr_str}<br>{pct_5yr_str}<br>{pct_10yr_str}"
    return hover_text


@st.cache_data
def get_color_for_column(column_name):
    column_lower = column_name.lower()

    # Net position colors - using a distinct color scheme
    if 'commercial net' in column_lower:
        return '#d62728'  # Red for Commercial Net
    elif 'large speculator net' in column_lower or 'money_manager net' in column_lower:
        return '#1f77b4'  # Blue for Large Speculator Net
    elif 'small speculator net' in column_lower or 'non_reportable net' in column_lower:
        return '#B8860B'  # Dark goldenrod (burnt yellow)
    elif 'other reportables net' in column_lower:
        return '#9467bd'  # Purple for Other Reportables Net
    elif 'swap dealer net' in column_lower:
        return '#ff7f0e'  # Orange for Swap Dealer Net

    # Original colors for standard columns with long/short differentiation
    elif 'non_commercial' in column_lower or 'money_manager' in column_lower:
        if 'longs' in column_lower:
            return '#1f77b4'  # Standard blue for longs
        elif 'shorts' in column_lower:
            return '#7bafd4'  # Lighter blue for shorts
        else:
            return '#1f77b4'  # Default blue for other variations
    elif 'commercial' in column_lower or 'producer' in column_lower:
        if 'longs' in column_lower:
            return '#d62728'  # Standard red for longs
        elif 'shorts' in column_lower:
            return '#ff9999'  # Lighter red for shorts
        else:
            return '#d62728'  # Default red for other variations
    elif 'non_reportable' in column_lower:
        if 'longs' in column_lower:
            return '#B8860B'  # Dark goldenrod for longs
        elif 'shorts' in column_lower:
            return '#DAA520'  # Regular goldenrod (lighter) for shorts
        else:
            return '#B8860B'  # Default dark goldenrod for other variations
    elif 'swap' in column_lower:
        if 'longs' in column_lower:
            return '#ff7f0e'  # Standard orange for longs
        elif 'shorts' in column_lower:
            return '#ffbb78'  # Lighter orange for shorts
        else:
            return '#ff7f0e'  # Default orange for other variations
    elif 'dealer' in column_lower:
        if 'longs' in column_lower:
            return '#2ca02c'  # Standard green for longs
        elif 'shorts' in column_lower:
            return '#98df8a'  # Lighter green for shorts
        else:
            return '#2ca02c'  # Default green for other variations
    elif 'other' in column_lower:
        if 'longs' in column_lower:
            return '#9467bd'  # Standard purple for longs
        elif 'shorts' in column_lower:
            return '#c5b0d5'  # Lighter purple for shorts
        else:
            return '#9467bd'  # Default purple for other variations
    else:
        return None  # Use default Plotly colors


@st.cache_data
def process_cftc_data(data, is_all_data=True):
    """
    Process CFTC data and calculate net positions only for ALL data

    Parameters:
    data (pd.DataFrame): The CFTC data to process
    is_all_data (bool): Whether this is ALL data (True) or CHG data (False)
    """
    # Ensure data is sorted by date
    data = data.sort_values('date')

    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    # Fill missing values with previous values (for continuous visualization)
    data = data.fillna(method='ffill')

    # Only calculate net positions if this is ALL data, not CHG data
    if is_all_data:
        # Add net position columns based on dataset type
        # For QDL/FON format (disaggregated)
        if ('producer_merchant_processor_user_longs' in data.columns and
                'producer_merchant_processor_user_shorts' in data.columns):

            # Add net position columns for FON format
            data['Commercial Net'] = data['producer_merchant_processor_user_longs'] - data[
                'producer_merchant_processor_user_shorts']

            if 'money_manager_longs' in data.columns and 'money_manager_shorts' in data.columns:
                data['Large Speculator Net'] = data['money_manager_longs'] - data['money_manager_shorts']

            if 'other_reportable_longs' in data.columns and 'other_reportable_shorts' in data.columns:
                data['Other Reportables Net'] = data['other_reportable_longs'] - data['other_reportable_shorts']

            if 'swap_dealer_longs' in data.columns and 'swap_dealer_shorts' in data.columns:
                data['Swap Dealer Net'] = data['swap_dealer_longs'] - data['swap_dealer_shorts']

        # For QDL/LFON format (legacy)
        elif ('commercial_longs' in data.columns and 'commercial_shorts' in data.columns):

            # Add net position columns for LFON format
            data['Commercial Net'] = data['commercial_longs'] - data['commercial_shorts']

            if 'non_commercial_longs' in data.columns and 'non_commercial_shorts' in data.columns:
                data['Large Speculator Net'] = data['non_commercial_longs'] - data['non_commercial_shorts']

            if 'non_reportable_longs' in data.columns and 'non_reportable_shorts' in data.columns:
                data['Small Speculator Net'] = data['non_reportable_longs'] - data['non_reportable_shorts']

        # If 'spreading' is available for either format, subtract it from the Large Speculator Net
        if 'spreading' in data.columns and 'Large Speculator Net' in data.columns:
            data['Large Speculator Net'] = data['Large Speculator Net'] - data['spreading']

    return data


# Keep all the other cached functions the same...
@st.cache_data
def calculate_data_changes(data):
    # Calculate change from previous week for all numerical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # Create new DataFrame with changes
    data_change = data.copy()
    for col in numeric_cols:
        # Calculate absolute change
        data_change[f'{col}_change'] = data_change[col].diff()
        # Apply the custom percentage calculation
        data_change[f'{col}_change_pct'] = calculate_correct_pct_change(data_change[col])

        # Calculate percentile ranks for different time periods
        # Add 10-year percentile (52 weeks * 10 = 520)
        if len(data) >= 520:  # If we have at least 10 years of data
            data_change[f'{col}_pct_10yr'] = data_change[col].rolling(520).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100).round(1)
        else:
            data_change[f'{col}_pct_10yr'] = np.nan

        if len(data) >= 260:  # If we have at least 5 years of data (52 weeks * 5)
            data_change[f'{col}_pct_5yr'] = data_change[col].rolling(260).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100).round(1)
        else:
            data_change[f'{col}_pct_5yr'] = np.nan

        if len(data) >= 104:  # If we have at least 2 years of data
            data_change[f'{col}_pct_2yr'] = data_change[col].rolling(104).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100).round(1)
        else:
            data_change[f'{col}_pct_2yr'] = np.nan

        if len(data) >= 52:  # If we have at least 1 year of data
            data_change[f'{col}_pct_1yr'] = data_change[col].rolling(52).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100).round(1)
        else:
            data_change[f'{col}_pct_1yr'] = np.nan

        # Year to date percentile (using first date of current year as reference)
        current_year = data['date'].max().year
        ytd_start = pd.to_datetime(f'{current_year}-01-01')
        ytd_data = data[data['date'] >= ytd_start]

        if not ytd_data.empty and len(ytd_data) > 1:
            # Create a temporary series for YTD values
            ytd_values = ytd_data[col]

            # Calculate percentile rank within YTD values
            data_change.loc[ytd_data.index, f'{col}_pct_ytd'] = ytd_values.rank(pct=True) * 100
        else:
            data_change[f'{col}_pct_ytd'] = np.nan

    return data_change


@st.cache_data
def filter_data_by_date_range(data, data_change, start_date, end_date):
    """Filter data based on selected date range"""
    plot_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    plot_data_change = data_change[(data_change['date'] >= start_date) & (data_change['date'] <= end_date)]
    return plot_data, plot_data_change


@st.cache_data
def prepare_hover_texts(plot_data_change, selected_cols):
    """Prepare hover texts for each data point"""
    for col in selected_cols:
        plot_data_change[f'{col}_hover'] = plot_data_change.apply(
            lambda row: create_hover_text(row, col), axis=1
        )
    return plot_data_change


@st.cache_data
def create_annotations(plot_data_change, selected_cols, num_periods, separate_plots, show_changes):
    """Create annotations for the change percentages"""
    annotations = []

    # Only create annotations if show_changes is True AND there are 90 or fewer periods
    if show_changes and num_periods <= 90:
        for col in selected_cols:
            for i, (date, value, change_pct) in enumerate(zip(
                    plot_data_change['date'],
                    plot_data_change[col],
                    plot_data_change[f'{col}_change_pct']
            )):
                if not pd.isna(change_pct):
                    # Format the change percentage
                    change_text = f"{change_pct:+.1f}%"

                    # Create annotation with color based on value
                    annotation = dict(
                        x=date,
                        y=value,
                        text=change_text,
                        showarrow=False,
                        font=dict(
                            size=12,
                            color="green" if change_pct > 0 else "red"
                        ),
                        xanchor="center",
                        yanchor="bottom",
                    )

                    if separate_plots:
                        # For separate plots, specify which subplot
                        subplot_idx = selected_cols.index(col) + 1
                        annotation["xref"] = f"x{subplot_idx}" if subplot_idx > 1 else "x"
                        annotation["yref"] = f"y{subplot_idx}" if subplot_idx > 1 else "y"

                    annotations.append(annotation)
    return annotations


def plot_cftc_data(data):
    """
    Create interactive plots for CFTC data with week-over-week changes and net positions

    Parameters:
    data (pd.DataFrame): The CFTC data to plot
    """
    st.subheader("Data Visualization")

    # Check if this is CHG data by examining the type column
    is_chg_data = False
    if 'type' in data.columns and not data.empty:
        type_value = str(data['type'].iloc[0]) if len(data) > 0 else ""
        is_chg_data = '_CHG' in type_value



    # Check if net positions should be calculated
    # Net positions only for QDL/FON or QDL/LFON with plain ALL (no suffixes)
    should_calculate_net = False

    # First check if we have the right dataset
    if 'dataset_code' in st.session_state:
        dataset_code = st.session_state.get('dataset_code', '')
        if dataset_code in ['QDL/FON', 'QDL/LFON']:
            # Now check the type column for plain ALL without suffixes
            if 'type' in data.columns and not data.empty:
                type_value = str(data['type'].iloc[0]) if len(data) > 0 else ""
                # Check if it's plain ALL (F_ALL, FO_ALL, F_L_ALL, FO_L_ALL)
                # without any suffixes like _NT, _OI, _CR
                type_patterns = ['F_ALL', 'FO_ALL', 'F_L_ALL', 'FO_L_ALL']
                should_calculate_net = type_value in type_patterns

    # Use cached functions for data processing
    data = process_cftc_data(data, is_all_data=not is_chg_data)  # Don't calculate net for CHG data
    # Only calculate changes if not CHG data
    if not is_chg_data:
        with st.spinner("Calculating data changes... This may take a moment."):
            data_change = calculate_data_changes(data)
    else:
        # For CHG data, create a minimal data_change dataframe without calculating changes
        # since the data itself already represents changes
        data_change = data.copy()
        # Add empty columns that would normally be created by calculate_data_changes
        # to avoid errors in other parts of the code
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            data_change[f'{col}_change'] = np.nan
            data_change[f'{col}_change_pct'] = np.nan
            data_change[f'{col}_pct_10yr'] = np.nan  # Add this line
            data_change[f'{col}_pct_5yr'] = np.nan
            data_change[f'{col}_pct_2yr'] = np.nan
            data_change[f'{col}_pct_1yr'] = np.nan
            data_change[f'{col}_pct_ytd'] = np.nan

    # Create column selection options
    plot_cols = [col for col in data.columns if col not in ['None', 'contract_code', 'type', 'date']]
    regular_plot_cols = [col for col in plot_cols if col != 'market_participation']

    # Let user select which columns to plot
    st.write("### Select Columns to Plot")

    # First, create a section specifically for net position columns
    # Only show net position columns if they should be calculated
    if should_calculate_net:
        st.write("#### Net Position Columns")
        net_position_cols = [col for col in regular_plot_cols if 'Net' in col]
    else:
        # Provide specific feedback based on why net positions aren't available
        if 'dataset_code' in st.session_state:
            dataset_code = st.session_state.get('dataset_code', '')
            if dataset_code not in ['QDL/FON', 'QDL/LFON']:
                st.info("Net position columns are only available for QDL/FON and QDL/LFON datasets.")
            else:
                st.info(
                    "Net position columns are only available for plain ALL data without additional categories (NT, OI, CR).")
        else:
            st.info("Net position columns require plain ALL data from QDL/FON or QDL/LFON datasets.")
        net_position_cols = []

    selected_cols = []

    if net_position_cols:
        cols_per_row = 3
        net_rows = [net_position_cols[i:i + cols_per_row] for i in range(0, len(net_position_cols), cols_per_row)]

        for row in net_rows:
            cols = st.columns(cols_per_row)
            for i, col in enumerate(row):
                if i < len(row):
                    label_color = ""
                    col_lower = row[i].lower()

                    if 'commercial net' in col_lower:
                        label_color = "red"
                    elif 'large speculator net' in col_lower:
                        label_color = "blue"
                    elif 'small speculator net' in col_lower or 'non_reportable net' in col_lower:
                        label_color = "orange"
                    elif 'other reportables net' in col_lower:
                        label_color = "violet"
                    elif 'swap dealer net' in col_lower:
                        label_color = "orange"

                    if label_color:
                        if cols[i].checkbox(f":{label_color}[{row[i]}]", key=f"checkbox_{row[i]}"):
                            selected_cols.append(row[i])
                    else:
                        if cols[i].checkbox(row[i], key=f"checkbox_{row[i]}"):
                            selected_cols.append(row[i])

    # Then show the original columns section
    st.write("#### Original Position Columns")
    original_cols = [col for col in regular_plot_cols if 'Net' not in col]

    cols_per_row = 3
    original_rows = [original_cols[i:i + cols_per_row] for i in range(0, len(original_cols), cols_per_row)]

    for row in original_rows:
        cols = st.columns(cols_per_row)
        for i, col in enumerate(row):
            if i < len(row):
                label_color = ""
                col_lower = row[i].lower()
                if 'non_commercial' in col_lower or 'money_manager' in col_lower:
                    label_color = "blue"
                elif 'commercial' in col_lower or 'producer' in col_lower:
                    label_color = "red"
                elif 'non_reportable' in col_lower:
                    label_color = "orange"

                if label_color:
                    if cols[i].checkbox(f":{label_color}[{row[i]}]", key=f"checkbox_{row[i]}"):
                        selected_cols.append(row[i])
                else:
                    if cols[i].checkbox(row[i], key=f"checkbox_{row[i]}"):
                        selected_cols.append(row[i])

    # Add a separate checkbox for market participation
    include_market_participation = False
    if 'market_participation' in plot_cols:
        include_market_participation = st.checkbox("Include Market Participation (with separate scale)", value=False)

    if not selected_cols and not include_market_participation:
        st.warning("Please select at least one column to plot.")
        return

    # Chart options
    st.write("### Chart Settings")

    # Plot type selection and show changes toggle in the same row
    col1, col2 = st.columns([1, 2])
    with col1:
        plot_type = st.radio("Plot Type", ["Line", "Bar"], horizontal=True)
    with col2:
        # Only show the toggle for percentage changes if not CHG data
        if not is_chg_data:
            show_changes = st.toggle("Show percentage changes", value=False,
                                     help="Toggle to show/hide percentage changes on the plot. Only available for 90 or fewer periods.")
        else:
            # For CHG data, display an info message and set show_changes to False
            st.info(
                "Percentage changes are not shown for CHG data as the data already represents week-over-week changes.")
            show_changes = False

    # Additional options
    separate_plots = st.checkbox("Create Separate Plots for Each Column", value=False)

    # Set fixed height for plots at 600px
    height_per_plot = 600

    # Create a date range selector
    min_date = data['date'].min().date()
    max_date = data['date'].max().date()

    all_dates = sorted(data['date'].unique())
    default_start_date = all_dates[-min(89, len(all_dates))].date() if len(all_dates) > 1 else min_date
    default_end_date = max_date

    selected_start_date, selected_end_date = st.select_slider(
        "Select Date Range:",
        options=[d.date() for d in all_dates],
        value=(default_start_date, default_end_date)
    )

    selected_start_date = pd.to_datetime(selected_start_date)
    selected_end_date = pd.to_datetime(selected_end_date)

    plot_data, plot_data_change = filter_data_by_date_range(data, data_change, selected_start_date, selected_end_date)

    num_periods = len(plot_data)

    if num_periods > 90 and show_changes:
        st.info(
            f"Note: Percentage changes are only shown for 90 or fewer periods. Currently displaying {num_periods} periods. Changes are hidden.")
        show_changes = False

    st.write(
        f"## CFTC Data Visualization - {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")

    # For CHG data, modify how we prepare hover text
    if not is_chg_data:
        plot_data_change = prepare_hover_texts(plot_data_change, selected_cols)
    else:
        # For CHG data, create simpler hover text that doesn't display changes
        for col in selected_cols:
            plot_data_change[f'{col}_hover'] = plot_data_change.apply(
                lambda row: f"<b>{pd.to_datetime(row['date']).strftime('%Y-%m-%d')}</b><br>{col}: {row[col]:,.0f}",
                axis=1
            )

    annotations = create_annotations(plot_data_change, selected_cols, num_periods, separate_plots, show_changes)

    # Determine plot layout
    if separate_plots:
        fig = make_subplots(rows=len(selected_cols), cols=1,
                            subplot_titles=selected_cols,
                            shared_xaxes=True,
                            vertical_spacing=0.05)

        for i, col in enumerate(selected_cols):
            row = i + 1
            color = get_color_for_column(col)

            if plot_type == "Line":
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['date'],
                        y=plot_data[col],
                        mode='lines+markers',
                        name=col,
                        hoverinfo='text',
                        hovertext=plot_data_change[f'{col}_hover'],
                        line=dict(color=color) if color else dict(),
                    ),
                    row=row, col=1
                )
            else:  # Bar
                fig.add_trace(
                    go.Bar(
                        x=plot_data['date'],
                        y=plot_data[col],
                        name=col,
                        hoverinfo='text',
                        hovertext=plot_data_change[f'{col}_hover'],
                        marker=dict(color=color) if color else dict(),
                    ),
                    row=row, col=1
                )

            fig.update_yaxes(title_text=col, row=row, col=1)

        if include_market_participation and 'market_participation' in data.columns:
            for i in range(len(selected_cols)):
                row = i + 1
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['date'],
                        y=plot_data['market_participation'],
                        mode='lines',
                        name='Market Participation',
                        line=dict(color='gray', dash='dot'),
                        yaxis='y2'
                    ),
                    row=row, col=1
                )

                # Hide gridlines for secondary y-axis to avoid clutter
                fig.update_layout(**{
                    f'yaxis{row * 2}': dict(
                        title='Market Participation',
                        titlefont=dict(color='gray'),
                        tickfont=dict(color='gray'),
                        overlaying=f'y{row * 2 - 1}',
                        side='right',
                        position=1.0,
                        showgrid=False  # Hide gridlines for secondary axis
                    )
                })

        total_height = height_per_plot * len(selected_cols)

    else:
        # Create a single plot with all selected columns
        has_secondary_axis = include_market_participation and 'market_participation' in data.columns

        fig = go.Figure()

        for col in selected_cols:
            color = get_color_for_column(col)

            if plot_type == "Line":
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['date'],
                        y=plot_data[col],
                        mode='lines+markers',
                        name=col,
                        hoverinfo='text',
                        hovertext=plot_data_change[f'{col}_hover'],
                        line=dict(color=color) if color else dict(),
                    )
                )
            else:  # Bar
                fig.add_trace(
                    go.Bar(
                        x=plot_data['date'],
                        y=plot_data[col],
                        name=col,
                        hoverinfo='text',
                        hovertext=plot_data_change[f'{col}_hover'],
                        marker=dict(color=color) if color else dict(),
                    )
                )

        if has_secondary_axis:
            plot_data_change['market_participation_hover'] = plot_data_change.apply(
                lambda row: create_hover_text(row, 'market_participation'), axis=1
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_data['date'],
                    y=plot_data['market_participation'],
                    mode='lines',
                    name='Market Participation',
                    line=dict(color='gray', dash='dot'),
                    yaxis='y2',
                    hoverinfo='text',
                    hovertext=plot_data_change['market_participation_hover']
                )
            )

            # Configure the secondary y-axis and hide its gridlines
            fig.update_layout(
                yaxis2=dict(
                    title='Market Participation',
                    title_font=dict(color='gray'),
                    tickfont=dict(color='gray'),
                    anchor='x',
                    overlaying='y',
                    side='right',
                    position=1.0,
                    showgrid=False  # Hide gridlines for secondary axis
                )
            )

        total_height = height_per_plot

    # Create the layout with annotations based on the toggle state
    layout = {
        'annotations': annotations,
        'title': f"CFTC Data Visualization - {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')} ({num_periods} periods)",
        'xaxis_title': "Date",
        'legend_title': "Data Series",
        'height': total_height,
        'hovermode': "closest",
        'legend': {
            'orientation': "h",
            'yanchor': "bottom",
            'y': 1.02,
            'xanchor': "center",
            'x': 0.5
        }
    }

    # Apply layout with or without annotations based on toggle
    fig.update_layout(layout)

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Display some statistics about the selected columns
    st.write("### Data Statistics")

    # Only show recent changes table if not CHG data
    if not is_chg_data:
        st.write("#### Recent Week-over-Week Changes")
        show_cols = ['date']
        for col in selected_cols:
            if f'{col}_change_pct' in plot_data_change.columns:
                show_cols.append(f'{col}_change_pct')

        if len(show_cols) > 1:
            recent_changes = plot_data_change.iloc[-5:][show_cols]
            renamed_cols = {f'{col}_change_pct': f'{col} (% change)' for col in selected_cols if
                            f'{col}_change_pct' in plot_data_change.columns}
            recent_changes = recent_changes.rename(columns=renamed_cols)
            st.dataframe(
                recent_changes.style.format({col: "{:+.2f}%" for col in recent_changes.columns if col != 'date'}))
    else:
        # For CHG data, show a different message
        st.info(
            "Recent week-over-week changes table is not shown for CHG data as the data itself already represents changes.")

        # Instead, show the most recent values
        st.write("#### Most Recent Values")
        recent_values = plot_data.iloc[-5:][['date'] + selected_cols]
        st.dataframe(recent_values.style.format({col: "{:,.0f}" for col in selected_cols}))

    # Add download button for the plotted data
    columns_to_download = ['date'] + selected_cols
    csv = plot_data[columns_to_download].to_csv(index=False)
    st.download_button(
        label="Download Plotted Data as CSV",
        data=csv,
        file_name="cftc_plotted_data.csv",
        mime="text/csv",
    )
