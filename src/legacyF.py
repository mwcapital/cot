import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sodapy import Socrata
import json
from datetime import datetime
import numpy as np
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="CFTC COT Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load instruments database
@st.cache_data
def load_instruments_database():
    """Load the instruments JSON database"""
    try:
        with open('instruments_LegacyF.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ instruments_LegacyF.json file not found. Please run the fetch script first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading instruments database: {e}")
        return None


# Cache the CFTC data fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_cftc_data(instrument_name, api_token):
    """Fetch CFTC data for a specific instrument"""
    try:
        client = Socrata("publicreporting.cftc.gov", api_token)

        # Define the columns we want
        columns = [
            "report_date_as_yyyy_mm_dd",
            "market_and_exchange_names",
            "open_interest_all",
            "noncomm_positions_long_all",
            "noncomm_positions_short_all",
            "noncomm_postions_spread_all",
            "comm_positions_long_all",
            "comm_positions_short_all",
            "tot_rept_positions_long_all",
            "tot_rept_positions_short",
            "nonrept_positions_long_all",
            "nonrept_positions_short_all"
        ]

        results = client.get(
            "6dca-aqww",
            where=f"market_and_exchange_names='{instrument_name}'",
            select=",".join(columns),
            order="report_date_as_yyyy_mm_dd ASC",
            limit=3000  # As you mentioned you added
        )

        client.close()

        if not results:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Convert date column
        df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])

        # Convert numeric columns
        numeric_columns = [col for col in columns if
                           col != "report_date_as_yyyy_mm_dd" and col != "market_and_exchange_names"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate additional metrics
        df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
        df['net_comm_positions'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
        df['net_reportable_positions'] = df['tot_rept_positions_long_all'] - df['tot_rept_positions_short']

        return df.sort_values('report_date_as_yyyy_mm_dd')

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def create_plotly_chart(df, selected_columns, chart_title):
    """Create interactive Plotly chart with dual y-axis for open interest"""

    try:
        if df.empty or not selected_columns:
            return None

        # Separate open interest from other columns
        open_interest_cols = [col for col in selected_columns if 'open_interest' in col.lower()]
        other_cols = [col for col in selected_columns if 'open_interest' not in col.lower()]

        # Create subplot with secondary y-axis if open interest is included
        if open_interest_cols and other_cols:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add open interest on secondary y-axis
            for col in open_interest_cols:
                if col in df.columns and not df[col].isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[col],
                            name=col.replace('_', ' ').title(),
                            line=dict(width=3),
                            opacity=0.8
                        ),
                        secondary_y=True
                    )

            # Add other columns on primary y-axis
            for col in other_cols:
                if col in df.columns and not df[col].isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[col],
                            name=col.replace('_', ' ').title(),
                            line=dict(width=2)
                        ),
                        secondary_y=False
                    )

            # Update y-axis labels
            fig.update_yaxes(title_text="Positions (Contracts)", secondary_y=False)
            fig.update_yaxes(title_text="Open Interest (Contracts)", secondary_y=True)

        else:
            # Single y-axis chart
            fig = go.Figure()

            for col in selected_columns:
                if col in df.columns and not df[col].isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[col],
                            name=col.replace('_', ' ').title(),
                            line=dict(width=2),
                            mode='lines'
                        )
                    )

            fig.update_yaxes(title_text="Positions (Contracts)")

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
            height=600,
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

        return fig

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None


def create_seasonality_chart(df, column, lookback_years=5, show_previous_year=True, zone_type='percentile'):
    """Create seasonality chart showing historical patterns with smooth zones"""
    try:
        # Add day of year column
        df_season = df.copy()
        df_season['day_of_year'] = df_season['report_date_as_yyyy_mm_dd'].dt.dayofyear
        df_season['year'] = df_season['report_date_as_yyyy_mm_dd'].dt.year

        current_year = df_season['year'].max()

        # Filter based on lookback period
        if lookback_years == 'all':
            years_to_include = df_season['year'].unique()
        else:
            years_to_include = [y for y in range(current_year - lookback_years + 1, current_year + 1)
                                if y in df_season['year'].unique()]

        df_filtered = df_season[df_season['year'].isin(years_to_include)]

        # Create figure
        fig = go.Figure()

        # Calculate statistics for the zones
        historical_data = df_filtered[df_filtered['year'] < current_year]

        # Special handling for 5-year lookback - just show individual years
        if lookback_years == 5 and len(historical_data['year'].unique()) < 5:
            # Just plot each year as individual lines
            years_to_plot = sorted(historical_data['year'].unique())

            # Add each historical year
            for year in years_to_plot:
                year_data = df_season[df_season['year'] == year]
                if not year_data.empty:
                    fig.add_trace(go.Scatter(
                        x=year_data['day_of_year'],
                        y=year_data[column],
                        mode='lines',
                        name=str(year),
                        line=dict(width=2),
                        hovertemplate=f'{year}: %{{y:,.0f}}<extra></extra>'
                    ))

        elif not historical_data.empty and len(historical_data['year'].unique()) >= 2:
            # Group by day of year and calculate statistics
            daily_stats = historical_data.groupby('day_of_year')[column].agg([
                'count', 'mean', 'std', 'min', 'max',
                ('p10', lambda x: np.percentile(x, 10) if len(x) > 0 else np.nan),
                ('p25', lambda x: np.percentile(x, 25) if len(x) > 0 else np.nan),
                ('p50', lambda x: np.percentile(x, 50) if len(x) > 0 else np.nan),
                ('p75', lambda x: np.percentile(x, 75) if len(x) > 0 else np.nan),
                ('p90', lambda x: np.percentile(x, 90) if len(x) > 0 else np.nan)
            ]).reset_index()

            # Remove days with insufficient data
            daily_stats = daily_stats[daily_stats['count'] >= 2]

            if len(daily_stats) > 10:  # Only proceed if we have enough days with data
                # Create a complete day range for interpolation
                all_days = pd.DataFrame({'day_of_year': range(1, 366)})

                if zone_type == 'percentile':
                    # Use percentile values directly
                    stats = all_days.merge(
                        daily_stats[['day_of_year', 'p10', 'p25', 'p50', 'p75', 'p90']],
                        on='day_of_year',
                        how='left'
                    )
                elif zone_type == 'minmax':
                    # Use min/max for outer bounds
                    stats = all_days.merge(
                        daily_stats[['day_of_year', 'min', 'p25', 'p50', 'p75', 'max']],
                        on='day_of_year',
                        how='left'
                    )
                    stats.rename(columns={'min': 'p10', 'max': 'p90'}, inplace=True)
                else:  # std
                    # Calculate std zones, but ensure they're reasonable
                    stats = all_days.merge(
                        daily_stats[['day_of_year', 'mean', 'std']],
                        on='day_of_year',
                        how='left'
                    )
                    # Use a minimum std to avoid collapse
                    stats['std'] = stats['std'].fillna(stats['std'].mean())
                    stats['std'] = stats['std'].clip(lower=stats['mean'].std() * 0.1)

                    stats['p10'] = stats['mean'] - 1.5 * stats['std']
                    stats['p25'] = stats['mean'] - 0.5 * stats['std']
                    stats['p50'] = stats['mean']
                    stats['p75'] = stats['mean'] + 0.5 * stats['std']
                    stats['p90'] = stats['mean'] + 1.5 * stats['std']

                # Interpolate missing values
                for col in ['p10', 'p25', 'p50', 'p75', 'p90']:
                    stats[col] = stats[col].interpolate(method='linear', limit_direction='both')

                # Apply smoothing
                window = min(21, len(stats) // 10)  # Adaptive window size
                for col in ['p10', 'p25', 'p50', 'p75', 'p90']:
                    stats[col] = stats[col].rolling(
                        window=window, center=True, min_periods=1
                    ).mean()

                # Filter to only days with original data
                days_with_data = daily_stats['day_of_year'].unique()
                stats = stats[stats['day_of_year'].isin(days_with_data)]

                # Add shaded zones - outer zone first
                fig.add_trace(go.Scatter(
                    x=stats['day_of_year'],
                    y=stats['p90'],
                    fill=None,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))

                zone_name = {
                    'percentile': '10-90 Percentile Zone',
                    'minmax': 'Min-Max Range',
                    'std': '±1.5 Std Dev Zone'
                }[zone_type]

                fig.add_trace(go.Scatter(
                    x=stats['day_of_year'],
                    y=stats['p10'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    name=zone_name,
                    fillcolor='rgba(200,200,200,0.3)',
                    hoverinfo='skip'
                ))

                # Add inner zone
                inner_zone_name = {
                    'percentile': '25-75 Percentile Zone',
                    'minmax': '25-75 Percentile Zone',
                    'std': '±0.5 Std Dev Zone'
                }[zone_type]

                fig.add_trace(go.Scatter(
                    x=stats['day_of_year'],
                    y=stats['p75'],
                    fill=None,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))

                fig.add_trace(go.Scatter(
                    x=stats['day_of_year'],
                    y=stats['p25'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    name=inner_zone_name,
                    fillcolor='rgba(150,150,150,0.4)',
                    hoverinfo='skip'
                ))

                # Add median/mean line
                median_name = 'Historical Mean' if zone_type == 'std' else 'Historical Median'
                fig.add_trace(go.Scatter(
                    x=stats['day_of_year'],
                    y=stats['p50'],
                    mode='lines',
                    name=median_name,
                    line=dict(color='black', width=2, dash='dash'),
                    hovertemplate=f'{median_name}: %{{y:,.0f}}<extra></extra>'
                ))
            else:
                st.warning("Insufficient historical data for the selected period to create zones.")
        else:
            st.warning("Need at least 2 years of historical data to create seasonality zones.")

        # Add previous year if requested
        if show_previous_year and current_year - 1 in df_season['year'].values:
            prev_year_data = df_season[df_season['year'] == current_year - 1]
            fig.add_trace(go.Scatter(
                x=prev_year_data['day_of_year'],
                y=prev_year_data[column],
                mode='lines',
                name=f'{current_year - 1}',
                line=dict(color='lightblue', width=2),
                hovertemplate=f'{current_year - 1}: %{{y:,.0f}}<extra></extra>'
            ))

        # Add current year
        current_year_data = df_season[df_season['year'] == current_year]
        if not current_year_data.empty:
            fig.add_trace(go.Scatter(
                x=current_year_data['day_of_year'],
                y=current_year_data[column],
                mode='lines',
                name=f'{current_year} (Current)',
                line=dict(color='red', width=3),
                hovertemplate=f'{current_year}: %{{y:,.0f}}<extra></extra>'
            ))

        # Update layout
        fig.update_layout(
            title=f"Seasonality Analysis: {column.replace('_', ' ').title()}",
            xaxis_title="",
            yaxis_title="Value",
            hovermode='x unified',
            height=500,
            xaxis=dict(
                tickmode='array',
                tickvals=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                range=[1, 365]  # Fix x-axis range
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    except Exception as e:
        st.error(f"Error creating seasonality chart: {str(e)}")
        return None


def create_percentile_chart(df, column, lookback_years=5, chart_type='time_series'):
    """Create percentile chart - time series, distribution, or cumulative curve"""
    try:
        df_pct = df.copy()
        df_pct['year'] = df_pct['report_date_as_yyyy_mm_dd'].dt.year
        current_year = df_pct['year'].max()

        # Filter based on lookback period
        if lookback_years == 'all':
            df_historical = df_pct
        else:
            df_historical = df_pct[df_pct['year'] >= current_year - lookback_years]

        if chart_type == 'time_series':
            # Calculate rolling percentile rank
            window_size = 52  # 52 weeks = 1 year
            df_pct['percentile_rank'] = df_pct[column].rolling(window=window_size, min_periods=1).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) * 100
            )

            fig = go.Figure()

            # Add percentile rank line
            fig.add_trace(go.Scatter(
                x=df_pct['report_date_as_yyyy_mm_dd'],
                y=df_pct['percentile_rank'],
                mode='lines',
                name='Percentile Rank',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Percentile: %{y:.1f}%<extra></extra>'
            ))

            # Add reference lines
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Median")
            fig.add_hline(y=90, line_dash="dot", line_color="red", annotation_text="90th %ile")
            fig.add_hline(y=10, line_dash="dot", line_color="green", annotation_text="10th %ile")

            # Add shaded extreme zones
            fig.add_hrect(y0=90, y1=100, line_width=0, fillcolor="red", opacity=0.1)
            fig.add_hrect(y0=0, y1=10, line_width=0, fillcolor="green", opacity=0.1)

            fig.update_layout(
                title=f"Historical Percentile: {column.replace('_', ' ').title()}",
                xaxis_title="Date",
                yaxis_title="Percentile Rank (%)",
                yaxis=dict(range=[0, 100]),
                hovermode='x unified',
                height=400
            )

        elif chart_type == 'distribution':
            # Get historical values
            historical_values = df_historical[column].dropna().values
            current_value = df[column].iloc[-1] if not df.empty else np.nan

            # Create histogram with distribution curve
            fig = go.Figure()

            # Add histogram
            fig.add_trace(go.Histogram(
                x=historical_values,
                nbinsx=50,
                name='Historical Distribution',
                histnorm='probability density',
                marker_color='lightblue',
                opacity=0.7
            ))

            # Calculate and add KDE curve
            kde = stats.gaussian_kde(historical_values)
            x_range = np.linspace(historical_values.min(), historical_values.max(), 200)
            kde_values = kde(x_range)

            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_values,
                mode='lines',
                name='Distribution Curve',
                line=dict(color='blue', width=2)
            ))

            # Add current value marker
            if not np.isnan(current_value):
                fig.add_vline(
                    x=current_value,
                    line_dash="solid",
                    line_color="red",
                    line_width=3,
                    annotation_text=f"Current: {current_value:,.0f}"
                )

                # Calculate percentile
                percentile = (historical_values < current_value).sum() / len(historical_values) * 100

                # Add percentile annotation
                fig.add_annotation(
                    x=current_value,
                    y=kde(current_value)[0] if current_value in x_range else 0,
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
                    borderwidth=2
                )

            # Add percentile markers
            percentiles = [10, 25, 50, 75, 90]
            for p in percentiles:
                value = np.percentile(historical_values, p)
                fig.add_vline(
                    x=value,
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text=f"{p}%"
                )

            fig.update_layout(
                title=f"Distribution Analysis: {column.replace('_', ' ').title()}",
                xaxis_title="Value",
                yaxis_title="Density",
                showlegend=True,
                height=400
            )

        else:  # cumulative percentile curve
            # Get historical values and sort them
            historical_values = df_historical[column].dropna().values
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
                title=f"Cumulative Percentile Curve: {column.replace('_', ' ').title()}",
                xaxis_title=f"{column.replace('_', ' ').title()} Value",
                yaxis_title="Percentile (%)",
                yaxis=dict(range=[0, 100]),
                showlegend=True,
                height=400,
                hovermode='closest'
            )

        return fig

    except Exception as e:
        st.error(f"Error creating percentile chart: {str(e)}")
        return None


# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = None
if 'fetched_instrument' not in st.session_state:
    st.session_state.fetched_instrument = None


# Main app
def main():
    st.title("📊 CFTC Commitments of Traders Dashboard")
    st.markdown("Interactive analysis of CFTC COT data with hierarchical instrument selection")

    # Load instruments database
    instruments_db = load_instruments_database()
    if not instruments_db:
        st.stop()

    # API Token Configuration
    with st.expander("🔧 API Configuration", expanded=False):
        api_token = st.text_input(
            "API Token (optional):",
            value="3CKjkFN6jIIHgSkIJH19i7VhK",  # Pre-filled for testing
            type="password",
            help="Enter your CFTC API token for higher rate limits. Leave empty to use default limits."
        )

    st.markdown("---")

    # Instrument Selection Methods
    st.header("🎯 Instrument Selection")

    # Search method selection
    search_method = st.radio(
        "Choose search method:",
        ["Extensive Search", "Search by Commodity Subgroup", "Search by Commodity Type", "Free Text Search"],
        horizontal=True
    )

    selected_instrument = None

    if search_method == "Extensive Search":
        st.subheader("📂 Browse by Exchange Hierarchy")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            exchanges = list(instruments_db['exchanges'].keys())
            selected_exchange = st.selectbox("📍 Exchange:", exchanges)

        with col2:
            groups = list(instruments_db['exchanges'][selected_exchange].keys())
            selected_group = st.selectbox("📂 Group:", groups)

        with col3:
            subgroups = list(instruments_db['exchanges'][selected_exchange][selected_group].keys())
            selected_subgroup = st.selectbox("📁 Subgroup:", subgroups)

        with col4:
            commodities = list(instruments_db['exchanges'][selected_exchange][selected_group][selected_subgroup].keys())
            selected_commodity = st.selectbox("🔸 Commodity:", commodities)

        # Instrument selection
        instruments = instruments_db['exchanges'][selected_exchange][selected_group][selected_subgroup][
            selected_commodity]
        selected_instrument = st.selectbox("📊 Select Instrument:", instruments, key="hierarchy_instrument")

        # Show classification path
        st.info(
            f"📦 **{selected_exchange}** → **{selected_group}** → **{selected_subgroup}** → **{selected_commodity}**")

    elif search_method == "Search by Commodity Subgroup":
        st.subheader("📁 Search by Commodity Subgroup")
        col1, col2 = st.columns(2)

        with col1:
            subgroups = sorted(list(instruments_db['commodity_subgroups'].keys()))
            selected_subgroup = st.selectbox("📁 Select Commodity Subgroup:", subgroups)

        with col2:
            instruments = instruments_db['commodity_subgroups'][selected_subgroup]
            selected_instrument = st.selectbox("📊 Select Instrument:", sorted(instruments), key="subgroup_instrument")

        st.info(f"📁 **{selected_subgroup}** → {len(instruments)} available instruments")

    elif search_method == "Search by Commodity Type":
        st.subheader("🔸 Search by Commodity Type")
        col1, col2 = st.columns(2)

        with col1:
            commodities = sorted(list(instruments_db['commodities'].keys()))
            selected_commodity_type = st.selectbox("🔸 Select Commodity:", commodities)

        with col2:
            instruments = instruments_db['commodities'][selected_commodity_type]
            selected_instrument = st.selectbox("📊 Select Instrument:", sorted(instruments), key="commodity_instrument")

        st.info(f"🔸 **{selected_commodity_type}** → {len(instruments)} available instruments")

    else:  # Free Text Search
        st.subheader("🔍 Free Text Search")
        search_text = st.text_input(
            "Type instrument name or keyword:",
            placeholder="e.g., GOLD, CRUDE OIL, S&P 500, WHEAT..."
        )

        if search_text:
            # Filter instruments based on search text
            all_instruments = instruments_db['all_instruments']
            filtered_instruments = [
                inst for inst in all_instruments
                if search_text.upper() in inst.upper()
            ]

            if filtered_instruments:
                selected_instrument = st.selectbox(
                    f"📊 Select from {len(filtered_instruments)} matching instruments:",
                    sorted(filtered_instruments),
                    key="search_instrument"
                )
                st.success(f"✅ Found {len(filtered_instruments)} matching instruments")
            else:
                st.warning(f"⚠️ No instruments found matching '{search_text}'")
        else:
            st.info("💡 Start typing to search through all available instruments")

    # Fetch Data Button - Now positioned after instrument selection
    if selected_instrument:
        st.markdown("---")
        st.subheader(f"Selected: {selected_instrument}")

        # Check if we need to fetch new data
        need_new_fetch = (not st.session_state.data_fetched or
                          st.session_state.fetched_instrument != selected_instrument)

        if st.button("🚀 Fetch Data", type="primary", use_container_width=False, disabled=not need_new_fetch):
            with st.spinner(f"Fetching data for {selected_instrument}..."):
                df = fetch_cftc_data(selected_instrument, api_token)

            if df is not None and not df.empty:
                st.session_state.fetched_data = df
                st.session_state.data_fetched = True
                st.session_state.fetched_instrument = selected_instrument
                st.success(f"✅ Successfully fetched {len(df)} records")
            else:
                st.error("❌ No data available for the selected instrument")
                st.session_state.data_fetched = False

    # Display data if available
    if st.session_state.data_fetched and st.session_state.fetched_data is not None:
        df = st.session_state.fetched_data

        # Date range selection
        st.markdown("---")
        st.subheader("📅 Date Range Selection")

        min_date = df['report_date_as_yyyy_mm_dd'].min()
        max_date = df['report_date_as_yyyy_mm_dd'].max()

        # Create list of available dates for the slider
        available_dates = sorted(df['report_date_as_yyyy_mm_dd'].unique())

        # Single slider with two handles for date range
        date_range = st.select_slider(
            "Select Date Range:",
            options=range(len(available_dates)),
            value=(0, len(available_dates) - 1),
            format_func=lambda x: available_dates[x].strftime('%Y-%m-%d')
        )

        # Get selected dates
        start_date = available_dates[date_range[0]]
        end_date = available_dates[date_range[1]]

        # Filter dataframe based on selected date range
        filtered_df = df[
            (df['report_date_as_yyyy_mm_dd'] >= start_date) &
            (df['report_date_as_yyyy_mm_dd'] <= end_date)
            ].copy()

        st.info(
            f"📊 Showing {len(filtered_df)} records from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Column selection for plotting
        st.markdown("---")
        st.subheader("📈 Data Visualization")

        available_columns = [
            "open_interest_all",
            "noncomm_positions_long_all",
            "noncomm_positions_short_all",
            "noncomm_postions_spread_all",
            "comm_positions_long_all",
            "comm_positions_short_all",
            "tot_rept_positions_long_all",
            "tot_rept_positions_short",
            "nonrept_positions_long_all",
            "nonrept_positions_short_all",
            "net_noncomm_positions",
            "net_comm_positions",
            "net_reportable_positions"
        ]

        # Filter columns that exist in the dataframe
        existing_columns = [col for col in available_columns if col in filtered_df.columns]

        # Use checkboxes for column selection
        st.write("Select columns to plot:")

        # Create columns for better layout of checkboxes
        col1, col2, col3 = st.columns(3)

        selected_plot_columns = []

        for i, col in enumerate(existing_columns):
            with [col1, col2, col3][i % 3]:
                if st.checkbox(
                        col.replace('_', ' ').title(),
                        value=False,
                        key=f"checkbox_{col}"
                ):
                    selected_plot_columns.append(col)

        if not selected_plot_columns:
            st.info("👆 Please select one or more columns above to create a chart")
        else:
            # Chart type selection
            chart_type = st.radio(
                "Select Chart Type:",
                ["Time Series", "Seasonality Analysis", "Percentile Analysis"],
                horizontal=True
            )

            if chart_type == "Time Series":
                # Calculate and display KPIs
                st.markdown("### 📊 Key Performance Indicators")

                # Get latest values and calculate metrics
                latest_date = filtered_df['report_date_as_yyyy_mm_dd'].max()
                latest_data = filtered_df[filtered_df['report_date_as_yyyy_mm_dd'] == latest_date].iloc[0]

                # Calculate averages and changes
                kpi_cols = st.columns(5)

                for idx, col in enumerate(selected_plot_columns[:5]):  # Show max 5 KPIs
                    with kpi_cols[idx % 5]:
                        current_val = latest_data[col] if col in latest_data else 0

                        # Calculate change from previous period
                        if len(filtered_df) > 1:
                            prev_val = filtered_df[col].iloc[-2]
                            change = ((current_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
                            change_str = f"{change:+.1f}%"
                            delta_color = "normal" if abs(change) < 10 else "inverse"
                        else:
                            change_str = "N/A"
                            delta_color = "off"

                        # Format the metric name
                        metric_name = col.replace('_', ' ').title()
                        if len(metric_name) > 20:
                            metric_name = metric_name[:17] + "..."

                        st.metric(
                            label=metric_name,
                            value=f"{current_val:,.0f}",
                            delta=change_str,
                            delta_color=delta_color
                        )

                # Add period average
                st.markdown("#### Period Statistics")
                stat_cols = st.columns(4)

                with stat_cols[0]:
                    st.info(f"**Period Start:** {start_date.strftime('%Y-%m-%d')}")
                with stat_cols[1]:
                    st.info(f"**Period End:** {end_date.strftime('%Y-%m-%d')}")
                with stat_cols[2]:
                    st.info(f"**Data Points:** {len(filtered_df)}")
                with stat_cols[3]:
                    avg_oi = filtered_df[
                        'open_interest_all'].mean() if 'open_interest_all' in filtered_df.columns else 0
                    st.info(f"**Avg Open Interest:** {avg_oi:,.0f}")

                st.markdown("---")

                # Create and display standard time series chart
                chart_title = f"{st.session_state.fetched_instrument} - COT Data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
                fig = create_plotly_chart(filtered_df, selected_plot_columns, chart_title)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    # Chart download button
                    if st.button("💾 Download Chart as HTML"):
                        html_string = fig.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="Download Chart",
                            data=html_string,
                            file_name=f"cftc_chart_{st.session_state.fetched_instrument.replace(' ', '_').replace('-', '_')}.html",
                            mime="text/html"
                        )

            elif chart_type == "Seasonality Analysis":
                # Seasonality options
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    lookback_option = st.selectbox(
                        "Historical Period:",
                        ["Last 5 Years", "Last 10 Years", "All Time"],
                        key="seasonality_lookback"
                    )

                    if lookback_option == "Last 5 Years":
                        lookback_years = 5
                    elif lookback_option == "Last 10 Years":
                        lookback_years = 10
                    else:
                        lookback_years = 'all'

                with col2:
                    zone_type = st.selectbox(
                        "Zone Type:",
                        ["Percentile Zones", "Min-Max Range", "Standard Deviation"],
                        key="zone_type"
                    )

                    zone_type_map = {
                        "Percentile Zones": "percentile",
                        "Min-Max Range": "minmax",
                        "Standard Deviation": "std"
                    }
                    zone_type_value = zone_type_map[zone_type]

                with col3:
                    show_prev_year = st.checkbox("Show Previous Year", value=True)

                with col4:
                    # Select which column to analyze
                    selected_season_col = st.selectbox(
                        "Select Column to Analyze:",
                        selected_plot_columns,
                        key="season_column"
                    )

                # Create seasonality chart
                fig_season = create_seasonality_chart(
                    df,  # Use full dataset for seasonality
                    selected_season_col,
                    lookback_years,
                    show_prev_year,
                    zone_type_value
                )

                if fig_season:
                    st.plotly_chart(fig_season, use_container_width=True)

                    # Download button
                    if st.button("💾 Download Seasonality Chart", key="download_season"):
                        html_string = fig_season.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="Download Chart",
                            data=html_string,
                            file_name=f"cftc_seasonality_{selected_season_col}_{st.session_state.fetched_instrument.replace(' ', '_').replace('-', '_')}.html",
                            mime="text/html"
                        )

            elif chart_type == "Percentile Analysis":
                # Percentile options
                col1, col2, col3 = st.columns(3)

                with col1:
                    lookback_option_pct = st.selectbox(
                        "Historical Period:",
                        ["Last 5 Years", "Last 10 Years", "All Time"],
                        key="percentile_lookback"
                    )

                    if lookback_option_pct == "Last 5 Years":
                        lookback_years_pct = 5
                    elif lookback_option_pct == "Last 10 Years":
                        lookback_years_pct = 10
                    else:
                        lookback_years_pct = 'all'

                with col2:
                    # Select which column to analyze
                    selected_pct_col = st.selectbox(
                        "Select Column to Analyze:",
                        selected_plot_columns,
                        key="pct_column"
                    )

                with col3:
                    # Chart view type
                    pct_view_type = st.radio(
                        "View Type:",
                        ["Time Series", "Distribution Curve", "Percentile Curve"],
                        key="pct_view"
                    )

                # Current value and historical stats
                current_value = filtered_df[selected_pct_col].iloc[-1] if not filtered_df.empty else np.nan
                historical_percentile = ((filtered_df[selected_pct_col] < current_value).sum() / len(
                    filtered_df) * 100) if not filtered_df.empty else np.nan

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Value", f"{current_value:,.0f}")
                with col2:
                    st.metric("Historical Percentile", f"{historical_percentile:.1f}%")
                with col3:
                    st.metric("Median", f"{filtered_df[selected_pct_col].median():,.0f}")
                with col4:
                    st.metric("Std Dev", f"{filtered_df[selected_pct_col].std():,.0f}")

                # Create percentile chart based on view type
                if pct_view_type == "Time Series":
                    chart_type_param = 'time_series'
                elif pct_view_type == "Distribution Curve":
                    chart_type_param = 'distribution'
                else:
                    chart_type_param = 'cumulative'

                fig_pct = create_percentile_chart(
                    df,  # Use full dataset
                    selected_pct_col,
                    lookback_years_pct,
                    chart_type_param
                )

                if fig_pct:
                    st.plotly_chart(fig_pct, use_container_width=True)

                    # Download button
                    if st.button("💾 Download Percentile Chart", key="download_pct"):
                        html_string = fig_pct.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="Download Chart",
                            data=html_string,
                            file_name=f"cftc_percentile_{selected_pct_col}_{st.session_state.fetched_instrument.replace(' ', '_').replace('-', '_')}.html",
                            mime="text/html"
                        )

        # Data table
        st.markdown("---")
        st.subheader("📋 Raw Data (Last 10 Rows)")

        # Show only selected columns and last 10 rows
        if selected_plot_columns:
            display_columns = ["report_date_as_yyyy_mm_dd"] + selected_plot_columns
            display_df = filtered_df[display_columns].tail(10)
        else:
            display_columns = ["report_date_as_yyyy_mm_dd"] + existing_columns[
                                                              :5]  # Show first 5 columns if none selected
            display_df = filtered_df[display_columns].tail(10)

        st.dataframe(display_df, use_container_width=True)

        # Download buttons
        col1, col2 = st.columns(2)

        with col1:
            # Download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Dataset as CSV",
                data=csv,
                file_name=f"cftc_data_{st.session_state.fetched_instrument.replace(' ', '_').replace('-', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col2:
            # Download full dataset
            full_csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset as CSV",
                data=full_csv,
                file_name=f"cftc_data_full_{st.session_state.fetched_instrument.replace(' ', '_').replace('-', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    elif selected_instrument and not st.session_state.data_fetched:
        st.info("👆 Click 'Fetch Data' to load the COT data for the selected instrument.")

    # Footer info
    st.markdown("---")
    with st.expander("ℹ️ About this Dashboard"):
        st.markdown("""
        **CFTC Commitments of Traders Dashboard**

        This dashboard provides interactive analysis of CFTC Commitments of Traders (COT) data using:
        - **Hierarchical Navigation**: Browse instruments by Exchange → Group → Subgroup → Commodity
        - **Interactive Charts**: Plotly charts with dual y-axis for open interest
        - **Data Export**: Download charts as HTML and data as CSV
        - **Real-time Data**: Direct API connection to CFTC public reporting environment

        **Data Fields:**
        - **Open Interest**: Total outstanding contracts
        - **Non-Commercial**: Large speculators (hedge funds, CTAs)
        - **Commercial**: Hedgers (producers, merchants)
        - **Reportable**: All large traders above reporting thresholds
        - **Non-Reportable**: Small traders below thresholds
        - **Net Positions**: Long minus Short positions

        **Chart Features:**
        - Open Interest plotted on separate y-axis when selected
        - Interactive zoom, pan, and hover
        - Time range selectors (1Y, 2Y, 5Y, All)
        - Downloadable as HTML for sharing
        """)


if __name__ == "__main__":
    main()