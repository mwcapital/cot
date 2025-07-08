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
            "nonrept_positions_short_all",
            # Trader count columns
            "traders_tot_all",
            "traders_noncomm_long_all",
            "traders_noncomm_short_all",
            "traders_noncomm_spread_all",
            "traders_comm_long_all",
            "traders_comm_short_all",
            "traders_tot_rept_long_all",
            "traders_tot_rept_short_all",
            # Concentration columns
            "conc_gross_le_4_tdr_long",
            "conc_gross_le_4_tdr_short",
            "conc_gross_le_8_tdr_long",
            "conc_gross_le_8_tdr_short",
            "conc_net_le_4_tdr_long_all",
            "conc_net_le_4_tdr_short_all",
            "conc_net_le_8_tdr_long_all",
            "conc_net_le_8_tdr_short_all",
            # Percentage of open interest columns
            "pct_of_open_interest_all",
            "pct_of_oi_noncomm_long_all",
            "pct_of_oi_noncomm_short_all",
            "pct_of_oi_noncomm_spread",
            "pct_of_oi_comm_long_all",
            "pct_of_oi_comm_short_all",
            "pct_of_oi_tot_rept_long_all",
            "pct_of_oi_tot_rept_short",
            "pct_of_oi_nonrept_long_all",
            "pct_of_oi_nonrept_short_all",
            # Change columns from API
            "change_in_open_interest_all",
            "change_in_noncomm_long_all",
            "change_in_noncomm_short_all",
            "change_in_noncomm_spead_all",
            "change_in_comm_long_all",
            "change_in_comm_short_all",
            "change_in_tot_rept_long_all",
            "change_in_tot_rept_short",
            "change_in_nonrept_long_all",
            "change_in_nonrept_short_all"
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
        
        # Debug: Check which percentage columns were actually returned
        pct_cols_requested = [col for col in columns if 'pct_of_oi' in col]
        pct_cols_received = [col for col in df.columns if 'pct_of_oi' in col]
        if pct_cols_requested and not pct_cols_received:
            st.warning(f"Requested {len(pct_cols_requested)} percentage columns but received {len(pct_cols_received)}")
            st.info("This might be due to the API not returning these fields for this instrument.")

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

        # Enable dynamic y-axis scaling - TradingView style
        # This will automatically adjust y-axis based on visible data range
        
        # Configure y-axes for dynamic scaling
        if open_interest_cols and other_cols:
            # For dual y-axis charts
            fig.update_yaxes(
                autorange=True, 
                fixedrange=False,
                automargin=True,
                secondary_y=False
            )
            fig.update_yaxes(
                autorange=True, 
                fixedrange=False,
                automargin=True,
                secondary_y=True
            )
        else:
            # For single y-axis charts
            fig.update_yaxes(
                autorange=True, 
                fixedrange=False,
                automargin=True
            )
        
        # Enable Plotly's built-in dynamic y-axis scaling on zoom
        fig.update_layout(
            xaxis_rangeslider_visible=True,
            dragmode='zoom',
            hovermode='x unified'
        )
        
        # Configure all y-axes for responsive scaling
        # This is crucial for TradingView-like behavior
        fig.update_yaxes(
            autorange=True,
            fixedrange=False,
            # This ensures y-axis updates when x-axis range changes
            scaleanchor=None,
            constraintoward='middle'
        )
        
        # For secondary y-axis if it exists
        if open_interest_cols and other_cols:
            fig.update_yaxes(
                autorange=True,
                fixedrange=False,
                scaleanchor=None,
                constraintoward='middle',
                secondary_y=True
            )
        
        # Additional configuration for better zoom behavior
        config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': chart_title.replace(' ', '_'),
                'height': 600,
                'width': 1200,
                'scale': 2
            },
            # Enable responsive y-axis
            'responsive': True
        }
        
        # Store config for later use
        fig._config = config

        return fig

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None


def create_share_of_oi_chart(df, calculation_side, chart_title):
    """Create share of open interest time series chart"""
    try:
        if df.empty:
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Use API percentages directly
        df_calc = df.copy()
        
        # Check if API percentage columns exist
        api_pct_cols = ['pct_of_oi_comm_long_all', 'pct_of_oi_noncomm_long_all', 
                       'pct_of_oi_noncomm_spread', 'pct_of_oi_nonrept_long_all',
                       'pct_of_oi_comm_short_all', 'pct_of_oi_noncomm_short_all',
                       'pct_of_oi_nonrept_short_all']
        
        has_api_pct = all(col in df.columns for col in api_pct_cols)
        
        if has_api_pct:
            # Use API percentages directly
            if calculation_side == "Long Side":
                df_calc['comm_pct'] = df_calc['pct_of_oi_comm_long_all']
                df_calc['noncomm_pct'] = df_calc['pct_of_oi_noncomm_long_all']
                df_calc['nonrept_pct'] = df_calc['pct_of_oi_nonrept_long_all']
                df_calc['spread_pct'] = df_calc['pct_of_oi_noncomm_spread']
            else:  # Short Side
                df_calc['comm_pct'] = df_calc['pct_of_oi_comm_short_all']
                df_calc['noncomm_pct'] = df_calc['pct_of_oi_noncomm_short_all']
                df_calc['nonrept_pct'] = df_calc['pct_of_oi_nonrept_short_all']
                df_calc['spread_pct'] = df_calc['pct_of_oi_noncomm_spread']
        else:
            # Fallback to manual calculation if API columns not available
            st.warning("API percentage columns not found. Using manual calculation.")
            
            if calculation_side == "Long Side":
                # Long side total: NonComm Long + Spread + Comm Long + NonRep Long
                df_calc['total_side'] = (
                    df_calc['noncomm_positions_long_all'] + 
                    df_calc['noncomm_postions_spread_all'] +
                    df_calc['comm_positions_long_all'] + 
                    df_calc['nonrept_positions_long_all']
                )
                
                # Calculate percentages
                df_calc['comm_pct'] = (df_calc['comm_positions_long_all'] / df_calc['total_side']) * 100
                df_calc['noncomm_pct'] = (df_calc['noncomm_positions_long_all'] / df_calc['total_side']) * 100
                df_calc['nonrept_pct'] = (df_calc['nonrept_positions_long_all'] / df_calc['total_side']) * 100
            else:  # Short Side
                # Short side total: NonComm Short + Spread + Comm Short + NonRep Short
                df_calc['total_side'] = (
                    df_calc['noncomm_positions_short_all'] + 
                    df_calc['noncomm_postions_spread_all'] +
                    df_calc['comm_positions_short_all'] + 
                    df_calc['nonrept_positions_short_all']
                )
                
                # Calculate percentages
                df_calc['comm_pct'] = (df_calc['comm_positions_short_all'] / df_calc['total_side']) * 100
                df_calc['noncomm_pct'] = (df_calc['noncomm_positions_short_all'] / df_calc['total_side']) * 100
                df_calc['nonrept_pct'] = (df_calc['nonrept_positions_short_all'] / df_calc['total_side']) * 100
            
            # Spread percentage is same for both sides
            df_calc['spread_pct'] = (df_calc['noncomm_postions_spread_all'] / df_calc['total_side']) * 100
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=df_calc['report_date_as_yyyy_mm_dd'],
            y=df_calc['comm_pct'],
            name='Commercial',
            line=dict(width=2, color='blue'),
            mode='lines',
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_calc['report_date_as_yyyy_mm_dd'],
            y=df_calc['noncomm_pct'],
            name='Non-Commercial',
            line=dict(width=2, color='green'),
            mode='lines',
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_calc['report_date_as_yyyy_mm_dd'],
            y=df_calc['spread_pct'],
            name='Non-Commercial Spread',
            line=dict(width=2, color='orange'),
            mode='lines',
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_calc['report_date_as_yyyy_mm_dd'],
            y=df_calc['nonrept_pct'],
            name='Non-Reportable',
            line=dict(width=2, color='gray'),
            mode='lines',
            stackgroup='one'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{chart_title} - Share of Open Interest ({calculation_side})",
            xaxis_title="Date",
            yaxis=dict(
                title="Share of Open Interest (%)",
                tickformat=".1f",
                ticksuffix="%",
                range=[0, 100]
            ),
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
                type='date'
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating share of OI chart: {str(e)}")
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
        elif lookback_years == 'ytd':
            # For YTD, only include the current year
            years_to_include = [current_year]
        else:
            years_to_include = [y for y in range(current_year - lookback_years + 1, current_year + 1)
                                if y in df_season['year'].unique()]

        df_filtered = df_season[df_season['year'].isin(years_to_include)]

        # Create figure
        fig = go.Figure()

        # Calculate statistics for the zones
        historical_data = df_filtered[df_filtered['year'] < current_year]

        # Special handling for small lookback periods or YTD
        if lookback_years == 'ytd' or (isinstance(lookback_years, int) and lookback_years <= 2) or \
           (lookback_years == 5 and len(historical_data['year'].unique()) < 5):
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

        # Always show all data for display
        # The time range buttons in the chart will handle filtering
        df_display = df_pct.copy()

        if chart_type == 'time_series':
            # Calculate rolling percentile rank
            # Use the specified lookback period for the rolling window
            if lookback_years == 'all':
                window_days = 252 * 20  # Use 20 years for 'all time'
            else:
                window_days = 252 * lookback_years  # Trading days based on lookback period
            
            df_display['percentile_rank'] = np.nan
            df_display['actual_value'] = df_display[column]
            
            # Calculate percentile for each point
            for idx in range(len(df_display)):
                current_date = df_display.iloc[idx]['report_date_as_yyyy_mm_dd']
                current_value = df_display.iloc[idx][column]
                
                # Define the lookback window
                window_start = current_date - pd.Timedelta(days=window_days)
                
                # Get all historical values in the window (including from full dataset)
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
                    f'Percentile Rank (Rolling {lookback_years if lookback_years != "all" else "20"}-Year Window)', 
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
            # Get historical data with dates for time-based coloring
            if lookback_years == 'all':
                df_hist = df_pct[[column, 'report_date_as_yyyy_mm_dd']].dropna()
            else:
                lookback_start = df_pct['report_date_as_yyyy_mm_dd'].max() - pd.DateOffset(years=lookback_years)
                df_hist = df_pct[df_pct['report_date_as_yyyy_mm_dd'] >= lookback_start][[column, 'report_date_as_yyyy_mm_dd']].dropna()
            
            historical_values = df_hist[column].values
            historical_dates = df_hist['report_date_as_yyyy_mm_dd']
            current_value = df[column].iloc[-1] if not df.empty else np.nan
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

        else:  # cumulative percentile curve
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
            
        if chart_type == 'rate_of_change':
            # Calculate rate of change
            df_filtered = df_filtered.copy()
            df_filtered['rate_of_change'] = df_filtered[column].pct_change() * 100
            df_filtered['rate_of_change_ma'] = df_filtered['rate_of_change'].rolling(window=4, min_periods=1).mean()
            
            # Create subplot with rate of change
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.6, 0.4],
                vertical_spacing=0.05,
                subplot_titles=(f'{column.replace("_", " ").title()} - Actual Values', 'Rate of Change (%)')
            )
            
            # Main chart - actual values
            fig.add_trace(
                go.Scatter(
                    x=df_filtered['report_date_as_yyyy_mm_dd'],
                    y=df_filtered[column],
                    mode='lines',
                    name='Actual Value',
                    line=dict(color='blue', width=2),
                    hovertemplate='Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Rate of change chart
            fig.add_trace(
                go.Bar(
                    x=df_filtered['report_date_as_yyyy_mm_dd'],
                    y=df_filtered['rate_of_change'],
                    name='Rate of Change',
                    marker=dict(
                        color=df_filtered['rate_of_change'],
                        colorscale='RdYlGn',
                        cmin=-10,
                        cmax=10,
                        colorbar=dict(
                            title="RoC %",
                            y=0.2,
                            len=0.35
                        )
                    ),
                    hovertemplate='Date: %{x}<br>RoC: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add moving average of rate of change
            fig.add_trace(
                go.Scatter(
                    x=df_filtered['report_date_as_yyyy_mm_dd'],
                    y=df_filtered['rate_of_change_ma'],
                    mode='lines',
                    name='RoC MA(4)',
                    line=dict(color='black', width=2, dash='dash'),
                    hovertemplate='Date: %{x}<br>MA RoC: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add zero line for rate of change
            fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title=f"Rate of Change Analysis: {column.replace('_', ' ').title()} (Last {lookback_years} Years)" if lookback_years != 'all' else f"Rate of Change Analysis: {column.replace('_', ' ').title()} (All Time)",
                hovermode='x unified',
                height=700,
                showlegend=True,
                xaxis2=dict(
                    rangeslider=dict(visible=True, thickness=0.05),
                    type='date'
                ),
                yaxis=dict(
                    title=column.replace('_', ' ').title(),
                    autorange=True,
                    fixedrange=False
                ),
                yaxis2=dict(
                    title="Rate of Change (%)",
                    autorange=True,
                    fixedrange=False
                ),
                dragmode='zoom'
            )

        return fig

    except Exception as e:
        st.error(f"Error creating percentile chart: {str(e)}")
        return None


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
        fig.update_xaxes(matches='x', row=1, col=2)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating trader breakdown charts: {str(e)}")
        return None


def create_single_variable_momentum_dashboard(df, variable_name, change_col):
    """Create focused momentum dashboard for a single variable"""
    try:
        # Prepare data
        df_momentum = df.copy()
        df_momentum = df_momentum.sort_values('report_date_as_yyyy_mm_dd')
        
        # Get the actual values and changes
        actual_values = df_momentum[variable_name].fillna(0)
        change_values = df_momentum[change_col].fillna(0)
        
        # Calculate z-scores using 52-week rolling window
        rolling_mean = change_values.rolling(window=52, min_periods=1).mean()
        rolling_std = change_values.rolling(window=52, min_periods=1).std()
        
        # Calculate z-score: (current - rolling mean) / rolling std
        z_scores = pd.Series(index=change_values.index, dtype=float)
        for i in range(len(change_values)):
            if i == 0:
                z_scores.iloc[i] = 0
            else:
                # Use up to 52 weeks of history
                lookback_start = max(0, i - 51)
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
            vertical_spacing=0.06,
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
            title=f"Momentum Dashboard - {variable_name.replace('_', ' ').title()}",
            height=1000,
            showlegend=False,
            hovermode='x unified',
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
                rangeslider=dict(visible=True, thickness=0.05),
                title='Date',
                autorange=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    bgcolor='rgba(255,255,255,0.9)',
                    activecolor='lightblue',
                    x=0.01,
                    y=1.0
                )
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


def create_cross_asset_analysis(selected_instruments, trader_category, api_token, lookback_start, show_week_ago, instruments_db):
    """Create cross-asset positioning analysis chart"""
    try:
        # Store data for all instruments
        instrument_data = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data for each instrument
        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Fetching data for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))
            
            # Fetch data
            df = fetch_cftc_data(instrument, api_token)
            
            if df is not None and not df.empty:
                # Filter data from lookback start date
                df = df[df['report_date_as_yyyy_mm_dd'] >= pd.Timestamp(lookback_start)]
                
                if not df.empty:
                    # Define column mappings
                    category_columns = {
                        "Non-Commercial": {
                            "long": "noncomm_positions_long_all",
                            "short": "noncomm_positions_short_all",
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
                    
                    cols = category_columns[trader_category]
                    
                    # Calculate net if not present
                    if trader_category == "Non-Reportable" and "net" not in cols:
                        df['net_nonrept_positions'] = df[cols['long']] - df[cols['short']]
                        cols['net'] = 'net_nonrept_positions'
                    
                    # Calculate net as % of open interest
                    if 'open_interest_all' in df.columns and cols.get('net') in df.columns:
                        df['net_pct_oi'] = (df[cols['net']] / df['open_interest_all'] * 100).fillna(0)
                        
                        # Calculate z-score
                        mean_pct = df['net_pct_oi'].mean()
                        std_pct = df['net_pct_oi'].std()
                        
                        if std_pct > 0:
                            # Get latest and week-ago values
                            latest_pct = df['net_pct_oi'].iloc[-1] if len(df) > 0 else 0
                            latest_z = (latest_pct - mean_pct) / std_pct
                            
                            week_ago_z = None
                            if show_week_ago and len(df) > 1:
                                week_ago_pct = df['net_pct_oi'].iloc[-2]
                                week_ago_z = (week_ago_pct - mean_pct) / std_pct
                            
                            instrument_data[instrument] = {
                                'z_score': latest_z,
                                'week_ago_z': week_ago_z,
                                'net_pct': latest_pct,
                                'mean': mean_pct,
                                'std': std_pct
                            }
        
        progress_bar.empty()
        status_text.empty()
        
        if not instrument_data:
            st.error("No valid data found for selected instruments")
            return None
        
        # Sort by z-score
        sorted_instruments = sorted(instrument_data.items(), key=lambda x: x[1]['z_score'], reverse=True)
        
        # Create the chart
        fig = go.Figure()
        
        # Prepare data for plotting
        instruments_full = [item[0] for item in sorted_instruments]
        # Shorten instrument names - take only the part before first "-"
        instruments_short = [name.split('-')[0].strip() for name in instruments_full]
        z_scores = [item[1]['z_score'] for item in sorted_instruments]
        
        # Nice mint/salad green color
        bar_color = '#7DCEA0'  # Soft mint green
        
        # Add bars
        fig.add_trace(go.Bar(
            x=instruments_short,
            y=z_scores,
            name='Current Z-Score',
            marker=dict(color=bar_color),
            text=[f"{z:.2f}" for z in z_scores],
            textposition='outside',
            hovertemplate='<b>%{customdata[3]}</b><br>' +  # Show full name in hover
                         'Z-Score: %{y:.2f}<br>' +
                         'Net %: %{customdata[0]:.1f}%<br>' +
                         'Mean %: %{customdata[1]:.1f}%<br>' +
                         'Std %: %{customdata[2]:.1f}%<extra></extra>',
            customdata=[[item[1]['net_pct'], item[1]['mean'], item[1]['std'], full_name] 
                       for item, full_name in zip(sorted_instruments, instruments_full)]
        ))
        
        # Add week-ago markers if requested
        if show_week_ago:
            week_ago_zs = [item[1]['week_ago_z'] for item in sorted_instruments]
            valid_week_ago = [(i, z) for i, z in enumerate(week_ago_zs) if z is not None]
            
            if valid_week_ago:
                indices, week_z_values = zip(*valid_week_ago)
                fig.add_trace(go.Scatter(
                    x=[instruments_short[i] for i in indices],
                    y=week_z_values,
                    mode='markers',
                    name='Week Ago',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color='purple',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='<b>%{customdata}</b><br>Week Ago Z-Score: %{y:.2f}<extra></extra>',
                    customdata=[instruments_full[i] for i in indices]
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{trader_category} Net Positioning Z-Scores (since {lookback_start.year})",
            xaxis_title="",
            yaxis_title="Z-Score",
            height=600,
            showlegend=True,
            hovermode='x unified',
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                gridcolor='lightgray'
            ),
            xaxis=dict(
                tickangle=-45
            ),
            plot_bgcolor='white',
            bargap=0.2
        )
        
        # Add reference lines
        fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1, 
                     annotation_text="", annotation_position="right")
        fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1,
                     annotation_text="", annotation_position="right")
        fig.add_hline(y=1, line_dash="dot", line_color="gray", line_width=1)
        fig.add_hline(y=-1, line_dash="dot", line_color="gray", line_width=1)
        
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating cross-asset analysis: {str(e)}")
        return None


def create_cross_asset_wow_changes(selected_instruments, trader_category, api_token, instruments_db):
    """Create week-over-week changes analysis chart"""
    try:
        # Store data for all instruments
        instrument_data = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data for each instrument
        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Fetching data for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))
            
            # Fetch data
            df = fetch_cftc_data(instrument, api_token)
            
            if df is not None and not df.empty:
                # Sort by date
                df = df.sort_values('report_date_as_yyyy_mm_dd')
                
                if len(df) >= 2:  # Need at least 2 observations for change
                    # Define column mappings
                    category_columns = {
                        "Non-Commercial": {
                            "long": "noncomm_positions_long_all",
                            "short": "noncomm_positions_short_all",
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
                    
                    cols = category_columns[trader_category]
                    
                    # Calculate net if not present
                    if trader_category == "Non-Reportable" and "net" not in cols:
                        df['net_nonrept_positions'] = df[cols['long']] - df[cols['short']]
                        cols['net'] = 'net_nonrept_positions'
                    
                    # Calculate week-over-week change
                    if 'open_interest_all' in df.columns and cols.get('net') in df.columns:
                        # Get latest and previous week data
                        latest_net = df[cols['net']].iloc[-1]
                        prev_net = df[cols['net']].iloc[-2]
                        latest_oi = df['open_interest_all'].iloc[-1]
                        
                        # Calculate change as % of open interest
                        if latest_oi > 0:
                            change_pct_oi = ((latest_net - prev_net) / latest_oi) * 100
                            
                            # Get date info
                            latest_date = df['report_date_as_yyyy_mm_dd'].iloc[-1]
                            prev_date = df['report_date_as_yyyy_mm_dd'].iloc[-2]
                            
                            instrument_data[instrument] = {
                                'change_pct_oi': change_pct_oi,
                                'latest_net': latest_net,
                                'prev_net': prev_net,
                                'latest_oi': latest_oi,
                                'latest_date': latest_date,
                                'prev_date': prev_date
                            }
        
        progress_bar.empty()
        status_text.empty()
        
        if not instrument_data:
            st.error("No valid data found for selected instruments")
            return None
        
        # Sort by change percentage (largest positive to largest negative)
        sorted_instruments = sorted(instrument_data.items(), 
                                  key=lambda x: x[1]['change_pct_oi'], 
                                  reverse=True)
        
        # Create the chart
        fig = go.Figure()
        
        # Prepare data for plotting
        instruments_full = [item[0] for item in sorted_instruments]
        # Shorten instrument names
        instruments_short = [name.split('-')[0].strip() for name in instruments_full]
        changes = [item[1]['change_pct_oi'] for item in sorted_instruments]
        
        # Use same mint green color as z-score chart
        bar_color = '#7DCEA0'  # Soft mint green
        
        # Add bars
        fig.add_trace(go.Bar(
            x=instruments_short,
            y=changes,
            name='WoW Change',
            marker=dict(color=bar_color),
            text=[f"{c:.2f}%" for c in changes],
            textposition='outside',
            hovertemplate='<b>%{customdata[0]}</b><br>' +  # Show full name
                         'WoW Change: %{y:.2f}% of OI<br>' +
                         'Latest Net: %{customdata[1]:,.0f}<br>' +
                         'Previous Net: %{customdata[2]:,.0f}<br>' +
                         'Open Interest: %{customdata[3]:,.0f}<br>' +
                         'Period: %{customdata[4]} to %{customdata[5]}<extra></extra>',
            customdata=[[full_name, 
                        item[1]['latest_net'], 
                        item[1]['prev_net'],
                        item[1]['latest_oi'],
                        item[1]['prev_date'].strftime('%Y-%m-%d'),
                        item[1]['latest_date'].strftime('%Y-%m-%d')] 
                       for item, full_name in zip(sorted_instruments, instruments_full)]
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{trader_category} Week-over-Week Changes (as % of Open Interest)",
            xaxis_title="",
            yaxis_title="Change (% of OI)",
            height=600,
            showlegend=False,
            hovermode='x unified',
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                gridcolor='lightgray',
                ticksuffix='%'
            ),
            xaxis=dict(
                tickangle=-45
            ),
            plot_bgcolor='white',
            bargap=0.2
        )
        
        # Add reference line at zero
        fig.add_hline(y=0, line_width=2, line_color="black")
        
        # Add annotation with latest date
        if sorted_instruments:
            latest_date = sorted_instruments[0][1]['latest_date'].strftime('%Y-%m-%d')
            fig.add_annotation(
                text=f"Data as of: {latest_date}",
                xref="paper", yref="paper",
                x=0.99, y=0.99,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right", yanchor="top"
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating week-over-week changes analysis: {str(e)}")
        return None


def create_positioning_concentration_charts(selected_instruments, trader_category, api_token, instruments_db):
    """Create time series and bar charts for positioning concentration analysis"""
    try:
        # Store data for all instruments
        all_data = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data for each instrument
        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Fetching data for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))
            
            # Fetch data
            df = fetch_cftc_data(instrument, api_token)
            
            if df is not None and not df.empty:
                # Sort by date
                df = df.sort_values('report_date_as_yyyy_mm_dd')
                
                # Define column mappings
                category_columns = {
                    "Non-Commercial": {
                        "long": "noncomm_positions_long_all",
                        "short": "noncomm_positions_short_all",
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
                
                cols = category_columns[trader_category]
                
                # Calculate net if not present
                if trader_category == "Non-Reportable" and "net" not in cols:
                    df['net_nonrept_positions'] = df[cols['long']] - df[cols['short']]
                    cols['net'] = 'net_nonrept_positions'
                
                # Calculate net as % of open interest
                if 'open_interest_all' in df.columns and cols.get('net') in df.columns:
                    df['net_pct_oi'] = (df[cols['net']] / df['open_interest_all'] * 100).fillna(0)
                    
                    # Store the data
                    all_data[instrument] = {
                        'dates': df['report_date_as_yyyy_mm_dd'],
                        'net_pct_oi': df['net_pct_oi'],
                        'latest_pct': df['net_pct_oi'].iloc[-1] if len(df) > 0 else 0
                    }
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            st.error("No valid data found for selected instruments")
            return None, None
        
        # Create time series chart
        time_series_fig = go.Figure()
        
        # Define a color palette for different instruments
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Add traces for each instrument
        for idx, (instrument, data) in enumerate(all_data.items()):
            # Shorten name for display
            short_name = instrument.split('-')[0].strip()
            
            time_series_fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data['net_pct_oi'],
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate=f'<b>{instrument}</b><br>' +
                             'Date: %{x}<br>' +
                             'Net % of OI: %{y:.1f}%<extra></extra>'
            ))
        
        # Update time series layout
        time_series_fig.update_layout(
            title=f"{trader_category} Positioning as % of Open Interest",
            xaxis_title="Date",
            yaxis_title="Net Positioning (% of OI)",
            height=500,
            hovermode='x unified',
            yaxis=dict(
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                gridcolor='lightgray',
                ticksuffix='%'
            ),
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
                type='date'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Create bar chart for latest values
        bar_chart_fig = go.Figure()
        
        # Sort by latest percentage
        sorted_instruments = sorted(all_data.items(), 
                                  key=lambda x: x[1]['latest_pct'], 
                                  reverse=True)
        
        # Prepare data for bar chart
        instruments_full = [item[0] for item in sorted_instruments]
        instruments_short = [name.split('-')[0].strip() for name in instruments_full]
        latest_values = [item[1]['latest_pct'] for item in sorted_instruments]
        
        # Use mint green for positive, red for negative
        colors_bar = ['#7DCEA0' if v > 0 else '#E74C3C' for v in latest_values]
        
        # Add bars
        bar_chart_fig.add_trace(go.Bar(
            x=instruments_short,
            y=latest_values,
            marker=dict(color=colors_bar),
            text=[f"{v:.1f}%" for v in latest_values],
            textposition='outside',
            hovertemplate='<b>%{customdata}</b><br>' +
                         'Net % of OI: %{y:.1f}%<extra></extra>',
            customdata=instruments_full
        ))
        
        # Update bar chart layout
        bar_chart_fig.update_layout(
            title=f"Latest {trader_category} Positioning (% of OI)",
            xaxis_title="",
            yaxis_title="Net Positioning (% of OI)",
            height=400,
            showlegend=False,
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                gridcolor='lightgray',
                ticksuffix='%'
            ),
            xaxis=dict(
                tickangle=-45
            ),
            bargap=0.2
        )
        
        # Add annotation with data date
        if sorted_instruments:
            latest_date = all_data[sorted_instruments[0][0]]['dates'].iloc[-1]
            bar_chart_fig.add_annotation(
                text=f"Data as of: {latest_date.strftime('%Y-%m-%d')}",
                xref="paper", yref="paper",
                x=0.99, y=0.99,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right", yanchor="top"
            )
        
        return time_series_fig, bar_chart_fig
        
    except Exception as e:
        st.error(f"Error creating positioning concentration charts: {str(e)}")
        return None, None


def calculate_percentiles_for_column(df, column, lookback_days):
    """Calculate rolling percentiles for a specific column"""
    percentiles = []
    for i in range(len(df)):
        current_date = df.iloc[i]['report_date_as_yyyy_mm_dd']
        current_val = df.iloc[i][column]
        
        if pd.isna(current_val):
            percentiles.append(50)
            continue
            
        if lookback_days:
            lookback_start = current_date - pd.Timedelta(days=lookback_days)
            window_data = df[(df['report_date_as_yyyy_mm_dd'] >= lookback_start) & 
                           (df['report_date_as_yyyy_mm_dd'] <= current_date)]
        else:
            window_data = df[df['report_date_as_yyyy_mm_dd'] <= current_date]
        
        valid_data = window_data[column].dropna()
        if len(valid_data) > 0:
            percentile = (valid_data < current_val).sum() / len(valid_data) * 100
            percentiles.append(percentile)
        else:
            percentiles.append(50)
    
    return percentiles


def create_participation_density_dashboard(df, instrument_name, percentile_data=None, lookback_days=None):
    """Create comprehensive avg position per trader dashboard for all categories"""
    try:
        # Copy and prepare data
        df_plot = df.copy()
        df_plot = df_plot.sort_values('report_date_as_yyyy_mm_dd')
        
        # Calculate average position per trader for overall and each category
        # Overall
        df_plot['avg_pos_per_trader'] = df_plot['open_interest_all'] / df_plot['traders_tot_all']
        
        # Non-Commercial
        df_plot['avg_noncomm_long'] = df_plot['noncomm_positions_long_all'] / df_plot['traders_noncomm_long_all']
        df_plot['avg_noncomm_short'] = df_plot['noncomm_positions_short_all'] / df_plot['traders_noncomm_short_all']
        df_plot['avg_noncomm_spread'] = df_plot['noncomm_postions_spread_all'] / df_plot['traders_noncomm_spread_all']
        
        # Commercial
        df_plot['avg_comm_long'] = df_plot['comm_positions_long_all'] / df_plot['traders_comm_long_all']
        df_plot['avg_comm_short'] = df_plot['comm_positions_short_all'] / df_plot['traders_comm_short_all']
        
        # Total Reportable
        df_plot['avg_rept_long'] = df_plot['tot_rept_positions_long_all'] / df_plot['traders_tot_rept_long_all']
        df_plot['avg_rept_short'] = df_plot['tot_rept_positions_short'] / df_plot['traders_tot_rept_short_all']
        
        # Define categories to plot
        categories = [
            ('avg_pos_per_trader', 'Overall Average', 'traders_tot_all'),
            ('avg_noncomm_long', 'Non-Commercial Long', 'traders_noncomm_long_all'),
            ('avg_noncomm_short', 'Non-Commercial Short', 'traders_noncomm_short_all'),
            ('avg_noncomm_spread', 'Non-Commercial Spread', 'traders_noncomm_spread_all'),
            ('avg_comm_long', 'Commercial Long', 'traders_comm_long_all'),
            ('avg_comm_short', 'Commercial Short', 'traders_comm_short_all'),
            ('avg_rept_long', 'Total Long', 'traders_tot_rept_long_all'),
            ('avg_rept_short', 'Total Short', 'traders_tot_rept_short_all')
        ]
        
        # Calculate percentiles for each category
        percentile_data_dict = {}
        for col, title, trader_col in categories:
            percentile_data_dict[col] = calculate_percentiles_for_column(df_plot, col, lookback_days)
        
        # Create subplot figure - 2 rows per category (value + percentile)
        num_categories = len(categories)
        fig = make_subplots(
            rows=num_categories * 2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.15, 0.08] * num_categories,  # Alternating heights for value and percentile
            specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]] + 
                  [[{"secondary_y": True}]] + [[{"secondary_y": False}]],
            subplot_titles=[title for col, title, trader_col in categories for _ in range(2)],  # Title only on value charts
            horizontal_spacing=0.01
        )
        
        # Plot each category
        for idx, (col, title, trader_col) in enumerate(categories):
            row_value = idx * 2 + 1  # Value chart row
            row_percentile = idx * 2 + 2  # Percentile chart row
            
            # Plot average position bars (green)
            fig.add_trace(
                go.Bar(
                    x=df_plot['report_date_as_yyyy_mm_dd'],
                    y=df_plot[col],
                    name=title,
                    marker_color='#90EE90',  # Light green
                    marker_line_width=0,
                    showlegend=False
                ),
                row=row_value, col=1, secondary_y=False
            )
        
            # Plot trader count on secondary axis (blue)
            fig.add_trace(
                go.Scatter(
                    x=df_plot['report_date_as_yyyy_mm_dd'],
                    y=df_plot[trader_col],
                    name=f'{title} Traders',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ),
                row=row_value, col=1, secondary_y=True
            )
            
            # Plot percentile chart (purple)
            fig.add_trace(
                go.Scatter(
                    x=df_plot['report_date_as_yyyy_mm_dd'],
                    y=percentile_data_dict[col],
                    name=f'{title} Percentile',
                    fill='tozeroy',
                    line=dict(color='purple', width=1),
                    fillcolor='rgba(128, 0, 128, 0.1)',
                    showlegend=False
                ),
                row=row_percentile, col=1
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Avg Pos", row=row_value, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Traders", row=row_value, col=1, secondary_y=True)
            fig.update_yaxes(title_text="%ile", row=row_percentile, col=1, range=[0, 100])
        
        # Update layout
        fig.update_layout(
            title=f"Comprehensive Trader Participation Analysis - {instrument_name}",
            height=150 * num_categories * 2,  # Dynamic height based on number of charts
            showlegend=False,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Add range selector only on the last x-axis
        last_row = num_categories * 2
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(label="All", step="all")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.02),
            row=last_row, col=1
        )
        
        # Configure all x-axes to be linked
        for i in range(1, last_row + 1):
            fig.update_xaxes(matches='x', row=i, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating participation density dashboard: {str(e)}")
        return None


def create_cross_asset_participation_comparison(selected_instruments, api_token, instruments_db):
    """Create cross-asset participation comparison showing trader count trends"""
    try:
        # Store data for all instruments
        all_data = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data for each instrument
        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Fetching data for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))
            
            # Fetch data
            df = fetch_cftc_data(instrument, api_token)
            
            if df is not None and not df.empty:
                # Sort by date
                df = df.sort_values('report_date_as_yyyy_mm_dd')
                
                # Store the data
                all_data[instrument] = {
                    'dates': df['report_date_as_yyyy_mm_dd'],
                    'total_traders': df['traders_tot_all'],
                    'noncomm_long': df['traders_noncomm_long_all'],
                    'noncomm_short': df['traders_noncomm_short_all'],
                    'comm_long': df['traders_comm_long_all'],
                    'comm_short': df['traders_comm_short_all'],
                    'open_interest': df['open_interest_all']
                }
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            st.error("No valid data found for selected instruments")
            return None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Trader Count', 'Trader Count YoY % Change', 
                           'Avg Position per Trader', 'Participation Score'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Define colors for instruments
        colors = px.colors.qualitative.Plotly[:len(selected_instruments)]
        
        # 1. Total trader count time series
        for idx, (instrument, data) in enumerate(all_data.items()):
            short_name = instrument.split('-')[0].strip()
            
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data['total_traders'],
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx], width=2),
                showlegend=True,
                legendgroup=short_name,
                hovertemplate=f'<b>{instrument}</b><br>Date: %{{x}}<br>Traders: %{{y:,.0f}}<extra></extra>'
            ), row=1, col=1)
        
        # 2. YoY % change in trader count
        for idx, (instrument, data) in enumerate(all_data.items()):
            short_name = instrument.split('-')[0].strip()
            
            # Calculate YoY change
            trader_series = pd.Series(data['total_traders'].values, index=data['dates'])
            yoy_change = trader_series.pct_change(52) * 100  # 52 weeks = 1 year
            
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=yoy_change,
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx], width=2),
                showlegend=False,
                legendgroup=short_name,
                hovertemplate=f'<b>{instrument}</b><br>Date: %{{x}}<br>YoY Change: %{{y:.1f}}%<extra></extra>'
            ), row=1, col=2)
        
        # 3. Average position per trader
        for idx, (instrument, data) in enumerate(all_data.items()):
            short_name = instrument.split('-')[0].strip()
            
            # Calculate avg position per trader
            avg_position = data['open_interest'] / data['total_traders']
            
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=avg_position,
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx], width=2),
                showlegend=False,
                legendgroup=short_name,
                hovertemplate=f'<b>{instrument}</b><br>Date: %{{x}}<br>Avg Position: %{{y:,.0f}}<extra></extra>'
            ), row=2, col=1)
        
        # 4. Participation score (traders as % of max historical)
        for idx, (instrument, data) in enumerate(all_data.items()):
            short_name = instrument.split('-')[0].strip()
            
            # Calculate participation score
            max_traders = data['total_traders'].max()
            participation_score = (data['total_traders'] / max_traders) * 100
            
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=participation_score,
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx], width=2),
                showlegend=False,
                legendgroup=short_name,
                hovertemplate=f'<b>{instrument}</b><br>Date: %{{x}}<br>Participation: %{{y:.1f}}%<extra></extra>'
            ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="Cross-Asset Participation Comparison",
            height=800,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Number of Traders", row=1, col=1)
        fig.update_yaxes(title_text="YoY Change %", row=1, col=2)
        fig.update_yaxes(title_text="Avg Position Size", row=2, col=1)
        fig.update_yaxes(title_text="Participation %", row=2, col=2)
        
        # Add zero line for YoY change
        fig.add_hline(y=0, row=1, col=2, line_dash="dash", line_color="gray")
        
        # Add range selector to first subplot
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            row=1, col=1
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating participation comparison: {str(e)}")
        return None


def create_relative_strength_matrix(selected_instruments, api_token, time_period, instruments_db):
    """Create relative strength heatmap matrix across instruments and time"""
    try:
        # Map time period to days
        period_map = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        days = period_map[time_period]
        
        # Store data for all instruments
        all_data = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data for each instrument
        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Fetching data for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))
            
            # Fetch data
            df = fetch_cftc_data(instrument, api_token)
            
            if df is not None and not df.empty:
                # Sort by date and get recent data
                df = df.sort_values('report_date_as_yyyy_mm_dd')
                cutoff_date = df['report_date_as_yyyy_mm_dd'].max() - pd.DateOffset(days=days)
                df_recent = df[df['report_date_as_yyyy_mm_dd'] >= cutoff_date]
                
                if not df_recent.empty:
                    all_data[instrument] = df_recent
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            st.error("No valid data found for selected instruments")
            return None
        
        # Create weekly time buckets
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(days=days)
        weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W')
        
        # Initialize matrix data
        matrix_data = []
        instrument_names = []
        
        # Calculate net positioning strength for each instrument and time bucket
        for instrument in selected_instruments:
            if instrument in all_data:
                df = all_data[instrument]
                instrument_names.append(instrument.split('-')[0].strip())
                
                weekly_strengths = []
                
                for i in range(len(weekly_dates) - 1):
                    week_start = weekly_dates[i]
                    week_end = weekly_dates[i + 1]
                    
                    # Get data for this week
                    week_data = df[(df['report_date_as_yyyy_mm_dd'] >= week_start) & 
                                  (df['report_date_as_yyyy_mm_dd'] < week_end)]
                    
                    if not week_data.empty:
                        # Use the latest data point in the week
                        latest = week_data.iloc[-1]
                        
                        # Calculate net non-commercial positioning
                        net_noncomm = latest['noncomm_positions_long_all'] - latest['noncomm_positions_short_all']
                        net_pct = (net_noncomm / latest['open_interest_all']) * 100
                        
                        # Calculate z-score based on historical data
                        historical_net = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
                        historical_pct = (historical_net / df['open_interest_all']) * 100
                        
                        mean = historical_pct.mean()
                        std = historical_pct.std()
                        
                        if std > 0:
                            z_score = (net_pct - mean) / std
                        else:
                            z_score = 0
                        
                        weekly_strengths.append(z_score)
                    else:
                        weekly_strengths.append(np.nan)
                
                matrix_data.append(weekly_strengths)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=[f"Week {i+1}" for i in range(len(weekly_dates) - 1)],
            y=instrument_names,
            colorscale='RdYlGn',
            zmid=0,
            zmin=-3,
            zmax=3,
            text=[[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in matrix_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{y}<br>%{x}<br>Z-Score: %{z:.2f}<extra></extra>',
            colorbar=dict(
                title="Z-Score",
                titleside="right",
                tickmode="linear",
                tick0=-3,
                dtick=1
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Relative Positioning Strength Matrix - {time_period}",
            xaxis=dict(title="Time Period", side="bottom"),
            yaxis=dict(title="Instruments", autorange="reversed"),
            height=400 + (len(instrument_names) * 30),  # Dynamic height based on instruments
            hovermode='closest'
        )
        
        # Add annotations for interpretation
        fig.add_annotation(
            text="🟢 Green = Bullish positioning | 🔴 Red = Bearish positioning",
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=12),
            xanchor="center", yanchor="top"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating relative strength matrix: {str(e)}")
        return None


def create_market_structure_matrix(all_instruments_data, selected_instruments):
    """Create market structure matrix showing instruments on 2x2 grid based on trader count and concentration"""
    try:
        # Prepare data for all selected instruments
        scatter_data = []
        
        for instrument in selected_instruments:
            if instrument in all_instruments_data:
                df = all_instruments_data[instrument]
                
                # Get latest data
                latest_idx = df['report_date_as_yyyy_mm_dd'].idxmax()
                latest_data = df.loc[latest_idx]
                
                # Calculate metrics
                trader_count = latest_data.get('traders_tot_all', 0)
                open_interest = latest_data.get('open_interest_all', 0)
                
                # Calculate concentration (average of long and short)
                conc_long = latest_data.get('conc_net_le_4_tdr_long_all', 0)
                conc_short = latest_data.get('conc_net_le_4_tdr_short_all', 0)
                concentration = (conc_long + conc_short) / 2 if conc_long and conc_short else 50
                
                # Add to scatter data
                scatter_data.append({
                    'instrument': instrument,
                    'trader_count': trader_count,
                    'concentration': concentration,
                    'open_interest': open_interest,
                    'short_name': instrument.split('-')[0].strip()
                })
        
        # Create scatter plot
        fig = go.Figure()
        
        # Define quadrant colors
        colors = []
        for item in scatter_data:
            if item['trader_count'] > np.median([d['trader_count'] for d in scatter_data]):
                if item['concentration'] < 50:
                    colors.append('#2ECC71')  # High traders, Low concentration (best)
                else:
                    colors.append('#F39C12')  # High traders, High concentration
            else:
                if item['concentration'] < 50:
                    colors.append('#3498DB')  # Low traders, Low concentration
                else:
                    colors.append('#E74C3C')  # Low traders, High concentration (worst)
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=[d['trader_count'] for d in scatter_data],
            y=[d['concentration'] for d in scatter_data],
            mode='markers+text',
            marker=dict(
                size=[np.sqrt(d['open_interest']/1000) for d in scatter_data],  # Scale bubble size
                color=colors,
                line=dict(width=2, color='white'),
                sizemode='diameter',
                sizemin=10,
                sizeref=2
            ),
            text=[d['short_name'] for d in scatter_data],
            textposition='top center',
            hovertemplate='<b>%{customdata}</b><br>' +
                         'Traders: %{x:,.0f}<br>' +
                         'Concentration: %{y:.1f}%<br>' +
                         'Open Interest: %{marker.size:,.0f}<extra></extra>',
            customdata=[d['instrument'] for d in scatter_data]
        ))
        
        # Add quadrant lines
        median_traders = np.median([d['trader_count'] for d in scatter_data])
        
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=median_traders, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(
            text="Low Concentration<br>Many Traders",
            xref="paper", yref="paper",
            x=0.75, y=0.25,
            showarrow=False,
            font=dict(size=12, color="green"),
            opacity=0.8
        )
        
        fig.add_annotation(
            text="High Concentration<br>Many Traders",
            xref="paper", yref="paper",
            x=0.75, y=0.75,
            showarrow=False,
            font=dict(size=12, color="orange"),
            opacity=0.8
        )
        
        fig.add_annotation(
            text="Low Concentration<br>Few Traders",
            xref="paper", yref="paper",
            x=0.25, y=0.25,
            showarrow=False,
            font=dict(size=12, color="blue"),
            opacity=0.8
        )
        
        fig.add_annotation(
            text="High Concentration<br>Few Traders",
            xref="paper", yref="paper",
            x=0.25, y=0.75,
            showarrow=False,
            font=dict(size=12, color="red"),
            opacity=0.8
        )
        
        # Update layout
        fig.update_layout(
            title="Market Structure Matrix",
            xaxis_title="Number of Traders",
            yaxis_title="Concentration Ratio (% held by top 4 traders)",
            height=600,
            showlegend=False,
            xaxis=dict(
                type='log',  # Log scale for better distribution
                gridcolor='lightgray',
                zeroline=False
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor='lightgray',
                zeroline=False
            )
        )
        
        # Add size legend
        fig.add_annotation(
            text="Bubble size = Open Interest",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10),
            xanchor="left", yanchor="top"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating market structure matrix: {str(e)}")
        return None


def create_category_flow_analysis(df, instrument_name):
    """Create category flow analysis showing week-over-week trader count changes"""
    try:
        # Calculate week-over-week changes for each trader category
        df = df.copy().sort_values('report_date_as_yyyy_mm_dd')
        
        # Trader categories to analyze
        trader_categories = {
            'Non-Commercial Long': 'traders_noncomm_long_all',
            'Non-Commercial Short': 'traders_noncomm_short_all',
            'Non-Commercial Spread': 'traders_noncomm_spread_all',
            'Commercial Long': 'traders_comm_long_all',
            'Commercial Short': 'traders_comm_short_all',
            'Total Reportable Long': 'traders_tot_rept_long_all',
            'Total Reportable Short': 'traders_tot_rept_short_all'
        }
        
        # Calculate changes
        for name, col in trader_categories.items():
            if col in df.columns:
                df[f'{col}_change'] = df[col].diff()
                df[f'{col}_pct_change'] = df[col].pct_change() * 100
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.4, 0.3, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                'Trader Count by Category',
                'Week-over-Week Change (Absolute)',
                'Week-over-Week Change (%)'
            )
        )
        
        # Color palette
        colors = ['#3498DB', '#E74C3C', '#F39C12', '#2ECC71', '#9B59B6', '#1ABC9C', '#34495E']
        
        # Plot 1: Absolute trader counts
        for idx, (name, col) in enumerate(trader_categories.items()):
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['report_date_as_yyyy_mm_dd'],
                        y=df[col],
                        name=name,
                        mode='lines',
                        line=dict(color=colors[idx % len(colors)], width=2),
                        hovertemplate=f'{name}<br>Date: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Absolute changes
        for idx, (name, col) in enumerate(trader_categories.items()):
            if f'{col}_change' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df['report_date_as_yyyy_mm_dd'],
                        y=df[f'{col}_change'],
                        name=name,
                        marker=dict(color=colors[idx % len(colors)]),
                        showlegend=False,
                        hovertemplate=f'{name}<br>Date: %{{x}}<br>Change: %{{y:+,.0f}}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Plot 3: Percentage changes
        for idx, (name, col) in enumerate(trader_categories.items()):
            if f'{col}_pct_change' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['report_date_as_yyyy_mm_dd'],
                        y=df[f'{col}_pct_change'],
                        name=name,
                        mode='lines+markers',
                        line=dict(color=colors[idx % len(colors)], width=1.5),
                        marker=dict(size=4),
                        showlegend=False,
                        hovertemplate=f'{name}<br>Date: %{{x}}<br>Change: %{{y:+.1f}}%<extra></extra>'
                    ),
                    row=3, col=1
                )
        
        # Add zero lines for reference
        fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Category Flow Analysis - {instrument_name}",
                y=0.98,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            height=900,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis3=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Trader Count", row=1, col=1, tickformat=",.0f")
        fig.update_yaxes(title_text="Absolute Change", row=2, col=1, tickformat="+,.0f")
        fig.update_yaxes(title_text="% Change", row=3, col=1, tickformat="+.1f%")
        
        # Update x-axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating category flow analysis: {str(e)}")
        return None


def create_institutional_vs_nonreportable(df, instrument_name):
    """Create institutional vs non-reportable analysis comparing large and small traders"""
    try:
        # Calculate non-reportable positions
        df = df.copy().sort_values('report_date_as_yyyy_mm_dd')
        
        # Calculate non-reportable traders (approximation)
        # Non-reportable = Total OI - Reportable positions
        df['nonrept_net'] = df['nonrept_positions_long_all'] - df['nonrept_positions_short_all']
        df['reportable_net'] = (df['tot_rept_positions_long_all'] - df['tot_rept_positions_short']).fillna(0)
        
        # Calculate percentages
        df['nonrept_pct_oi'] = (df['nonrept_net'] / df['open_interest_all']) * 100
        df['reportable_pct_oi'] = (df['reportable_net'] / df['open_interest_all']) * 100
        
        # Estimate non-reportable trader count (total - reportable)
        df['nonrept_traders_est'] = df['traders_tot_all'] - (df['traders_tot_rept_long_all'] + df['traders_tot_rept_short_all']) / 2
        df['nonrept_traders_est'] = df['nonrept_traders_est'].clip(lower=0)  # Ensure non-negative
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.35, 0.35, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                'Net Positioning: Institutional vs Non-Reportable',
                'Positioning as % of Open Interest',
                'Divergence Indicator'
            ),
            specs=[[{"secondary_y": True}], 
                   [{"secondary_y": False}], 
                   [{"secondary_y": False}]]
        )
        
        # Plot 1: Net positions with dual axis
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['reportable_net'],
                name='Institutional (Reportable)',
                mode='lines',
                line=dict(color='#3498DB', width=3),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)',
                hovertemplate='Date: %{x}<br>Net Position: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['nonrept_net'],
                name='Non-Reportable (Retail)',
                mode='lines',
                line=dict(color='#E74C3C', width=2),
                yaxis='y2',
                hovertemplate='Date: %{x}<br>Net Position: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Plot 2: Percentage of OI
        fig.add_trace(
            go.Bar(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['reportable_pct_oi'],
                name='Institutional % of OI',
                marker=dict(color='#3498DB'),
                hovertemplate='Date: %{x}<br>% of OI: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['nonrept_pct_oi'],
                name='Non-Reportable % of OI',
                marker=dict(color='#E74C3C'),
                hovertemplate='Date: %{x}<br>% of OI: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Plot 3: Divergence indicator
        # Calculate correlation and divergence
        window = 20
        df['inst_direction'] = df['reportable_net'].diff().rolling(window=window).mean()
        df['retail_direction'] = df['nonrept_net'].diff().rolling(window=window).mean()
        df['divergence'] = (df['inst_direction'] * df['retail_direction'] < 0).astype(int)
        
        # Color based on who's buying vs selling
        colors = []
        for idx in range(len(df)):
            if df['divergence'].iloc[idx] == 1:
                if df['inst_direction'].iloc[idx] > 0:
                    colors.append('#2ECC71')  # Green - Inst buying, retail selling
                else:
                    colors.append('#E74C3C')  # Red - Inst selling, retail buying
            else:
                colors.append('#95A5A6')  # Gray - Both same direction
        
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['divergence'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    symbol='diamond'
                ),
                name='Divergence Events',
                hovertemplate='Date: %{x}<br>Divergence: %{y}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add annotations for divergence
        fig.add_annotation(
            text="🟢 Institutions Buying / Retail Selling",
            xref="paper", yref="paper",
            x=0.02, y=0.35,
            showarrow=False,
            font=dict(size=10, color='green'),
            xanchor="left"
        )
        
        fig.add_annotation(
            text="🔴 Institutions Selling / Retail Buying",
            xref="paper", yref="paper",
            x=0.02, y=0.30,
            showarrow=False,
            font=dict(size=10, color='red'),
            xanchor="left"
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Institutional vs Non-Reportable Analysis - {instrument_name}",
                y=0.98,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            height=900,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            barmode='group',
            xaxis3=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        
        # Update axes
        fig.update_yaxes(title_text="Institutional Net Position", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Non-Reportable Net Position", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="% of Open Interest", row=2, col=1)
        fig.update_yaxes(title_text="Divergence Signal", row=3, col=1, range=[-0.1, 1.1])
        
        # Add zero lines
        fig.add_hline(y=0, row=1, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating institutional vs non-reportable analysis: {str(e)}")
        return None


def create_concentration_risk_monitor(df, instrument_name):
    """Create concentration risk monitor showing top trader concentration metrics"""
    try:
        df = df.copy().sort_values('report_date_as_yyyy_mm_dd')
        
        # Create figure with subplots
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=[0.25, 0.25, 0.25, 0.25],
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                'Top 4 Traders Concentration (Net Positions)',
                'Top 8 Traders Concentration (Net Positions)',
                'Concentration Risk Score',
                'Market Dominance Indicator'
            )
        )
        
        # Plot 1: Top 4 concentration
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['conc_net_le_4_tdr_long_all'],
                name='Top 4 Long',
                mode='lines',
                line=dict(color='#2ECC71', width=2),
                hovertemplate='Date: %{x}<br>Concentration: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['conc_net_le_4_tdr_short_all'],
                name='Top 4 Short',
                mode='lines',
                line=dict(color='#E74C3C', width=2),
                hovertemplate='Date: %{x}<br>Concentration: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Plot 2: Top 8 concentration
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['conc_net_le_8_tdr_long_all'],
                name='Top 8 Long',
                mode='lines',
                line=dict(color='#2ECC71', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Concentration: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['conc_net_le_8_tdr_short_all'],
                name='Top 8 Short',
                mode='lines',
                line=dict(color='#E74C3C', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Concentration: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Plot 3: Risk Score (combining multiple factors)
        # Calculate risk score based on concentration levels
        df['avg_conc_4'] = (df['conc_net_le_4_tdr_long_all'] + df['conc_net_le_4_tdr_short_all']) / 2
        df['avg_conc_8'] = (df['conc_net_le_8_tdr_long_all'] + df['conc_net_le_8_tdr_short_all']) / 2
        
        # Risk score: higher concentration = higher risk
        df['risk_score'] = (df['avg_conc_4'] * 0.6 + df['avg_conc_8'] * 0.4)
        
        # Color based on risk level
        colors = []
        for score in df['risk_score']:
            if score < 40:
                colors.append('#2ECC71')  # Low risk
            elif score < 60:
                colors.append('#F39C12')  # Medium risk
            else:
                colors.append('#E74C3C')  # High risk
        
        fig.add_trace(
            go.Bar(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['risk_score'],
                name='Risk Score',
                marker=dict(color=colors),
                hovertemplate='Date: %{x}<br>Risk Score: %{y:.1f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add risk level lines
        fig.add_hline(y=40, row=3, col=1, line_dash="dash", line_color="green", 
                      annotation_text="Low Risk", annotation_position="right")
        fig.add_hline(y=60, row=3, col=1, line_dash="dash", line_color="orange", 
                      annotation_text="High Risk", annotation_position="right")
        
        # Plot 4: Market Dominance (ratio of top 4 to total traders)
        df['dominance_ratio'] = (4 / df['traders_tot_all']) * 100 * df['avg_conc_4']
        
        fig.add_trace(
            go.Scatter(
                x=df['report_date_as_yyyy_mm_dd'],
                y=df['dominance_ratio'],
                mode='lines+markers',
                line=dict(color='#9B59B6', width=2),
                marker=dict(size=4),
                fill='tozeroy',
                fillcolor='rgba(155, 89, 182, 0.2)',
                name='Dominance Ratio',
                hovertemplate='Date: %{x}<br>Dominance: %{y:.2f}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Concentration Risk Monitor - {instrument_name}",
                y=0.98,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            height=1000,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis4=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="% of OI", row=1, col=1, range=[0, 100])
        fig.update_yaxes(title_text="% of OI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Risk Score", row=3, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Dominance", row=4, col=1)
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating concentration risk monitor: {str(e)}")
        return None


def get_asset_class(instrument_name, instruments_db):
    """Determine asset class from instrument name"""
    # Simple classification based on common patterns
    instrument_upper = instrument_name.upper()
    
    if any(keyword in instrument_upper for keyword in ['CRUDE', 'GAS', 'OIL', 'RBOB', 'HEATING']):
        return 'Energy'
    elif any(keyword in instrument_upper for keyword in ['GOLD', 'SILVER', 'COPPER', 'PLATINUM', 'PALLADIUM']):
        return 'Metals'
    elif any(keyword in instrument_upper for keyword in ['WHEAT', 'CORN', 'SOYBEAN', 'COTTON', 'SUGAR', 'COFFEE', 'COCOA', 'CATTLE', 'HOGS']):
        return 'Agriculture'
    elif any(keyword in instrument_upper for keyword in ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'DOLLAR INDEX', 'DX']):
        return 'Currencies'
    elif any(keyword in instrument_upper for keyword in ['S&P', 'NASDAQ', 'DOW', 'RUSSELL', 'VIX', 'DJIA']):
        return 'Indices'
    elif any(keyword in instrument_upper for keyword in ['BOND', 'NOTE', 'BILL', 'TREASURY']):
        return 'Bonds'
    else:
        return 'Other'


def create_momentum_dashboard(df, change_columns):
    """Create momentum dashboard with heat strips and advanced analytics"""
    try:
        # Validate that we have change columns
        if not change_columns:
            st.warning("No change columns available for momentum analysis")
            return None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.4, 0.3, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Heat Map: Weekly Changes', 'Z-Score Analysis', 'Regime Detection')
        )
        
        # Prepare data
        df_momentum = df.copy()
        df_momentum = df_momentum.sort_values('report_date_as_yyyy_mm_dd')
        
        # 1. Heat Map for all change columns
        for i, col in enumerate(change_columns):
            # Calculate z-scores for coloring
            values = df_momentum[col].fillna(0)
            z_scores = (values - values.mean()) / values.std()
            
            # Create heat strip
            # Clean up column name for display
            display_name = col.replace('change_in_', '').replace('calculated_change_', '').replace('_', ' ').title()
            
            fig.add_trace(
                go.Scatter(
                    x=df_momentum['report_date_as_yyyy_mm_dd'],
                    y=[i] * len(df_momentum),
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=z_scores,
                        colorscale='RdYlGn',
                        cmin=-3,
                        cmax=3,
                        colorbar=dict(
                            title="Z-Score",
                            x=1.05,
                            len=0.3,
                            y=0.85
                        ) if i == 0 else None,
                        symbol='square',
                        line=dict(width=0)
                    ),
                    name=display_name,
                    hovertemplate='%{text}<br>Date: %{x}<br>Z-Score: %{marker.color:.2f}<extra></extra>',
                    text=[f"{display_name}: {v:,.0f}" for v in values],
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Update y-axis for heat map
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(change_columns))),
            ticktext=[col.replace('change_in_', '').replace('calculated_change_', '').replace('_', ' ').title() for col in change_columns],
            row=1, col=1
        )
        
        # 2. Z-Score time series for selected columns (max 3)
        colors = ['blue', 'red', 'green']
        for i, col in enumerate(change_columns[:3]):
            values = df_momentum[col].fillna(0)
            z_scores = (values - values.mean()) / values.std()
            
            fig.add_trace(
                go.Scatter(
                    x=df_momentum['report_date_as_yyyy_mm_dd'],
                    y=z_scores,
                    mode='lines',
                    name=col.replace('change_in_', '').replace('calculated_change_', '').replace('_', ' ').title(),
                    line=dict(color=colors[i % 3], width=2),
                    hovertemplate='%{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add reference lines for z-scores
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)
        fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
        
        # 3. Regime Detection (using first change column)
        if change_columns:
            main_col = change_columns[0]
            values = df_momentum[main_col].fillna(0)
            
            # Simple regime detection using rolling statistics
            rolling_mean = values.rolling(window=12, min_periods=1).mean()
            rolling_std = values.rolling(window=12, min_periods=1).std()
            
            # Detect trending vs mean-reverting regimes
            # Trending: sustained moves above/below mean
            # Mean-reverting: oscillating around mean
            regime = pd.Series(index=df_momentum.index, dtype='object')
            
            for i in range(len(values)):
                if i < 12:
                    regime.iloc[i] = 'Insufficient Data'
                else:
                    recent_values = values.iloc[max(0, i-12):i]
                    mean = recent_values.mean()
                    
                    # Count how many values are above/below mean
                    above_mean = (recent_values > mean).sum()
                    below_mean = (recent_values < mean).sum()
                    
                    if above_mean > 8 or below_mean > 8:
                        regime.iloc[i] = 'Trending'
                    else:
                        regime.iloc[i] = 'Mean-Reverting'
            
            # Plot regime
            regime_colors = {'Trending': 1, 'Mean-Reverting': 0, 'Insufficient Data': 0.5}
            regime_numeric = regime.map(regime_colors)
            
            fig.add_trace(
                go.Scatter(
                    x=df_momentum['report_date_as_yyyy_mm_dd'],
                    y=regime_numeric,
                    mode='lines',
                    fill='tozeroy',
                    name='Market Regime',
                    line=dict(color='purple', width=2),
                    fillcolor='rgba(128, 0, 128, 0.3)',
                    hovertemplate='Regime: %{text}<extra></extra>',
                    text=regime
                ),
                row=3, col=1
            )
            
            # Add momentum strength indicator
            momentum_strength = values.rolling(window=4, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=df_momentum['report_date_as_yyyy_mm_dd'],
                    y=momentum_strength,
                    mode='lines',
                    name='Momentum Strength',
                    line=dict(color='orange', width=2),
                    yaxis='y2',
                    hovertemplate='Momentum: %{y:,.0f}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Momentum Dashboard - {', '.join([col.replace('change_in_', '').replace('calculated_change_', '').replace('_', ' ').title() for col in change_columns[:2]])}{'...' if len(change_columns) > 2 else ''}",
            height=900,
            showlegend=True,
            hovermode='x unified',
            xaxis3=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(label="All", step="all")
                    ])
                ),
                rangeslider=dict(visible=True, thickness=0.05)
            ),
            yaxis2=dict(title="", range=[-0.1, 1.1], showticklabels=False),
            yaxis4=dict(title="Z-Score", zeroline=True),
            yaxis5=dict(title="Regime", range=[-0.1, 1.1]),
            yaxis6=dict(title="Momentum", overlaying='y5', side='right')
        )
        
        # Configure axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating momentum dashboard: {str(e)}")
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
    st.markdown("Interactive analysis of CFTC COT data")

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

    # Chart Type Selection FIRST
    st.header("📈 Select Analysis Type")
    
    col_single, col_multi = st.columns(2)
    
    with col_single:
        st.markdown("### Single Instrument Analysis")
        single_chart_type = st.segmented_control(
            "Select chart type",
            ["Time Series", "Percentile", "Momentum", "Trader Participation"],
            selection_mode="single",
            default=None,
            key="single_chart_type",
            label_visibility="collapsed"
        )
    
    with col_multi:
        st.markdown("### Multi-Instrument Analysis")
        multi_chart_type = st.segmented_control(
            "Select chart type",
            ["Cross-Asset", "Market Matrix", "Asset Concentration", "Z-Score Analysis", "WoW Changes", "Positioning Conc.", "Participation", "Strength Matrix"],
            selection_mode="single", 
            default=None,
            key="multi_chart_type",
            label_visibility="collapsed"
        )
    
    # Clear the other selection when one is selected
    if single_chart_type and multi_chart_type:
        if st.session_state.get('last_selection') == 'single':
            multi_chart_type = None
        else:
            single_chart_type = None
    
    # Track the last selection
    if single_chart_type:
        st.session_state['last_selection'] = 'single'
    elif multi_chart_type:
        st.session_state['last_selection'] = 'multi'
    
    st.markdown("---")
    
    # Determine which type of analysis was selected
    if single_chart_type:
        # Single Instrument Flow
        handle_single_instrument_flow(single_chart_type, instruments_db, api_token)
    elif multi_chart_type:
        # Multi-Instrument Flow
        handle_multi_instrument_flow(multi_chart_type, instruments_db, api_token)
    else:
        # No chart type selected yet
        st.info("👆 Please select an analysis type from the options above to begin")


def handle_single_instrument_flow(chart_type, instruments_db, api_token):
    """Handle single instrument selection and analysis"""
    st.header("🎯 Select Instrument")
    
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
        st.markdown("---")
        
        # Display chart based on pre-selected chart type
        if chart_type == "Time Series":
            display_time_series_chart(df, selected_instrument)
        elif chart_type == "Percentile":
            display_percentile_chart(df, selected_instrument)
        elif chart_type == "Momentum":
            display_momentum_chart(df, selected_instrument)
        elif chart_type == "Trader Participation":
            display_trader_participation_chart(df, selected_instrument)


# Display functions for single instrument charts
def display_time_series_chart(df, instrument_name):
    """Display time series analysis"""
    st.subheader("📈 Time Series Analysis")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Standard Time Series", "Share of Open Interest", "Seasonality"])
    
    with tab1:
        # Date range selection
        min_date = df['report_date_as_yyyy_mm_dd'].min()
        max_date = df['report_date_as_yyyy_mm_dd'].max()
        available_dates = sorted(df['report_date_as_yyyy_mm_dd'].unique())
        
        date_range = st.select_slider(
            "Select Date Range:",
            options=range(len(available_dates)),
            value=(0, len(available_dates) - 1),
            format_func=lambda x: available_dates[x].strftime('%Y-%m-%d')
        )
        
        start_date = available_dates[date_range[0]]
        end_date = available_dates[date_range[1]]
        
        filtered_df = df[
            (df['report_date_as_yyyy_mm_dd'] >= start_date) &
            (df['report_date_as_yyyy_mm_dd'] <= end_date)
        ].copy()
        
        st.info(f"📊 Showing {len(filtered_df)} records from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Column selection with checkboxes
        st.markdown("#### Select data series to plot:")
        
        # Define column display names
        column_display_names = {
            "open_interest_all": "Open Interest",
            "noncomm_positions_long_all": "Non-Commercial Long", 
            "noncomm_positions_short_all": "Non-Commercial Short",
            "comm_positions_long_all": "Commercial Long",
            "comm_positions_short_all": "Commercial Short",
            "net_noncomm_positions": "Net Non-Commercial",
            "net_comm_positions": "Net Commercial",
            "net_reportable_positions": "Net Reportable"
        }
        
        # Create columns for checkboxes
        col1, col2, col3 = st.columns(3)
        selected_columns = []
        
        # Get available columns in the data
        available_columns = list(column_display_names.keys())
        existing_columns = [col for col in available_columns if col in filtered_df.columns]
        
        # Distribute checkboxes across columns
        for idx, col in enumerate(existing_columns):
            with [col1, col2, col3][idx % 3]:
                if st.checkbox(column_display_names.get(col, col), 
                              value=(col == "open_interest_all"),
                              key=f"ts_checkbox_{col}"):
                    selected_columns.append(col)
        
        if selected_columns:
            fig = create_plotly_chart(filtered_df, selected_columns, f"{instrument_name} - Time Series Analysis")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one data series to plot")
    
    with tab2:
        # Share of Open Interest view
        st.markdown("#### Share of Open Interest Over Time")
        
        # Add explanation
        st.info("""
        This chart shows how open interest is distributed among different trader categories as a percentage of total.
        
        **Calculation Method:**
        - **Long Side**: NonComm Long + Spread + Comm Long + NonRep Long = 100%
        - **Short Side**: NonComm Short + Spread + Comm Short + NonRep Short = 100%
        """)
        
        # Calculation side selector
        calculation_side = st.selectbox(
            "Calculate percentages using:",
            ["Long Side", "Short Side"],
            index=0,
            key="share_oi_side_selector"
        )
        
        # Create and display the chart
        fig = create_share_of_oi_chart(df, calculation_side, instrument_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Seasonality Analysis
        st.markdown("#### Seasonality Analysis")
        
        # Create grouped metrics structure
        metric_groups = {
            "📊 Core Metrics": {
                "open_interest_all": "Open Interest"
            },
            "📈 Long Positions": {
                "noncomm_positions_long_all": "Non-Commercial Long",
                "comm_positions_long_all": "Commercial Long",
                "tot_rept_positions_long_all": "Total Reportable Long",
                "nonrept_positions_long_all": "Non-Reportable Long"
            },
            "📉 Short Positions": {
                "noncomm_positions_short_all": "Non-Commercial Short",
                "comm_positions_short_all": "Commercial Short",
                "tot_rept_positions_short": "Total Reportable Short",
                "nonrept_positions_short_all": "Non-Reportable Short"
            },
            "🔄 Spread Positions": {
                "noncomm_postions_spread_all": "Non-Commercial Spread"
            },
            "⚖️ Net Positioning": {
                "net_noncomm_positions": "Net Non-Commercial",
                "net_comm_positions": "Net Commercial",
                "net_reportable_positions": "Net Reportable"
            },
            "👥 Trader Counts": {
                "traders_tot_all": "Total Traders",
                "traders_noncomm_long_all": "Non-Commercial Long Traders",
                "traders_noncomm_short_all": "Non-Commercial Short Traders",
                "traders_noncomm_spread_all": "Non-Commercial Spread Traders",
                "traders_comm_long_all": "Commercial Long Traders",
                "traders_comm_short_all": "Commercial Short Traders",
                "traders_tot_rept_long_all": "Total Reportable Long Traders",
                "traders_tot_rept_short_all": "Total Reportable Short Traders"
            },
            "📊 % of Open Interest": {
                "pct_of_oi_noncomm_long_all": "Non-Commercial Long",
                "pct_of_oi_noncomm_short_all": "Non-Commercial Short",
                "pct_of_oi_noncomm_spread": "Non-Commercial Spread",
                "pct_of_oi_comm_long_all": "Commercial Long",
                "pct_of_oi_comm_short_all": "Commercial Short",
                "pct_of_oi_nonrept_long_all": "Non-Reportable Long",
                "pct_of_oi_nonrept_short_all": "Non-Reportable Short"
            }
        }
        
        # Build options list with available columns only
        options = []
        option_labels = {}
        
        for group_name, group_metrics in metric_groups.items():
            group_options = [(k, v) for k, v in group_metrics.items() if k in df.columns]
            if group_options:
                # Add group header
                for key, label in group_options:
                    options.append(key)
                    option_labels[key] = f"{group_name} → {label}"
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Find default index
            default_key = 'net_noncomm_positions' if 'net_noncomm_positions' in options else (options[0] if options else None)
            default_index = options.index(default_key) if default_key and default_key in options else 0
            
            seasonality_column = st.selectbox(
                "Select metric:",
                options,
                format_func=lambda x: option_labels.get(x, x),
                index=default_index,
                key="seasonality_metric_selector"
            )
        
        with col2:
            lookback_years = st.selectbox(
                "Lookback period:",
                [5, 10, 'all'],
                format_func=lambda x: f"{x} Years" if x != 'all' else "All Time",
                index=0,
                key="seasonality_lookback"
            )
        
        with col3:
            zone_type = st.radio(
                "Zone type:",
                ['percentile', 'std'],
                format_func=lambda x: 'Percentile' if x == 'percentile' else 'Std Dev',
                index=0,
                key="seasonality_zone"
            )
        
        show_previous = st.checkbox("Show previous year", value=True, key="seasonality_prev_year")
        
        if seasonality_column in df.columns:
            fig = create_seasonality_chart(df, seasonality_column, lookback_years, show_previous, zone_type)
            if fig:
                st.plotly_chart(fig, use_container_width=True)


def display_percentile_chart(df, instrument_name):
    """Display percentile analysis"""
    st.subheader("📊 Percentile Analysis")
    
    # Create grouped metrics structure (same as seasonality)
    metric_groups = {
        "📊 Core Metrics": {
            "open_interest_all": "Open Interest"
        },
        "📈 Long Positions": {
            "noncomm_positions_long_all": "Non-Commercial Long",
            "comm_positions_long_all": "Commercial Long",
            "tot_rept_positions_long_all": "Total Reportable Long",
            "nonrept_positions_long_all": "Non-Reportable Long"
        },
        "📉 Short Positions": {
            "noncomm_positions_short_all": "Non-Commercial Short",
            "comm_positions_short_all": "Commercial Short",
            "tot_rept_positions_short": "Total Reportable Short",
            "nonrept_positions_short_all": "Non-Reportable Short"
        },
        "🔄 Spread Positions": {
            "noncomm_postions_spread_all": "Non-Commercial Spread"
        },
        "⚖️ Net Positioning": {
            "net_noncomm_positions": "Net Non-Commercial",
            "net_comm_positions": "Net Commercial",
            "net_reportable_positions": "Net Reportable"
        },
        "👥 Trader Counts": {
            "traders_tot_all": "Total Traders",
            "traders_noncomm_long_all": "Non-Commercial Long Traders",
            "traders_noncomm_short_all": "Non-Commercial Short Traders",
            "traders_noncomm_spread_all": "Non-Commercial Spread Traders",
            "traders_comm_long_all": "Commercial Long Traders",
            "traders_comm_short_all": "Commercial Short Traders",
            "traders_tot_rept_long_all": "Total Reportable Long Traders",
            "traders_tot_rept_short_all": "Total Reportable Short Traders"
        },
        "🎯 Concentration (Gross)": {
            "conc_gross_le_4_tdr_long": "Top 4 Long Traders",
            "conc_gross_le_4_tdr_short": "Top 4 Short Traders",
            "conc_gross_le_8_tdr_long": "Top 8 Long Traders",
            "conc_gross_le_8_tdr_short": "Top 8 Short Traders"
        },
        "🎯 Concentration (% of OI)": {
            "conc_net_le_4_tdr_long_all": "Top 4 Long Traders",
            "conc_net_le_4_tdr_short_all": "Top 4 Short Traders",
            "conc_net_le_8_tdr_long_all": "Top 8 Long Traders",
            "conc_net_le_8_tdr_short_all": "Top 8 Short Traders"
        },
        "📊 % of Open Interest": {
            "pct_of_oi_noncomm_long_all": "Non-Commercial Long",
            "pct_of_oi_noncomm_short_all": "Non-Commercial Short",
            "pct_of_oi_noncomm_spread": "Non-Commercial Spread",
            "pct_of_oi_comm_long_all": "Commercial Long",
            "pct_of_oi_comm_short_all": "Commercial Short",
            "pct_of_oi_nonrept_long_all": "Non-Reportable Long",
            "pct_of_oi_nonrept_short_all": "Non-Reportable Short"
        }
    }
    
    # Build options list with available columns only
    options = []
    option_labels = {}
    
    for group_name, group_metrics in metric_groups.items():
        group_options = [(k, v) for k, v in group_metrics.items() if k in df.columns]
        if group_options:
            # Add group header
            for key, label in group_options:
                options.append(key)
                option_labels[key] = f"{group_name} → {label}"
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Find default index
        default_key = 'net_noncomm_positions' if 'net_noncomm_positions' in options else (options[0] if options else None)
        default_index = options.index(default_key) if default_key and default_key in options else 0
        
        selected_metric = st.selectbox(
            "Select metric for analysis:",
            options,
            format_func=lambda x: option_labels.get(x, x),
            index=default_index,
            key="percentile_metric_selector"
        )
    
    with col2:
        lookback_period = st.selectbox(
            "Historical lookback:",
            ["1 Year", "2 Years", "5 Years", "10 Years", "All Time"],
            index=0
        )
    
    chart_type = st.radio(
        "Chart type:",
        ["time_series", "distribution", "cumulative"],
        format_func=lambda x: x.replace('_', ' ').title(),
        index=0,
        horizontal=True
    )
    
    # Map lookback to years
    lookback_map = {
        "1 Year": 1,
        "2 Years": 2,
        "5 Years": 5,
        "10 Years": 10,
        "All Time": 'all'
    }
    
    fig = create_percentile_chart(df, selected_metric, lookback_map[lookback_period], chart_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def display_momentum_chart(df, instrument_name):
    """Display momentum dashboard"""
    st.subheader("🚀 Momentum Dashboard")
    
    # Add date range selector for adaptive view
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Define momentum variables with their corresponding API change columns
        momentum_vars = {
            'open_interest_all': {
                'display': 'Open Interest',
                'change_col': 'change_in_open_interest_all'
            },
            'noncomm_positions_long_all': {
                'display': 'Non-Commercial Long',
                'change_col': 'change_in_noncomm_long_all'
            },
            'noncomm_positions_short_all': {
                'display': 'Non-Commercial Short',
                'change_col': 'change_in_noncomm_short_all'
            },
            'comm_positions_long_all': {
                'display': 'Commercial Long',
                'change_col': 'change_in_comm_long_all'
            },
            'comm_positions_short_all': {
                'display': 'Commercial Short',
                'change_col': 'change_in_comm_short_all'
            },
            'tot_rept_positions_long_all': {
                'display': 'Total Reportable Long',
                'change_col': 'change_in_tot_rept_long_all'
            },
            'tot_rept_positions_short': {
                'display': 'Total Reportable Short',
                'change_col': 'change_in_tot_rept_short'
            },
            'nonrept_positions_long_all': {
                'display': 'Non-Reportable Long',
                'change_col': 'change_in_nonrept_long_all'
            },
            'nonrept_positions_short_all': {
                'display': 'Non-Reportable Short',
                'change_col': 'change_in_nonrept_short_all'
            }
        }
        
        # Filter to only available position columns
        available_vars = {k: v for k, v in momentum_vars.items() 
                         if k in df.columns and v['change_col'] in df.columns}
        
        selected_var = st.selectbox(
            "Select variable for momentum analysis:",
            list(available_vars.keys()),
            format_func=lambda x: available_vars[x]['display'],
            index=0
        )
    
    with col2:
        # Date range quick selector
        date_range_option = st.selectbox(
            "Time Period:",
            ["All Time", "5 Years", "2 Years", "1 Year", "6 Months", "3 Months", "Custom"],
            index=0,
            key="momentum_date_range"
        )
    
    # Filter data based on selection
    df_filtered = df.copy()
    
    if date_range_option != "All Time":
        end_date = df['report_date_as_yyyy_mm_dd'].max()
        
        if date_range_option == "5 Years":
            start_date = end_date - pd.DateOffset(years=5)
        elif date_range_option == "2 Years":
            start_date = end_date - pd.DateOffset(years=2)
        elif date_range_option == "1 Year":
            start_date = end_date - pd.DateOffset(years=1)
        elif date_range_option == "6 Months":
            start_date = end_date - pd.DateOffset(months=6)
        elif date_range_option == "3 Months":
            start_date = end_date - pd.DateOffset(months=3)
        elif date_range_option == "Custom":
            with col3:
                # Custom date range
                min_date = df['report_date_as_yyyy_mm_dd'].min()
                max_date = df['report_date_as_yyyy_mm_dd'].max()
                
                date_range = st.date_input(
                    "Select dates:",
                    value=(max_date - pd.DateOffset(years=1), max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="momentum_custom_dates"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = end_date - pd.DateOffset(years=1)
        
        df_filtered = df_filtered[
            (df_filtered['report_date_as_yyyy_mm_dd'] >= pd.Timestamp(start_date)) &
            (df_filtered['report_date_as_yyyy_mm_dd'] <= pd.Timestamp(end_date))
        ]
    
    # Get the corresponding API change column
    change_col = available_vars[selected_var]['change_col']
    
    # Use the position variable for display and API change column for calculations
    display_var = selected_var
    
    # Show data info
    st.info(f"📊 Showing {len(df_filtered)} data points from {df_filtered['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')} to {df_filtered['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')}")
    
    fig = create_single_variable_momentum_dashboard(df_filtered, display_var, change_col)
    if fig:
        # Configure plotly to show autoscale buttons
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'momentum_{selected_var}',
                'height': 1000,
                'width': 1400,
                'scale': 1
            }
        }
        st.plotly_chart(fig, use_container_width=True, config=config)


def display_trader_participation_chart(df, instrument_name):
    """Display trader participation analysis"""
    st.subheader("👥 Trader Participation Analysis")
    
    # Check if trader count columns exist
    if 'traders_tot_all' not in df.columns:
        st.error("⚠️ Trader count data not available for this instrument.")
        return
    
    # Sub-analysis selection
    analysis_type = st.radio(
        "Select analysis type:",
        ["Participation Density Dashboard", "Market Concentration Flow", "Concentration Risk Heatmap", 
         "Market Structure Quadrant", "Concentration Divergence", "Heterogeneity Index", "Regime Detection Dashboard",
         "Concentration Momentum", "Participant Behavior Clusters", "Market Microstructure Analysis"],
        key="trader_analysis_type",
        horizontal=True
    )
    
    if analysis_type == "Participation Density Dashboard":
        st.markdown("#### 📊 Average Position per Trader Analysis")
        
        # Percentile lookback selector
        col_lookback, col_empty = st.columns([1, 3])
        with col_lookback:
            lookback_period = st.selectbox(
                "Percentile Lookback:",
                ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "All Time"],
                index=2
            )
        
        # Map lookback to days
        lookback_map = {
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825,
            "All Time": None
        }
        lookback_days = lookback_map[lookback_period]
        
        # Create participation density chart with lookback period
        density_fig = create_participation_density_dashboard(df, instrument_name, None, lookback_days)
        
        if density_fig:
            st.plotly_chart(density_fig, use_container_width=True)
    
    elif analysis_type == "Market Concentration Flow":
        st.markdown("#### 🌊 Market Concentration Flow Analysis")
        
        # Explanation expander
        with st.expander("📖 Understanding Market Concentration", expanded=False):
            st.markdown("""
            **What is Market Concentration?**
            
            Market concentration measures whether positions are controlled by a few large traders (concentrated) 
            or distributed among many smaller traders (democratic/dispersed).
            
            **How we measure it:**
            - **Trader Count Percentile (T%)**: Where current trader count ranks vs historical (since 2010)
            - **Average Position Percentile (P%)**: Where average position size ranks vs historical
            
            **Concentration Levels:**
            - 🔴 **High**: Few traders (≤33%ile) with large positions (≥67%ile) - Market dominated by few
            - 🟠 **Medium-High**: Either few traders OR large positions - Somewhat concentrated
            - 🟡 **Medium**: Middle range for both metrics - Balanced market
            - 🟢 **Medium-Low**: Either many traders OR small positions - Somewhat dispersed
            - 🟢 **Low**: Many traders (≥67%ile) with small positions (≤33%ile) - Democratic market
            
            **Why it matters:**
            High concentration suggests potential for larger price moves as few traders control the market.
            Low concentration indicates a more stable, democratized market with diverse participation.
            """)
        
        # Time period selection
        col1, col2 = st.columns([1, 3])
        with col1:
            flow_lookback = st.selectbox(
                "Compare periods:",
                ["Week over Week", "Month over Month", "Quarter over Quarter"],
                index=0
            )
        
        # Map to days
        lookback_map = {"Week over Week": 7, "Month over Month": 30, "Quarter over Quarter": 90}
        lookback_days = lookback_map[flow_lookback]
        
        # Get current and previous period data
        latest_date = df['report_date_as_yyyy_mm_dd'].max()
        previous_date = latest_date - pd.Timedelta(days=lookback_days)
        
        # Find closest available dates
        df['date_diff_prev'] = abs(df['report_date_as_yyyy_mm_dd'] - previous_date)
        prev_idx = df['date_diff_prev'].idxmin()
        prev_data = df.loc[prev_idx]
        
        current_data = df[df['report_date_as_yyyy_mm_dd'] == latest_date].iloc[0]
        
        # Create a grouped bar chart instead of Sankey for better visibility
        categories = [
            ('Non-Comm Long', 'noncomm_positions_long_all', 'traders_noncomm_long_all'),
            ('Non-Comm Short', 'noncomm_positions_short_all', 'traders_noncomm_short_all'),
            ('Commercial Long', 'comm_positions_long_all', 'traders_comm_long_all'),
            ('Commercial Short', 'comm_positions_short_all', 'traders_comm_short_all')
        ]
        
        # Prepare data for visualization
        data_for_plot = []
        for cat_name, pos_col, trader_col in categories:
            # Previous period
            prev_traders = float(prev_data[trader_col]) if pd.notna(prev_data[trader_col]) else 0
            prev_avg = float(prev_data[pos_col]) / prev_traders if prev_traders > 0 else 0
            
            # Current period
            curr_traders = float(current_data[trader_col]) if pd.notna(current_data[trader_col]) else 0
            curr_avg = float(current_data[pos_col]) / curr_traders if curr_traders > 0 else 0
            
            # Classify concentration levels based on historical percentiles
            def get_concentration_level(avg_pos, trader_count, pos_col, trader_col, df):
                # Calculate historical percentiles for this category
                # Use all data since 2010 for percentile calculation
                lookback_date = pd.Timestamp('2010-01-01')
                hist_data = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()
                
                # Calculate average positions for historical data
                hist_data['avg_pos'] = hist_data[pos_col] / hist_data[trader_col]
                hist_data = hist_data[hist_data[trader_col] > 0]  # Filter out zero traders
                
                # Get percentiles
                trader_percentile = stats.percentileofscore(hist_data[trader_col], trader_count)
                avg_pos_percentile = stats.percentileofscore(hist_data['avg_pos'], avg_pos)
                
                # High concentration: Few traders (low percentile) with large positions (high percentile)
                # Low concentration: Many traders (high percentile) with small positions (low percentile)
                
                if trader_percentile <= 33 and avg_pos_percentile >= 67:
                    return "High"  # Few traders, large positions
                elif trader_percentile >= 67 and avg_pos_percentile <= 33:
                    return "Low"   # Many traders, small positions
                elif trader_percentile <= 33 or avg_pos_percentile >= 67:
                    return "Medium-High"  # Either few traders OR large positions
                elif trader_percentile >= 67 or avg_pos_percentile <= 33:
                    return "Medium-Low"   # Either many traders OR small positions
                else:
                    return "Medium"  # Middle range for both
            
            prev_level = get_concentration_level(prev_avg, prev_traders, pos_col, trader_col, df)
            curr_level = get_concentration_level(curr_avg, curr_traders, pos_col, trader_col, df)
            
            # Calculate percentiles for display
            lookback_date = pd.Timestamp('2010-01-01')
            hist_data = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()
            hist_data['avg_pos'] = hist_data[pos_col] / hist_data[trader_col]
            hist_data = hist_data[hist_data[trader_col] > 0]
            
            prev_trader_pct = stats.percentileofscore(hist_data[trader_col], prev_traders)
            prev_pos_pct = stats.percentileofscore(hist_data['avg_pos'], prev_avg)
            curr_trader_pct = stats.percentileofscore(hist_data[trader_col], curr_traders)
            curr_pos_pct = stats.percentileofscore(hist_data['avg_pos'], curr_avg)
            
            data_for_plot.append({
                'Category': cat_name,
                'Period': 'Previous',
                'Concentration': prev_level,
                'Avg Position': prev_avg,
                'Trader Count': prev_traders,
                'Total Position': float(prev_data[pos_col]),
                'Trader Percentile': prev_trader_pct,
                'Position Percentile': prev_pos_pct
            })
            
            data_for_plot.append({
                'Category': cat_name,
                'Period': 'Current',
                'Concentration': curr_level,
                'Avg Position': curr_avg,
                'Trader Count': curr_traders,
                'Total Position': float(current_data[pos_col]),
                'Trader Percentile': curr_trader_pct,
                'Position Percentile': curr_pos_pct
            })
        
        # Create DataFrame
        plot_df = pd.DataFrame(data_for_plot)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Position per Trader', 'Trader Count', 
                          'Concentration Levels (T%=Traders, P%=Position)', 'Total Positions'),
            vertical_spacing=0.18,
            horizontal_spacing=0.12,
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Color mapping
        colors = {'Previous': '#90EE90', 'Current': '#4169E1'}
        concentration_colors = {
            'High': '#DC143C',         # Crimson red - high concentration risk
            'Medium-High': '#FF8C00',  # Dark orange
            'Medium': '#FFD700',       # Gold
            'Medium-Low': '#9ACD32',   # Yellow green
            'Low': '#32CD32'           # Lime green - low concentration (more democratic)
        }
        
        # Plot 1: Average Position per Trader
        for period in ['Previous', 'Current']:
            period_data = plot_df[plot_df['Period'] == period]
            fig.add_trace(
                go.Bar(
                    x=period_data['Category'],
                    y=period_data['Avg Position'],
                    name=period,
                    marker_color=colors[period],
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Plot 2: Trader Count
        for period in ['Previous', 'Current']:
            period_data = plot_df[plot_df['Period'] == period]
            fig.add_trace(
                go.Bar(
                    x=period_data['Category'],
                    y=period_data['Trader Count'],
                    name=period,
                    marker_color=colors[period],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: Concentration Level with Percentiles
        for cat in ['Non-Comm Long', 'Non-Comm Short', 'Commercial Long', 'Commercial Short']:
            prev_data_cat = plot_df[(plot_df['Category'] == cat) & (plot_df['Period'] == 'Previous')].iloc[0]
            curr_data_cat = plot_df[(plot_df['Category'] == cat) & (plot_df['Period'] == 'Current')].iloc[0]
            
            # Create text with percentiles
            prev_text = f"T:{prev_data_cat['Trader Percentile']:.0f}%<br>P:{prev_data_cat['Position Percentile']:.0f}%"
            curr_text = f"T:{curr_data_cat['Trader Percentile']:.0f}%<br>P:{curr_data_cat['Position Percentile']:.0f}%"
            
            # Plot previous period
            fig.add_trace(
                go.Scatter(
                    x=[cat],
                    y=['Previous'],
                    mode='markers+text',
                    marker=dict(
                        size=40,
                        color=concentration_colors[prev_data_cat['Concentration']],
                        line=dict(width=2, color='black')
                    ),
                    text=[prev_text],
                    textposition='middle center',
                    textfont=dict(size=10, color='black'),
                    showlegend=False,
                    hovertemplate=f"{cat}<br>Concentration: {prev_data_cat['Concentration']}<br>Traders: {prev_data_cat['Trader Percentile']:.1f}%ile<br>Avg Pos: {prev_data_cat['Position Percentile']:.1f}%ile<extra></extra>"
                ),
                row=2, col=1
            )
            
            # Plot current period
            fig.add_trace(
                go.Scatter(
                    x=[cat],
                    y=['Current'],
                    mode='markers+text',
                    marker=dict(
                        size=40,
                        color=concentration_colors[curr_data_cat['Concentration']],
                        line=dict(width=2, color='black')
                    ),
                    text=[curr_text],
                    textposition='middle center',
                    textfont=dict(size=10, color='black'),
                    showlegend=False,
                    hovertemplate=f"{cat}<br>Concentration: {curr_data_cat['Concentration']}<br>Traders: {curr_data_cat['Trader Percentile']:.1f}%ile<br>Avg Pos: {curr_data_cat['Position Percentile']:.1f}%ile<extra></extra>"
                ),
                row=2, col=1
            )
            
        
        # Plot 4: Total Positions
        for period in ['Previous', 'Current']:
            period_data = plot_df[plot_df['Period'] == period]
            fig.add_trace(
                go.Bar(
                    x=period_data['Category'],
                    y=period_data['Total Position'],
                    name=period,
                    marker_color=colors[period],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Market Concentration Flow Analysis ({flow_lookback})",
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(title_text="Avg Position", row=1, col=1, tickformat=",.0f")
        fig.update_yaxes(title_text="Trader Count", row=1, col=2, tickformat=",.0f")
        fig.update_yaxes(title_text="Period", row=2, col=1, categoryorder="array", categoryarray=["Previous", "Current"])
        fig.update_xaxes(row=2, col=1, tickangle=-25)
        fig.update_yaxes(title_text="Total Position", row=2, col=2, tickformat=",.0f")
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Previous Date", prev_data['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'))
        with col2:
            st.metric("Current Date", current_data['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'))
        with col3:
            # Calculate total trader change from the plot data
            prev_total = plot_df[plot_df['Period'] == 'Previous']['Trader Count'].sum()
            curr_total = plot_df[plot_df['Period'] == 'Current']['Trader Count'].sum()
            total_trader_change = int(curr_total - prev_total)
            st.metric("Total Trader Change", f"{total_trader_change:+d}")
    
    elif analysis_type == "Concentration Risk Heatmap":
        st.markdown("#### 🔥 Concentration Risk Heatmap")
        
        # Explanation expander
        with st.expander("📖 Understanding the Concentration Risk Heatmap", expanded=False):
            st.markdown("""
            **What does each square represent?**
            
            Each square shows a **Risk Score (0-100)** for a specific trader category at a specific time period.
            
            **How to read the heatmap:**
            - **X-axis**: Time periods (weekly/monthly/quarterly)
            - **Y-axis**: Trader categories
            - **Color**: Risk level (🟢 Green = Low risk → 🔴 Red = High risk)
            
            **Risk Score Calculation:**
            ```
            Risk Score = (Position Concentration × 70%) + (Inverse Trader Participation × 30%)
            ```
            
            **Components explained:**
            
            1. **Position Concentration (70% weight)**
               - Percentage of total positions held by top 4 or 8 traders
               - Example: If top 4 traders hold 45% of all long positions = High concentration
               
            2. **Inverse Trader Participation (30% weight)**
               - Formula: (100 - Trader Count Percentile)
               - Percentile calculated vs ALL data since 2010
               - Example: If current trader count is at 20th percentile = Few traders = High risk
            
            **Color Scale:**
            - 🟢 **Green (0-25)**: Low risk - Many traders, well-distributed positions
            - 🟡 **Gold (25-50)**: Medium risk - Moderate concentration
            - 🟠 **Orange (50-75)**: High risk - Significant concentration
            - 🔴 **Red (75-100)**: Very high risk - Market dominated by few large traders
            
            **Example Interpretation:**
            A red square for "Non-Commercial Long" in Week 15 means:
            - Long speculative positions are highly concentrated
            - Few traders control large percentage of positions
            - Higher volatility risk as these traders could move markets
            
            **Important Notes:**
            - Display period: Shows 1-10 years based on your selection
            - Percentile baseline: Always uses full history since 2010 for consistency
            - Non-reportable categories: Use concentration metrics only (no trader counts)
            """)
        
        st.info("Visualizes concentration risk over time by combining trader participation rates with position concentration metrics")
        
        # Configuration columns
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Concentration metric selection
            conc_metric = st.selectbox(
                "Position Concentration Metric:",
                [
                    ("4 or Less Traders - Gross Long", "conc_gross_le_4_tdr_long"),
                    ("4 or Less Traders - Gross Short", "conc_gross_le_4_tdr_short"),
                    ("8 or Less Traders - Gross Long", "conc_gross_le_8_tdr_long"),
                    ("8 or Less Traders - Gross Short", "conc_gross_le_8_tdr_short"),
                    ("4 or Less Traders - Net Long", "conc_net_le_4_tdr_long_all"),
                    ("4 or Less Traders - Net Short", "conc_net_le_4_tdr_short_all"),
                    ("8 or Less Traders - Net Long", "conc_net_le_8_tdr_long_all"),
                    ("8 or Less Traders - Net Short", "conc_net_le_8_tdr_short_all")
                ],
                format_func=lambda x: x[0],
                index=0
            )
            selected_conc_col = conc_metric[1]
        
        with col2:
            # Time aggregation
            time_agg = st.selectbox(
                "Time Aggregation:",
                ["Weekly", "Monthly", "Quarterly"],
                index=0
            )
        
        with col3:
            # Lookback period
            lookback_years = st.slider(
                "Years of History:",
                min_value=1,
                max_value=10,
                value=3,
                step=1
            )
        
        # Calculate lookback date
        lookback_date = df['report_date_as_yyyy_mm_dd'].max() - pd.DateOffset(years=lookback_years)
        df_heatmap = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()
        
        # Define trader categories for the heatmap
        categories = [
            ('Non-Commercial Long', 'noncomm_positions_long_all', 'traders_noncomm_long_all'),
            ('Non-Commercial Short', 'noncomm_positions_short_all', 'traders_noncomm_short_all'),
            ('Commercial Long', 'comm_positions_long_all', 'traders_comm_long_all'),
            ('Commercial Short', 'comm_positions_short_all', 'traders_comm_short_all'),
            ('Non-Reportable Long', 'nonrept_positions_long_all', None),
            ('Non-Reportable Short', 'nonrept_positions_short_all', None)
        ]
        
        # Prepare data for heatmap
        heatmap_data = []
        
        # Time aggregation
        if time_agg == "Weekly":
            df_heatmap['period'] = df_heatmap['report_date_as_yyyy_mm_dd'].dt.to_period('W')
        elif time_agg == "Monthly":
            df_heatmap['period'] = df_heatmap['report_date_as_yyyy_mm_dd'].dt.to_period('M')
        else:  # Quarterly
            df_heatmap['period'] = df_heatmap['report_date_as_yyyy_mm_dd'].dt.to_period('Q')
        
        # Group by period and calculate metrics
        for period, period_data in df_heatmap.groupby('period'):
            # Get the last observation in the period
            last_obs = period_data.iloc[-1]
            
            for cat_name, pos_col, trader_col in categories:
                # Calculate concentration risk score
                # 1. Position concentration from selected metric
                try:
                    if selected_conc_col in last_obs.index and pd.notna(last_obs[selected_conc_col]):
                        pos_concentration = float(last_obs[selected_conc_col])
                    else:
                        pos_concentration = 0
                    
                    # 2. Trader participation rate (if available)
                    if trader_col and trader_col in last_obs.index:
                        if pd.notna(last_obs[trader_col]) and float(last_obs[trader_col]) > 0:
                            traders = float(last_obs[trader_col])
                            
                            # Calculate historical percentiles
                            hist_data = df[df['report_date_as_yyyy_mm_dd'] >= pd.Timestamp('2010-01-01')]
                            if trader_col in hist_data.columns:
                                trader_values = hist_data[trader_col].dropna()
                                if len(trader_values) > 0:
                                    trader_percentile = stats.percentileofscore(trader_values, traders)
                                else:
                                    trader_percentile = 50
                            else:
                                trader_percentile = 50
                            
                            # Risk score: High concentration + Low trader participation = High risk
                            # Normalize concentration to 0-100 scale
                            risk_score = (pos_concentration * 0.7) + ((100 - trader_percentile) * 0.3)
                        else:
                            # No traders, use concentration only
                            risk_score = pos_concentration
                    else:
                        # For non-reportable, use only concentration
                        risk_score = pos_concentration
                        
                except Exception as e:
                    # If any error, default to concentration only
                    risk_score = pos_concentration if 'pos_concentration' in locals() else 0
                
                heatmap_data.append({
                    'Period': str(period),
                    'Category': cat_name,
                    'Risk_Score': risk_score,
                    'Concentration': pos_concentration,
                    'Date': period.to_timestamp()
                })
        
        # Create DataFrame
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Pivot for heatmap
        pivot_df = heatmap_df.pivot(index='Category', columns='Period', values='Risk_Score')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale=[
                [0, '#00FF00'],      # Green - Low risk
                [0.25, '#90EE90'],   # Light green
                [0.5, '#FFD700'],    # Gold - Medium risk
                [0.75, '#FF8C00'],   # Dark orange
                [1, '#DC143C']       # Crimson - High risk
            ],
            colorbar=dict(
                title="Risk Score",
                tickmode="linear",
                tick0=0,
                dtick=20,
                thickness=20,
                len=0.9
            ),
            hovertemplate='Category: %{y}<br>Period: %{x}<br>Risk Score: %{z:.1f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Concentration Risk Heatmap - {conc_metric[0]}",
            height=500,
            xaxis=dict(
                title="Time Period",
                tickangle=-45,
                side="bottom"
            ),
            yaxis=dict(
                title="Trader Category",
                tickmode="linear"
            )
        )
        
        # Display heatmap
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        if len(heatmap_df) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                current_period = heatmap_df['Period'].max()
                current_avg_risk = heatmap_df[heatmap_df['Period'] == current_period]['Risk_Score'].mean()
                st.metric("Current Avg Risk", f"{current_avg_risk:.1f}")
            
            with col2:
                if len(heatmap_df) > 0 and heatmap_df['Risk_Score'].notna().any():
                    max_risk_row = heatmap_df.loc[heatmap_df['Risk_Score'].idxmax()]
                    st.metric("Highest Risk", f"{max_risk_row['Category']} ({max_risk_row['Period']})")
                else:
                    st.metric("Highest Risk", "N/A")
            
            with col3:
                risk_trend = heatmap_df.groupby('Period')['Risk_Score'].mean()
                if len(risk_trend) > 1:
                    trend_change = risk_trend.iloc[-1] - risk_trend.iloc[-2]
                    st.metric("Risk Trend", f"{trend_change:+.1f}", delta_color="inverse")
                else:
                    st.metric("Risk Trend", "N/A")
        else:
            st.warning("No data available for the selected parameters")
        
    elif analysis_type == "Market Structure Quadrant":
        st.markdown("#### 🎯 Market Structure Quadrant Analysis")
        
        # Explanation expander
        with st.expander("📖 Understanding Market Structure Quadrants", expanded=False):
            st.markdown("""
            **What is Market Structure Quadrant Analysis?**
            
            This visualization classifies market conditions into four distinct regimes based on:
            - **X-axis**: Trader Participation (percentile vs historical since 2010)
            - **Y-axis**: Position Concentration (% held by top traders)
            
            **The Four Quadrants:**
            
            🔴 **Oligopolistic (Top-Left)**: Few traders + High concentration
            - Market dominated by small group of large players
            - High volatility risk, susceptible to manipulation
            
            🟠 **Crowded Concentration (Top-Right)**: Many traders + High concentration  
            - Lots of participants but positions still concentrated
            - Herding behavior risk, sudden reversals possible
            
            🟡 **Specialized (Bottom-Left)**: Few traders + Low concentration
            - Small market with evenly distributed positions  
            - Low liquidity risk, price gaps possible
            
            🟢 **Democratic (Bottom-Right)**: Many traders + Low concentration
            - Ideal market structure with broad participation
            - Better price discovery, lower volatility
            
            **How to interpret:**
            - Each dot represents a trader category at a specific time
            - Lines show movement over time
            - Size of dot indicates total open interest
            """)
        
        # Configuration
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Concentration metric selection
            conc_metric = st.selectbox(
                "Concentration Metric:",
                [
                    ("4 or Less Traders - Long", "conc_gross_le_4_tdr_long"),
                    ("4 or Less Traders - Short", "conc_gross_le_4_tdr_short"),
                    ("8 or Less Traders - Long", "conc_gross_le_8_tdr_long"),
                    ("8 or Less Traders - Short", "conc_gross_le_8_tdr_short")
                ],
                format_func=lambda x: x[0],
                index=0,
                key="quad_conc_metric"
            )
            selected_conc_col = conc_metric[1]
        
        with col2:
            # Time window
            time_window = st.selectbox(
                "Time Window:",
                ["Last 3 Months", "Last 6 Months", "Last Year", "Last 2 Years"],
                index=1
            )
        
        with col3:
            # Show trail
            show_trail = st.checkbox("Show Historical Trail", value=True)
        
        # Map time window to days
        window_map = {
            "Last 3 Months": 90,
            "Last 6 Months": 180,
            "Last Year": 365,
            "Last 2 Years": 730
        }
        lookback_days = window_map[time_window]
        
        # Filter data
        cutoff_date = df['report_date_as_yyyy_mm_dd'].max() - pd.DateOffset(days=lookback_days)
        df_quad = df[df['report_date_as_yyyy_mm_dd'] >= cutoff_date].copy()
        
        # Calculate historical baselines (since 2010)
        hist_baseline = df[df['report_date_as_yyyy_mm_dd'] >= pd.Timestamp('2010-01-01')]
        
        # Categories to plot
        categories = [
            ('Non-Commercial', 'noncomm_positions_long_all', 'traders_noncomm_long_all', 'noncomm_positions_short_all', 'traders_noncomm_short_all'),
            ('Commercial', 'comm_positions_long_all', 'traders_comm_long_all', 'comm_positions_short_all', 'traders_comm_short_all')
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add quadrant backgrounds
        fig.add_shape(type="rect", x0=0, y0=50, x1=50, y1=100,
                      fillcolor="rgba(255,0,0,0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=50, y0=50, x1=100, y1=100,
                      fillcolor="rgba(255,165,0,0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50,
                      fillcolor="rgba(255,255,0,0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=50, y0=0, x1=100, y1=50,
                      fillcolor="rgba(0,255,0,0.1)", line=dict(width=0))
        
        # Add quadrant labels
        fig.add_annotation(x=25, y=75, text="<b>Oligopolistic</b><br>Few traders<br>High concentration",
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=75, y=75, text="<b>Crowded<br>Concentration</b><br>Many traders<br>High concentration",
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=25, y=25, text="<b>Specialized</b><br>Few traders<br>Low concentration",
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=75, y=25, text="<b>Democratic</b><br>Many traders<br>Low concentration",
                          showarrow=False, font=dict(size=10))
        
        # Plot data for each category
        colors = {'Non-Commercial': 'blue', 'Commercial': 'red'}
        
        for cat_name, long_pos_col, long_trader_col, short_pos_col, short_trader_col in categories:
            # Prepare data points
            plot_data = []
            
            # Determine which columns to use based on concentration metric
            if 'long' in selected_conc_col:
                trader_col = long_trader_col
                pos_col = long_pos_col
                label_suffix = " (Long)"
            else:
                trader_col = short_trader_col
                pos_col = short_pos_col
                label_suffix = " (Short)"
            
            # Calculate metrics for each time point
            for idx, row in df_quad.iterrows():
                if pd.notna(row[trader_col]) and pd.notna(row[selected_conc_col]):
                    # X-axis: Trader participation percentile
                    trader_count = float(row[trader_col])
                    trader_percentile = stats.percentileofscore(
                        hist_baseline[trader_col].dropna(), trader_count
                    )
                    
                    # Y-axis: Concentration percentage
                    concentration = float(row[selected_conc_col])
                    
                    # Size: Total open interest
                    oi = float(row['open_interest_all'])
                    
                    plot_data.append({
                        'date': row['report_date_as_yyyy_mm_dd'],
                        'x': trader_percentile,
                        'y': concentration,
                        'oi': oi,
                        'traders': trader_count
                    })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data).sort_values('date')
                
                # Plot trail if enabled
                if show_trail and len(plot_df) > 1:
                    fig.add_trace(go.Scatter(
                        x=plot_df['x'],
                        y=plot_df['y'],
                        mode='lines',
                        line=dict(color=colors[cat_name], width=1, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Plot points
                fig.add_trace(go.Scatter(
                    x=plot_df['x'],
                    y=plot_df['y'],
                    mode='markers',
                    name=cat_name + label_suffix,
                    marker=dict(
                        size=plot_df['oi'] / plot_df['oi'].max() * 30 + 10,
                        color=colors[cat_name],
                        line=dict(width=1, color='white')
                    ),
                    customdata=np.column_stack((plot_df['date'].dt.strftime('%Y-%m-%d'), 
                                               plot_df['traders'], 
                                               plot_df['oi'])),
                    hovertemplate='%{name}<br>' +
                                 'Date: %{customdata[0]}<br>' +
                                 'Trader Percentile: %{x:.1f}%<br>' +
                                 'Concentration: %{y:.1f}%<br>' +
                                 'Traders: %{customdata[1]:,.0f}<br>' +
                                 'Open Interest: %{customdata[2]:,.0f}<extra></extra>'
                ))
                
                # Add arrow showing direction
                if len(plot_df) > 1:
                    last_point = plot_df.iloc[-1]
                    second_last = plot_df.iloc[-2]
                    
                    fig.add_annotation(
                        x=second_last['x'],
                        y=second_last['y'],
                        ax=last_point['x'],
                        ay=last_point['y'],
                        xref='x',
                        yref='y',
                        axref='x',
                        ayref='y',
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=colors[cat_name]
                    )
        
        # Add center lines
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title=f"Market Structure Quadrant - {conc_metric[0]}",
            xaxis=dict(
                title="Trader Participation Percentile →<br>(Few Traders ... Many Traders)",
                range=[0, 100],
                ticksuffix="%"
            ),
            yaxis=dict(
                title="Position Concentration →<br>(Low ... High)",
                range=[0, 100],
                ticksuffix="%"
            ),
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Current position summary
        st.markdown("### Current Market Structure")
        col1, col2 = st.columns(2)
        
        latest_date = df_quad['report_date_as_yyyy_mm_dd'].max()
        latest_data = df_quad[df_quad['report_date_as_yyyy_mm_dd'] == latest_date].iloc[0]
        
        for i, (cat_name, long_pos_col, long_trader_col, short_pos_col, short_trader_col) in enumerate(categories):
            with col1 if i == 0 else col2:
                if 'long' in selected_conc_col:
                    trader_col = long_trader_col
                    label_suffix = " (Long)"
                else:
                    trader_col = short_trader_col
                    label_suffix = " (Short)"
                
                if pd.notna(latest_data[trader_col]) and pd.notna(latest_data[selected_conc_col]):
                    trader_count = float(latest_data[trader_col])
                    trader_percentile = stats.percentileofscore(
                        hist_baseline[trader_col].dropna(), trader_count
                    )
                    concentration = float(latest_data[selected_conc_col])
                    
                    # Determine quadrant
                    if trader_percentile < 50 and concentration >= 50:
                        quadrant = "🔴 Oligopolistic"
                    elif trader_percentile >= 50 and concentration >= 50:
                        quadrant = "🟠 Crowded Concentration"
                    elif trader_percentile < 50 and concentration < 50:
                        quadrant = "🟡 Specialized"
                    else:
                        quadrant = "🟢 Democratic"
                    
                    st.metric(
                        f"{cat_name}{label_suffix}",
                        quadrant,
                        f"Traders: {trader_percentile:.0f}%ile, Conc: {concentration:.1f}%"
                    )
        
    elif analysis_type == "Concentration Divergence":
        st.markdown("#### 📊 Concentration Divergence Analysis")
        
        # Explanation expander
        with st.expander("📖 Understanding Concentration Divergence", expanded=False):
            st.markdown("""
            **What is Concentration Divergence?**
            
            This analysis identifies when different trader groups or market sides show opposing concentration patterns, 
            which often signals important market dynamics.
            
            **Three Types of Divergence:**
            
            1. **Category Divergence** (Commercial vs Non-Commercial)
               - When hedgers and speculators show opposite concentration trends
               - Often indicates smart money positioning differently from crowd
               
            2. **Directional Divergence** (Long vs Short)
               - When concentration differs significantly between long and short sides
               - May signal asymmetric risk or positioning imbalances
               
            3. **Historical Divergence** (Current vs Historical Norm)
               - When current concentration deviates significantly from historical average
               - Identifies unusual market conditions
            
            **How to interpret:**
            - **Positive values**: Higher concentration than comparison
            - **Negative values**: Lower concentration than comparison
            - **Large divergences** (>20 points): Significant market imbalance
            - **Trend changes**: Watch for divergence narrowing/widening
            
            **Why it matters:**
            - Divergences often precede major market moves
            - Shows when smart money disagrees with speculators
            - Identifies structural market imbalances
            """)
        
        # Configuration
        col1, col2 = st.columns([1, 1])
        
        with col1:
            divergence_type = st.selectbox(
                "Divergence Type:",
                ["Category Divergence (Commercial vs Non-Commercial)",
                 "Directional Divergence (Long vs Short)",
                 "Historical Divergence (Current vs Average)"],
                index=0
            )
        
        with col2:
            lookback_period = st.selectbox(
                "Analysis Period:",
                ["3 Months", "6 Months", "1 Year", "2 Years"],
                index=2
            )
        
        # Map lookback to days
        lookback_map = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730}
        lookback_days = lookback_map[lookback_period]
        
        # Filter data
        cutoff_date = df['report_date_as_yyyy_mm_dd'].max() - pd.DateOffset(days=lookback_days)
        df_div = df[df['report_date_as_yyyy_mm_dd'] >= cutoff_date].copy()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Concentration Levels', 'Divergence Score', 'Divergence Distribution'),
            row_heights=[0.4, 0.35, 0.25],
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        if divergence_type == "Category Divergence (Commercial vs Non-Commercial)":
            # Use API-provided percentage of open interest data
            # First, check which concentration side to analyze
            conc_side = st.radio(
                "Analyze concentration for:",
                ["Long Positions", "Short Positions", "Net (Long - Short)"],
                horizontal=True,
                key="cat_div_side"
            )
            
            if conc_side == "Long Positions":
                # Check if percentage columns exist, otherwise calculate them
                if 'pct_of_oi_comm_long_all' in df_div.columns:
                    df_div['comm_position_share'] = df_div['pct_of_oi_comm_long_all']
                    df_div['noncomm_position_share'] = df_div['pct_of_oi_noncomm_long_all']
                else:
                    # Calculate position shares manually
                    df_div['comm_position_share'] = (df_div['comm_positions_long_all'] / df_div['open_interest_all']) * 100
                    df_div['noncomm_position_share'] = (df_div['noncomm_positions_long_all'] / df_div['open_interest_all']) * 100
                
                # Calculate trader share for each category
                df_div['comm_trader_share'] = df_div['traders_comm_long_all'] / df_div['traders_tot_all'] * 100
                df_div['noncomm_trader_share'] = df_div['traders_noncomm_long_all'] / df_div['traders_tot_all'] * 100
                
            elif conc_side == "Short Positions":
                # Check if percentage columns exist, otherwise calculate them
                if 'pct_of_oi_comm_short_all' in df_div.columns:
                    df_div['comm_position_share'] = df_div['pct_of_oi_comm_short_all']
                    df_div['noncomm_position_share'] = df_div['pct_of_oi_noncomm_short_all']
                else:
                    # Calculate position shares manually
                    df_div['comm_position_share'] = (df_div['comm_positions_short_all'] / df_div['open_interest_all']) * 100
                    df_div['noncomm_position_share'] = (df_div['noncomm_positions_short_all'] / df_div['open_interest_all']) * 100
                
                # Calculate trader share for each category
                df_div['comm_trader_share'] = df_div['traders_comm_short_all'] / df_div['traders_tot_all'] * 100
                df_div['noncomm_trader_share'] = df_div['traders_noncomm_short_all'] / df_div['traders_tot_all'] * 100
                
            else:  # Net positions
                # Calculate net positions
                df_div['comm_net'] = df_div['comm_positions_long_all'] - df_div['comm_positions_short_all']
                df_div['noncomm_net'] = df_div['noncomm_positions_long_all'] - df_div['noncomm_positions_short_all']
                
                # Calculate trader share (average of long and short)
                df_div['comm_trader_share'] = ((df_div['traders_comm_long_all'] + df_div['traders_comm_short_all']) / 2) / df_div['traders_tot_all'] * 100
                df_div['noncomm_trader_share'] = ((df_div['traders_noncomm_long_all'] + df_div['traders_noncomm_short_all']) / 2) / df_div['traders_tot_all'] * 100
                
                # Calculate net position shares
                df_div['comm_position_share'] = abs(df_div['comm_net']) / df_div['open_interest_all'] * 100
                df_div['noncomm_position_share'] = abs(df_div['noncomm_net']) / df_div['open_interest_all'] * 100
            
            # Calculate concentration score: Position Share / Trader Share
            # Higher score = more concentrated (larger positions per trader)
            # Handle division by zero
            df_div['comm_concentration'] = np.where(
                df_div['comm_trader_share'] > 0,
                df_div['comm_position_share'] / df_div['comm_trader_share'],
                0
            )
            df_div['noncomm_concentration'] = np.where(
                df_div['noncomm_trader_share'] > 0,
                df_div['noncomm_position_share'] / df_div['noncomm_trader_share'],
                0
            )
            
            # Replace inf and nan values with 0
            df_div['comm_concentration'] = df_div['comm_concentration'].replace([np.inf, -np.inf], 0).fillna(0)
            df_div['noncomm_concentration'] = df_div['noncomm_concentration'].replace([np.inf, -np.inf], 0).fillna(0)
            
            # Calculate divergence (scaled for visibility)
            df_div['divergence'] = (df_div['noncomm_concentration'] - df_div['comm_concentration']) * 10
            
            # Plot 1: Concentration scores
            fig.add_trace(go.Scatter(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['comm_concentration'],
                name='Commercial',
                line=dict(color='red', width=2),
                hovertemplate='Commercial<br>Date: %{x}<br>Concentration Score: %{y:.2f}<br>Position Share: %{customdata[0]:.1f}%<br>Trader Share: %{customdata[1]:.1f}%<extra></extra>',
                customdata=np.column_stack((df_div['comm_position_share'], df_div['comm_trader_share']))
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['noncomm_concentration'],
                name='Non-Commercial',
                line=dict(color='blue', width=2),
                hovertemplate='Non-Commercial<br>Date: %{x}<br>Concentration Score: %{y:.2f}<br>Position Share: %{customdata[0]:.1f}%<br>Trader Share: %{customdata[1]:.1f}%<extra></extra>',
                customdata=np.column_stack((df_div['noncomm_position_share'], df_div['noncomm_trader_share']))
            ), row=1, col=1)
            
            # Plot 2: Divergence score
            colors = ['green' if x < 0 else 'red' for x in df_div['divergence']]
            fig.add_trace(go.Bar(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['divergence'],
                name='Divergence',
                marker_color=colors,
                hovertemplate='Date: %{x}<br>Divergence: %{y:.1f}%<extra></extra>'
            ), row=2, col=1)
            
            # Add zero line
            fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add significance bands
            fig.add_hrect(y0=20, y1=100, row=2, col=1, fillcolor="red", opacity=0.1, line_width=0)
            fig.add_hrect(y0=-100, y1=-20, row=2, col=1, fillcolor="green", opacity=0.1, line_width=0)
            
        elif divergence_type == "Directional Divergence (Long vs Short)":
            # Calculate concentration for long vs short
            df_div['long_concentration'] = (df_div['conc_gross_le_4_tdr_long'] + df_div['conc_net_le_4_tdr_long_all']) / 2
            df_div['short_concentration'] = (df_div['conc_gross_le_4_tdr_short'] + df_div['conc_net_le_4_tdr_short_all']) / 2
            df_div['divergence'] = df_div['long_concentration'] - df_div['short_concentration']
            
            # Plot 1: Concentration levels
            fig.add_trace(go.Scatter(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['long_concentration'],
                name='Long Side',
                line=dict(color='green', width=2),
                hovertemplate='Long Side<br>Date: %{x}<br>Concentration: %{y:.1f}%<extra></extra>'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['short_concentration'],
                name='Short Side',
                line=dict(color='red', width=2),
                hovertemplate='Short Side<br>Date: %{x}<br>Concentration: %{y:.1f}%<extra></extra>'
            ), row=1, col=1)
            
            # Plot 2: Divergence score
            colors = ['green' if x > 0 else 'red' for x in df_div['divergence']]
            fig.add_trace(go.Bar(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['divergence'],
                name='Divergence',
                marker_color=colors,
                hovertemplate='Date: %{x}<br>Divergence: %{y:.1f}%<extra></extra>'
            ), row=2, col=1)
            
            # Add zero line and bands
            fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_hrect(y0=15, y1=100, row=2, col=1, fillcolor="green", opacity=0.1, line_width=0)
            fig.add_hrect(y0=-100, y1=-15, row=2, col=1, fillcolor="red", opacity=0.1, line_width=0)
            
        else:  # Historical Divergence
            # Calculate average concentration over full history
            hist_data = df[df['report_date_as_yyyy_mm_dd'] >= pd.Timestamp('2010-01-01')]
            
            # Use average of all concentration metrics
            df_div['current_concentration'] = (
                df_div['conc_gross_le_4_tdr_long'] + 
                df_div['conc_gross_le_4_tdr_short'] + 
                df_div['conc_net_le_4_tdr_long_all'] + 
                df_div['conc_net_le_4_tdr_short_all']
            ) / 4
            
            # Calculate rolling average
            df_div['historical_avg'] = df_div['current_concentration'].rolling(window=52, min_periods=26).mean()
            
            # Calculate z-score divergence
            rolling_std = df_div['current_concentration'].rolling(window=52, min_periods=26).std()
            df_div['divergence'] = (df_div['current_concentration'] - df_div['historical_avg']) / rolling_std
            df_div['divergence'] = df_div['divergence'] * 10  # Scale for visibility
            
            # Plot 1: Concentration levels
            fig.add_trace(go.Scatter(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['current_concentration'],
                name='Current',
                line=dict(color='blue', width=2),
                hovertemplate='Current<br>Date: %{x}<br>Concentration: %{y:.1f}%<extra></extra>'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['historical_avg'],
                name='Historical Avg',
                line=dict(color='gray', width=2, dash='dash'),
                hovertemplate='Historical Avg<br>Date: %{x}<br>Concentration: %{y:.1f}%<extra></extra>'
            ), row=1, col=1)
            
            # Plot 2: Divergence z-score
            colors = ['red' if abs(x) > 20 else 'orange' if abs(x) > 10 else 'gray' for x in df_div['divergence']]
            fig.add_trace(go.Bar(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['divergence'],
                name='Divergence',
                marker_color=colors,
                hovertemplate='Date: %{x}<br>Z-Score: %{y:.1f}<extra></extra>'
            ), row=2, col=1)
            
            # Add significance lines
            fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_hline(y=20, row=2, col=1, line_dash="dot", line_color="red", opacity=0.5)
            fig.add_hline(y=-20, row=2, col=1, line_dash="dot", line_color="red", opacity=0.5)
        
        # Plot 3: Divergence distribution histogram
        fig.add_trace(go.Histogram(
            x=df_div['divergence'].dropna(),
            nbinsx=30,
            name='Distribution',
            marker_color='lightblue',
            showlegend=False
        ), row=3, col=1)
        
        # Add normal distribution overlay
        mean_div = df_div['divergence'].mean()
        std_div = df_div['divergence'].std()
        x_range = np.linspace(df_div['divergence'].min(), df_div['divergence'].max(), 100)
        normal_dist = stats.norm.pdf(x_range, mean_div, std_div) * len(df_div) * (df_div['divergence'].max() - df_div['divergence'].min()) / 30
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            name='Normal',
            line=dict(color='red', width=2),
            showlegend=False
        ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Concentration Divergence Analysis - {divergence_type.split('(')[0]}",
            height=900,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        if divergence_type.startswith("Category"):
            fig.update_yaxes(title_text="Concentration Score<br>(Position Share / Trader Share)", row=1, col=1)
        else:
            fig.update_yaxes(title_text="Concentration %", row=1, col=1)
        fig.update_yaxes(title_text="Divergence", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Divergence Value", row=3, col=1)
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        current_divergence = df_div['divergence'].iloc[-1] if len(df_div) > 0 else 0
        avg_divergence = df_div['divergence'].mean() if len(df_div) > 0 else 0
        max_divergence = df_div['divergence'].max() if len(df_div) > 0 else 0
        min_divergence = df_div['divergence'].min() if len(df_div) > 0 else 0
        
        with col1:
            st.metric("Current Divergence", f"{current_divergence:.1f}")
        with col2:
            st.metric("Average", f"{avg_divergence:.1f}")
        with col3:
            st.metric("Maximum", f"{max_divergence:.1f}")
        with col4:
            st.metric("Minimum", f"{min_divergence:.1f}")
        
    elif analysis_type == "Heterogeneity Index":
        st.markdown("#### 🔀 Market Heterogeneity Index")
        
        # Explanation expander
        with st.expander("📖 Understanding Heterogeneity Index", expanded=False):
            st.markdown("""
            **What is the Heterogeneity Index?**
            
            A percentile-based measure of divergence between commercial and non-commercial trader behavior. All components use historical percentile rankings to identify unusual market conditions.
            
            **Methodology:**
            - Each component calculates a raw divergence metric
            - Converts to percentile rank (0-100) based on historical window
            - Scales to 25% of total index
            - Final index = sum of four components (0-100)
            
            **Component 1: Directional Opposition (25%)**
            - Variables: `comm_net` = comm_long - comm_short, `noncomm_net` = noncomm_long - noncomm_short
            - Z-scores: Calculate 52-week z-score for each group's net position
            - Raw metric: `abs(comm_net_zscore - noncomm_net_zscore)`
            - Score: 52-week percentile of raw metric × 25
            
            **Component 2: Flow Divergence (25%)**
            - Variables: `comm_flow` = weekly change in comm_net, `noncomm_flow` = weekly change in noncomm_net
            - Normalized: `comm_flow_pct` = comm_flow / open_interest × 100
            - Raw metric: `abs(comm_flow_pct - noncomm_flow_pct)`
            - Score: 26-week percentile of raw metric × 25
            
            **Component 3: Commitment Divergence (25%)**
            - Variables: `comm_imbalance` = comm_long_traders - comm_short_traders
            - Variables: `noncomm_imbalance` = noncomm_long_traders - noncomm_short_traders
            - Raw metric: `abs(noncomm_imbalance - comm_imbalance)`
            - Score: 52-week percentile of raw metric × 25
            
            **Component 4: Directional Bias Divergence (25%)**
            - Variables: `comm_bias` = (comm_long - comm_short) / (comm_long + comm_short)
            - Variables: `noncomm_bias` = (noncomm_long - noncomm_short) / (noncomm_long + noncomm_short)
            - Raw metric: `abs(noncomm_bias - comm_bias)`
            - Score: 52-week percentile of raw metric × 25
            
            **Index Scale (0-100):**
            - **0-25**: Low divergence - Normal market behavior
            - **25-50**: Moderate divergence - Some unusual patterns
            - **50-75**: High divergence - Significant disagreement
            - **75-100**: Extreme divergence - Historic divergence levels
            """)
        
        # Calculate heterogeneity components
        df_hetero = df.copy()
        
        # Component 1: Directional Opposition (25%)
        # Measures divergence in positioning relative to each group's own history
        window = 52  # 52-week lookback
        min_periods = 26  # Need at least 6 months of data
        
        # Calculate net positioning for each group
        df_hetero['comm_net'] = df_hetero['comm_positions_long_all'] - df_hetero['comm_positions_short_all']
        df_hetero['noncomm_net'] = df_hetero['noncomm_positions_long_all'] - df_hetero['noncomm_positions_short_all']
        
        # Calculate z-scores of net positions
        df_hetero['comm_net_mean'] = df_hetero['comm_net'].rolling(window, min_periods=min_periods).mean()
        df_hetero['comm_net_std'] = df_hetero['comm_net'].rolling(window, min_periods=min_periods).std()
        df_hetero['comm_net_zscore'] = np.where(
            df_hetero['comm_net_std'] > 0,
            (df_hetero['comm_net'] - df_hetero['comm_net_mean']) / df_hetero['comm_net_std'],
            0
        )
        
        df_hetero['noncomm_net_mean'] = df_hetero['noncomm_net'].rolling(window, min_periods=min_periods).mean()
        df_hetero['noncomm_net_std'] = df_hetero['noncomm_net'].rolling(window, min_periods=min_periods).std()
        df_hetero['noncomm_net_zscore'] = np.where(
            df_hetero['noncomm_net_std'] > 0,
            (df_hetero['noncomm_net'] - df_hetero['noncomm_net_mean']) / df_hetero['noncomm_net_std'],
            0
        )
        
        # Divergence = how differently positioned they are
        df_hetero['directional_divergence_raw'] = abs(df_hetero['comm_net_zscore'] - df_hetero['noncomm_net_zscore'])
        
        # Get percentile ranking
        df_hetero['directional_percentile'] = df_hetero['directional_divergence_raw'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # Scale to component (0-25)
        df_hetero['directional_opposition'] = (df_hetero['directional_percentile'] / 100) * 25
        
        # Component 2: Flow Divergence (25%)
        # Measures divergence in weekly position changes
        flow_window = 26  # 26-week for more responsive flow metrics
        flow_min_periods = 13
        
        # Calculate week-over-week changes in net positions
        df_hetero['comm_flow'] = df_hetero['comm_net'].diff()
        df_hetero['noncomm_flow'] = df_hetero['noncomm_net'].diff()
        
        # Normalize flows by open interest
        df_hetero['comm_flow_pct'] = (df_hetero['comm_flow'] / df_hetero['open_interest_all']) * 100
        df_hetero['noncomm_flow_pct'] = (df_hetero['noncomm_flow'] / df_hetero['open_interest_all']) * 100
        
        # Flow divergence = difference in normalized flows
        df_hetero['flow_divergence_raw'] = abs(df_hetero['comm_flow_pct'] - df_hetero['noncomm_flow_pct'])
        
        # Get percentile ranking
        df_hetero['flow_percentile'] = df_hetero['flow_divergence_raw'].rolling(flow_window, min_periods=flow_min_periods).rank(pct=True) * 100
        
        # For display purposes, keep track of whether flows oppose
        df_hetero['flows_oppose'] = (df_hetero['comm_flow'] * df_hetero['noncomm_flow']) < 0
        
        # Scale to component weight (0-25)
        df_hetero['flow_divergence'] = (df_hetero['flow_percentile'] / 100) * 25
        
        # Component 3: Commitment Divergence (25%)
        # Measures trader participation imbalance between groups
        
        # Calculate trader imbalances for each group
        df_hetero['noncomm_trader_imbalance'] = df_hetero['traders_noncomm_long_all'] - df_hetero['traders_noncomm_short_all']
        df_hetero['comm_trader_imbalance'] = df_hetero['traders_comm_long_all'] - df_hetero['traders_comm_short_all']
        
        # Calculate absolute divergence in trader imbalances
        df_hetero['trader_imbalance_divergence'] = abs(df_hetero['noncomm_trader_imbalance'] - df_hetero['comm_trader_imbalance'])
        
        # Get percentile ranking
        df_hetero['commitment_percentile'] = df_hetero['trader_imbalance_divergence'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # For display purposes, rename to match UI
        df_hetero['divergence_percentile'] = df_hetero['commitment_percentile']
        
        # Scale to component weight (0-25)
        df_hetero['commitment_divergence'] = (df_hetero['commitment_percentile'] / 100) * 25
        
        # Component 4: Directional Bias Divergence (25%)
        # Measures divergence in position ratios
        
        # Calculate directional biases (-1 to +1)
        df_hetero['noncomm_total'] = df_hetero['noncomm_positions_long_all'] + df_hetero['noncomm_positions_short_all']
        df_hetero['comm_total'] = df_hetero['comm_positions_long_all'] + df_hetero['comm_positions_short_all']
        
        df_hetero['noncomm_bias'] = np.where(
            df_hetero['noncomm_total'] > 0,
            (df_hetero['noncomm_positions_long_all'] - df_hetero['noncomm_positions_short_all']) / df_hetero['noncomm_total'],
            0
        )
        
        df_hetero['comm_bias'] = np.where(
            df_hetero['comm_total'] > 0,
            (df_hetero['comm_positions_long_all'] - df_hetero['comm_positions_short_all']) / df_hetero['comm_total'],
            0
        )
        
        # Bias divergence (0 to 2)
        df_hetero['bias_divergence_raw'] = abs(df_hetero['noncomm_bias'] - df_hetero['comm_bias'])
        
        # Get percentile ranking
        df_hetero['bias_percentile'] = df_hetero['bias_divergence_raw'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # Scale to component weight (0-25)
        df_hetero['directional_bias_divergence'] = (df_hetero['bias_percentile'] / 100) * 25
        
        # Combine all components into final index
        df_hetero['heterogeneity_index'] = (
            df_hetero['directional_opposition'] +
            df_hetero['flow_divergence'] +
            df_hetero['commitment_divergence'] +
            df_hetero['directional_bias_divergence']
        ).clip(0, 100)
        
        # Create visualization
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.3, 0.2],
            subplot_titles=[
                'Heterogeneity Index (0-100)',
                'Component Breakdown',
                'Regime Classification'
            ]
        )
        
        # 1. Main Heterogeneity Index
        fig.add_trace(
            go.Scatter(
                x=df_hetero['report_date_as_yyyy_mm_dd'],
                y=df_hetero['heterogeneity_index'],
                name='Heterogeneity Index',
                line=dict(color='black', width=3),
                fill='tozeroy',
                fillcolor='rgba(0,0,0,0.1)'
            ),
            row=1, col=1
        )
        
        # Add regime zones
        fig.add_hrect(y0=0, y1=25, fillcolor="green", opacity=0.1, row=1, col=1)
        fig.add_hrect(y0=25, y1=50, fillcolor="yellow", opacity=0.1, row=1, col=1)
        fig.add_hrect(y0=50, y1=75, fillcolor="orange", opacity=0.1, row=1, col=1)
        fig.add_hrect(y0=75, y1=100, fillcolor="red", opacity=0.1, row=1, col=1)
        
        # 2. Component Breakdown
        components = ['directional_opposition', 'flow_divergence', 'commitment_divergence', 'directional_bias_divergence']
        colors = ['red', 'blue', 'green', 'orange']
        labels = ['Directional Opposition (25%)', 'Flow Divergence (25%)', 'Commitment Divergence (25%)', 'Directional Bias (25%)']
        
        for comp, color, label in zip(components, colors, labels):
            fig.add_trace(
                go.Scatter(
                    x=df_hetero['report_date_as_yyyy_mm_dd'],
                    y=df_hetero[comp],
                    name=label,
                    stackgroup='components',
                    fillcolor=color,
                    line=dict(color=color, width=0.5)
                ),
                row=2, col=1
            )
        
        # 3. Regime Classification
        df_hetero['regime'] = pd.cut(
            df_hetero['heterogeneity_index'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low Divergence', 'Moderate', 'High Divergence', 'Extreme']
        )
        
        # Create regime indicator
        regime_colors = {
            'Low Divergence': 'green',
            'Moderate': 'yellow',
            'High Divergence': 'orange',
            'Extreme': 'red'
        }
        
        for regime, color in regime_colors.items():
            mask = df_hetero['regime'] == regime
            fig.add_trace(
                go.Scatter(
                    x=df_hetero.loc[mask, 'report_date_as_yyyy_mm_dd'],
                    y=[1] * mask.sum(),
                    mode='markers',
                    marker=dict(color=color, size=10, symbol='square'),
                    name=regime,
                    showlegend=True
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_yaxes(title_text="Index Value", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="Contribution", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Regime", showticklabels=False, row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        fig.update_layout(
            height=800,
            title=f"Market Heterogeneity Analysis - {instrument_name}",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        current_index = df_hetero['heterogeneity_index'].iloc[-1]
        avg_index = df_hetero['heterogeneity_index'].mean()
        current_regime = df_hetero['regime'].iloc[-1]
        
        with col1:
            st.metric("Current Index", f"{current_index:.1f}", 
                     delta=f"{current_index - df_hetero['heterogeneity_index'].iloc[-2]:.1f}")
        with col2:
            st.metric("Average Index", f"{avg_index:.1f}")
        with col3:
            st.metric("Current Regime", current_regime)
        with col4:
            percentile = (df_hetero['heterogeneity_index'] < current_index).sum() / len(df_hetero) * 100
            st.metric("Historical Percentile", f"{percentile:.0f}%")
        
        # Component breakdown details
        with st.expander("📊 Detailed Component Breakdown", expanded=True):
            latest = df_hetero.iloc[-1]
            
            # Create a detailed breakdown table
            st.markdown("### Current Index Calculation")
            
            # Component 1
            st.markdown("**1. Directional Opposition**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Comm Z-score", f"{latest['comm_net_zscore']:.2f}")
            with col2:
                st.metric("NonComm Z-score", f"{latest['noncomm_net_zscore']:.2f}")
            with col3:
                st.metric("Divergence", f"{latest['directional_divergence_raw']:.2f}")
            with col4:
                st.metric("Percentile → Score", f"{latest['directional_percentile']:.0f}th → {latest['directional_opposition']:.1f}")
            
            # Component 2
            st.markdown("**2. Flow Divergence**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Comm Flow %", f"{latest['comm_flow_pct']:.3f}%")
            with col2:
                st.metric("NonComm Flow %", f"{latest['noncomm_flow_pct']:.3f}%")
            with col3:
                st.metric("Divergence", f"{latest['flow_divergence_raw']:.3f}%")
            with col4:
                st.metric("Percentile → Score", f"{latest['flow_percentile']:.0f}th → {latest['flow_divergence']:.1f}")
            
            # Component 3
            st.markdown("**3. Commitment Divergence**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Comm Imbalance", f"{latest['comm_trader_imbalance']:.0f}")
            with col2:
                st.metric("NonComm Imbalance", f"{latest['noncomm_trader_imbalance']:.0f}")
            with col3:
                st.metric("Divergence", f"{latest['trader_imbalance_divergence']:.0f}")
            with col4:
                st.metric("Percentile → Score", f"{latest['commitment_percentile']:.0f}th → {latest['commitment_divergence']:.1f}")
            
            # Component 4
            st.markdown("**4. Directional Bias**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Comm Bias", f"{latest['comm_bias']:.3f}")
            with col2:
                st.metric("NonComm Bias", f"{latest['noncomm_bias']:.3f}")
            with col3:
                st.metric("Divergence", f"{latest['bias_divergence_raw']:.3f}")
            with col4:
                st.metric("Percentile → Score", f"{latest['bias_percentile']:.0f}th → {latest['directional_bias_divergence']:.1f}")
            
            # Summary
            st.markdown("---")
            st.markdown("### Total Index Calculation")
            components_sum = (latest['directional_opposition'] + latest['flow_divergence'] + 
                            latest['commitment_divergence'] + latest['directional_bias_divergence'])
            st.info(f"""
            **Total Score**: {latest['directional_opposition']:.1f} + {latest['flow_divergence']:.1f} + {latest['commitment_divergence']:.1f} + {latest['directional_bias_divergence']:.1f} = **{components_sum:.1f}**
            
            **Current Regime**: {latest['regime']}
            """)
    
    elif analysis_type == "Regime Detection Dashboard":
        st.markdown("#### 🎯 Market Regime Detection")
        
        # Explanation
        with st.expander("📖 Understanding Regime Detection - Complete Guide", expanded=False):
            st.markdown("""
            **What is Regime Detection?**
            
            A comprehensive system that identifies distinct market states by analyzing extreme readings across seven key dimensions of trader behavior. This helps identify when market conditions deviate significantly from historical norms.
            
            **Detailed Methodology:**
            
            **Step 1: Calculate Percentile Rankings (52-week rolling window)**
            - Each metric is ranked against its own 52-week history
            - Percentiles range from 0 (lowest in 52 weeks) to 100 (highest in 52 weeks)
            - Minimum 26 weeks of data required to begin calculations
            
            **Step 2: Calculate Individual Metric Components**
            
            1. **Long Concentration Percentile** = `rank(conc_gross_le_4_tdr_long) / count * 100`
               - Measures: How concentrated long positions are among top 4 traders
               - High percentile (>85): Few traders control most long positions
               
            2. **Short Concentration Percentile** = `rank(conc_gross_le_4_tdr_short) / count * 100`
               - Measures: How concentrated short positions are among top 4 traders
               - High percentile (>85): Few traders control most short positions
            
            3. **Commercial Net Percentile** = `rank(comm_long - comm_short) / count * 100`
               - Measures: How extreme commercial net positioning is
               - High percentile (>85): Commercials unusually long
               - Low percentile (<15): Commercials unusually short
            
            4. **Non-Commercial Net Percentile** = `rank(noncomm_long - noncomm_short) / count * 100`
               - Measures: How extreme speculative net positioning is
               - High percentile (>85): Speculators unusually long
               - Low percentile (<15): Speculators unusually short
            
            5. **Flow Intensity Percentile** = `rank(|comm_flow| + |noncomm_flow|) / count * 100`
               - Measures: Magnitude of weekly position changes
               - High percentile (>85): Unusually large position adjustments
            
            6. **Total Traders Percentile** = `rank(traders_tot_all) / count * 100`
               - Measures: Market participation level
               - High percentile (>70): High participation
               - Low percentile (<30): Low participation
            
            7. **Heterogeneity Percentile** = Currently fixed at 50 (placeholder)
               - Would measure: Inter-group behavioral divergence
            
            **Step 3: Calculate Extremity Score (0-100)**
            
            ```
            distance_from_center(x) = |x - 50| × 2
            
            Extremity = 0.25 × max(distance_from_center(long_conc), distance_from_center(short_conc))
                      + 0.25 × max(distance_from_center(comm_net), distance_from_center(noncomm_net))
                      + 0.25 × flow_intensity_percentile
                      + 0.25 × heterogeneity_percentile
            ```
            
            **Step 4: Regime Classification Rules**
            
            The system checks conditions in order and assigns the first matching regime:
            
            1. **🔴 Long Concentration Extreme**
               - Condition: Long concentration >85th AND Short concentration <30th percentile
               - Meaning: Long side dominated by few large traders, short side distributed
               - Risk: Potential long squeeze if large longs liquidate
            
            2. **🔴 Short Concentration Extreme**
               - Condition: Short concentration >85th AND Long concentration <30th percentile
               - Meaning: Short side dominated by few large traders, long side distributed
               - Risk: Potential short squeeze if large shorts cover
            
            3. **🟠 Bilateral Concentration**
               - Condition: Both Long AND Short concentration >85th percentile
               - Meaning: Both sides dominated by large institutional players
               - Risk: Volatile moves when either side adjusts positions
            
            4. **🔴 Speculative Long Extreme**
               - Condition: Non-commercial net >85th AND Commercial net <15th percentile
               - Meaning: Speculators extremely long, commercials extremely short
               - Risk: Classic overbought condition, vulnerable to reversal
            
            5. **🟠 Commercial Long Extreme**
               - Condition: Non-commercial net <15th AND Commercial net >85th percentile
               - Meaning: Commercials extremely long, speculators extremely short
               - Risk: Potential bottom, commercials often early
            
            6. **🟡 High Flow Volatility**
               - Condition: Flow intensity >85th percentile
               - Meaning: Unusually large week-over-week position changes
               - Risk: Market in transition, direction uncertain
            
            7. **🔴 Maximum Divergence**
               - Condition: Heterogeneity >85th percentile
               - Meaning: Trader groups behaving very differently from each other
               - Risk: Fundamental disagreement, potential for large moves
            
            8. **🟢 Balanced Market**
               - Condition: Extremity score <40
               - Meaning: All metrics within normal ranges
               - Risk: Low - normal market conditions
            
            9. **⚪ Transitional**
               - Condition: Some elevation but no specific extreme pattern
               - Meaning: Market between regimes
               - Risk: Moderate - watch for emerging patterns
            
            **Extremity Score Interpretation:**
            - **0-40**: Normal conditions (Green zone)
            - **40-70**: Elevated conditions (Yellow zone)
            - **70-100**: Extreme conditions (Red zone)
            
            **How to Use This Dashboard:**
            
            1. **Check Current Regime**: Identifies the dominant market characteristic
            2. **Monitor Duration**: Longer durations suggest persistent conditions
            3. **Review Extremity Score**: Higher scores = more unusual conditions
            4. **Analyze Spider Chart**: See which specific metrics are extreme
            5. **Study Timeline**: Understand regime persistence and transitions
            
            **Key Insights:**
            - Multiple regimes can flash warnings before major moves
            - Regime changes often precede trend changes
            - Persistent extreme regimes suggest strong trends
            - Rapid regime cycling indicates unstable conditions
            """)
        
        # Calculate regime metrics
        df_regime = df.copy()
        window = 52
        min_periods = 26
        
        # Step 1: Calculate all percentile metrics
        df_regime['long_conc_pct'] = df_regime['conc_gross_le_4_tdr_long'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        df_regime['short_conc_pct'] = df_regime['conc_gross_le_4_tdr_short'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # Net positions
        df_regime['comm_net'] = df_regime['comm_positions_long_all'] - df_regime['comm_positions_short_all']
        df_regime['noncomm_net'] = df_regime['noncomm_positions_long_all'] - df_regime['noncomm_positions_short_all']
        df_regime['comm_net_pct'] = df_regime['comm_net'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        df_regime['noncomm_net_pct'] = df_regime['noncomm_net'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # Flow intensity
        df_regime['comm_flow'] = df_regime['comm_net'].diff()
        df_regime['noncomm_flow'] = df_regime['noncomm_net'].diff()
        df_regime['flow_intensity'] = abs(df_regime['comm_flow']) + abs(df_regime['noncomm_flow'])
        df_regime['flow_pct'] = df_regime['flow_intensity'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # Trader participation
        df_regime['trader_total_pct'] = df_regime['traders_tot_all'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # We need to recalculate heterogeneity percentile if not already done
        # Using simplified version for now
        df_regime['heterogeneity_pct'] = 50  # Placeholder - would use actual heterogeneity index
        
        # Step 2: Calculate regime extremity score
        def distance_from_center(pct):
            return abs(pct - 50) * 2
        
        df_regime['regime_extremity'] = df_regime.apply(lambda row: 
            max(distance_from_center(row['long_conc_pct']), 
                distance_from_center(row['short_conc_pct'])) * 0.25 +
            max(distance_from_center(row['comm_net_pct']), 
                distance_from_center(row['noncomm_net_pct'])) * 0.25 +
            row['flow_pct'] * 0.25 +
            row['heterogeneity_pct'] * 0.25
        , axis=1)
        
        # Step 3: Detect regime
        def detect_regime(row):
            EXTREME_HIGH = 85
            EXTREME_LOW = 15
            MODERATE_HIGH = 70
            MODERATE_LOW = 30
            
            if pd.isna(row['long_conc_pct']):
                return "Insufficient Data", "gray"
            
            # Check patterns
            if row['long_conc_pct'] > EXTREME_HIGH and row['short_conc_pct'] < MODERATE_LOW:
                return "Long Concentration Extreme", "red"
            elif row['short_conc_pct'] > EXTREME_HIGH and row['long_conc_pct'] < MODERATE_LOW:
                return "Short Concentration Extreme", "red"
            elif row['long_conc_pct'] > EXTREME_HIGH and row['short_conc_pct'] > EXTREME_HIGH:
                return "Bilateral Concentration", "orange"
            elif row['noncomm_net_pct'] > EXTREME_HIGH and row['comm_net_pct'] < EXTREME_LOW:
                return "Speculative Long Extreme", "red"
            elif row['noncomm_net_pct'] < EXTREME_LOW and row['comm_net_pct'] > EXTREME_HIGH:
                return "Commercial Long Extreme", "orange"
            elif row['flow_pct'] > EXTREME_HIGH:
                return "High Flow Volatility", "yellow"
            elif row['heterogeneity_pct'] > EXTREME_HIGH:
                return "Maximum Divergence", "red"
            elif row['regime_extremity'] < 40:
                return "Balanced Market", "green"
            else:
                return "Transitional", "gray"
        
        df_regime[['regime', 'regime_color']] = df_regime.apply(
            lambda row: pd.Series(detect_regime(row)), axis=1
        )
        
        # Create visualization
        latest = df_regime.iloc[-1]
        
        # Main metrics display
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            # Gauge-style display for extremity
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = latest['regime_extremity'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Market Extremity"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col2:
            # Spider chart of all metrics
            categories = ['Long Conc', 'Short Conc', 'Comm Net', 'NonComm Net', 
                         'Flow', 'Traders', 'Heterogeneity']
            values = [
                latest['long_conc_pct'],
                latest['short_conc_pct'],
                latest['comm_net_pct'],
                latest['noncomm_net_pct'],
                latest['flow_pct'],
                latest['trader_total_pct'],
                latest['heterogeneity_pct']
            ]
            
            fig_spider = go.Figure()
            fig_spider.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Current',
                line_color='blue'
            ))
            
            # Add reference circles
            fig_spider.add_trace(go.Scatterpolar(
                r=[50]*7,
                theta=categories,
                name='Normal (50th)',
                line=dict(color='gray', dash='dash')
            ))
            
            fig_spider.add_trace(go.Scatterpolar(
                r=[85]*7,
                theta=categories,
                name='Extreme (85th)',
                line=dict(color='red', dash='dot')
            ))
            
            fig_spider.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Percentile Rankings",
                height=300
            )
            st.plotly_chart(fig_spider, use_container_width=True)
            
        with col3:
            # Current regime display
            st.markdown("### Current Regime")
            regime_color_map = {
                'red': '🔴',
                'orange': '🟠',
                'yellow': '🟡',
                'green': '🟢',
                'gray': '⚪'
            }
            st.markdown(f"## {regime_color_map.get(latest['regime_color'], '⚪')} {latest['regime']}")
            
            # Regime duration
            current_regime = latest['regime']
            regime_duration = 1
            for i in range(2, min(len(df_regime), 20)):
                if df_regime.iloc[-i]['regime'] == current_regime:
                    regime_duration += 1
                else:
                    break
            
            st.metric("Duration", f"{regime_duration} weeks")
            st.metric("Extremity Score", f"{latest['regime_extremity']:.1f} / 100")
        
        # Detailed metrics table
        st.markdown("### Current Metrics")
        metrics_data = {
            'Metric': ['Long Concentration', 'Short Concentration', 'Commercial Net', 
                      'Non-Commercial Net', 'Flow Intensity', 'Total Traders'],
            'Value': [
                f"{latest['conc_gross_le_4_tdr_long']:.1f}%",
                f"{latest['conc_gross_le_4_tdr_short']:.1f}%",
                f"{latest['comm_net']:,.0f}",
                f"{latest['noncomm_net']:,.0f}",
                f"{latest['flow_intensity']:,.0f}",
                f"{latest['traders_tot_all']:.0f}"
            ],
            'Percentile': [
                f"{latest['long_conc_pct']:.0f}th",
                f"{latest['short_conc_pct']:.0f}th",
                f"{latest['comm_net_pct']:.0f}th",
                f"{latest['noncomm_net_pct']:.0f}th",
                f"{latest['flow_pct']:.0f}th",
                f"{latest['trader_total_pct']:.0f}th"
            ],
            'Status': [
                '↑↑' if latest['long_conc_pct'] > 85 else '↓↓' if latest['long_conc_pct'] < 15 else '→',
                '↑↑' if latest['short_conc_pct'] > 85 else '↓↓' if latest['short_conc_pct'] < 15 else '→',
                '↑↑' if latest['comm_net_pct'] > 85 else '↓↓' if latest['comm_net_pct'] < 15 else '→',
                '↑↑' if latest['noncomm_net_pct'] > 85 else '↓↓' if latest['noncomm_net_pct'] < 15 else '→',
                '↑↑' if latest['flow_pct'] > 85 else '↓' if latest['flow_pct'] < 15 else '→',
                '↑' if latest['trader_total_pct'] > 70 else '↓' if latest['trader_total_pct'] < 30 else '→'
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Regime Legend
        st.markdown("### Regime Color Legend")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔴 Red Regimes (High Risk)**")
            st.caption("• Long Concentration Extreme")
            st.caption("• Short Concentration Extreme")
            st.caption("• Speculative Long Extreme")
            st.caption("• Maximum Divergence")
        
        with col2:
            st.markdown("**🟠 Orange Regimes (Moderate Risk)**")
            st.caption("• Bilateral Concentration")
            st.caption("• Commercial Long Extreme")
            st.markdown("**🟡 Yellow Regimes**")
            st.caption("• High Flow Volatility")
        
        with col3:
            st.markdown("**🟢 Green Regimes (Low Risk)**")
            st.caption("• Balanced Market")
            st.markdown("**⚪ Gray Regimes**")
            st.caption("• Transitional")
            st.caption("• Insufficient Data")
        
        # Regime timeline
        st.markdown("### Regime History")
        
        # Create regime timeline chart
        fig_timeline = go.Figure()
        
        # Get last 52 weeks of regime data
        timeline_data = df_regime.tail(52).copy()
        
        # Create color mapping
        color_map = {
            'Long Concentration Extreme': 'darkred',
            'Short Concentration Extreme': 'darkred',
            'Bilateral Concentration': 'orange',
            'Speculative Long Extreme': 'red',
            'Commercial Long Extreme': 'darkorange',
            'High Flow Volatility': 'gold',
            'Maximum Divergence': 'darkred',
            'Balanced Market': 'green',
            'Transitional': 'gray',
            'Insufficient Data': 'lightgray'
        }
        
        # Add all possible regimes to ensure complete legend
        all_regimes = [
            'Long Concentration Extreme',
            'Short Concentration Extreme', 
            'Bilateral Concentration',
            'Speculative Long Extreme',
            'Commercial Long Extreme',
            'High Flow Volatility',
            'Maximum Divergence',
            'Balanced Market',
            'Transitional',
            'Insufficient Data'
        ]
        
        # Add regime bars - include all regimes for complete legend
        for regime in all_regimes:
            regime_mask = timeline_data['regime'] == regime
            if regime_mask.sum() > 0:
                # Regime exists in data
                fig_timeline.add_trace(go.Bar(
                    x=timeline_data.loc[regime_mask, 'report_date_as_yyyy_mm_dd'],
                    y=[1] * regime_mask.sum(),
                    name=regime,
                    marker_color=color_map.get(regime, 'gray'),
                    hovertemplate='%{x}<br>' + regime + '<extra></extra>',
                    showlegend=True
                ))
            else:
                # Add empty trace for legend
                fig_timeline.add_trace(go.Bar(
                    x=[],
                    y=[],
                    name=regime,
                    marker_color=color_map.get(regime, 'gray'),
                    showlegend=True
                ))
        
        fig_timeline.update_layout(
            barmode='stack',
            showlegend=True,
            height=200,
            yaxis=dict(showticklabels=False, title=''),
            xaxis=dict(title='Date'),
            title='52-Week Regime Timeline'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    elif analysis_type == "Concentration Momentum":
        st.markdown("#### 📈 Concentration Momentum Analysis")
        
        # Explanation
        with st.expander("📖 Understanding Concentration Momentum", expanded=False):
            st.markdown("""
            **What is Concentration Momentum?**
            
            Tracks the rate of change (velocity) and acceleration of market concentration to identify when control is shifting between many traders (democratic) and few traders (oligopolistic).
            
            **Key Concepts:**
            - **Concentration Level**: Current % of market controlled by top traders
            - **Momentum (Velocity)**: Rate of change in concentration
            - **Acceleration**: Change in momentum (speeding up/slowing down)
            
            **Concentration Types:**
            - **Gross**: Total long OR short positions (shows raw market power)
            - **Net**: Long minus short positions (shows directional commitment)
            - **Top 4**: Most concentrated measure (oligopoly indicator)
            - **Top 8**: Broader institutional control measure
            
            **Regime Types:**
            - **Bilateral Concentration Building**: Both sides concentrating (institutional battle forming)
            - **Market Democratizing**: Both sides distributing (more participants entering)
            - **Asymmetric Shift**: One side concentrating while other distributes (squeeze setup)
            - **Power Balance Shifting**: Large change in concentration spread
            - **Stable Structure**: Little change in concentration
            
            **Risk Factors:**
            1. High absolute concentration (>40% for top 4, >60% for top 8)
            2. Rapid concentration changes (high momentum)
            3. Accelerating concentration (momentum increasing)
            4. Large imbalances between long/short concentration
            """)
        
        # Concentration type selector
        col1, col2 = st.columns([2, 3])
        
        with col1:
            concentration_type = st.selectbox(
                "Select Concentration Metric:",
                ["Top 4 Gross", "Top 4 Net", "Top 8 Gross", "Top 8 Net"],
                help="""
                • Gross: Total positions (shows market power)
                • Net: Long-short (shows directional bias)
                • Top 4: Most concentrated (oligopoly risk)
                • Top 8: Broader institutional view
                """
            )
        
        with col2:
            # Show comparison checkbox
            show_comparison = st.checkbox("Show all concentration metrics comparison", value=False)
        
        # Map selection to data columns
        concentration_map = {
            "Top 4 Gross": ("conc_gross_le_4_tdr_long", "conc_gross_le_4_tdr_short"),
            "Top 4 Net": ("conc_net_le_4_tdr_long_all", "conc_net_le_4_tdr_short_all"),
            "Top 8 Gross": ("conc_gross_le_8_tdr_long", "conc_gross_le_8_tdr_short"),
            "Top 8 Net": ("conc_net_le_8_tdr_long_all", "conc_net_le_8_tdr_short_all")
        }
        
        long_conc_col, short_conc_col = concentration_map[concentration_type]
        
        # Calculate momentum metrics
        df_momentum = df.copy()
        window = 52
        min_periods = 26
        
        # Core concentration metrics
        df_momentum['long_concentration'] = df_momentum[long_conc_col]
        df_momentum['short_concentration'] = df_momentum[short_conc_col]
        df_momentum['concentration_spread'] = abs(df_momentum['long_concentration'] - df_momentum['short_concentration'])
        df_momentum['max_concentration'] = df_momentum[['long_concentration', 'short_concentration']].max(axis=1)
        
        # Momentum calculations (4-week)
        df_momentum['long_momentum_4w'] = df_momentum['long_concentration'].diff(4)
        df_momentum['short_momentum_4w'] = df_momentum['short_concentration'].diff(4)
        df_momentum['spread_momentum'] = df_momentum['concentration_spread'].diff(4)
        
        # Calculate z-scores for momentum
        df_momentum['long_momentum_mean'] = df_momentum['long_momentum_4w'].rolling(window, min_periods=min_periods).mean()
        df_momentum['long_momentum_std'] = df_momentum['long_momentum_4w'].rolling(window, min_periods=min_periods).std()
        df_momentum['long_momentum_zscore'] = np.where(
            df_momentum['long_momentum_std'] > 0,
            (df_momentum['long_momentum_4w'] - df_momentum['long_momentum_mean']) / df_momentum['long_momentum_std'],
            0
        )
        
        df_momentum['short_momentum_mean'] = df_momentum['short_momentum_4w'].rolling(window, min_periods=min_periods).mean()
        df_momentum['short_momentum_std'] = df_momentum['short_momentum_4w'].rolling(window, min_periods=min_periods).std()
        df_momentum['short_momentum_zscore'] = np.where(
            df_momentum['short_momentum_std'] > 0,
            (df_momentum['short_momentum_4w'] - df_momentum['short_momentum_mean']) / df_momentum['short_momentum_std'],
            0
        )
        
        # Spread momentum z-score
        df_momentum['spread_momentum_mean'] = df_momentum['spread_momentum'].rolling(window, min_periods=min_periods).mean()
        df_momentum['spread_momentum_std'] = df_momentum['spread_momentum'].rolling(window, min_periods=min_periods).std()
        df_momentum['spread_momentum_zscore'] = np.where(
            df_momentum['spread_momentum_std'] > 0,
            (df_momentum['spread_momentum'] - df_momentum['spread_momentum_mean']) / df_momentum['spread_momentum_std'],
            0
        )
        
        # Acceleration (1-week momentum change)
        df_momentum['long_momentum_1w'] = df_momentum['long_concentration'].diff(1)
        df_momentum['short_momentum_1w'] = df_momentum['short_concentration'].diff(1)
        df_momentum['long_acceleration'] = df_momentum['long_momentum_1w'].diff(1)
        df_momentum['short_acceleration'] = df_momentum['short_momentum_1w'].diff(1)
        
        # Regime detection
        def detect_momentum_regime(row):
            long_z = row['long_momentum_zscore']
            short_z = row['short_momentum_zscore']
            spread_z = row['spread_momentum_zscore']
            
            # Adjust thresholds based on concentration type
            if "Net" in concentration_type:
                high_threshold = 2.0
                low_threshold = -2.0
            else:
                high_threshold = 1.5
                low_threshold = -1.5
            
            if pd.isna(long_z) or pd.isna(short_z):
                return "Insufficient Data", "gray"
            
            if long_z > high_threshold and short_z > high_threshold:
                return "Bilateral Concentration Building", "red"
            elif long_z < low_threshold and short_z < low_threshold:
                return "Market Democratizing", "green"
            elif long_z > high_threshold and short_z < low_threshold:
                return "Long Concentration / Short Distribution", "orange"
            elif long_z < low_threshold and short_z > high_threshold:
                return "Short Concentration / Long Distribution", "orange"
            elif abs(spread_z) > 2.0:
                return "Power Balance Shifting", "yellow"
            elif max(abs(long_z), abs(short_z)) < 0.5:
                return "Stable Structure", "lightgreen"
            else:
                return "Transitional", "gray"
        
        df_momentum[['momentum_regime', 'regime_color']] = df_momentum.apply(
            lambda row: pd.Series(detect_momentum_regime(row)), axis=1
        )
        
        # Calculate regime duration
        current_regime = df_momentum['momentum_regime'].iloc[-1]
        regime_duration = 1
        for i in range(2, min(len(df_momentum), 20)):
            if df_momentum.iloc[-i]['momentum_regime'] == current_regime:
                regime_duration += 1
            else:
                break
        
        # Risk calculations
        latest = df_momentum.iloc[-1]
        
        # Percentile-based risk metrics
        long_pct = df_momentum['long_concentration'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-1] * 100
        short_pct = df_momentum['short_concentration'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-1] * 100
        spread_pct = df_momentum['concentration_spread'].rolling(window, min_periods=min_periods).rank(pct=True).iloc[-1] * 100
        
        # Adaptive risk scoring
        concentration_risk = max(long_pct, short_pct) / 100
        momentum_risk = min(max(abs(latest['long_momentum_zscore']), abs(latest['short_momentum_zscore'])) / 2, 1)
        imbalance_risk = spread_pct / 100
        
        composite_risk = (concentration_risk * 0.4 + momentum_risk * 0.4 + imbalance_risk * 0.2) * 100
        
        # Create visualizations
        if show_comparison:
            # Comparison of all concentration types
            fig_compare = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Top 4 Gross', 'Top 4 Net', 'Top 8 Gross', 'Top 8 Net'],
                shared_xaxes=True,
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for idx, (metric_name, (long_col, short_col)) in enumerate(concentration_map.items()):
                row = idx // 2 + 1
                col = idx % 2 + 1
                
                if long_col in df.columns and short_col in df.columns:
                    fig_compare.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[long_col],
                            name=f'Long',
                            line=dict(color='green', width=2),
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )
                    
                    fig_compare.add_trace(
                        go.Scatter(
                            x=df['report_date_as_yyyy_mm_dd'],
                            y=df[short_col],
                            name=f'Short',
                            line=dict(color='red', width=2),
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )
                    
                    # Add title annotations
                    fig_compare.add_annotation(
                        text=metric_name,
                        xref=f"x{idx+1} domain",
                        yref=f"y{idx+1} domain",
                        x=0.5, y=1.05,
                        showarrow=False,
                        font=dict(size=12, color='black')
                    )
            
            fig_compare.update_layout(
                height=600,
                title="Concentration Metrics Comparison",
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
        
        # Main momentum visualization
        fig_main = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.35, 0.25, 0.25, 0.15],
            subplot_titles=[
                f'{concentration_type} Concentration Levels',
                'Momentum Z-Scores (4-week change)',
                'Acceleration (Rate of change)',
                'Momentum Regime'
            ]
        )
        
        # Plot 1: Concentration levels
        fig_main.add_trace(
            go.Scatter(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=df_momentum['long_concentration'],
                name='Long Concentration',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        fig_main.add_trace(
            go.Scatter(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=df_momentum['short_concentration'],
                name='Short Concentration',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Add spread as area
        fig_main.add_trace(
            go.Scatter(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=df_momentum['concentration_spread'],
                name='Spread',
                fill='tozeroy',
                fillcolor='rgba(128, 128, 128, 0.2)',
                line=dict(color='gray', width=1)
            ),
            row=1, col=1
        )
        
        # Plot 2: Momentum z-scores
        fig_main.add_trace(
            go.Scatter(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=df_momentum['long_momentum_zscore'],
                name='Long Momentum Z',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig_main.add_trace(
            go.Scatter(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=df_momentum['short_momentum_zscore'],
                name='Short Momentum Z',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Add threshold lines
        fig_main.add_hline(y=1.5, row=2, col=1, line_dash="dash", line_color="orange", opacity=0.5)
        fig_main.add_hline(y=-1.5, row=2, col=1, line_dash="dash", line_color="orange", opacity=0.5)
        fig_main.add_hline(y=0, row=2, col=1, line_dash="solid", line_color="gray", opacity=0.3)
        
        # Plot 3: Acceleration
        fig_main.add_trace(
            go.Bar(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=df_momentum['long_acceleration'],
                name='Long Acceleration',
                marker_color='green',
                opacity=0.6
            ),
            row=3, col=1
        )
        
        fig_main.add_trace(
            go.Bar(
                x=df_momentum['report_date_as_yyyy_mm_dd'],
                y=df_momentum['short_acceleration'],
                name='Short Acceleration',
                marker_color='red',
                opacity=0.6
            ),
            row=3, col=1
        )
        
        # Plot 4: Regime timeline
        for regime in df_momentum['momentum_regime'].unique():
            regime_mask = df_momentum['momentum_regime'] == regime
            color = df_momentum.loc[regime_mask, 'regime_color'].iloc[0] if regime_mask.sum() > 0 else 'gray'
            
            fig_main.add_trace(
                go.Bar(
                    x=df_momentum.loc[regime_mask, 'report_date_as_yyyy_mm_dd'],
                    y=[1] * regime_mask.sum(),
                    name=regime,
                    marker_color=color,
                    showlegend=False,
                    hovertemplate='%{x}<br>' + regime + '<extra></extra>'
                ),
                row=4, col=1
            )
        
        # Update layout
        fig_main.update_yaxes(title_text="Concentration %", row=1, col=1)
        fig_main.update_yaxes(title_text="Z-Score", row=2, col=1)
        fig_main.update_yaxes(title_text="% Change", row=3, col=1)
        fig_main.update_yaxes(showticklabels=False, row=4, col=1)
        fig_main.update_xaxes(title_text="Date", row=4, col=1)
        
        fig_main.update_layout(
            height=900,
            title=f"Concentration Momentum Analysis - {instrument_name}",
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Risk and regime dashboard
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Risk gauge
            fig_risk = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = composite_risk,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Composite Risk Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_risk.update_layout(height=250)
            st.plotly_chart(fig_risk, use_container_width=True)
            
            st.metric("Current Regime", current_regime)
            st.metric("Regime Duration", f"{regime_duration} weeks")
        
        with col2:
            # Momentum quadrant chart
            fig_quad = go.Figure()
            
            # Add quadrant backgrounds
            fig_quad.add_shape(
                type="rect", x0=-3, x1=0, y0=0, y1=3,
                fillcolor="lightgreen", opacity=0.2,
                line=dict(width=0)
            )
            fig_quad.add_shape(
                type="rect", x0=0, x1=3, y0=0, y1=3,
                fillcolor="lightcoral", opacity=0.2,
                line=dict(width=0)
            )
            fig_quad.add_shape(
                type="rect", x0=-3, x1=0, y0=-3, y1=0,
                fillcolor="lightblue", opacity=0.2,
                line=dict(width=0)
            )
            fig_quad.add_shape(
                type="rect", x0=0, x1=3, y0=-3, y1=0,
                fillcolor="lightyellow", opacity=0.2,
                line=dict(width=0)
            )
            
            # Add current position
            fig_quad.add_trace(go.Scatter(
                x=[latest['long_momentum_zscore']],
                y=[latest['short_momentum_zscore']],
                mode='markers+text',
                marker=dict(size=20, color='darkblue'),
                text=['Current'],
                textposition='top center',
                showlegend=False
            ))
            
            # Add historical trail
            trail_data = df_momentum.tail(12)
            fig_quad.add_trace(go.Scatter(
                x=trail_data['long_momentum_zscore'],
                y=trail_data['short_momentum_zscore'],
                mode='lines+markers',
                line=dict(color='blue', width=1),
                marker=dict(size=5),
                opacity=0.5,
                name='12-week trail'
            ))
            
            fig_quad.update_xaxes(
                title="Long Momentum Z-Score",
                range=[-3, 3],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black'
            )
            fig_quad.update_yaxes(
                title="Short Momentum Z-Score",
                range=[-3, 3],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black'
            )
            
            fig_quad.update_layout(
                title="Momentum Quadrant Analysis",
                height=300,
                annotations=[
                    dict(x=-1.5, y=2.5, text="Short Building<br>Long Unwinding", showarrow=False),
                    dict(x=1.5, y=2.5, text="Both Building<br>(Bilateral)", showarrow=False),
                    dict(x=-1.5, y=-2.5, text="Both Unwinding<br>(Democratizing)", showarrow=False),
                    dict(x=1.5, y=-2.5, text="Long Building<br>Short Unwinding", showarrow=False)
                ]
            )
            
            st.plotly_chart(fig_quad, use_container_width=True)
        
        with col3:
            # Current metrics
            st.markdown("### Current Metrics")
            st.metric("Long Concentration", f"{latest['long_concentration']:.1f}%", 
                     f"{latest['long_momentum_4w']:+.1f}% (4w)")
            st.metric("Short Concentration", f"{latest['short_concentration']:.1f}%",
                     f"{latest['short_momentum_4w']:+.1f}% (4w)")
            st.metric("Concentration Spread", f"{latest['concentration_spread']:.1f}%",
                     f"{latest['spread_momentum']:+.1f}% (4w)")
        
        # Type-specific insights
        st.markdown("### Concentration Type Insights")
        
        if "Gross" in concentration_type:
            st.info(f"""
            **{concentration_type}** measures total positions held by top traders.
            - Current Long: {latest['long_concentration']:.1f}% ({long_pct:.0f}th percentile)
            - Current Short: {latest['short_concentration']:.1f}% ({short_pct:.0f}th percentile)
            - Shows raw market power regardless of hedging
            """)
        else:  # Net
            st.info(f"""
            **{concentration_type}** measures net exposure (long - short) of top traders.
            - Current Long: {latest['long_concentration']:.1f}% ({long_pct:.0f}th percentile)
            - Current Short: {latest['short_concentration']:.1f}% ({short_pct:.0f}th percentile)
            - Shows directional commitment after hedging
            """)
        
        # Generate signals
        signals = []
        
        if long_pct > 90 and latest['long_momentum_zscore'] > 1:
            signals.append(("⚠️ Long Concentration Critical", "Long side dangerously concentrated and still rising"))
        
        if short_pct > 90 and latest['short_momentum_zscore'] > 1:
            signals.append(("⚠️ Short Concentration Critical", "Short side dangerously concentrated and still rising"))
        
        if long_pct > 85 and latest['long_momentum_zscore'] < -1:
            signals.append(("📉 Long Distribution Beginning", "High long concentration starting to unwind"))
        
        if abs(latest['long_momentum_zscore'] - latest['short_momentum_zscore']) > 3:
            signals.append(("🔄 Extreme Asymmetry", "Unprecedented divergence in concentration momentum"))
        
        if regime_duration > 8 and "Building" in current_regime:
            signals.append(("⏰ Persistent Concentration", f"{current_regime} has persisted for {regime_duration} weeks"))
        
        if signals:
            st.markdown("### Current Signals")
            for signal, description in signals:
                st.warning(f"{signal}: {description}")
                
    elif analysis_type == "Participant Behavior Clusters":
        st.markdown("#### 🎯 Participant Behavior Clusters")
        
        # Explanation expander
        with st.expander("📖 Understanding Participant Behavior Clusters", expanded=False):
            st.markdown("""
            **What are Participant Behavior Clusters?**
            
            This analysis uses clustering algorithms to identify groups of market participants with similar behavior patterns.
            We analyze multiple dimensions to create behavioral profiles.
            
            **Features Analyzed:**
            1. **Position Size**: Average position per trader (normalized)
            2. **Directional Bias**: Long vs Short positioning preference
            3. **Activity Level**: Trading frequency and position changes
            4. **Concentration**: How concentrated their positions are
            5. **Trend Following**: Correlation with price movements
            
            **Clustering Method:**
            - Uses K-means clustering to identify 3-5 distinct behavior groups
            - Each cluster represents a different trading style/approach
            - Tracks how cluster membership changes over time
            
            **Common Cluster Types:**
            - **Trend Followers**: Move with price, high activity
            - **Contrarians**: Counter-trend positioning
            - **Hedgers**: Balanced long/short, stable positions
            - **Speculators**: Large directional bets, high concentration
            - **Market Makers**: Small positions, high activity
            
            **Why It Matters:**
            - Identifies dominant trading strategies in the market
            - Shows shifts in participant behavior over time
            - Helps predict market dynamics based on cluster changes
            """)
        
        # Configuration
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            cluster_count = st.selectbox(
                "Number of Clusters:",
                [3, 4, 5],
                index=1,
                help="More clusters = more granular behavior groups"
            )
        
        with col2:
            lookback_weeks = st.selectbox(
                "Analysis Window:",
                ["13 Weeks", "26 Weeks", "52 Weeks"],
                index=1
            )
            lookback_map = {"13 Weeks": 13, "26 Weeks": 26, "52 Weeks": 52}
            weeks = lookback_map[lookback_weeks]
        
        with col3:
            show_transitions = st.checkbox(
                "Show Cluster Transitions",
                value=True,
                help="Track how traders move between clusters"
            )
        
        # Prepare data for clustering
        df_cluster = df.copy()
        
        # Calculate features for each trader category
        categories = [
            ('Commercial Long', 'comm_positions_long_all', 'traders_comm_long_all'),
            ('Commercial Short', 'comm_positions_short_all', 'traders_comm_short_all'),
            ('Non-Commercial Long', 'noncomm_positions_long_all', 'traders_noncomm_long_all'),
            ('Non-Commercial Short', 'noncomm_positions_short_all', 'traders_noncomm_short_all')
        ]
        
        # Get latest data window
        latest_date = df_cluster['report_date_as_yyyy_mm_dd'].max()
        start_date = latest_date - pd.Timedelta(weeks=weeks)
        df_window = df_cluster[df_cluster['report_date_as_yyyy_mm_dd'] >= start_date].copy()
        
        # Calculate features for each category over the window
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            sklearn_available = True
        except ImportError:
            sklearn_available = False
            st.warning("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")
            st.info("Clustering visualization requires scikit-learn for advanced analysis.")
            return
        
        clustering_data = []
        
        for date in df_window['report_date_as_yyyy_mm_dd'].unique():
            date_data = df_window[df_window['report_date_as_yyyy_mm_dd'] == date]
            
            for cat_name, pos_col, trader_col in categories:
                if trader_col in date_data.columns:
                    row = date_data.iloc[0]
                    
                    # Calculate features
                    traders = float(row[trader_col]) if pd.notna(row[trader_col]) and float(row[trader_col]) > 0 else 0.001
                    positions = float(row[pos_col]) if pd.notna(row[pos_col]) else 0
                    
                    # Feature 1: Average position size (normalized by total OI)
                    avg_position = positions / traders if traders > 0 else 0
                    avg_position_norm = avg_position / float(row['open_interest_all']) * 100
                    
                    # Feature 2: Directional bias
                    if 'Long' in cat_name:
                        direction = 1
                    else:
                        direction = -1
                    
                    # Feature 3: Activity level (week-over-week change)
                    prev_week = date - pd.Timedelta(days=7)
                    prev_data = df_window[df_window['report_date_as_yyyy_mm_dd'] <= prev_week].tail(1)
                    if len(prev_data) > 0 and pos_col in prev_data.columns:
                        prev_positions = float(prev_data.iloc[0][pos_col]) if pd.notna(prev_data.iloc[0][pos_col]) else 0
                        activity = abs(positions - prev_positions) / float(row['open_interest_all']) * 100
                    else:
                        activity = 0
                    
                    # Feature 4: Concentration (using percentile)
                    hist_data = df_cluster[df_cluster['report_date_as_yyyy_mm_dd'] >= pd.Timestamp('2010-01-01')]
                    if trader_col in hist_data.columns:
                        trader_pct = stats.percentileofscore(hist_data[trader_col].dropna(), traders)
                    else:
                        trader_pct = 50
                    
                    concentration = 100 - trader_pct  # High percentile = many traders = low concentration
                    
                    # Feature 5: Relative strength (position as % of category total)
                    if 'comm' in pos_col:
                        total_comm = float(row['comm_positions_long_all']) + float(row['comm_positions_short_all'])
                        relative_strength = positions / total_comm * 100 if total_comm > 0 else 0
                    else:
                        total_noncomm = float(row['noncomm_positions_long_all']) + float(row['noncomm_positions_short_all'])
                        relative_strength = positions / total_noncomm * 100 if total_noncomm > 0 else 0
                    
                    clustering_data.append({
                        'date': date,
                        'category': cat_name,
                        'avg_position_norm': avg_position_norm,
                        'direction': direction,
                        'activity': activity,
                        'concentration': concentration,
                        'relative_strength': relative_strength,
                        'traders': traders,
                        'positions': positions
                    })
        
        # Create DataFrame
        cluster_df = pd.DataFrame(clustering_data)
        
        if len(cluster_df) > 0:
            # Prepare features for clustering
            feature_cols = ['avg_position_norm', 'direction', 'activity', 'concentration', 'relative_strength']
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(cluster_df[feature_cols])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
            cluster_df['cluster'] = kmeans.fit_predict(features_scaled)
            
            # Analyze clusters
            cluster_profiles = []
            for i in range(cluster_count):
                cluster_data = cluster_df[cluster_df['cluster'] == i]
                profile = {
                    'cluster': i,
                    'size': len(cluster_data),
                    'avg_position': cluster_data['avg_position_norm'].mean(),
                    'direction_bias': cluster_data['direction'].mean(),
                    'activity': cluster_data['activity'].mean(),
                    'concentration': cluster_data['concentration'].mean(),
                    'relative_strength': cluster_data['relative_strength'].mean()
                }
                
                # Classify cluster type
                if profile['activity'] > cluster_df['activity'].quantile(0.75):
                    if abs(profile['direction_bias']) > 0.5:
                        profile['type'] = "🚀 Aggressive Traders"
                    else:
                        profile['type'] = "⚡ Market Makers"
                elif profile['concentration'] > cluster_df['concentration'].quantile(0.75):
                    profile['type'] = "🐋 Large Speculators"
                elif abs(profile['direction_bias']) < 0.2 and profile['activity'] < cluster_df['activity'].quantile(0.25):
                    profile['type'] = "🛡️ Hedgers"
                else:
                    profile['type'] = "📊 Trend Followers"
                
                cluster_profiles.append(profile)
            
            # Create visualizations
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cluster Distribution by Category', 'Cluster Characteristics',
                              'Temporal Evolution', 'Cluster Profiles'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "table"}]]
            )
            
            # Plot 1: Cluster distribution
            cluster_counts = cluster_df.groupby(['category', 'cluster']).size().reset_index(name='count')
            for i in range(cluster_count):
                cluster_i = cluster_counts[cluster_counts['cluster'] == i]
                fig.add_trace(go.Bar(
                    x=cluster_i['category'],
                    y=cluster_i['count'],
                    name=f"Cluster {i}",
                    showlegend=True
                ), row=1, col=1)
            
            # Plot 2: Cluster characteristics (2D projection)
            # Use first two principal components for visualization
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_scaled)
            
            for i in range(cluster_count):
                mask = cluster_df['cluster'] == i
                profile = next(p for p in cluster_profiles if p['cluster'] == i)
                fig.add_trace(go.Scatter(
                    x=features_2d[mask, 0],
                    y=features_2d[mask, 1],
                    mode='markers',
                    name=profile['type'],
                    marker=dict(size=8),
                    showlegend=False
                ), row=1, col=2)
            
            # Plot 3: Temporal evolution
            temporal_counts = cluster_df.groupby(['date', 'cluster']).size().reset_index(name='count')
            for i in range(cluster_count):
                cluster_i = temporal_counts[temporal_counts['cluster'] == i]
                profile = next(p for p in cluster_profiles if p['cluster'] == i)
                fig.add_trace(go.Scatter(
                    x=cluster_i['date'],
                    y=cluster_i['count'],
                    name=profile['type'],
                    mode='lines+markers',
                    showlegend=False
                ), row=2, col=1)
            
            # Plot 4: Cluster profiles table
            profile_data = []
            for p in cluster_profiles:
                profile_data.append([
                    p['type'],
                    f"{p['size']}",
                    f"{p['avg_position']:.1f}%",
                    f"{p['direction_bias']:+.2f}",
                    f"{p['activity']:.1f}%",
                    f"{p['concentration']:.0f}%"
                ])
            
            fig.add_trace(go.Table(
                header=dict(
                    values=['Cluster Type', 'Members', 'Avg Position', 'Direction', 'Activity', 'Concentration'],
                    fill_color='lightgray',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*profile_data)),
                    align='left'
                )
            ), row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title=f"Participant Behavior Clustering Analysis ({lookback_weeks})",
                height=900,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Category", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_xaxes(title_text="Component 1", row=1, col=2)
            fig.update_yaxes(title_text="Component 2", row=1, col=2)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Participants", row=2, col=1)
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Show transitions if requested
            if show_transitions:
                st.markdown("### Cluster Transitions")
                
                # Calculate week-over-week transitions
                latest_week = cluster_df['date'].max()
                prev_week = latest_week - pd.Timedelta(days=7)
                
                latest_clusters = cluster_df[cluster_df['date'] == latest_week][['category', 'cluster']]
                prev_clusters = cluster_df[cluster_df['date'] <= prev_week].groupby('category').last()[['cluster']].reset_index()
                
                transitions = latest_clusters.merge(prev_clusters, on='category', suffixes=('_current', '_previous'))
                transitions = transitions[transitions['cluster_current'] != transitions['cluster_previous']]
                
                if len(transitions) > 0:
                    for _, trans in transitions.iterrows():
                        prev_profile = next(p for p in cluster_profiles if p['cluster'] == trans['cluster_previous'])
                        curr_profile = next(p for p in cluster_profiles if p['cluster'] == trans['cluster_current'])
                        st.info(f"{trans['category']}: {prev_profile['type']} → {curr_profile['type']}")
                else:
                    st.success("No significant cluster transitions in the last week")
        
        else:
            st.warning("Insufficient data for clustering analysis")
            
    elif analysis_type == "Market Microstructure Analysis":
        st.markdown("#### 🔬 Market Microstructure Analysis")
        
        # Explanation expander
        with st.expander("📖 Understanding Market Microstructure", expanded=False):
            st.markdown("""
            **What is Market Microstructure Analysis?**
            
            A granular view of how different trader sizes and types interact in the market.
            This analysis reveals the "ecosystem" of market participants.
            
            **Key Metrics:**
            
            1. **Size Distribution**
               - Small traders: Bottom 50% by position size
               - Medium traders: 50-90th percentile
               - Large traders: 90-95th percentile
               - Whales: Top 5%
               
            2. **Market Impact Potential**
               - Measures how much each group could move the market
               - Based on position size × concentration
               
            3. **Liquidity Provision**
               - Identifies who provides vs consumes liquidity
               - Based on position stability and two-sided exposure
               
            4. **Position Persistence**
               - How long different groups maintain positions
               - Indicates trading style (scalping vs investing)
               
            **Microstructure Patterns:**
            - **Healthy**: Diverse participation across all sizes
            - **Top-Heavy**: Dominated by large traders
            - **Bifurcated**: Missing middle (only large and small)
            - **Fragmented**: Many small traders, few large
            
            **Why It Matters:**
            - Reveals market vulnerability to large trader actions
            - Shows liquidity depth and resilience
            - Identifies potential manipulation risks
            - Helps predict volatility patterns
            """)
        
        # Configuration
        col1, col2 = st.columns([1, 1])
        
        with col1:
            weeks_back = st.selectbox(
                "Compare with:",
                [0, 1, 2, 3, 4],
                format_func=lambda x: "Current Week" if x == 0 else f"{x} Week{'s' if x > 1 else ''} Ago",
                index=0,
                help="Show data from selected week for comparison"
            )
        
        with col2:
            # Only show evolution checkbox for current week
            if weeks_back == 0:
                show_evolution = st.checkbox(
                    "Show Evolution Over Time",
                    value=True,
                    help="Track how microstructure changes"
                )
            else:
                show_evolution = False
                st.empty()  # Keep layout consistent
        
        # Prepare data
        df_micro = df.copy()
        
        # Get the analysis window - single week based on selection
        sorted_dates = sorted(df_micro['report_date_as_yyyy_mm_dd'].unique(), reverse=True)
        
        if weeks_back < len(sorted_dates):
            selected_date = sorted_dates[weeks_back]
            analysis_data = df_micro[df_micro['report_date_as_yyyy_mm_dd'] == selected_date]
            periods = 1
        else:
            # Fallback to latest available
            selected_date = sorted_dates[0]
            analysis_data = df_micro[df_micro['report_date_as_yyyy_mm_dd'] == selected_date]
            periods = 1
        
        if len(analysis_data) > 0:
            # CORRECTED MICROSTRUCTURE CALCULATION
            # Using concentration as % of total OI, not category positions
            
            # Get values for the selected week
            avg_oi = float(analysis_data['open_interest_all'].iloc[0])
            
            # Calculate top trader metrics from OI-based concentration
            top_traders_metrics = {}
            
            for side in ['long', 'short']:
                # Use gross concentration (includes all traders)
                if side == 'long':
                    conc_4_pct = float(analysis_data['conc_gross_le_4_tdr_long'].iloc[0])
                    conc_8_pct = float(analysis_data['conc_gross_le_8_tdr_long'].iloc[0])
                else:
                    conc_4_pct = float(analysis_data['conc_gross_le_4_tdr_short'].iloc[0])
                    conc_8_pct = float(analysis_data['conc_gross_le_8_tdr_short'].iloc[0])
                
                # Calculate contracts held by top traders
                # These percentages are of TOTAL OPEN INTEREST
                top_4_contracts = avg_oi * (conc_4_pct / 100)
                top_4_avg = top_4_contracts / 4
                
                # Top 8 total contracts
                top_8_contracts = avg_oi * (conc_8_pct / 100)
                
                # Next 4 (5-8) by subtraction
                next_4_contracts = top_8_contracts - top_4_contracts
                next_4_avg = next_4_contracts / 4
                
                top_traders_metrics[side] = {
                    'top_4_avg': top_4_avg,
                    'next_4_avg': next_4_avg,
                    'top_4_pct_of_oi': conc_4_pct,
                    'top_8_pct_of_oi': conc_8_pct
                }
            
            # Calculate category averages for comparison
            category_metrics = []
            
            categories = [
                ('Commercial Long', 'comm_positions_long_all', 'traders_comm_long_all', 'red'),
                ('Commercial Short', 'comm_positions_short_all', 'traders_comm_short_all', 'darkred'),
                ('Non-Commercial Long', 'noncomm_positions_long_all', 'traders_noncomm_long_all', 'blue'),
                ('Non-Commercial Short', 'noncomm_positions_short_all', 'traders_noncomm_short_all', 'darkblue')
            ]
            
            for cat_name, pos_col, trader_col, color in categories:
                if trader_col in analysis_data.columns and pos_col in analysis_data.columns:
                    avg_positions = float(analysis_data[pos_col].iloc[0])
                    avg_traders = float(analysis_data[trader_col].iloc[0])
                    
                    if avg_traders > 0:
                        avg_position = avg_positions / avg_traders
                        position_pct_of_oi = (avg_positions / avg_oi) * 100
                        
                        # Determine side for comparison
                        side = 'long' if 'Long' in cat_name else 'short'
                        
                        # Calculate ratios to top traders
                        top_4_avg = top_traders_metrics[side]['top_4_avg']
                        ratio_to_top_4 = top_4_avg / avg_position if avg_position > 0 else 0
                        
                        category_metrics.append({
                            'category': cat_name,
                            'total_traders': avg_traders,
                            'total_positions': avg_positions,
                            'avg_position': avg_position,
                            'position_pct_of_oi': position_pct_of_oi,
                            'ratio_to_top_4': ratio_to_top_4,
                            'color': color,
                            'side': side
                        })
            
            # Combine top trader and category data for visualization
            micro_metrics = []
            
            # Create visualizations with corrected approach
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Average Position Comparison', 'Market Control Breakdown',
                              'Top Trader Dominance', 'Category vs Top Trader Analysis'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatter"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )
            
            # Plot 1: Average Position Comparison
            # Shows top traders vs category averages
            
            # Prepare data for comparison
            groups = []
            avg_positions = []
            colors = []
            labels = []
            
            # Add top traders (both sides)
            for side in ['long', 'short']:
                metrics = top_traders_metrics[side]
                groups.extend([f'Top 4 {side.title()}', f'Next 4 {side.title()}'])
                avg_positions.extend([metrics['top_4_avg'], metrics['next_4_avg']])
                colors.extend(['gold' if side == 'long' else 'darkgoldenrod', 
                             'orange' if side == 'long' else 'darkorange'])
                labels.extend([f"{metrics['top_4_avg']:,.0f}", f"{metrics['next_4_avg']:,.0f}"])
            
            # Add category averages
            for cat_metric in category_metrics:
                groups.append(cat_metric['category'])
                avg_positions.append(cat_metric['avg_position'])
                colors.append(cat_metric['color'])
                labels.append(f"{cat_metric['avg_position']:,.0f}")
            
            fig.add_trace(go.Bar(
                x=groups,
                y=avg_positions,
                marker_color=colors,
                text=labels,
                textposition='auto',
                showlegend=False,
                hovertemplate='%{x}<br>Avg Position: %{y:,.0f} contracts<extra></extra>'
            ), row=1, col=1)
            
            # Add ratio annotations
            for i, cat_metric in enumerate(category_metrics):
                if cat_metric['ratio_to_top_4'] > 0:
                    fig.add_annotation(
                        x=cat_metric['category'],
                        y=cat_metric['avg_position'],
                        text=f"{cat_metric['ratio_to_top_4']:.1f}x",
                        showarrow=True,
                        arrowhead=2,
                        yshift=20,
                        row=1, col=1
                    )
            
            # Plot 2: Contract Distribution Breakdown
            # Shows actual contract numbers from concentration data
            
            # Get total reportable positions for each side
            total_long = float(analysis_data['tot_rept_positions_long_all'].iloc[0]) if 'tot_rept_positions_long_all' in analysis_data.columns else 0
            total_short = float(analysis_data['tot_rept_positions_short'].iloc[0]) if 'tot_rept_positions_short' in analysis_data.columns else 0
            
            # Calculate actual contracts from OI percentages
            long_metrics = top_traders_metrics['long']
            short_metrics = top_traders_metrics['short']
            
            # Top 4 contracts (from % of total OI)
            long_top4_contracts = avg_oi * (long_metrics['top_4_pct_of_oi'] / 100)
            short_top4_contracts = avg_oi * (short_metrics['top_4_pct_of_oi'] / 100)
            
            # Top 8 total contracts
            long_top8_contracts = avg_oi * (long_metrics['top_8_pct_of_oi'] / 100)
            short_top8_contracts = avg_oi * (short_metrics['top_8_pct_of_oi'] / 100)
            
            # Next 4 (5-8) contracts
            long_next4_contracts = long_top8_contracts - long_top4_contracts
            short_next4_contracts = short_top8_contracts - short_top4_contracts
            
            # Remaining contracts (total reportable minus top 8)
            long_remaining = total_long - long_top8_contracts if total_long > long_top8_contracts else 0
            short_remaining = total_short - short_top8_contracts if total_short > short_top8_contracts else 0
            
            # Create stacked bar showing contract distribution
            fig.add_trace(go.Bar(
                name='Top 4 Traders',
                x=['Long Contracts', 'Short Contracts'],
                y=[long_top4_contracts, short_top4_contracts],
                text=[f'{long_top4_contracts:,.0f}<br>({long_metrics["top_4_pct_of_oi"]:.1f}% of OI)', 
                      f'{short_top4_contracts:,.0f}<br>({short_metrics["top_4_pct_of_oi"]:.1f}% of OI)'],
                textposition='inside',
                marker_color='darkred',
                hovertemplate='%{x}<br>Top 4: %{y:,.0f} contracts<extra></extra>'
            ), row=1, col=2)
            
            fig.add_trace(go.Bar(
                name='Next 4 Traders (5-8)',
                x=['Long Contracts', 'Short Contracts'],
                y=[long_next4_contracts, short_next4_contracts],
                text=[f'{long_next4_contracts:,.0f}<br>({(long_metrics["top_8_pct_of_oi"]-long_metrics["top_4_pct_of_oi"]):.1f}% of OI)', 
                      f'{short_next4_contracts:,.0f}<br>({(short_metrics["top_8_pct_of_oi"]-short_metrics["top_4_pct_of_oi"]):.1f}% of OI)'],
                textposition='inside',
                marker_color='orange',
                hovertemplate='%{x}<br>Next 4: %{y:,.0f} contracts<extra></extra>'
            ), row=1, col=2)
            
            fig.add_trace(go.Bar(
                name='Remaining Traders',
                x=['Long Contracts', 'Short Contracts'],
                y=[long_remaining, short_remaining],
                text=[f'{long_remaining:,.0f}', f'{short_remaining:,.0f}'],
                textposition='inside',
                marker_color='lightblue',
                hovertemplate='%{x}<br>Remaining: %{y:,.0f} contracts<extra></extra>'
            ), row=1, col=2)
            
            # Add annotations with calculations
            fig.add_annotation(
                text=f"Total OI: {avg_oi:,.0f}",
                xref="paper", yref="paper",
                x=0.75, y=0.95,
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="lightyellow",
                row=1, col=2
            )
            
            # Update layout for stacked bars
            fig.update_yaxes(title_text="Number of Contracts", row=1, col=2)
            fig.update_xaxes(title_text="", row=1, col=2)
            fig.update_layout(barmode='stack')
            
            # Plot 3: Top Trader Dominance
            # Shows how concentration changes from top 4 to top 8
            
            # Create data for both sides
            dominance_data = []
            
            for side in ['long', 'short']:
                metrics = top_traders_metrics[side]
                # Top 4
                dominance_data.append({
                    'side': side.title(),
                    'traders': 4,
                    'concentration': metrics['top_4_pct_of_oi'],
                    'label': f"Top 4 {side.title()}"
                })
                # Top 8
                dominance_data.append({
                    'side': side.title(),
                    'traders': 8,
                    'concentration': metrics['top_8_pct_of_oi'],
                    'label': f"Top 8 {side.title()}"
                })
            
            dominance_df = pd.DataFrame(dominance_data)
            
            for side in ['Long', 'Short']:
                side_data = dominance_df[dominance_df['side'] == side]
                fig.add_trace(go.Scatter(
                    x=side_data['traders'],
                    y=side_data['concentration'],
                    name=side,
                    mode='lines+markers',
                    marker=dict(size=12),
                    line=dict(width=3, color='green' if side == 'Long' else 'red'),
                    hovertemplate='%{text}<br>%{y:.1f}% of Total OI<extra></extra>',
                    text=side_data['label']
                ), row=2, col=1)
            
            # Add annotations for concentration increase
            for side, color in [('long', 'green'), ('short', 'red')]:
                metrics = top_traders_metrics[side]
                increase = metrics['top_8_pct_of_oi'] - metrics['top_4_pct_of_oi']
                fig.add_annotation(
                    x=6,
                    y=(metrics['top_4_pct_of_oi'] + metrics['top_8_pct_of_oi']) / 2,
                    text=f"+{increase:.1f}%",
                    showarrow=False,
                    font=dict(color=color, size=10),
                    row=2, col=1
                )
            
            # Plot 4: Category vs Top Trader Analysis
            # Shows which categories likely contain the top traders
            
            if category_metrics:
                # Calculate max value first for positioning
                max_val = max([m['avg_position'] for m in category_metrics] + 
                             [top_traders_metrics['long']['top_4_avg'], 
                              top_traders_metrics['short']['top_4_avg']])
                
                # Prepare scatter data
                for i, cat_metric in enumerate(category_metrics):
                    # Get corresponding top trader average
                    side = cat_metric['side']
                    top_4_avg = top_traders_metrics[side]['top_4_avg']
                    
                    # Scale down bubble sizes and add minimum size
                    bubble_size = max(15, min(50, cat_metric['total_traders'] / 4))
                    
                    fig.add_trace(go.Scatter(
                        x=[cat_metric['avg_position']],
                        y=[top_4_avg],
                        mode='markers',
                        marker=dict(
                            size=bubble_size,
                            color=cat_metric['color'],
                            line=dict(width=2, color='black'),
                            opacity=0.8
                        ),
                        name=cat_metric['category'],
                        showlegend=False,
                        hovertemplate='%{text}<br>Category Avg: %{x:,.0f}<br>Top 4 Avg: %{y:,.0f}<br>Ratio: %{customdata[0]:.1f}x<br>Traders: %{customdata[1]:.0f}<extra></extra>',
                        text=[cat_metric['category']],
                        customdata=[[cat_metric['ratio_to_top_4'], cat_metric['total_traders']]]
                    ), row=2, col=2)
                    
                    # Add text annotations with arrows pointing to bubbles
                    # Position text based on quadrant to avoid overlaps
                    x_pos = cat_metric['avg_position']
                    y_pos = top_4_avg
                    
                    # Determine text positioning based on location
                    if i == 0:  # Top-left
                        text_x = x_pos - max_val * 0.15
                        text_y = y_pos + max_val * 0.15
                        ax = 20
                        ay = -20
                    elif i == 1:  # Top-right
                        text_x = x_pos + max_val * 0.15
                        text_y = y_pos + max_val * 0.15
                        ax = -20
                        ay = -20
                    elif i == 2:  # Bottom-left
                        text_x = x_pos - max_val * 0.15
                        text_y = y_pos - max_val * 0.15
                        ax = 20
                        ay = 20
                    else:  # Bottom-right
                        text_x = x_pos + max_val * 0.15
                        text_y = y_pos - max_val * 0.15
                        ax = -20
                        ay = 20
                    
                    # Add category label with arrow
                    fig.add_annotation(
                        x=x_pos,
                        y=y_pos,
                        text=f"<b>{cat_metric['category']}</b><br>{cat_metric['total_traders']:.0f} traders",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=cat_metric['color'],
                        ax=ax,
                        ay=ay,
                        bgcolor="white",
                        bordercolor=cat_metric['color'],
                        borderwidth=2,
                        opacity=0.9,
                        font=dict(size=10),
                        row=2, col=2
                    )
                
                # Add diagonal line (y=x)
                fig.add_trace(go.Scatter(
                    x=[0, max_val * 1.1],
                    y=[0, max_val * 1.1],
                    mode='lines',
                    line=dict(dash='dash', color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=2, col=2)
                
                # Add guide text
                fig.add_annotation(
                    x=max_val * 0.55,
                    y=max_val * 0.45,
                    text="← Categories below line have<br>larger positions than top 4 avg",
                    showarrow=False,
                    font=dict(size=9, color='gray'),
                    opacity=0.7,
                    row=2, col=2
                )
            
            # Update layout
            week_label = "Current Week" if weeks_back == 0 else f"{weeks_back} Week{'s' if weeks_back > 1 else ''} Ago"
            fig.update_layout(
                title=f"Market Microstructure Analysis - {week_label} ({selected_date.strftime('%Y-%m-%d')})",
                height=900,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Trader Group", row=1, col=1, tickangle=-45)
            fig.update_yaxes(title_text="Average Position (Contracts)", row=1, col=1)
            fig.update_xaxes(
                title_text="Number of Top Traders", 
                row=2, col=1,
                tickmode='array',
                tickvals=[4, 8],
                ticktext=['4', '8']
            )
            fig.update_yaxes(title_text="% of Total Open Interest", row=2, col=1)
            fig.update_xaxes(title_text="Category Average Position", row=2, col=2)
            fig.update_yaxes(title_text="Top 4 Average Position", row=2, col=2)
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics with corrected calculations
            st.markdown("### Market Structure Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # Calculate total traders from categories
            total_traders = sum(m['total_traders'] for m in category_metrics)
            
            # Calculate what % of each side the average top 4 trader controls
            # Get total positions for each side
            total_long = float(analysis_data['tot_rept_positions_long_all'].iloc[0]) if 'tot_rept_positions_long_all' in analysis_data.columns else 0
            total_short = float(analysis_data['tot_rept_positions_short'].iloc[0]) if 'tot_rept_positions_short' in analysis_data.columns else 0
            
            # Calculate percentage of side controlled by each top 4 trader
            long_top4_avg = top_traders_metrics['long']['top_4_avg']
            short_top4_avg = top_traders_metrics['short']['top_4_avg']
            
            long_pct_of_side = (long_top4_avg / total_long * 100) if total_long > 0 else 0
            short_pct_of_side = (short_top4_avg / total_short * 100) if total_short > 0 else 0
            
            # Average of both sides
            avg_top4_control_of_side = (long_pct_of_side + short_pct_of_side) / 2
            
            # Determine likely composition
            if category_metrics:
                min_ratio = min(m['ratio_to_top_4'] for m in category_metrics)
                likely_category = next(m['category'] for m in category_metrics if m['ratio_to_top_4'] == min_ratio)
            else:
                likely_category = "Unknown"
            
            # Calculate structure for each week (moved up to use in current status)
            structure_data = []
            weeks_data = df_micro.tail(52).copy()
            
            # Get latest date for calculations
            latest_date = df_micro['report_date_as_yyyy_mm_dd'].max()
            
            # Calculate empirical percentiles for market structure determination
            # Get historical data for the past 52 weeks
            historical_window = df_micro[df_micro['report_date_as_yyyy_mm_dd'] >= (latest_date - pd.Timedelta(weeks=52))]
            
            if len(historical_window) > 10:  # Need enough data for percentiles
                # Calculate historical control percentages
                historical_controls = []
                for _, week_data in historical_window.iterrows():
                    week_oi = week_data['open_interest_all']
                    week_total_long = week_data['tot_rept_positions_long_all'] if 'tot_rept_positions_long_all' in week_data else 0
                    week_total_short = week_data['tot_rept_positions_short'] if 'tot_rept_positions_short' in week_data else 0
                    
                    # Long side control
                    if week_total_long > 0 and week_oi > 0:
                        week_long_top4_contracts = week_oi * (week_data['conc_gross_le_4_tdr_long'] / 100)
                        week_long_top4_avg = week_long_top4_contracts / 4
                        week_long_pct = (week_long_top4_avg / week_total_long * 100)
                        historical_controls.append(week_long_pct)
                    
                    # Short side control
                    if week_total_short > 0 and week_oi > 0:
                        week_short_top4_contracts = week_oi * (week_data['conc_gross_le_4_tdr_short'] / 100)
                        week_short_top4_avg = week_short_top4_contracts / 4
                        week_short_pct = (week_short_top4_avg / week_total_short * 100)
                        historical_controls.append(week_short_pct)
                
                # Calculate percentiles
                p75 = np.percentile(historical_controls, 75)  # Top 25% most concentrated
                p50 = np.percentile(historical_controls, 50)  # Median
                
                # Now calculate structure for each week for the timeline
                for _, week in weeks_data.iterrows():
                    week_oi = week['open_interest_all']
                    week_total_long = week['tot_rept_positions_long_all'] if 'tot_rept_positions_long_all' in week else 0
                    week_total_short = week['tot_rept_positions_short'] if 'tot_rept_positions_short' in week else 0
                    
                    # Calculate control percentages for this week
                    if week_total_long > 0 and week_oi > 0:
                        week_long_top4_contracts = week_oi * (week['conc_gross_le_4_tdr_long'] / 100)
                        week_long_top4_avg = week_long_top4_contracts / 4
                        week_long_pct = (week_long_top4_avg / week_total_long * 100)
                    else:
                        week_long_pct = 0
                        
                    if week_total_short > 0 and week_oi > 0:
                        week_short_top4_contracts = week_oi * (week['conc_gross_le_4_tdr_short'] / 100)
                        week_short_top4_avg = week_short_top4_contracts / 4
                        week_short_pct = (week_short_top4_avg / week_total_short * 100)
                    else:
                        week_short_pct = 0
                    
                    # Use max for structure determination
                    week_max_control = max(week_long_pct, week_short_pct)
                    
                    # Assign structure based on percentiles
                    if week_max_control > p75:
                        structure_value = 2  # Highly Concentrated
                        structure_label = "High"
                    elif week_max_control > p50:
                        structure_value = 1  # Moderately Concentrated
                        structure_label = "Moderate"
                    else:
                        structure_value = 0  # Well Distributed
                        structure_label = "Low"
                    
                    structure_data.append({
                        'date': week['report_date_as_yyyy_mm_dd'],
                        'structure_value': structure_value,
                        'structure_label': structure_label,
                        'control_pct': week_max_control
                    })
                
                # Determine market structure
                # For current week, always use the timeline's last entry for consistency
                if weeks_back == 0 and structure_data:
                    # Use the most recent week from timeline
                    latest_week = structure_data[-1]
                    
                    if latest_week['structure_value'] == 2:
                        structure = f"🔴 Highly Concentrated (>{p75:.1f}%)"
                    elif latest_week['structure_value'] == 1:
                        structure = f"🟠 Moderately Concentrated ({p50:.1f}-{p75:.1f}%)"
                    else:
                        structure = f"🟢 Well Distributed (<{p50:.1f}%)"
                else:
                    # For historical weeks or if no structure data, calculate based on values
                    max_control = max(long_pct_of_side, short_pct_of_side)
                    if max_control > p75:
                        structure = f"🔴 Highly Concentrated (>{p75:.1f}%)"
                    elif max_control > p50:
                        structure = f"🟠 Moderately Concentrated ({p50:.1f}-{p75:.1f}%)"
                    else:
                        structure = f"🟢 Well Distributed (<{p50:.1f}%)"
            else:
                # Fallback to fixed thresholds if not enough historical data
                max_control = max(long_pct_of_side, short_pct_of_side)
                if max_control > 6:
                    structure = "🔴 Highly Concentrated"
                elif max_control > 4:
                    structure = "🟠 Moderately Concentrated"
                else:
                    structure = "🟢 Well Distributed"
            
            with col1:
                st.metric("Total Traders", f"{int(total_traders):,}")
            with col2:
                st.metric("Top-4 Long Control", f"{long_pct_of_side:.1f}% of Longs")
            with col3:
                st.metric("Top-4 Short Control", f"{short_pct_of_side:.1f}% of Shorts")
            with col4:
                st.metric("Top 4 Likely Type", likely_category)
            with col5:
                st.metric("Market Structure", structure)
            
            # Detailed insights
            st.markdown("### Key Insights")
            insights_cols = st.columns(2)
            
            with insights_cols[0]:
                st.markdown("**Top Trader Analysis:**")
                long_metrics = top_traders_metrics['long']
                short_metrics = top_traders_metrics['short']
                
                st.write(f"• Top 4 Long traders: {long_metrics['top_4_avg']:,.0f} contracts each")
                st.write(f"• Top 4 Short traders: {short_metrics['top_4_avg']:,.0f} contracts each")
                st.write(f"• Each top Long controls: {long_pct_of_side:.1f}% of all longs")
                st.write(f"• Each top Short controls: {short_pct_of_side:.1f}% of all shorts")
            
            with insights_cols[1]:
                st.markdown("**Category Comparison:**")
                if category_metrics:
                    # Sort by ratio to find most likely categories
                    sorted_categories = sorted(category_metrics, key=lambda x: x['ratio_to_top_4'])
                    
                    st.write(f"• Most likely top 4: {sorted_categories[0]['category']} ({sorted_categories[0]['ratio_to_top_4']:.1f}x)")
                    st.write(f"• Least likely top 4: {sorted_categories[-1]['category']} ({sorted_categories[-1]['ratio_to_top_4']:.1f}x)")
                    
                    # Average positions
                    comm_avg = np.mean([m['avg_position'] for m in category_metrics if 'Commercial' in m['category']])
                    noncomm_avg = np.mean([m['avg_position'] for m in category_metrics if 'Non-Commercial' in m['category']])
                    
                    st.write(f"• Commercial avg: {comm_avg:,.0f} contracts")
                    st.write(f"• Non-Commercial avg: {noncomm_avg:,.0f} contracts")
            
            # Market Structure History Heatmap
            st.markdown("### Market Structure History (52 Weeks)")
            
            # Create heatmap
            if structure_data:
                # Prepare data for heatmap
                dates = [d['date'] for d in structure_data]
                values = [d['structure_value'] for d in structure_data]
                labels = [d['structure_label'] for d in structure_data]
                control_pcts = [d['control_pct'] for d in structure_data]
                
                # Create custom hover text
                hover_text = []
                for i in range(len(dates)):
                    hover_text.append(
                        f"Date: {dates[i].strftime('%Y-%m-%d')}<br>" +
                        f"Status: {labels[i]}<br>" +
                        f"Max Control: {control_pcts[i]:.1f}%"
                    )
                
                # Create the timeline
                fig_timeline = go.Figure()
                
                # Create a single row heatmap
                fig_timeline.add_trace(go.Heatmap(
                    z=[values],  # Single row
                    text=[hover_text],
                    hovertemplate='%{text}<extra></extra>',
                    colorscale=[
                        [0, '#4CAF50'],      # 0 = Green - Well Distributed
                        [0.33, '#4CAF50'],   # Still green
                        [0.34, '#FF9800'],   # 1 = Orange - Moderately Concentrated
                        [0.66, '#FF9800'],   # Still orange
                        [0.67, '#F44336'],   # 2 = Red - Highly Concentrated
                        [1, '#F44336']       # Still red
                    ],
                    showscale=False,
                    xgap=2,
                    ygap=0,
                    zmin=0,
                    zmax=2
                ))
                
                # Add x-axis labels for months
                # Get unique months from dates
                month_positions = []
                month_labels = []
                current_month = None
                
                for i, date in enumerate(dates):
                    month_year = date.strftime('%b %Y')
                    if month_year != current_month:
                        month_positions.append(i)
                        month_labels.append(month_year)
                        current_month = month_year
                
                # Update layout
                fig_timeline.update_layout(
                    title="52-Week Market Concentration Timeline",
                    height=120,
                    xaxis=dict(
                        tickmode='array',
                        tickvals=month_positions,
                        ticktext=month_labels,
                        tickangle=0,
                        showgrid=False,
                        side='bottom'
                    ),
                    yaxis=dict(
                        showticklabels=False, 
                        showgrid=False,
                        fixedrange=True
                    ),
                    plot_bgcolor='white',
                    margin=dict(t=60, b=40, l=20, r=20)
                )
                
                # Add legend
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("🟢 **Well Distributed** (<" + f"{p50:.1f}%)")
                with col2:
                    st.markdown("🟠 **Moderately Concentrated** (" + f"{p50:.1f}-{p75:.1f}%)")
                with col3:
                    st.markdown("🔴 **Highly Concentrated** (>" + f"{p75:.1f}%)")
                
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Show evolution if requested
            if show_evolution:
                st.markdown("### Microstructure Evolution")
                
                # Calculate metrics over time
                evolution_data = []
                
                for date in df_micro['report_date_as_yyyy_mm_dd'].tail(52).unique():
                    date_data = df_micro[df_micro['report_date_as_yyyy_mm_dd'] == date]
                    
                    # Calculate key metrics
                    total_traders = 0
                    total_positions = 0
                    
                    # Sum up traders and positions across categories
                    for cat_name, pos_col, trader_col, color in categories:
                        if trader_col in date_data.columns and pos_col in date_data.columns:
                            traders = float(date_data.iloc[0][trader_col]) if pd.notna(date_data.iloc[0][trader_col]) else 0
                            positions = float(date_data.iloc[0][pos_col]) if pd.notna(date_data.iloc[0][pos_col]) else 0
                            total_traders += traders
                            total_positions += positions
                    
                    # Get market concentration - use maximum as indicator of concentration risk
                    long_conc = float(date_data.iloc[0]['conc_gross_le_4_tdr_long']) if 'conc_gross_le_4_tdr_long' in date_data.columns else 0
                    short_conc = float(date_data.iloc[0]['conc_gross_le_4_tdr_short']) if 'conc_gross_le_4_tdr_short' in date_data.columns else 0
                    max_concentration = max(long_conc, short_conc)
                    
                    # Calculate average position size
                    avg_position_size = total_positions / total_traders if total_traders > 0 else 0
                    
                    evolution_data.append({
                        'date': date,
                        'total_traders': total_traders,
                        'max_concentration': max_concentration,
                        'long_concentration': long_conc,
                        'short_concentration': short_conc,
                        'avg_position_size': avg_position_size
                    })
                
                evolution_df = pd.DataFrame(evolution_data)
                
                # Create evolution chart
                fig_evo = make_subplots(
                    rows=1, cols=1,
                    specs=[[{"secondary_y": True}]]
                )
                
                fig_evo.add_trace(go.Scatter(
                    x=evolution_df['date'],
                    y=evolution_df['total_traders'],
                    name='Total Traders',
                    line=dict(color='blue', width=2)
                ), secondary_y=False)
                
                fig_evo.add_trace(go.Scatter(
                    x=evolution_df['date'],
                    y=evolution_df['long_concentration'],
                    name='Long Concentration',
                    line=dict(color='green', width=2)
                ), secondary_y=True)
                
                fig_evo.add_trace(go.Scatter(
                    x=evolution_df['date'],
                    y=evolution_df['short_concentration'],
                    name='Short Concentration',
                    line=dict(color='red', width=2)
                ), secondary_y=True)
                
                fig_evo.update_layout(
                    title="Market Microstructure Evolution (52 weeks)",
                    height=400
                )
                fig_evo.update_xaxes(title_text="Date")
                fig_evo.update_yaxes(title_text="Total Traders", secondary_y=False)
                fig_evo.update_yaxes(title_text="Top 4 Concentration (% of OI)", secondary_y=True)
                
                st.plotly_chart(fig_evo, use_container_width=True)
        
        else:
            st.warning("Insufficient data for microstructure analysis")


def handle_multi_instrument_flow(chart_type, instruments_db, api_token):
    """Handle multi-instrument selection and analysis"""
    st.header("🎯 Select Multiple Instruments")
    
    # Search method selection
    search_method = st.radio(
        "Choose search method:",
        ["Search by Commodity Subgroup", "Search by Commodity Type", "Free Text Search"],
        horizontal=True
    )
    
    # Add clear selection button
    if 'multi_selected_instruments' in st.session_state and st.session_state.multi_selected_instruments:
        if st.button("🗑️ Clear All Selections", type="secondary"):
            st.session_state.multi_selected_instruments = []
            st.rerun()
    
    selected_instruments = []
    
    if search_method == "Search by Commodity Subgroup":
        st.subheader("📁 Search by Commodity Subgroup")
        
        # Initialize session state for selected instruments if not exists
        if 'multi_selected_instruments' not in st.session_state:
            st.session_state.multi_selected_instruments = []
        
        # Allow selection of multiple subgroups
        subgroups = sorted(list(instruments_db['commodity_subgroups'].keys()))
        selected_subgroups = st.multiselect("📁 Select Commodity Subgroups:", subgroups)
        
        if selected_subgroups:
            # Gather all instruments from selected subgroups
            available_instruments = []
            for subgroup in selected_subgroups:
                available_instruments.extend(instruments_db['commodity_subgroups'][subgroup])
            
            # Combine with previously selected instruments
            combined_options = sorted(set(st.session_state.multi_selected_instruments + available_instruments))
            
            selected_instruments = st.multiselect(
                f"📊 Select from {len(available_instruments)} instruments:",
                combined_options,
                default=st.session_state.multi_selected_instruments,
                max_selections=15,
                key="subgroup_multiselect"
            )
            
            # Update session state
            st.session_state.multi_selected_instruments = selected_instruments
        else:
            # Show previously selected instruments
            if st.session_state.multi_selected_instruments:
                selected_instruments = st.multiselect(
                    "📊 Previously selected instruments:",
                    st.session_state.multi_selected_instruments,
                    default=st.session_state.multi_selected_instruments,
                    max_selections=15,
                    key="subgroup_multiselect_prev"
                )
    
    elif search_method == "Search by Commodity Type":
        st.subheader("🔸 Search by Commodity Type")
        
        # Initialize session state for selected instruments if not exists
        if 'multi_selected_instruments' not in st.session_state:
            st.session_state.multi_selected_instruments = []
        
        # Allow selection of multiple commodity types
        commodities = sorted(list(instruments_db['commodities'].keys()))
        selected_commodities = st.multiselect("🔸 Select Commodities:", commodities)
        
        if selected_commodities:
            # Gather all instruments from selected commodities
            available_instruments = []
            for commodity in selected_commodities:
                available_instruments.extend(instruments_db['commodities'][commodity])
            
            # Combine with previously selected instruments
            combined_options = sorted(set(st.session_state.multi_selected_instruments + available_instruments))
            
            selected_instruments = st.multiselect(
                f"📊 Select from {len(available_instruments)} instruments:",
                combined_options,
                default=st.session_state.multi_selected_instruments,
                max_selections=15,
                key="commodity_multiselect"
            )
            
            # Update session state
            st.session_state.multi_selected_instruments = selected_instruments
        else:
            # Show previously selected instruments
            if st.session_state.multi_selected_instruments:
                selected_instruments = st.multiselect(
                    "📊 Previously selected instruments:",
                    st.session_state.multi_selected_instruments,
                    default=st.session_state.multi_selected_instruments,
                    max_selections=15,
                    key="commodity_multiselect_prev"
                )
    
    elif search_method == "Free Text Search":
        st.subheader("🔍 Free Text Search")
        
        # Initialize session state for selected instruments if not exists
        if 'multi_selected_instruments' not in st.session_state:
            st.session_state.multi_selected_instruments = []
        
        search_text = st.text_input(
            "Type keywords (comma-separated for multiple):",
            placeholder="e.g., GOLD, CRUDE OIL, WHEAT, EURO..."
        )
        
        if search_text:
            # Split by comma and search for each term
            search_terms = [term.strip() for term in search_text.split(',')]
            all_instruments = instruments_db['all_instruments']
            
            # Find instruments matching any search term
            filtered_instruments = []
            for inst in all_instruments:
                for term in search_terms:
                    if term.upper() in inst.upper():
                        filtered_instruments.append(inst)
                        break
            
            # Combine previously selected instruments with new search results
            combined_options = sorted(set(st.session_state.multi_selected_instruments + filtered_instruments))
            
            if combined_options:
                selected_instruments = st.multiselect(
                    f"📊 Select instruments (showing {len(filtered_instruments)} new matches):",
                    combined_options,
                    default=st.session_state.multi_selected_instruments,
                    max_selections=15,
                    key="free_text_multiselect"
                )
                
                # Update session state
                st.session_state.multi_selected_instruments = selected_instruments
                
                if filtered_instruments:
                    st.success(f"✅ Found {len(set(filtered_instruments))} matching instruments")
            else:
                st.warning(f"⚠️ No instruments found matching your search terms")
        else:
            # Show previously selected instruments even when search is empty
            if st.session_state.multi_selected_instruments:
                selected_instruments = st.multiselect(
                    "📊 Previously selected instruments:",
                    st.session_state.multi_selected_instruments,
                    default=st.session_state.multi_selected_instruments,
                    max_selections=15,
                    key="free_text_multiselect_prev"
                )
    
    
    if selected_instruments and len(selected_instruments) >= 2:
        if st.button("🚀 Fetch Data for All Instruments", type="primary"):
            st.markdown("---")
            
            if chart_type == "Cross-Asset":
                # Original Cross-asset comparison
                st.subheader("🔄 Cross-Asset Comparison")
                
                # Trader category selection
                trader_category = st.selectbox(
                    "Select trader category:",
                    ["Non-Commercial", "Commercial", "Non-Reportable"],
                    index=0
                )
                
                # Lookback period
                lookback_period = st.selectbox(
                    "Lookback period for Z-score calculation:",
                    ["1 Year", "2 Years", "3 Years", "5 Years"],
                    index=1
                )
                
                lookback_map = {
                    "1 Year": pd.Timestamp.now() - pd.DateOffset(years=1),
                    "2 Years": pd.Timestamp.now() - pd.DateOffset(years=2),
                    "3 Years": pd.Timestamp.now() - pd.DateOffset(years=3),
                    "5 Years": pd.Timestamp.now() - pd.DateOffset(years=5)
                }
                
                # Create cross-asset analysis
                fig = create_cross_asset_analysis(
                    selected_instruments, 
                    trader_category,
                    api_token,
                    lookback_start=lookback_map[lookback_period],
                    show_week_ago=True,
                    instruments_db=instruments_db
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif chart_type == "Z-Score Analysis":
                # Cross-asset Z-score comparison
                st.subheader("📊 Cross-Asset Z-Score Analysis")
                st.info("Compare current positioning z-scores across multiple instruments to identify stretched markets")
                
                # Trader category selection
                trader_category = st.selectbox(
                    "Select trader category:",
                    ["Non-Commercial", "Commercial", "Non-Reportable"],
                    index=0
                )
                
                # Lookback period
                lookback_period = st.selectbox(
                    "Lookback period for Z-score calculation:",
                    ["1 Year", "2 Years", "3 Years", "5 Years"],
                    index=1
                )
                
                # Show week-ago option
                show_week_ago = st.checkbox("Show week-ago positions", value=True)
                
                lookback_map = {
                    "1 Year": pd.Timestamp.now() - pd.DateOffset(years=1),
                    "2 Years": pd.Timestamp.now() - pd.DateOffset(years=2),
                    "3 Years": pd.Timestamp.now() - pd.DateOffset(years=3),
                    "5 Years": pd.Timestamp.now() - pd.DateOffset(years=5)
                }
                
                # Create cross-asset analysis
                fig = create_cross_asset_analysis(
                    selected_instruments, 
                    trader_category,
                    api_token,
                    lookback_start=lookback_map[lookback_period],
                    show_week_ago=show_week_ago,
                    instruments_db=instruments_db
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif chart_type == "WoW Changes":
                # Week-over-week changes
                st.subheader("📈 Week-over-Week Changes Analysis")
                st.info("Shows weekly positioning changes across instruments as % of open interest")
                
                # Trader category selection
                trader_category = st.selectbox(
                    "Select trader category:",
                    ["Non-Commercial", "Commercial", "Non-Reportable"],
                    index=0
                )
                
                # Create WoW changes chart
                fig = create_cross_asset_wow_changes(
                    selected_instruments,
                    trader_category,
                    api_token,
                    instruments_db
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif chart_type == "Positioning Conc.":
                # Positioning concentration
                st.subheader("📊 Cross-Asset Positioning Concentration")
                st.info("Compares positioning as % of open interest across instruments")
                
                # Trader category selection
                trader_category = st.selectbox(
                    "Select trader category:",
                    ["Non-Commercial", "Commercial", "Non-Reportable"],
                    index=0
                )
                
                # Create positioning concentration charts
                time_series_fig, bar_chart_fig = create_positioning_concentration_charts(
                    selected_instruments,
                    trader_category,
                    api_token,
                    instruments_db
                )
                
                if time_series_fig and bar_chart_fig:
                    st.plotly_chart(time_series_fig, use_container_width=True)
                    st.plotly_chart(bar_chart_fig, use_container_width=True)
                    
            elif chart_type == "Participation":
                # Cross-asset participation comparison
                st.subheader("👥 Cross-Asset Participation Analysis")
                st.info("Compares trader participation trends across instruments")
                
                # Create participation comparison chart
                fig = create_cross_asset_participation_comparison(
                    selected_instruments,
                    api_token,
                    instruments_db
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif chart_type == "Strength Matrix":
                # Relative strength matrix
                st.subheader("🔥 Relative Strength Matrix")
                st.info("Heatmap showing positioning strength across instruments and time periods")
                
                # Time period selection
                time_period = st.selectbox(
                    "Select time period:",
                    ["1 Month", "3 Months", "6 Months", "1 Year"],
                    index=1
                )
                
                # Create relative strength matrix
                fig = create_relative_strength_matrix(
                    selected_instruments,
                    api_token,
                    time_period,
                    instruments_db
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif chart_type == "Market Matrix":
                # Market structure matrix
                st.subheader("🎯 Market Structure Matrix")
                st.info("Visualizes market structure by plotting instruments on a 2x2 grid based on trader count (x-axis) and concentration ratio (y-axis).")
                
                # Fetch data for all instruments
                all_instruments_data = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, instrument in enumerate(selected_instruments):
                    status_text.text(f"Fetching data for {instrument}...")
                    progress_bar.progress((idx + 1) / len(selected_instruments))
                    
                    df = fetch_cftc_data(instrument, api_token)
                    if df is not None and not df.empty:
                        all_instruments_data[instrument] = df
                
                progress_bar.empty()
                status_text.empty()
                
                if all_instruments_data:
                    fig = create_market_structure_matrix(all_instruments_data, selected_instruments)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to fetch data for the selected instruments")
                    
            elif chart_type == "Asset Concentration":
                # Asset concentration analysis
                st.subheader("🎯 Asset Positioning Concentration")
                st.info("Analyzes concentration of positions across multiple assets to identify crowded trades.")
                
                # Concentration metric selection
                concentration_metric = st.selectbox(
                    "Select concentration metric:",
                    ["Top 4 Traders %", "Top 8 Traders %", "Herfindahl Index"],
                    index=0
                )
                
                # TODO: Implement asset concentration analysis
                st.info("Asset Concentration analysis functionality will be implemented here")
                    
    elif selected_instruments and len(selected_instruments) < 2:
        st.warning("⚠️ Please select at least 2 instruments for comparison")
    else:
        st.info("👆 Please select instruments from the list above to begin analysis")


if __name__ == "__main__":
    main()