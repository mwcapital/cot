"""
Dashboard overview module for displaying key commodity metrics
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_fetcher import fetch_cftc_data_ytd_only, fetch_cftc_data_2year, fetch_cftc_data
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define key instruments for dashboard with color groups
KEY_INSTRUMENTS = {
    "Agriculture": {
        "CT": "COTTON NO. 2 - ICE FUTURES U.S. (033661)",
        "CC": "COCOA - ICE FUTURES U.S. (073732)",
        "JO": "FRZN CONCENTRATED ORANGE JUICE - ICE FUTURES U.S. (040701)",
        "KC": "COFFEE C - ICE FUTURES U.S. (083731)",
        "LB": "LUMBER - CHICAGO MERCANTILE EXCHANGE (058643)",
        "MW": "WHEAT-HRSpring - MIAX FUTURES EXCHANGE (001626)",
        "SB": "SUGAR NO. 11 - ICE FUTURES U.S. (080732)",
        "W": "WHEAT-SRW - CHICAGO BOARD OF TRADE (001602)",
        "ZC": "CORN - CHICAGO BOARD OF TRADE (002602)",
        "ZF": "FEEDER CATTLE - CHICAGO MERCANTILE EXCHANGE (061641)",
        "ZL": "SOYBEAN OIL - CHICAGO BOARD OF TRADE (007601)",
        "ZM": "SOYBEAN MEAL - CHICAGO BOARD OF TRADE (026603)",
        "ZS": "SOYBEANS - CHICAGO BOARD OF TRADE (005602)",
        "ZZ": "LEAN HOGS - CHICAGO MERCANTILE EXCHANGE (054642)",
    },
    "Currency": {
        "BN": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE (096742)",
        "DX": "USD INDEX - ICE FUTURES U.S. (098662)",
        "FN": "EURO FX - CHICAGO MERCANTILE EXCHANGE (099741)",
        "JN": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE (097741)",
    },
    "Equity": {
        "ES": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE (13874A)",
        "EN": "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE (209742)",
        "ER": "RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE (239742)",
    },
    "Energy": {
        "ZU": "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE (067651)",
        "ZB": "GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE (111659)",
        "ZH": "GULF JET NY HEAT OIL SPR - NEW YORK MERCANTILE EXCHANGE (86465A)",
        "NG": "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE (023651)",
    },
    "Metals": {
        "ZG": "GOLD - COMMODITY EXCHANGE INC. (088691)",
        "ZI": "SILVER - COMMODITY EXCHANGE INC. (084691)",
        "ZK": "COPPER- #1 - COMMODITY EXCHANGE INC. (085692)",
        "ZA": "PALLADIUM - NEW YORK MERCANTILE EXCHANGE (075651)",
        "ZP": "PLATINUM - NEW YORK MERCANTILE EXCHANGE (076651)",
    },
}

# Color mapping for each category
CATEGORY_COLORS = {
    "Energy": "#FF6B6B",  # Red
    "Metals": "#4ECDC4",  # Teal
    "Agriculture": "#95E77E",  # Green
    "Currency": "#FFD93D",  # Yellow
    "Equity": "#6A7FDB",  # Blue
}


def calculate_ytd_percentile(series, current_value):
    """Calculate YTD percentile for a given value"""
    if series.empty:
        return np.nan
    return (series <= current_value).mean() * 100


def create_sparkline(df, column_name):
    """Create a simple sparkline chart for YTD data"""
    if df.empty or column_name not in df.columns:
        return None
    
    # Data is already YTD only
    ytd_data = df.copy()
    
    if ytd_data.empty:
        return None
    
    # Create sparkline
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ytd_data['report_date_as_yyyy_mm_dd'],
        y=ytd_data[column_name],
        mode='lines',
        line=dict(color='#1f77b4', width=1),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)',
        showlegend=False,
        hovertemplate='%{y:,.0f}<br>%{x|%b %d}<extra></extra>'
    ))
    
    # Minimal layout for sparkline
    fig.update_layout(
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
        ),
        hovermode='x',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def fetch_single_instrument_data(category, instrument, api_token):
    """Helper function to fetch data for a single instrument"""
    try:
        # Fetch YTD data only (optimized for dashboard)
        df = fetch_cftc_data_ytd_only(instrument, api_token)
        
        if df is not None and not df.empty:
            # Get latest data point
            latest = df.iloc[-1]
            
            # Data is already YTD only from optimized fetch
            ytd_df = df
            
            # Create sparkline data as a list of values for LineChartColumn
            ytd_sparkline = None
            if not ytd_df.empty and 'net_noncomm_positions' in ytd_df.columns:
                # Sort by date and get values as list
                ytd_sorted = ytd_df.sort_values('report_date_as_yyyy_mm_dd')
                ytd_sparkline = ytd_sorted['net_noncomm_positions'].fillna(0).tolist()
            
            # Calculate metrics
            # Extract ticker symbol from comments or use simplified name
            ticker_map = {
                "COTTON NO. 2": "CT",
                "BRITISH POUND": "BN",
                "COCOA": "CC",
                "USD INDEX": "DX",
                "E-MINI S&P 500": "ES",
                "NASDAQ MINI": "EN",
                "RUSSELL E-MINI": "ER",
                "EURO FX": "FN",
                "JAPANESE YEN": "JN",
                "FRZN CONCENTRATED ORANGE JUICE": "JO",
                "COFFEE C": "KC",
                "LUMBER": "LB",
                "WHEAT-HRSpring": "MW",
                "SUGAR NO. 11": "SB",
                "WHEAT-SRW": "W",
                "CORN": "ZC",
                "FEEDER CATTLE": "ZF",
                "GOLD": "ZG",
                "SILVER": "ZI",
                "COPPER- #1": "ZK",
                "SOYBEAN OIL": "ZL",
                "SOYBEAN MEAL": "ZM",
                "NAT GAS NYME": "NG",
                "SOYBEANS": "ZS",
                "WTI-PHYSICAL": "ZU",
                "GASOLINE RBOB": "ZB",
                "GULF JET NY HEAT OIL SPR": "ZH",
                "LEAN HOGS": "ZZ",
                "PALLADIUM": "ZA",
                "PLATINUM": "ZP",
            }
            
            instrument_name = instrument.split(' - ')[0]
            ticker = ticker_map.get(instrument_name, instrument_name[:2].upper())
            
            row_data = {
                'Category': category,
                'Ticker': ticker,
                'Instrument': instrument_name,  # Simplified name
                
                # Non-Commercial Net Positions
                'NC Net Position': latest.get('net_noncomm_positions', np.nan),
                'NC Net YTD %ile': calculate_ytd_percentile(
                    ytd_df.get('net_noncomm_positions', pd.Series()),
                    latest.get('net_noncomm_positions', np.nan)
                ),
                'NC Correlation': '',  # Placeholder for future price correlation
                
                # Commercial Longs
                'Comm Long': latest.get('comm_positions_long_all', np.nan),
                'Comm Long YTD %ile': calculate_ytd_percentile(
                    ytd_df.get('comm_positions_long_all', pd.Series()),
                    latest.get('comm_positions_long_all', np.nan)
                ),
                'Comm Long Corr': '',  # Placeholder
                
                # Commercial Shorts
                'Comm Short': latest.get('comm_positions_short_all', np.nan),
                'Comm Short YTD %ile': calculate_ytd_percentile(
                    ytd_df.get('comm_positions_short_all', pd.Series()),
                    latest.get('comm_positions_short_all', np.nan)
                ),
                'Comm Short Corr': '',  # Placeholder
                
                # Concentration (4 traders)
                'Conc Long 4T': latest.get('conc_net_le_4_tdr_long_all', np.nan),
                'Conc Short 4T': latest.get('conc_net_le_4_tdr_short_all', np.nan),
                
                # YTD NC Net Position sparkline
                'YTD NC Net Trend': ytd_sparkline if ytd_sparkline else [],
                
                # Last update
                'Last Update': latest['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'),
            }
            
            return row_data
            
    except Exception as e:
        return None


@st.cache_data(ttl=3600)
def fetch_dashboard_data(api_token):
    """Fetch YTD-only data for all dashboard instruments (optimized with parallel fetching)"""
    dashboard_data = []
    
    # Flatten instrument list
    all_instruments = []
    for category, instruments in KEY_INSTRUMENTS.items():
        for ticker, cot_name in instruments.items():
            all_instruments.append((category, cot_name))
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Fetching dashboard data in parallel...")
    
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all fetch tasks
        future_to_instrument = {
            executor.submit(fetch_single_instrument_data, cat, inst, api_token): (cat, inst) 
            for cat, inst in all_instruments
        }
        
        completed = 0
        # Process completed futures as they finish
        for future in as_completed(future_to_instrument):
            completed += 1
            progress_bar.progress(completed / len(all_instruments))
            
            category, instrument = future_to_instrument[future]
            try:
                row_data = future.result()
                if row_data:
                    dashboard_data.append(row_data)
            except Exception as e:
                st.warning(f"Could not fetch data for {instrument}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(dashboard_data)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_zscore_data_parallel(api_token, trader_category):
    """Fetch and calculate Z-scores for all instruments in parallel"""

    # Get all instruments from KEY_INSTRUMENTS and create ticker map
    all_instruments = []
    ticker_map = {}
    for category, instruments in KEY_INSTRUMENTS.items():
        for ticker, cot_name in instruments.items():
            all_instruments.append(cot_name)
            ticker_map[cot_name] = ticker

    # Store data for all instruments
    instrument_data = {}

    # Set 2-year lookback period
    lookback_start = datetime.now() - timedelta(days=730)  # 2 years

    def process_single_instrument(instrument):
        """Process a single instrument and return its z-score data"""
        try:
            # Use optimized 2-year fetch function
            df = fetch_cftc_data_2year(instrument, api_token)

            if df is not None and not df.empty:
                # Handle both old and new naming conventions
                trader_cat = trader_category.replace(" Net", "")  # Remove " Net" suffix if present

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

                cols = category_columns[trader_cat]

                # Calculate net positions
                if trader_cat == "Non-Commercial":
                    df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
                elif trader_cat == "Commercial":
                    df['net_comm_positions'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
                elif trader_cat == "Non-Reportable":
                    df['net_nonrept_positions'] = df['nonrept_positions_long_all'] - df['nonrept_positions_short_all']
                    cols['net'] = 'net_nonrept_positions'

                if cols.get('net') in df.columns:
                    # Calculate Z-score of raw net positions
                    net_mean = df[cols['net']].mean()
                    net_std = df[cols['net']].std()

                    # Calculate Z-score of net positions as % of OI
                    net_pct_z = None
                    net_pct_week_z = None
                    latest_pct = None

                    if 'open_interest_all' in df.columns:
                        df['net_pct_oi'] = (df[cols['net']] / df['open_interest_all'] * 100).fillna(0)
                        mean_pct = df['net_pct_oi'].mean()
                        std_pct = df['net_pct_oi'].std()

                        if std_pct > 0 and len(df) > 0:
                            latest_pct = df['net_pct_oi'].iloc[-1]
                            net_pct_z = (latest_pct - mean_pct) / std_pct

                            if len(df) > 1:
                                week_ago_pct = df['net_pct_oi'].iloc[-2]
                                net_pct_week_z = (week_ago_pct - mean_pct) / std_pct

                    if net_std > 0 and len(df) > 0:
                        # Get latest and week-ago values for raw net positions
                        latest_net = df[cols['net']].iloc[-1]
                        net_z = (latest_net - net_mean) / net_std

                        week_ago_net_z = None
                        if len(df) > 1:
                            week_ago_net = df[cols['net']].iloc[-2]
                            week_ago_net_z = (week_ago_net - net_mean) / net_std

                        # Extract ticker from instrument name
                        ticker = instrument.split(' - ')[0]
                        # Use the ticker mapping to get short ticker
                        for inst_name, tick in ticker_map.items():
                            if inst_name in ticker:
                                ticker = tick
                                break

                        return {
                            'ticker': ticker,
                            'z_score': net_z,  # Z-score of raw net positions
                            'z_score_pct': net_pct_z,  # Z-score of % of OI
                            'week_ago_z': week_ago_net_z,  # Week ago Z-score of raw net
                            'week_ago_z_pct': net_pct_week_z,  # Week ago Z-score of % of OI
                            'net_pct': latest_pct,  # Latest % of OI
                            'mean': net_mean,
                            'std': net_std,
                            'full_name': instrument
                        }
        except Exception as e:
            return None
        return None

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_instrument, inst) for inst in all_instruments]

        for future in as_completed(futures):
            result = future.result()
            if result:
                instrument_data[result['ticker']] = {
                    'z_score': result['z_score'],
                    'z_score_pct': result['z_score_pct'],
                    'week_ago_z': result['week_ago_z'],
                    'week_ago_z_pct': result['week_ago_z_pct'],
                    'net_pct': result['net_pct'],
                    'mean': result['mean'],
                    'std': result['std'],
                    'full_name': result['full_name']
                }

    return instrument_data


def display_cross_asset_zscore(api_token, trader_category, display_mode="Raw"):
    """Display cross-asset comparison using dashboard instruments"""

    # Fetch data with caching and parallel processing
    with st.spinner("Loading cross-asset analysis..."):
        instrument_data = fetch_zscore_data_parallel(api_token, trader_category)

    if not instrument_data:
        st.warning("No valid data found for cross-asset comparison")
        return

    # Sort by z-score from most positive to most negative (left to right)
    if display_mode == "Raw":
        # Use Z-score of raw net positions
        sorted_instruments = sorted(instrument_data.items(),
                                  key=lambda x: x[1]['z_score'] if x[1]['z_score'] is not None else -999,
                                  reverse=True)  # Most positive first
        y_values = [item[1]['z_score'] if item[1]['z_score'] is not None else 0 for item in sorted_instruments]
        week_ago_values = [item[1]['week_ago_z'] for item in sorted_instruments]
        y_title = "Z-Score"
        text_format = "{:.2f}"
    else:  # as % of Open Interest
        # Use Z-score of net as % of OI
        sorted_instruments = sorted(instrument_data.items(),
                                  key=lambda x: x[1]['z_score_pct'] if x[1]['z_score_pct'] is not None else -999,
                                  reverse=True)  # Most positive first
        y_values = [item[1]['z_score_pct'] if item[1]['z_score_pct'] is not None else 0 for item in sorted_instruments]
        week_ago_values = [item[1]['week_ago_z_pct'] for item in sorted_instruments]
        y_title = "Z-Score (% of OI)"
        text_format = "{:.2f}"

    # Create the chart
    fig = go.Figure()

    # Prepare data for plotting
    tickers = [item[0] for item in sorted_instruments]

    # Create a reverse mapping from instrument names to categories
    instrument_to_category = {}
    for category, instruments in KEY_INSTRUMENTS.items():
        for ticker, full_name in instruments.items():
            # Map both ticker and the instrument name part (before exchange) to category
            instrument_to_category[ticker] = category
            # Also map the instrument name (first part before " - ")
            instrument_name = full_name.split(' - ')[0]
            instrument_to_category[instrument_name] = category

    # Get colors based on category
    colors = []
    for item in sorted_instruments:
        # The ticker here is actually the displayed name from the chart
        display_name = item[0]

        # Try to find category by exact match or by instrument name
        category_found = instrument_to_category.get(display_name)

        if not category_found:
            # Try matching against full instrument names
            for category, instruments in KEY_INSTRUMENTS.items():
                for ticker, full_name in instruments.items():
                    if display_name in full_name or full_name.startswith(display_name):
                        category_found = category
                        break
                if category_found:
                    break

        # Default to Agriculture if not found
        if not category_found:
            category_found = "Agriculture"

        color = CATEGORY_COLORS.get(category_found, '#95E77E')
        colors.append(color)

    # Add bars with category colors
    fig.add_trace(go.Bar(
        x=tickers,
        y=y_values,
        name='Current',
        marker=dict(color=colors),
        text=[text_format.format(y) for y in y_values],
        textposition='outside',
        hoverinfo='skip'  # Disable hover tooltip
    ))

    # Add week-ago markers if available
    valid_week_ago = [(i, z) for i, z in enumerate(week_ago_values) if z is not None]

    if valid_week_ago:
        indices, week_z_values = zip(*valid_week_ago)
        fig.add_trace(go.Scatter(
            x=[tickers[i] for i in indices],
            y=week_z_values,
            mode='markers',
            name='Week Ago',
            marker=dict(
                symbol='diamond',
                size=10,
                color='purple',
                line=dict(width=2, color='white')
            ),
            hoverinfo='skip'  # Disable hover tooltip
        ))

    # Update layout based on display mode
    title = (f"{trader_category} Net Positioning Z-Scores (2-year lookback)"
             if display_mode == "Raw"
             else f"{trader_category} Net Positioning Z-Scores (% of OI basis, 2-year lookback)")

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_title,
        height=500,
        showlegend=False,
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

    # Add reference lines for both modes (both show Z-scores)
    fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=1, line_dash="dot", line_color="gray", line_width=1)
    fig.add_hline(y=-1, line_dash="dot", line_color="gray", line_width=1)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_wow_changes_data(api_token, trader_category):
    """Fetch week-over-week changes for all dashboard instruments"""

    # Get all instruments from KEY_INSTRUMENTS
    all_instruments = []
    ticker_map = {}
    for category, instruments in KEY_INSTRUMENTS.items():
        for ticker, cot_name in instruments.items():
            all_instruments.append(cot_name)
            ticker_map[cot_name] = ticker

    # Store data for all instruments
    instrument_data = {}

    def process_single_instrument(instrument):
        """Process a single instrument and return its WoW change data"""
        try:
            # Use optimized 2-year fetch function for consistency
            df = fetch_cftc_data_2year(instrument, api_token)

            if df is not None and not df.empty and len(df) >= 2:
                # Define column mappings
                # Handle both old and new naming conventions
                trader_cat = trader_category.replace(" Net", "")  # Remove " Net" suffix if present

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

                cols = category_columns[trader_cat]

                # Calculate net positions
                if trader_cat == "Non-Commercial":
                    df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
                elif trader_cat == "Commercial":
                    df['net_comm_positions'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
                elif trader_cat == "Non-Reportable":
                    df['net_nonrept_positions'] = df['nonrept_positions_long_all'] - df['nonrept_positions_short_all']
                    cols['net'] = 'net_nonrept_positions'

                if cols.get('net') in df.columns:
                    # Sort by date to ensure proper order
                    df = df.sort_values('report_date_as_yyyy_mm_dd')

                    # Get latest and previous values
                    latest_net = df[cols['net']].iloc[-1]
                    prev_net = df[cols['net']].iloc[-2]

                    # Calculate raw change
                    raw_change = latest_net - prev_net

                    # Calculate change as % of OI if available
                    change_pct_oi = None
                    if 'open_interest_all' in df.columns:
                        latest_oi = df['open_interest_all'].iloc[-1]
                        if latest_oi > 0:
                            change_pct_oi = (raw_change / latest_oi) * 100

                    # Extract ticker from instrument name
                    ticker = instrument.split(' - ')[0]
                    # Use the ticker mapping to get short ticker
                    for inst_name, tick in ticker_map.items():
                        if inst_name in ticker:
                            ticker = tick
                            break

                    return {
                        'ticker': ticker,
                        'raw_change': raw_change,
                        'change_pct_oi': change_pct_oi,
                        'latest_net': latest_net,
                        'prev_net': prev_net,
                        'full_name': instrument
                    }
        except Exception as e:
            return None
        return None

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_instrument, inst) for inst in all_instruments]

        for future in as_completed(futures):
            result = future.result()
            if result:
                instrument_data[result['ticker']] = result

    return instrument_data


def display_wow_changes(api_token, trader_category, display_mode="Raw"):
    """Display week-over-week changes using dashboard instruments"""

    # Fetch data with caching and parallel processing
    with st.spinner("Loading week-over-week changes..."):
        instrument_data = fetch_wow_changes_data(api_token, trader_category)

    if not instrument_data:
        st.warning("No valid data found for week-over-week changes")
        return

    # Sort by change value (most positive to most negative)
    if display_mode == "Raw":
        # Calculate Z-score of raw changes
        raw_changes = [v['raw_change'] for v in instrument_data.values() if v['raw_change'] is not None]

        if raw_changes:
            mean_change = np.mean(raw_changes)
            std_change = np.std(raw_changes)

            # Calculate Z-scores for each instrument
            z_scores = {}
            for ticker, data in instrument_data.items():
                if data['raw_change'] is not None and std_change > 0:
                    z_scores[ticker] = (data['raw_change'] - mean_change) / std_change
                else:
                    z_scores[ticker] = 0

            # Sort by Z-score
            sorted_instruments = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
            y_values = [item[1] for item in sorted_instruments]
            y_title = "WoW Change Z-Score"
            text_format = "{:.2f}"
        else:
            sorted_instruments = sorted(instrument_data.items(),
                                      key=lambda x: x[1]['raw_change'] if x[1]['raw_change'] is not None else 0,
                                      reverse=True)
            y_values = [0 for _ in sorted_instruments]
            y_title = "WoW Change Z-Score"
            text_format = "{:.2f}"
    else:  # as % of Open Interest
        # Sort by change as % of OI
        sorted_instruments = sorted(instrument_data.items(),
                                  key=lambda x: x[1]['change_pct_oi'] if x[1]['change_pct_oi'] is not None else -999,
                                  reverse=True)
        y_values = [item[1]['change_pct_oi'] if item[1]['change_pct_oi'] is not None else 0 for item in sorted_instruments]
        y_title = "Change (% of OI)"
        text_format = "{:.2f}%"

    # Create the chart
    fig = go.Figure()

    # Prepare data for plotting
    tickers = [item[0] for item in sorted_instruments]

    # Create a reverse mapping from instrument names to categories
    instrument_to_category = {}
    for category, instruments in KEY_INSTRUMENTS.items():
        for ticker, full_name in instruments.items():
            instrument_to_category[ticker] = category
            # Also map the instrument name (first part before " - ")
            instrument_name = full_name.split(' - ')[0]
            instrument_to_category[instrument_name] = category

    # Get colors based on category
    colors = []
    for item in sorted_instruments:
        ticker = item[0]
        category_found = instrument_to_category.get(ticker)

        if not category_found:
            # Try matching against full instrument names
            for category, instruments in KEY_INSTRUMENTS.items():
                for tick, full_name in instruments.items():
                    if ticker in full_name or full_name.startswith(ticker):
                        category_found = category
                        break
                if category_found:
                    break

        # Default to Agriculture if not found
        if not category_found:
            category_found = "Agriculture"

        color = CATEGORY_COLORS.get(category_found, '#95E77E')
        colors.append(color)

    # Add bars with category colors
    fig.add_trace(go.Bar(
        x=tickers,
        y=y_values,
        name='WoW Change',
        marker=dict(color=colors),
        text=[text_format.format(y) for y in y_values],
        textposition='outside',
        hoverinfo='skip'  # Disable hover tooltip
    ))

    # Update layout based on display mode
    title = (f"{trader_category} Week-over-Week Position Changes"
             if display_mode == "Raw"
             else f"{trader_category} Week-over-Week Position Changes (% of OI)")

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_title,
        height=700,
        showlegend=False,
        hovermode='x unified',
        yaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            gridcolor='lightgray'
        ),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=12)
        ),
        plot_bgcolor='white',
        bargap=0.2
    )

    # Add a zero line for reference
    fig.add_hline(y=0, line_width=2, line_color="black")

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_market_matrix_data(api_token, concentration_metric):
    """Fetch data for Market Matrix using dashboard instruments"""

    # Get all instruments from KEY_INSTRUMENTS
    all_instruments = []
    for category, instruments in KEY_INSTRUMENTS.items():
        for ticker, cot_name in instruments.items():
            all_instruments.append(cot_name)

    # Store data for all instruments
    all_instruments_data = {}

    def process_single_instrument(instrument):
        """Process a single instrument and return its data"""
        try:
            # Use full historical data fetch for 5-year percentile calculations
            df = fetch_cftc_data(instrument, api_token)
            if df is not None and not df.empty:
                return {instrument: df}
        except Exception as e:
            return None
        return None

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_instrument, inst) for inst in all_instruments]

        for future in as_completed(futures):
            result = future.result()
            if result:
                all_instruments_data.update(result)

    return all_instruments_data


def create_dashboard_market_matrix(api_token, concentration_metric='conc_gross_le_4_tdr_long'):
    """Create market structure matrix for dashboard instruments"""
    try:
        # Fetch data for all dashboard instruments
        with st.spinner("Loading market structure data..."):
            all_instruments_data = fetch_market_matrix_data(api_token, concentration_metric)

        if not all_instruments_data:
            st.warning("No data available for market matrix")
            return None

        # Prepare data for all selected instruments
        scatter_data = []

        # Calculate 2-year lookback date
        lookback_date = pd.Timestamp.now() - pd.DateOffset(years=2)

        # Get all instruments list
        dashboard_instruments = []
        for category, instruments in KEY_INSTRUMENTS.items():
            for ticker, cot_name in instruments.items():
                dashboard_instruments.append(cot_name)

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, instrument in enumerate(dashboard_instruments):
            status_text.text(f"Calculating percentiles for {instrument}...")
            progress_bar.progress((idx + 1) / len(dashboard_instruments))

            if instrument in all_instruments_data:
                df = all_instruments_data[instrument]

                # Filter data for 2-year lookback
                df_2yr = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()

                if len(df_2yr) < 10:  # Need sufficient data for percentiles
                    continue

                # Get latest data
                latest_idx = df['report_date_as_yyyy_mm_dd'].idxmax()
                latest_data = df.loc[latest_idx]

                # Get raw values
                trader_count = latest_data.get('traders_tot_all', 0)
                open_interest = latest_data.get('open_interest_all', 0)

                # Get concentration based on selected metric
                concentration = latest_data.get(concentration_metric, 0)

                # Calculate percentiles for trader count
                trader_percentile = (df_2yr['traders_tot_all'] <= trader_count).sum() / len(df_2yr) * 100

                # Calculate percentiles for concentration metric
                if concentration_metric in df_2yr.columns:
                    conc_percentile = (df_2yr[concentration_metric] <= concentration).sum() / len(df_2yr) * 100
                else:
                    conc_percentile = 50  # Default if column not found

                # Get category and ticker for coloring
                category = None
                ticker = None
                for cat, instruments in KEY_INSTRUMENTS.items():
                    for tick, cot_name in instruments.items():
                        if cot_name == instrument:
                            category = cat
                            ticker = tick
                            break
                    if category:
                        break

                # Create commodity name mapping from instrument names
                commodity_name_mapping = {
                    "COTTON NO. 2": "Cotton",
                    "BRITISH POUND": "GBP",
                    "COCOA": "Cocoa",
                    "USD INDEX": "DXY",
                    "E-MINI S&P 500": "S&P 500",
                    "NASDAQ MINI": "Nasdaq",
                    "RUSSELL E-MINI": "Russell",
                    "EURO FX": "EUR",
                    "JAPANESE YEN": "JPY",
                    "FRZN CONCENTRATED ORANGE JUICE": "Orange Juice",
                    "COFFEE C": "Coffee",
                    "LUMBER": "Lumber",
                    "WHEAT-HRSpring": "Wheat (HRS)",
                    "SUGAR NO. 11": "Sugar",
                    "WHEAT-SRW": "Wheat (SRW)",
                    "CORN": "Corn",
                    "FEEDER CATTLE": "Feeder Cattle",
                    "GOLD": "Gold",
                    "SILVER": "Silver",
                    "COPPER- #1": "Copper",
                    "SOYBEAN OIL": "Soybean Oil",
                    "SOYBEAN MEAL": "Soybean Meal",
                    "NAT GAS NYME": "Natural Gas",
                    "SOYBEANS": "Soybeans",
                    "WTI-PHYSICAL": "Crude Oil",
                    "GASOLINE RBOB": "Gasoline",
                    "GULF JET NY HEAT OIL SPR": "Heating Oil",
                    "LEAN HOGS": "Lean Hogs",
                    "PALLADIUM": "Palladium",
                    "PLATINUM": "Platinum",
                }

                # Get the commodity name from the instrument
                instrument_base = instrument.split(' - ')[0]
                commodity_name = commodity_name_mapping.get(instrument_base, instrument_base)

                # Add to scatter data
                scatter_data.append({
                    'instrument': instrument,
                    'trader_count': trader_count,
                    'concentration': concentration,
                    'trader_percentile': trader_percentile,
                    'conc_percentile': conc_percentile,
                    'open_interest': open_interest,
                    'short_name': commodity_name,
                    'category': category or 'Unknown'
                })

        progress_bar.empty()
        status_text.empty()

        if not scatter_data:
            st.warning("Insufficient historical data for percentile calculations")
            return None

        # Create scatter plot
        fig = go.Figure()

        # Define quadrant colors based on percentiles (50th percentile as threshold)
        colors = []
        for item in scatter_data:
            # Use category colors from CATEGORY_COLORS
            category_color = CATEGORY_COLORS.get(item['category'], '#95E77E')
            colors.append(category_color)

        # Calculate smart text positions to avoid overlaps and edge clipping
        def get_smart_text_position(x, y, index, data_points):
            """Determine optimal text position based on bubble location and nearby points"""
            # Check edges first - avoid clipping at chart boundaries
            if x < 15:  # Near left edge
                if y > 85:  # Top-left corner
                    return 'bottom right'
                elif y < 15:  # Bottom-left corner
                    return 'top right'
                else:  # Left edge
                    return 'middle right'
            elif x > 85:  # Near right edge
                if y > 85:  # Top-right corner
                    return 'bottom left'
                elif y < 15:  # Bottom-right corner
                    return 'top left'
                else:  # Right edge
                    return 'middle left'
            elif y > 85:  # Near top edge
                return 'bottom center'
            elif y < 15:  # Near bottom edge
                return 'top center'
            else:
                # Interior points - use quadrant-based positioning with collision avoidance
                positions = ['top center', 'bottom center', 'middle left', 'middle right',
                           'top left', 'top right', 'bottom left', 'bottom right']

                # Check for nearby points to avoid overlaps
                min_distance = float('inf')
                best_position = 'top center'

                for pos in positions:
                    # Calculate if this position would overlap with nearby points
                    overlap_count = 0
                    for i, other in enumerate(data_points):
                        if i != index:
                            other_x = other['trader_percentile']
                            other_y = other['conc_percentile']
                            distance = ((x - other_x) ** 2 + (y - other_y) ** 2) ** 0.5

                            # If points are close, consider this position
                            if distance < 20:  # Within proximity threshold
                                overlap_count += 1

                    # Prefer positions with fewer overlaps
                    if overlap_count < min_distance:
                        min_distance = overlap_count
                        best_position = pos

                return best_position

        # Calculate text positions for each point
        text_positions = []
        for i, d in enumerate(scatter_data):
            position = get_smart_text_position(d['trader_percentile'], d['conc_percentile'], i, scatter_data)
            text_positions.append(position)

        # Add scatter points using percentiles with smart text positioning
        fig.add_trace(go.Scatter(
            x=[d['trader_percentile'] for d in scatter_data],
            y=[d['conc_percentile'] for d in scatter_data],
            mode='markers+text',
            marker=dict(
                size=15,  # Uniform size for all bubbles
                color=colors,
                line=dict(width=2, color='white')
            ),
            text=[d['short_name'] for d in scatter_data],
            textposition=text_positions,  # Use calculated smart positions
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Category: %{customdata[1]}<br>' +
                         'Trader Count Percentile: %{x:.1f}%<br>' +
                         'Concentration Percentile: %{y:.1f}%<br>' +
                         'Actual Traders: %{customdata[2]:,.0f}<br>' +
                         'Actual Concentration: %{customdata[3]:.1f}%<br>' +
                         'Open Interest: %{customdata[4]:,.0f}<extra></extra>',
            customdata=[[d['instrument'], d['category'], d['trader_count'], d['concentration'], d['open_interest']]
                       for d in scatter_data]
        ))

        # Add quadrant lines at 50th percentile
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)

        # Add quadrant labels
        fig.add_annotation(
            text="Below Median Concentration<br>Above Median Traders",
            xref="paper", yref="paper",
            x=0.75, y=0.25,
            showarrow=False,
            font=dict(size=10, color="green"),
            opacity=0.8
        )

        fig.add_annotation(
            text="Above Median Concentration<br>Above Median Traders",
            xref="paper", yref="paper",
            x=0.75, y=0.75,
            showarrow=False,
            font=dict(size=10, color="orange"),
            opacity=0.8
        )

        fig.add_annotation(
            text="Below Median Concentration<br>Below Median Traders",
            xref="paper", yref="paper",
            x=0.25, y=0.25,
            showarrow=False,
            font=dict(size=10, color="blue"),
            opacity=0.8
        )

        fig.add_annotation(
            text="Above Median Concentration<br>Below Median Traders",
            xref="paper", yref="paper",
            x=0.25, y=0.75,
            showarrow=False,
            font=dict(size=10, color="red"),
            opacity=0.8
        )

        # Get concentration metric label
        metric_labels = {
            'conc_gross_le_4_tdr_long': 'Gross Top 4 Long',
            'conc_gross_le_4_tdr_short': 'Gross Top 4 Short',
            'conc_gross_le_8_tdr_long': 'Gross Top 8 Long',
            'conc_gross_le_8_tdr_short': 'Gross Top 8 Short',
            'conc_net_le_4_tdr_long_all': 'Net Top 4 Long',
            'conc_net_le_4_tdr_short_all': 'Net Top 4 Short',
            'conc_net_le_8_tdr_long_all': 'Net Top 8 Long',
            'conc_net_le_8_tdr_short_all': 'Net Top 8 Short'
        }
        metric_label = metric_labels.get(concentration_metric, concentration_metric)

        # Update layout
        fig.update_layout(
            title=f"Market Structure Matrix - {metric_label} (2-Year Percentiles)",
            xaxis_title="Trader Count Percentile (2-Year)",
            yaxis_title=f"Concentration Percentile ({metric_label}, 2-Year)",
            height=700,
            showlegend=False,
            xaxis=dict(
                range=[0, 100],
                gridcolor='lightgray',
                zeroline=False,
                ticksuffix='%'
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor='lightgray',
                zeroline=False,
                ticksuffix='%'
            )
        )

        # Add size legend with more detailed explanation
        fig.add_annotation(
            text="Bubble size ∝ Open Interest (larger bubbles = more market activity)<br>Color = Asset Category",
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


def display_dashboard(api_token):
    """Display the main dashboard overview"""
    st.header("Commodity Markets Overview")
    
    # Fetch dashboard data
    with st.spinner("Loading market data..."):
        df = fetch_dashboard_data(api_token)
    
    if df.empty:
        st.warning("No data available for dashboard instruments")
        return
    
    # Format the dataframe for display
    display_df = df.copy()
    
    # Format numeric columns
    numeric_cols = ['NC Net Position', 'Comm Long', 'Comm Short']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    
    # Format percentile columns
    percentile_cols = ['NC Net YTD %ile', 'Comm Long YTD %ile', 'Comm Short YTD %ile']
    for col in percentile_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")
    
    # Format concentration columns
    conc_cols = ['Conc Long 4T', 'Conc Short 4T']
    for col in conc_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    
    # Configure column display with LineChartColumn for sparklines
    column_config = {
        "Category": st.column_config.TextColumn("Category", width="small"),
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Instrument": st.column_config.TextColumn("Instrument", width="medium"),
        "NC Net Position": st.column_config.TextColumn("NC Net", help="Non-Commercial Net Position"),
        "NC Net YTD %ile": st.column_config.TextColumn("YTD %", help="YTD Percentile"),
        "NC Correlation": st.column_config.TextColumn("Corr", help="Price Correlation (TBD)", width="small"),
        "Comm Long": st.column_config.TextColumn("Comm L", help="Commercial Long Positions"),
        "Comm Long YTD %ile": st.column_config.TextColumn("YTD %", help="YTD Percentile"),
        "Comm Long Corr": st.column_config.TextColumn("Corr", help="Price Correlation (TBD)", width="small"),
        "Comm Short": st.column_config.TextColumn("Comm S", help="Commercial Short Positions"),
        "Comm Short YTD %ile": st.column_config.TextColumn("YTD %", help="YTD Percentile"),
        "Comm Short Corr": st.column_config.TextColumn("Corr", help="Price Correlation (TBD)", width="small"),
        "Conc Long 4T": st.column_config.TextColumn("4T L%", help="% held by 4 largest long traders"),
        "Conc Short 4T": st.column_config.TextColumn("4T S%", help="% held by 4 largest short traders"),
        "YTD NC Net Trend": st.column_config.LineChartColumn(
            "YTD NC Net Trend",
            help="Year-to-date Non-Commercial Net Position trend",
            width="medium",
            y_min=None,
            y_max=None
        ),
        "Last Update": st.column_config.TextColumn("Updated", width="small"),
    }
    
    # Sort by category to group them visually
    display_df = display_df.sort_values('Category', ascending=True)
    
    # Apply row coloring based on category using pandas styler
    def color_rows(row):
        """Apply background color based on category"""
        category = row['Category']
        color = CATEGORY_COLORS.get(category, '#FFFFFF')
        # Use light version of color with transparency
        return [f'background-color: {color}20' for _ in row]
    
    # Use pandas styler for coloring
    styled_df = display_df.style.apply(color_rows, axis=1)
    
    st.dataframe(
        styled_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=500
    )

    # Add Cross-Asset Z-Score Comparison below the table
    st.markdown("---")
    st.subheader("Cross-Asset Futures Positioning")

    # Controls row
    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        st.markdown("Select trader category:")
        trader_category = st.selectbox(
            "Trader category",
            ["Non-Commercial Net", "Commercial Net", "Non-Reportable Net"],
            index=0,
            label_visibility="collapsed",
            key="dashboard_trader_category"
        )

    with col2:
        st.markdown("Display mode:")
        display_mode = st.radio(
            "Display mode",
            ["Raw", "as % of Open Interest"],
            index=0,
            label_visibility="collapsed",
            key="dashboard_display_mode",
            horizontal=True
        )

    # Display the cross-asset Z-score chart
    display_cross_asset_zscore(api_token, trader_category, display_mode)

    # Add disclaimer about lookback period
    st.markdown(
        """
        <p style='color: #888; font-size: 0.9em; font-style: italic; margin-top: 10px;'>
        Note: Z-scores are calculated using a 2-year lookback period. Values above +2 or below -2 indicate extreme positioning relative to the historical average.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Add Week-over-Week Changes section
    st.markdown("---")
    st.subheader("Week-over-Week Position Changes")

    # Controls row for WoW chart
    col1_wow, col2_wow, col3_wow = st.columns([2, 2, 4])
    with col1_wow:
        st.markdown("Select trader category:")
        wow_trader_category = st.selectbox(
            "WoW Trader category",
            ["Non-Commercial Net", "Commercial Net", "Non-Reportable Net"],
            index=0,
            label_visibility="collapsed",
            key="wow_trader_category"
        )

    with col2_wow:
        st.markdown("Display mode:")
        wow_display_mode = st.radio(
            "WoW Display mode",
            ["Raw", "as % of Open Interest"],
            index=0,
            label_visibility="collapsed",
            key="wow_display_mode",
            horizontal=True
        )

    # Display the WoW changes chart
    display_wow_changes(api_token, wow_trader_category, wow_display_mode)

    # Add disclaimer about WoW calculation
    st.markdown(
        """
        <p style='color: #888; font-size: 0.9em; font-style: italic; margin-top: 10px;'>
        Week-over-week changes show the difference in positions from the previous week's report.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Add Market Matrix section
    st.markdown("---")
    st.subheader("Market Structure Matrix")

    # Controls row for Market Matrix
    col1_matrix, col2_matrix, col3_matrix = st.columns([2, 2, 4])
    with col1_matrix:
        st.markdown("Select concentration metric:")
        concentration_options = {
            'conc_gross_le_4_tdr_long': 'Gross Top 4 Traders Long',
            'conc_gross_le_4_tdr_short': 'Gross Top 4 Traders Short',
            'conc_gross_le_8_tdr_long': 'Gross Top 8 Traders Long',
            'conc_gross_le_8_tdr_short': 'Gross Top 8 Traders Short',
            'conc_net_le_4_tdr_long_all': 'Net Top 4 Traders Long',
            'conc_net_le_4_tdr_short_all': 'Net Top 4 Traders Short',
            'conc_net_le_8_tdr_long_all': 'Net Top 8 Traders Long',
            'conc_net_le_8_tdr_short_all': 'Net Top 8 Traders Short'
        }

        matrix_concentration_metric = st.selectbox(
            "Matrix concentration metric",
            options=list(concentration_options.keys()),
            format_func=lambda x: concentration_options[x],
            index=0,
            label_visibility="collapsed",
            key="matrix_concentration_metric"
        )

    # Display the Market Matrix chart
    matrix_fig = create_dashboard_market_matrix(api_token, matrix_concentration_metric)
    if matrix_fig:
        st.plotly_chart(matrix_fig, use_container_width=True, config={'displayModeBar': False})

    # Add disclaimer about Market Matrix
    st.markdown(
        """
        <p style='color: #888; font-size: 0.9em; font-style: italic; margin-top: 10px;'>
        Market Structure Matrix shows where each instrument ranks relative to its own 2-year history.
        Bubble sizes are proportional to Open Interest (market activity level).
        Percentiles allow fair comparison across different markets with varying scales.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Add refresh button
    if st.button("🔄 Refresh Dashboard Data"):
        st.cache_data.clear()
        st.rerun()