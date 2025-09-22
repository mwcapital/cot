"""
Dashboard overview module for displaying key commodity metrics
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_fetcher import fetch_cftc_data_ytd_only, fetch_cftc_data_2year
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
        "ZA": "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE (023651)",
    },
    "Metals": {
        "ZG": "GOLD - COMMODITY EXCHANGE INC. (088691)",
        "ZN": "SILVER - COMMODITY EXCHANGE INC. (084691)",
        "ZI": "COPPER- #1 - COMMODITY EXCHANGE INC. (085692)",
        "ZK": "PALLADIUM - NEW YORK MERCANTILE EXCHANGE (075651)",
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
                "NAT GAS NYME": "ZN",
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
        for instrument in instruments:
            all_instruments.append((category, instrument))
    
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

                # Calculate net positions
                if trader_category == "Non-Commercial":
                    df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
                elif trader_category == "Commercial":
                    df['net_comm_positions'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
                elif trader_category == "Non-Reportable":
                    df['net_nonrept_positions'] = df['nonrept_positions_long_all'] - df['nonrept_positions_short_all']
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
                        if len(df) > 1:
                            week_ago_pct = df['net_pct_oi'].iloc[-2]
                            week_ago_z = (week_ago_pct - mean_pct) / std_pct

                        # Extract ticker from instrument name
                        ticker = instrument.split(' - ')[0]
                        # Use the ticker mapping to get short ticker
                        for inst_name, tick in ticker_map.items():
                            if inst_name in ticker:
                                ticker = tick
                                break

                        return {
                            'ticker': ticker,
                            'z_score': latest_z,
                            'week_ago_z': week_ago_z,
                            'net_pct': latest_pct,
                            'mean': mean_pct,
                            'std': std_pct,
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
                    'week_ago_z': result['week_ago_z'],
                    'net_pct': result['net_pct'],
                    'mean': result['mean'],
                    'std': result['std'],
                    'full_name': result['full_name']
                }

    return instrument_data


def display_cross_asset_zscore(api_token, trader_category):
    """Display cross-asset Z-score comparison using dashboard instruments"""

    # Fetch data with caching and parallel processing
    with st.spinner("Loading cross-asset analysis..."):
        instrument_data = fetch_zscore_data_parallel(api_token, trader_category)

    if not instrument_data:
        st.warning("No valid data found for cross-asset comparison")
        return

    # Sort by z-score
    sorted_instruments = sorted(instrument_data.items(), key=lambda x: x[1]['z_score'], reverse=True)

    # Create the chart
    fig = go.Figure()

    # Prepare data for plotting
    tickers = [item[0] for item in sorted_instruments]
    z_scores = [item[1]['z_score'] for item in sorted_instruments]

    # Nice mint/salad green color
    bar_color = '#7DCEA0'  # Soft mint green

    # Add bars
    fig.add_trace(go.Bar(
        x=tickers,
        y=z_scores,
        name='Current Z-Score',
        marker=dict(color=bar_color),
        text=[f"{z:.2f}" for z in z_scores],
        textposition='outside',
        hovertemplate='<b>%{customdata[3]}</b><br>' +
                     'Z-Score: %{y:.2f}<br>' +
                     'Net %: %{customdata[0]:.1f}%<br>' +
                     'Mean %: %{customdata[1]:.1f}%<br>' +
                     'Std %: %{customdata[2]:.1f}%<extra></extra>',
        customdata=[[item[1]['net_pct'], item[1]['mean'], item[1]['std'], item[1]['full_name'].split(' - ')[0]]
                   for item in sorted_instruments]
    ))

    # Add week-ago markers if available
    week_ago_zs = [item[1]['week_ago_z'] for item in sorted_instruments]
    valid_week_ago = [(i, z) for i, z in enumerate(week_ago_zs) if z is not None]

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
            hovertemplate='Week Ago Z-Score: %{y:.2f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title=f"{trader_category} Net Positioning Z-Scores (since 2023)",
        xaxis_title="",
        yaxis_title="Z-Score",
        height=500,
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
    fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=1, line_dash="dot", line_color="gray", line_width=1)
    fig.add_hline(y=-1, line_dash="dot", line_color="gray", line_width=1)

    st.plotly_chart(fig, use_container_width=True)


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
    st.subheader("🎯 Cross-Asset Comparison")

    # Trader category selector
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("Select trader category:")
        trader_category = st.selectbox(
            "Trader category",
            ["Non-Commercial", "Commercial", "Non-Reportable"],
            index=0,
            label_visibility="collapsed",
            key="dashboard_trader_category"
        )

    # Display the cross-asset Z-score chart
    display_cross_asset_zscore(api_token, trader_category)

    # Add refresh button
    if st.button("🔄 Refresh Dashboard Data"):
        st.cache_data.clear()
        st.rerun()