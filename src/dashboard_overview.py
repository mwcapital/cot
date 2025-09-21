"""
Dashboard overview module for displaying key commodity metrics
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_fetcher import fetch_cftc_data_ytd_only
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define key instruments for dashboard with color groups
KEY_INSTRUMENTS = {
    "Energy": [
        "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE (067651)",  # CL - WTI Oil
        "HENRY HUB - NEW YORK MERCANTILE EXCHANGE (03565B)",  # NG - Natural Gas
        "GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE (111659)",  # RBE - Gasoline
        # Note: Heating Oil and TTF Dutch not found in available instruments
        # Note: Brent futures may be under different contract code
    ],
    "Metals": [
        "GOLD - COMMODITY EXCHANGE INC. (088691)",  # GC - Gold
        "SILVER - COMMODITY EXCHANGE INC. (084691)",  # SI - Silver
        "COPPER- #1 - COMMODITY EXCHANGE INC. (085692)",  # HG - High Grade Copper
        "PLATINUM - NEW YORK MERCANTILE EXCHANGE (076651)",  # PL - Platinum
        "PALLADIUM - NEW YORK MERCANTILE EXCHANGE (075651)",  # PA - Palladium
    ],
    "Agriculture": [
        "COFFEE C - ICE FUTURES U.S. (083731)",  # KC - Coffee
        "COCOA - ICE FUTURES U.S. (073732)",  # CC - Cocoa
        "CORN - CHICAGO BOARD OF TRADE (002602)",  # ZC - Corn
        "SUGAR NO. 11 - ICE FUTURES U.S. (080732)",  # SB - Sugar No.11
        "WHEAT-SRW - CHICAGO BOARD OF TRADE (001602)",  # ZW - Wheat (Soft Red Winter)
        "SOYBEANS - CHICAGO BOARD OF TRADE (005602)",  # ZS - Soybeans
        "SOYBEAN OIL - CHICAGO BOARD OF TRADE (007601)",  # ZL - Soybean Oil
        "SOYBEAN MEAL - CHICAGO BOARD OF TRADE (026603)",  # ZM - Soybean Meal
    ],
}

# Color mapping for each category
CATEGORY_COLORS = {
    "Energy": "#FF6B6B",  # Red
    "Metals": "#4ECDC4",  # Teal
    "Agriculture": "#95E77E",  # Green
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
                "WTI-PHYSICAL": "CL",
                "HENRY HUB": "NG",
                "GASOLINE RBOB": "RB",
                "GOLD": "GC",
                "SILVER": "SI",
                "COPPER- #1": "HG",
                "PLATINUM": "PL",
                "PALLADIUM": "PA",
                "COFFEE C": "KC",
                "COCOA": "CC",
                "CORN": "ZC",
                "SUGAR NO. 11": "SB",
                "WHEAT-SRW": "ZW",
                "SOYBEANS": "ZS",
                "SOYBEAN OIL": "ZL",
                "SOYBEAN MEAL": "ZM",
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
    
    # Add refresh button
    if st.button("🔄 Refresh Dashboard Data"):
        st.cache_data.clear()
        st.rerun()