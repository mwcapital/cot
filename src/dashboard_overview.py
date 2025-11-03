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
import os
import json
from supabase import create_client, Client
from dotenv import load_dotenv
from charts.cross_asset_analysis import create_positioning_concentration_charts, create_relative_strength_matrix, create_cross_asset_participation_comparison

# Load environment variables
load_dotenv()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_futures_instruments():
    """Load all futures instruments with COT mappings from futures_symbols_enhanced.json"""
    try:
        # Load the futures mapping
        mapping_path = 'instrument_management/futures/futures_symbols_enhanced.json'
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)

        instruments = {}

        # Process each futures symbol
        for symbol, symbol_data in mapping_data['futures_symbols'].items():
            # Only include symbols that have COT mapping
            if symbol_data.get('cot_mapping', {}).get('matched', False):
                category = symbol_data.get('category', 'Other')

                # Initialize category if not exists
                if category not in instruments:
                    instruments[category] = {}

                # Get the first COT instrument name for this symbol
                cot_instruments = symbol_data['cot_mapping'].get('instruments', [])
                if cot_instruments:
                    # Use the first instrument, or combine if multiple
                    cot_name = cot_instruments[0]
                    instruments[category][symbol] = cot_name

        return instruments
    except Exception as e:
        st.error(f"Error loading futures instruments: {e}")
        return {}

# Define key instruments for dashboard - now loaded dynamically
def get_key_instruments():
    """Get instruments from futures_symbols_enhanced.json"""
    return load_futures_instruments()

@st.cache_data(ttl=3600)
def get_ticker_mapping():
    """Create a mapping from COT instrument names to futures symbols"""
    try:
        mapping_path = 'instrument_management/futures/futures_symbols_enhanced.json'
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)

        ticker_map = {}
        for symbol, symbol_data in mapping_data['futures_symbols'].items():
            if symbol_data.get('cot_mapping', {}).get('matched', False):
                cot_instruments = symbol_data['cot_mapping'].get('instruments', [])
                for cot_instrument in cot_instruments:
                    # Map both full name and shortened name to ticker
                    ticker_map[cot_instrument] = symbol
                    # Also map just the instrument name part (before " - ")
                    instrument_name = cot_instrument.split(' - ')[0]
                    ticker_map[instrument_name] = symbol

        return ticker_map
    except Exception as e:
        return {}

@st.cache_data(ttl=3600)
def get_ticker_to_name_mapping():
    """Create a mapping from ticker symbols to display names"""
    try:
        mapping_path = 'instrument_management/futures/futures_symbols_enhanced.json'
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)

        name_map = {}
        for symbol, symbol_data in mapping_data['futures_symbols'].items():
            if symbol_data.get('cot_mapping', {}).get('matched', False):
                # Map ticker to the "name" field
                name_map[symbol] = symbol_data.get('name', symbol)

        return name_map
    except Exception as e:
        return {}

# Color mapping for each category
CATEGORY_COLORS = {
    "Energy": "#FF6B6B",  # Red
    "Metals": "#4ECDC4",  # Teal
    "Agricultural": "#95E77E",  # Green (note: changed from "Agriculture" to "Agricultural")
    "Currency": "#FFD93D",  # Yellow
    "Index": "#6A7FDB",  # Blue (was "Equity", now "Index")
    "Financial": "#9B59B6",  # Purple
}




def get_supabase_client():
    """Initialize Supabase client using secure configuration"""
    try:
        # Try Streamlit secrets first (production)
        if hasattr(st, 'secrets') and 'SUPABASE_URL' in st.secrets:
            url = st.secrets['SUPABASE_URL']
            key = st.secrets['SUPABASE_KEY']
        else:
            # Fallback to environment variables (local development)
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            return None

        return create_client(url, key)
    except Exception as e:
        return None


def get_futures_symbol_for_cot_instrument(instrument_name):
    """Map COT instrument name to futures symbol using the enhanced mapping"""
    try:
        # Load the futures mapping
        mapping_path = 'instrument_management/futures/futures_symbols_enhanced.json'
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)

        # Search for the instrument in the COT mappings
        for symbol, symbol_data in mapping_data['futures_symbols'].items():
            if 'cot_mapping' in symbol_data and symbol_data['cot_mapping'].get('matched', False):
                for cot_instrument in symbol_data['cot_mapping'].get('instruments', []):
                    if instrument_name in cot_instrument:
                        return symbol
        return None
    except Exception as e:
        return None


def calculate_enhanced_correlations(cot_data, futures_symbol, position_type, window_days=730):
    """
    Calculate enhanced correlations: Pearson, Spearman, and Lead-Lag between COT position CHANGES and weekly futures returns

    Args:
        cot_data: DataFrame with COT data
        futures_symbol: Futures symbol to get price data for
        position_type: 'net_noncomm', 'comm_long', or 'comm_short'
        window_days: Rolling window in days (default 730 = ~2 years)

    Returns:
        dict: {'pearson': float, 'spearman': float, 'lead_lag': float} or NaN values if insufficient data
    """
    from scipy.stats import pearsonr, spearmanr

    try:
        if cot_data.empty or not futures_symbol:
            return {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}

        supabase = get_supabase_client()
        if not supabase:
            return {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}

        # Get COT position column name
        position_columns = {
            'net_noncomm': 'net_noncomm_positions',
            'comm_long': 'comm_positions_long_all',
            'comm_short': 'comm_positions_short_all'
        }

        if position_type not in position_columns:
            return {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}

        position_col = position_columns[position_type]

        if position_col not in cot_data.columns:
            return {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}

        # Get 2+ years of futures price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days + 100)  # Extra buffer

        response = supabase.from_('futures_prices').select(
            'date, close'
        ).eq('symbol', futures_symbol).eq(
            'adjustment_method', 'NON'
        ).gte(
            'date', start_date.strftime('%Y-%m-%d')
        ).order('date', desc=False).execute()

        if not response.data:
            return {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}

        # Convert to DataFrame
        price_df = pd.DataFrame(response.data)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['close'] = pd.to_numeric(price_df['close'], errors='coerce')
        price_df = price_df.dropna().sort_values('date')

        if len(price_df) < 100:  # Need sufficient price data
            return {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}

        # Calculate weekly returns (Tuesday to Tuesday to align with COT reports)
        # Resample to weekly, ending on Tuesday
        price_df.set_index('date', inplace=True)
        weekly_df = price_df.resample('W-TUE').agg({'close': 'last'}).dropna()

        # Calculate weekly returns
        weekly_df['returns'] = weekly_df['close'].pct_change()
        weekly_df = weekly_df.dropna()

        if len(weekly_df) < 20:  # Need at least 20 weeks of data
            return {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}

        # Prepare COT data
        cot_df = cot_data.copy()
        cot_df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(cot_df['report_date_as_yyyy_mm_dd'])
        cot_df = cot_df.sort_values('report_date_as_yyyy_mm_dd')

        # Calculate position changes (key improvement!)
        cot_df['position_change'] = cot_df[position_col].diff()
        cot_df = cot_df.dropna()

        if len(cot_df) < 20:  # Need at least 20 weeks of COT data
            return {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}

        # Merge datasets on date (align COT report date with weekly price returns)
        weekly_df.reset_index(inplace=True)
        merged = pd.merge_asof(
            weekly_df.sort_values('date'),
            cot_df[['report_date_as_yyyy_mm_dd', 'position_change']].sort_values('report_date_as_yyyy_mm_dd'),
            left_on='date',
            right_on='report_date_as_yyyy_mm_dd',
            direction='nearest',
            tolerance=pd.Timedelta('7 days')
        ).dropna()

        if len(merged) < 15:  # Need at least 15 overlapping points
            return {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}

        returns = merged['returns'].values
        position_changes = merged['position_change'].values

        # 1. Pearson Correlation (position changes vs returns)
        try:
            pearson_corr = pearsonr(position_changes, returns)[0]
        except:
            pearson_corr = np.nan

        # 2. Spearman Correlation (position changes vs returns)
        try:
            spearman_corr = spearmanr(position_changes, returns)[0]
        except:
            spearman_corr = np.nan

        # 3. Lead-Lag Analysis (±2 weeks)
        best_lead_lag_corr = np.nan
        try:
            correlations = []

            # Test different lags: -2, -1, 0, +1, +2 weeks
            for lag in range(-2, 3):
                if lag == 0:
                    # Contemporaneous correlation
                    corr = pearsonr(position_changes, returns)[0]
                elif lag > 0:
                    # Position changes lead returns (position changes predict future returns)
                    if len(position_changes) > lag and len(returns) > lag:
                        corr = pearsonr(position_changes[:-lag], returns[lag:])[0]
                    else:
                        continue
                else:  # lag < 0
                    # Position changes lag returns (returns predict future position changes)
                    abs_lag = abs(lag)
                    if len(position_changes) > abs_lag and len(returns) > abs_lag:
                        corr = pearsonr(position_changes[abs_lag:], returns[:-abs_lag])[0]
                    else:
                        continue

                if not np.isnan(corr):
                    correlations.append(corr)

            if correlations:
                # Take the correlation with highest absolute value
                best_lead_lag_corr = max(correlations, key=abs)
        except:
            best_lead_lag_corr = np.nan

        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'lead_lag': best_lead_lag_corr
        }

    except Exception as e:
        print(f"Enhanced correlation calculation failed for {futures_symbol}: {e}")
        return {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}


def calculate_all_enhanced_correlations_optimized(cot_data, futures_symbol, window_days=730):
    """
    OPTIMIZED: Calculate all enhanced correlations at once, reusing price data fetch

    Args:
        cot_data: DataFrame with COT data
        futures_symbol: Futures symbol to get price data for
        window_days: Rolling window in days (default 730 = ~2 years)

    Returns:
        dict: {
            'nc': {'pearson': float, 'spearman': float, 'lead_lag': float},
            'cl': {'pearson': float, 'spearman': float, 'lead_lag': float},
            'cs': {'pearson': float, 'spearman': float, 'lead_lag': float}
        }
    """
    from scipy.stats import pearsonr, spearmanr
    from concurrent.futures import ThreadPoolExecutor

    # Initialize return structure
    result = {
        'nc': {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan},
        'cl': {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan},
        'cs': {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}
    }

    try:
        if cot_data.empty or not futures_symbol:
            return result

        supabase = get_supabase_client()
        if not supabase:
            print(f"❌ Supabase client is None for {futures_symbol} - check credentials")
            return result

        # OPTIMIZATION 1: Single price data fetch for all correlations
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days + 100)

        response = supabase.from_('futures_prices').select(
            'date, close'
        ).eq('symbol', futures_symbol).eq(
            'adjustment_method', 'NON'
        ).gte(
            'date', start_date.strftime('%Y-%m-%d')
        ).order('date', desc=False).execute()

        if not response.data:
            print(f"❌ No price data returned from Supabase for symbol: {futures_symbol}")
            return result

        # Convert to DataFrame and calculate weekly returns
        price_df = pd.DataFrame(response.data)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['close'] = pd.to_numeric(price_df['close'], errors='coerce')
        price_df = price_df.dropna().sort_values('date')

        if len(price_df) < 100:
            return result

        # Calculate weekly returns (Tuesday to Tuesday)
        price_df.set_index('date', inplace=True)
        weekly_df = price_df.resample('W-TUE').agg({'close': 'last'}).dropna()
        weekly_df['returns'] = weekly_df['close'].pct_change()
        weekly_df = weekly_df.dropna()

        if len(weekly_df) < 20:
            return result

        # Prepare COT data with all position columns
        cot_df = cot_data.copy()
        cot_df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(cot_df['report_date_as_yyyy_mm_dd'])
        cot_df = cot_df.sort_values('report_date_as_yyyy_mm_dd')

        # Calculate all position changes at once
        required_cols = ['net_noncomm_positions', 'comm_positions_long_all', 'comm_positions_short_all']
        for col in required_cols:
            if col in cot_df.columns:
                cot_df[f'{col}_change'] = cot_df[col].diff()

        cot_df = cot_df.dropna()

        if len(cot_df) < 20:
            return result

        # OPTIMIZATION 2: Single merge for all position types
        weekly_df.reset_index(inplace=True)
        merged_base = pd.merge_asof(
            weekly_df.sort_values('date'),
            cot_df[['report_date_as_yyyy_mm_dd', 'net_noncomm_positions_change',
                   'comm_positions_long_all_change', 'comm_positions_short_all_change']].sort_values('report_date_as_yyyy_mm_dd'),
            left_on='date',
            right_on='report_date_as_yyyy_mm_dd',
            direction='nearest',
            tolerance=pd.Timedelta('7 days')
        ).dropna()

        if len(merged_base) < 15:
            return result

        returns = merged_base['returns'].values

        # OPTIMIZATION 3: Calculate all correlations using parallel processing
        def calculate_correlations_for_position(position_changes, position_type):
            """Calculate Pearson, Spearman, and Lead-Lag for one position type"""
            pos_result = {'pearson': np.nan, 'spearman': np.nan, 'lead_lag': np.nan}

            try:
                # 1. Pearson
                pos_result['pearson'] = pearsonr(position_changes, returns)[0]

                # 2. Spearman
                pos_result['spearman'] = spearmanr(position_changes, returns)[0]

                # 3. Lead-Lag (find best correlation across ±2 weeks)
                correlations = []
                for lag in range(-2, 3):
                    if lag == 0:
                        corr = pearsonr(position_changes, returns)[0]
                    elif lag > 0:
                        if len(position_changes) > lag and len(returns) > lag:
                            corr = pearsonr(position_changes[:-lag], returns[lag:])[0]
                        else:
                            continue
                    else:  # lag < 0
                        abs_lag = abs(lag)
                        if len(position_changes) > abs_lag and len(returns) > abs_lag:
                            corr = pearsonr(position_changes[abs_lag:], returns[:-abs_lag])[0]
                        else:
                            continue

                    if not np.isnan(corr):
                        correlations.append(corr)

                if correlations:
                    pos_result['lead_lag'] = max(correlations, key=abs)

            except Exception as e:
                print(f"Correlation calculation failed for {position_type}: {e}")

            return pos_result

        # Run correlation calculations in parallel for all three position types
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                'nc': executor.submit(calculate_correlations_for_position,
                                    merged_base['net_noncomm_positions_change'].values, 'net_noncomm'),
                'cl': executor.submit(calculate_correlations_for_position,
                                    merged_base['comm_positions_long_all_change'].values, 'comm_long'),
                'cs': executor.submit(calculate_correlations_for_position,
                                    merged_base['comm_positions_short_all_change'].values, 'comm_short')
            }

            # Collect results
            for key, future in futures.items():
                result[key] = future.result()

        return result

    except Exception as e:
        print(f"Optimized correlation calculation failed for {futures_symbol}: {e}")
        return result


def calculate_rolling_correlation(cot_data, futures_symbol, position_type, window_days=730):
    """
    Legacy function - kept for backward compatibility
    Calculate rolling 2-year correlation between COT positions and weekly futures returns

    Args:
        cot_data: DataFrame with COT data
        futures_symbol: Futures symbol to get price data for
        position_type: 'net_noncomm', 'comm_long', or 'comm_short'
        window_days: Rolling window in days (default 730 = ~2 years)

    Returns:
        float: Latest correlation value, or NaN if insufficient data
    """
    try:
        if cot_data.empty or not futures_symbol:
            return np.nan

        supabase = get_supabase_client()
        if not supabase:
            return np.nan

        # Get COT position column name
        position_columns = {
            'net_noncomm': 'net_noncomm_positions',
            'comm_long': 'comm_positions_long_all',
            'comm_short': 'comm_positions_short_all'
        }

        if position_type not in position_columns:
            return np.nan

        position_col = position_columns[position_type]

        if position_col not in cot_data.columns:
            return np.nan

        # Get 2+ years of futures price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days + 100)  # Extra buffer

        print(f"Querying Supabase for symbol '{futures_symbol}' from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        response = supabase.from_('futures_prices').select(
            'date, close'
        ).eq('symbol', futures_symbol).eq(
            'adjustment_method', 'NON'
        ).gte(
            'date', start_date.strftime('%Y-%m-%d')
        ).order('date', desc=False).execute()

        print(f"Supabase returned {len(response.data) if response.data else 0} price records for {futures_symbol}")

        if not response.data:
            return np.nan

        # Convert to DataFrame
        price_df = pd.DataFrame(response.data)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['close'] = pd.to_numeric(price_df['close'], errors='coerce')
        price_df = price_df.dropna().sort_values('date')

        if len(price_df) < 100:  # Need sufficient price data
            return np.nan

        # Calculate weekly returns (Tuesday to Tuesday to align with COT reports)
        # COT reports on Tuesday for previous week ending Tuesday
        price_df['weekday'] = price_df['date'].dt.dayofweek  # 0=Monday, 1=Tuesday

        # Find Tuesdays (weekday=1) or closest day if Tuesday missing
        weekly_prices = []
        for week_start in pd.date_range(start=price_df['date'].min(),
                                      end=price_df['date'].max(),
                                      freq='W-TUE'):  # Week ending Tuesday
            week_data = price_df[
                (price_df['date'] >= week_start - timedelta(days=7)) &
                (price_df['date'] <= week_start)
            ]
            if not week_data.empty:
                # Get closest to Tuesday or last available price of the week
                closest_price = week_data.iloc[-1]  # Last price of the week
                weekly_prices.append({
                    'week_end': week_start,
                    'close': closest_price['close']
                })

        if len(weekly_prices) < 50:  # Need sufficient weekly data
            return np.nan

        weekly_df = pd.DataFrame(weekly_prices)
        weekly_df['weekly_return'] = weekly_df['close'].pct_change()
        weekly_df = weekly_df.dropna()

        # Prepare COT data - ensure it's sorted by date
        cot_df = cot_data.copy().sort_values('report_date_as_yyyy_mm_dd')

        # Align COT data with weekly returns
        # COT report date corresponds to the week ending that Tuesday
        aligned_data = []

        for _, cot_row in cot_df.iterrows():
            cot_date = pd.to_datetime(cot_row['report_date_as_yyyy_mm_dd'])

            # Find matching weekly return (same week ending date)
            matching_return = weekly_df[weekly_df['week_end'] == cot_date]

            if not matching_return.empty and not pd.isna(cot_row[position_col]):
                aligned_data.append({
                    'date': cot_date,
                    'position': cot_row[position_col],
                    'return': matching_return.iloc[0]['weekly_return']
                })

        if len(aligned_data) < 50:  # Need sufficient aligned data
            return np.nan

        aligned_df = pd.DataFrame(aligned_data)

        # Calculate rolling correlation (2-year window = ~104 weeks)
        correlation_window = min(104, len(aligned_df) - 1)

        if correlation_window < 20:  # Need minimum data for meaningful correlation
            return np.nan

        # Get the most recent correlation
        recent_data = aligned_df.tail(correlation_window)
        correlation = recent_data['position'].corr(recent_data['return'])

        return correlation if not pd.isna(correlation) else np.nan

    except Exception as e:
        return np.nan


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
        # Fetch YTD data for basic dashboard metrics
        df_ytd = fetch_cftc_data_ytd_only(instrument, api_token)

        # Fetch 2-year data for correlation calculations
        df_2year = fetch_cftc_data_2year(instrument, api_token)

        if df_ytd is not None and not df_ytd.empty:
            # Get latest data point
            latest = df_ytd.iloc[-1]

            # Data is already YTD only from optimized fetch
            ytd_df = df_ytd
            
            # Create sparkline data as a list of values for LineChartColumn
            ytd_sparkline = None
            if not ytd_df.empty and 'net_noncomm_positions' in ytd_df.columns:
                # Sort by date and get values as list
                ytd_sorted = ytd_df.sort_values('report_date_as_yyyy_mm_dd')
                ytd_sparkline = ytd_sorted['net_noncomm_positions'].fillna(0).tolist()
            
            # Calculate metrics
            # Get ticker symbol from dynamic mapping (use same logic as plot)
            ticker_map = get_ticker_mapping()
            name_map = get_ticker_to_name_mapping()

            # Try full instrument name first, then short name
            ticker = ticker_map.get(instrument)
            if not ticker:
                instrument_name = instrument.split(' - ')[0]
                ticker = ticker_map.get(instrument_name, instrument_name[:2].upper())

            # Get display name from the "name" field in JSON
            display_name = name_map.get(ticker, instrument.split(' - ')[0])

            # Get futures symbol for correlation calculations
            # The ticker IS the futures symbol we need
            futures_symbol = ticker

            # Initialize enhanced correlation variables
            nc_corr_p = nc_corr_s = nc_corr_ll = np.nan
            comm_long_corr_p = comm_long_corr_s = comm_long_corr_ll = np.nan
            comm_short_corr_p = comm_short_corr_s = comm_short_corr_ll = np.nan

            # Calculate enhanced correlations using 2-year data (OPTIMIZED - single price fetch)
            if df_2year is not None and not df_2year.empty and futures_symbol:
                try:
                    # OPTIMIZATION: Calculate all correlations at once to reuse price data
                    all_corrs = calculate_all_enhanced_correlations_optimized(df_2year, futures_symbol)

                    # Non-Commercial Net correlations
                    nc_corr_p = all_corrs['nc']['pearson']
                    nc_corr_s = all_corrs['nc']['spearman']
                    nc_corr_ll = all_corrs['nc']['lead_lag']

                    # Commercial Long correlations
                    comm_long_corr_p = all_corrs['cl']['pearson']
                    comm_long_corr_s = all_corrs['cl']['spearman']
                    comm_long_corr_ll = all_corrs['cl']['lead_lag']

                    # Commercial Short correlations
                    comm_short_corr_p = all_corrs['cs']['pearson']
                    comm_short_corr_s = all_corrs['cs']['spearman']
                    comm_short_corr_ll = all_corrs['cs']['lead_lag']

                except Exception as e:
                    # Keep correlations as NaN if calculation fails
                    pass  # Silent fail for performance

            row_data = {
                'Category': category,
                'Instrument': display_name,  # Display name from JSON "name" field

                # Open Interest from COT data (CFTC API)
                'Open Interest': latest.get('open_interest_all', np.nan),
                'OI Change': latest.get('change_in_open_interest_all', np.nan),

                # Non-Commercial Net Positions
                'NC Net Position': latest.get('net_noncomm_positions', np.nan),
                'NC Net YTD %ile': calculate_ytd_percentile(
                    ytd_df.get('net_noncomm_positions', pd.Series()),
                    latest.get('net_noncomm_positions', np.nan)
                ),
                'NC Corr_P': nc_corr_p,
                'NC Corr_S': nc_corr_s,
                'NC L_L': nc_corr_ll,

                # Commercial Longs
                'Comm Long': latest.get('comm_positions_long_all', np.nan),
                'Comm Long YTD %ile': calculate_ytd_percentile(
                    ytd_df.get('comm_positions_long_all', pd.Series()),
                    latest.get('comm_positions_long_all', np.nan)
                ),
                'CL Corr_P': comm_long_corr_p,
                'CL Corr_S': comm_long_corr_s,
                'CL L_L': comm_long_corr_ll,

                # Commercial Shorts
                'Comm Short': latest.get('comm_positions_short_all', np.nan),
                'Comm Short YTD %ile': calculate_ytd_percentile(
                    ytd_df.get('comm_positions_short_all', pd.Series()),
                    latest.get('comm_positions_short_all', np.nan)
                ),
                'CS Corr_P': comm_short_corr_p,
                'CS Corr_S': comm_short_corr_s,
                'CS L_L': comm_short_corr_ll,
                
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
    # Get instruments dynamically from futures file
    key_instruments = get_key_instruments()

    # Flatten instrument list
    all_instruments = []
    for category, instruments in key_instruments.items():
        for ticker, cot_name in instruments.items():
            all_instruments.append((category, cot_name))

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Processing instruments with optimized correlations...")

    # Use ThreadPoolExecutor for parallel fetching with optimized correlations
    with ThreadPoolExecutor(max_workers=6) as executor:  # Reduced workers for correlation-heavy tasks
        # Submit all fetch tasks
        future_to_instrument = {
            executor.submit(fetch_single_instrument_data, cat, inst, api_token): (cat, inst)
            for cat, inst in all_instruments
        }
        
        completed = 0
        # Process completed futures as they finish
        for future in as_completed(future_to_instrument):
            completed += 1
            progress = completed / len(all_instruments)
            progress_bar.progress(progress)
            status_text.text(f"Processing correlations... {completed}/{len(all_instruments)} ({progress:.0%})")

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

    # Get all instruments dynamically and create ticker map
    key_instruments = get_key_instruments()
    ticker_map = get_ticker_mapping()  # Use the standardized ticker mapping
    all_instruments = []

    # Build list of all COT instrument names
    for category, instruments in key_instruments.items():
        for ticker, cot_name in instruments.items():
            all_instruments.append(cot_name)

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

                        # Get the futures ticker from the ticker map using the full COT instrument name
                        ticker = ticker_map.get(instrument)

                        # Fallback: if not found, try without exchange name
                        if not ticker:
                            instrument_short = instrument.split(' - ')[0]
                            ticker = ticker_map.get(instrument_short)

                        # Final fallback: use short instrument name
                        if not ticker:
                            ticker = instrument.split(' - ')[0]

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


def display_cross_asset_zscore(api_token, trader_category):
    """Display cross-asset comparison using dashboard instruments - shows both Raw and % of OI charts"""

    # Fetch data with caching and parallel processing
    with st.spinner("Loading cross-asset analysis..."):
        instrument_data = fetch_zscore_data_parallel(api_token, trader_category)

    if not instrument_data:
        st.warning("No valid data found for cross-asset comparison")
        return

    # Sort once by Raw z-score and use same order for both charts
    sorted_instruments = sorted(instrument_data.items(),
                              key=lambda x: x[1]['z_score'] if x[1]['z_score'] is not None else -999,
                              reverse=True)  # Most positive first

    # Create both charts
    for display_mode in ["Raw", "as % of Open Interest"]:
        # Use the same sorted order for both charts
        if display_mode == "Raw":
            # Use Z-score of raw net positions
            y_values = [item[1]['z_score'] if item[1]['z_score'] is not None else 0 for item in sorted_instruments]
            week_ago_values = [item[1]['week_ago_z'] for item in sorted_instruments]
            y_title = "Z-Score"
            text_format = "{:.2f}"
        else:  # as % of Open Interest
            # Use Z-score of net as % of OI, but keep the same instrument order
            y_values = [item[1]['z_score_pct'] if item[1]['z_score_pct'] is not None else 0 for item in sorted_instruments]
            week_ago_values = [item[1]['week_ago_z_pct'] for item in sorted_instruments]
            y_title = "Z-Score (% of OI)"
            text_format = "{:.2f}"

        # Create the chart
        fig = go.Figure()

        # Get ticker to name mapping for display
        name_map = get_ticker_to_name_mapping()

        # Prepare data for plotting - map tickers to display names
        tickers = [item[0] for item in sorted_instruments]
        display_names = [name_map.get(ticker, ticker) for ticker in tickers]

        # Create a reverse mapping from instrument names to categories
        key_instruments = get_key_instruments()
        instrument_to_category = {}
        for category, instruments in key_instruments.items():
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
                for category, instruments in key_instruments.items():
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
            x=display_names,
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
                x=[display_names[i] for i in indices],
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
                gridcolor='lightgray',
                range=[-3, 3]  # Fixed range to show ±3 standard deviations
            ),
            xaxis=dict(
                tickangle=-45
            ),
            plot_bgcolor='white',
            bargap=0.2,
            margin=dict(l=50, r=50, t=80, b=150)  # Add margins to use available space
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

    # Get all instruments dynamically
    key_instruments = get_key_instruments()
    all_instruments = []
    ticker_map = {}
    for category, instruments in key_instruments.items():
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

                    # Calculate historical Z-score for this instrument's WoW changes
                    # Calculate all historical WoW changes for this instrument
                    df['wow_change'] = df[cols['net']].diff()
                    historical_changes = df['wow_change'].dropna()

                    # Calculate Z-score based on instrument's own history
                    z_score = None
                    z_score_pct_oi = None

                    if len(historical_changes) > 10:  # Need sufficient history
                        mean_change = historical_changes.mean()
                        std_change = historical_changes.std()

                        if std_change > 0:
                            z_score = (raw_change - mean_change) / std_change

                    # Calculate change as % of OI if available
                    change_pct_oi = None
                    if 'open_interest_all' in df.columns:
                        latest_oi = df['open_interest_all'].iloc[-1]
                        if latest_oi > 0:
                            change_pct_oi = (raw_change / latest_oi) * 100

                            # Also calculate Z-score for % of OI changes
                            df['wow_change_pct_oi'] = (df[cols['net']].diff() / df['open_interest_all']) * 100
                            historical_pct_changes = df['wow_change_pct_oi'].dropna()

                            if len(historical_pct_changes) > 10:
                                mean_pct_change = historical_pct_changes.mean()
                                std_pct_change = historical_pct_changes.std()

                                if std_pct_change > 0:
                                    z_score_pct_oi = (change_pct_oi - mean_pct_change) / std_pct_change

                    # Get the futures ticker from the ticker map using the full COT instrument name
                    ticker = ticker_map.get(instrument)

                    # Fallback: if not found, try without exchange name
                    if not ticker:
                        instrument_short = instrument.split(' - ')[0]
                        ticker = ticker_map.get(instrument_short)

                    # Final fallback: use short instrument name
                    if not ticker:
                        ticker = instrument.split(' - ')[0]

                    return {
                        'ticker': ticker,
                        'raw_change': raw_change,
                        'change_pct_oi': change_pct_oi,
                        'z_score': z_score,
                        'z_score_pct_oi': z_score_pct_oi,
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
        # Use pre-calculated historical Z-scores from fetch function
        sorted_instruments = sorted(instrument_data.items(),
                                  key=lambda x: x[1]['z_score'] if x[1]['z_score'] is not None else 0,
                                  reverse=True)
        y_values = [item[1]['z_score'] if item[1]['z_score'] is not None else 0 for item in sorted_instruments]
        y_title = "WoW Change Z-Score (Historical)"
        text_format = "{:.2f}"
    else:  # as % of Open Interest
        # Use pre-calculated historical Z-scores for % OI from fetch function
        sorted_instruments = sorted(instrument_data.items(),
                                  key=lambda x: x[1]['z_score_pct_oi'] if x[1]['z_score_pct_oi'] is not None else 0,
                                  reverse=True)
        y_values = [item[1]['z_score_pct_oi'] if item[1]['z_score_pct_oi'] is not None else 0 for item in sorted_instruments]
        y_title = "Change Z-Score (% of OI, Historical)"
        text_format = "{:.2f}"

    # Create the chart
    fig = go.Figure()

    # Get ticker to name mapping for display
    name_map = get_ticker_to_name_mapping()

    # Prepare data for plotting - map tickers to display names
    tickers = [item[0] for item in sorted_instruments]
    display_names = [name_map.get(ticker, ticker) for ticker in tickers]

    # Create a reverse mapping from instrument names to categories
    key_instruments = get_key_instruments()
    instrument_to_category = {}
    for category, instruments in key_instruments.items():
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
            for category, instruments in key_instruments.items():
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
        x=display_names,
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
        bargap=0.2,
        margin=dict(l=50, r=50, t=80, b=150)  # Add margins to use available space
    )

    # Add reference lines for Z-scores
    fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=1, line_dash="dot", line_color="gray", line_width=1)
    fig.add_hline(y=-1, line_dash="dot", line_color="gray", line_width=1)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_market_matrix_data(api_token, concentration_metric):
    """Fetch data for Market Matrix using dashboard instruments"""

    # Get all instruments dynamically
    key_instruments = get_key_instruments()
    all_instruments = []
    for category, instruments in key_instruments.items():
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


def create_dashboard_market_matrix(api_token, concentration_metric='conc_gross_le_4_tdr_long', category_filter='All Categories'):
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
        key_instruments = get_key_instruments()
        dashboard_instruments = []
        for category, instruments in key_instruments.items():
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

                # If insufficient 2-year data, use all available data as fallback
                if len(df_2yr) < 10:
                    # Use all available data if we have at least 10 records
                    if len(df) >= 10:
                        df_2yr = df.copy()
                        st.info(f"📊 Using all available data for {instrument} (insufficient 2-year history)")
                    else:
                        st.warning(f"⚠️ Skipping {instrument}: insufficient historical data ({len(df)} records)")
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
                key_instruments = get_key_instruments()
                ticker_map = get_ticker_mapping()
                name_map = get_ticker_to_name_mapping()

                category = None
                ticker = None
                for cat, instruments in key_instruments.items():
                    for tick, cot_name in instruments.items():
                        if cot_name == instrument:
                            category = cat
                            ticker = tick
                            break
                    if category:
                        break

                # If ticker not found, try to get it from the mapping
                if not ticker:
                    ticker = ticker_map.get(instrument)
                    if not ticker:
                        instrument_short = instrument.split(' - ')[0]
                        ticker = ticker_map.get(instrument_short)

                # Get the display name from the "name" field in JSON
                commodity_name = name_map.get(ticker, instrument.split(' - ')[0])

                # Apply category filter
                if category_filter == 'All Categories' or category == category_filter:
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
            st.warning(f"No data available for {category_filter if category_filter != 'All Categories' else 'selected instruments'}")
            return None

        # Show summary of included vs excluded instruments
        total_instruments = len(dashboard_instruments)
        included_count = len(scatter_data)
        excluded_count = total_instruments - included_count

        # Build info message with category filter info
        category_info = f" ({category_filter})" if category_filter != 'All Categories' else ""

        if excluded_count > 0:
            st.info(f"📊 Market Matrix: Showing {included_count}/{total_instruments} instruments{category_info} ({excluded_count} excluded due to data limitations)")
        else:
            if category_filter != 'All Categories':
                st.info(f"📊 Market Matrix: Showing {included_count} {category_filter} instruments")
            else:
                st.success(f"✅ Market Matrix: Showing all {included_count} instruments")

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


def display_positioning_concentration(api_token):
    """Display positioning concentration analysis for selected asset category"""

    st.markdown("---")
    st.subheader("Positioning Concentration Analysis")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Net positioning as percentage of open interest. Positive values indicate net long positions, negative values indicate net short positions."
        "</p>",
        unsafe_allow_html=True
    )

    # Get all categories
    key_instruments = get_key_instruments()
    all_categories = sorted(list(key_instruments.keys()))

    # Category selection
    col1_pos, col2_pos = st.columns([2, 6])
    with col1_pos:
        st.markdown("Select asset category:")
        selected_category = st.selectbox(
            "Category",
            options=all_categories,
            index=all_categories.index("Metals") if "Metals" in all_categories else 0,
            label_visibility="collapsed",
            key="positioning_category"
        )

    with col2_pos:
        st.markdown("Select trader category:")
        trader_category = st.selectbox(
            "Trader category",
            options=["Commercial", "Non-Commercial", "Non-Reportable"],
            index=0,
            label_visibility="collapsed",
            key="positioning_trader_category"
        )

    # Get all instruments for the selected category
    if selected_category in key_instruments:
        # Get COT instrument names for this category
        category_instruments = list(key_instruments[selected_category].values())

        if category_instruments:
            with st.spinner(f"Calculating positioning concentration for {selected_category} instruments..."):
                # Create positioning concentration charts
                # Note: instruments_db parameter is None since we're working with pre-loaded instruments
                fig_ts, fig_bar = create_positioning_concentration_charts(
                    category_instruments,
                    trader_category,
                    api_token,
                    None  # instruments_db not needed here
                )

            if fig_ts and fig_bar:
                # Display the two charts
                st.plotly_chart(fig_ts, use_container_width=True)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Download buttons
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("📥 Download Time Series Chart", key="download_positioning_ts"):
                        html_string = fig_ts.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="Download Time Series",
                            data=html_string,
                            file_name=f"cftc_positioning_timeseries_{selected_category}_{trader_category}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                            mime="text/html",
                            key="download_ts_btn"
                        )
                with col2:
                    if st.button("📥 Download Bar Chart", key="download_positioning_bar"):
                        html_string = fig_bar.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="Download Bar Chart",
                            data=html_string,
                            file_name=f"cftc_positioning_current_{selected_category}_{trader_category}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                            mime="text/html",
                            key="download_bar_btn"
                        )
            else:
                st.error("Unable to generate positioning concentration charts. Please check the data.")
        else:
            st.warning(f"No instruments found for category: {selected_category}")
    else:
        st.error(f"Category '{selected_category}' not found")


def display_strength_matrix(api_token):
    """Display relative strength matrix for selected asset category"""

    st.markdown("---")
    st.subheader("Relative Strength Matrix")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Positioning correlation matrix calculated using rolling window of Non-Commercial net positioning data."
        "</p>",
        unsafe_allow_html=True
    )

    # Get all categories
    key_instruments = get_key_instruments()
    all_categories = sorted(list(key_instruments.keys()))

    # Category and time period selection
    col1_str, col2_str = st.columns([2, 6])
    with col1_str:
        st.markdown("Select asset category:")
        selected_category = st.selectbox(
            "Category",
            options=all_categories,
            index=all_categories.index("Metals") if "Metals" in all_categories else 0,
            label_visibility="collapsed",
            key="strength_matrix_category"
        )

    with col2_str:
        st.markdown("Select time period:")
        time_period = st.selectbox(
            "Time period",
            options=["6 Months", "1 Year", "2 Years", "5 Years", "10 Years"],
            index=2,  # Default to 2 Years
            label_visibility="collapsed",
            key="strength_matrix_time_period"
        )

    # Get all instruments for the selected category
    if selected_category in key_instruments:
        # Get COT instrument names for this category
        category_instruments = list(key_instruments[selected_category].values())

        if category_instruments:
            with st.spinner(f"Calculating positioning correlations for {selected_category} instruments..."):
                # Create relative strength matrix
                fig = create_relative_strength_matrix(
                    category_instruments,
                    api_token,
                    time_period,
                    None  # instruments_db not needed here
                )

            if fig:
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Add explainer
                with st.expander("📊 Understanding the Strength Matrix", expanded=False):
                    st.markdown("""
                    **What This Matrix Shows:**
                    - Correlation between Non-Commercial net positioning (long - short) across different instruments
                    - Values range from -1 to +1, where:
                      - **+1** (dark blue): Perfect positive correlation - instruments move together
                      - **0** (white): No correlation - instruments move independently
                      - **-1** (dark red): Perfect negative correlation - instruments move opposite

                    **How to Use It:**
                    - **Portfolio Diversification**: Look for instruments with low or negative correlations
                    - **Risk Management**: High correlations mean similar market exposure
                    - **Trading Opportunities**: Divergence from typical correlations may signal opportunities
                    - **Market Regime**: Changing correlations can indicate shifts in market dynamics

                    **Example Interpretations:**
                    - If Gold and Silver show +0.8: They tend to move in the same direction
                    - If Oil and Bonds show -0.5: They often move in opposite directions
                    - If Wheat and Gold show 0.1: They have little relationship

                    **Time Period Impact:**
                    - Shorter periods (6M-1Y): Capture recent market dynamics and short-term relationships
                    - Medium periods (2Y): Balance between recent trends and historical patterns
                    - Longer periods (5Y-10Y): Show stable, long-term relationships and structural correlations
                    """)

                # Download button
                if st.button("📥 Download Strength Matrix", key="download_strength_matrix"):
                    html_string = fig.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        label="Download Chart",
                        data=html_string,
                        file_name=f"cftc_strength_matrix_{selected_category}_{time_period.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                        mime="text/html",
                        key="download_strength_btn"
                    )
            else:
                st.error("Unable to generate strength matrix. Please check the data.")
        else:
            st.warning(f"No instruments found for category: {selected_category}")
    else:
        st.error(f"Category '{selected_category}' not found")


def display_participation_comparison(api_token):
    """Display trader participation comparison for selected asset category"""

    st.markdown("---")
    st.subheader("Trader Participation Comparison")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Analyzes trader count trends, year-over-year changes, average positions per trader, and participation scores across instruments."
        "</p>",
        unsafe_allow_html=True
    )

    # Get all categories
    key_instruments = get_key_instruments()
    all_categories = sorted(list(key_instruments.keys()))

    # Category selection
    col1_part, col2_part = st.columns([2, 6])
    with col1_part:
        st.markdown("Select asset category:")
        selected_category = st.selectbox(
            "Category",
            options=all_categories,
            index=all_categories.index("Metals") if "Metals" in all_categories else 0,
            label_visibility="collapsed",
            key="participation_category"
        )

    # col2_part reserved for future controls (e.g., lookback period, aggregation method)

    # Get all instruments for the selected category
    if selected_category in key_instruments:
        # Get COT instrument names for this category
        category_instruments = list(key_instruments[selected_category].values())

        if category_instruments:
            with st.spinner(f"Analyzing trader participation for {selected_category} instruments..."):
                # Create participation comparison
                fig = create_cross_asset_participation_comparison(
                    category_instruments,
                    api_token,
                    None  # instruments_db not needed here
                )

            if fig:
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Add explainer
                with st.expander("📊 Understanding the Participation Charts", expanded=False):
                    st.markdown("""
                    **What These Charts Show:**

                    This analysis provides four complementary views of trader participation:

                    1. **Total Trader Count** (Top Left):
                       - Shows the absolute number of traders active in each market over time
                       - Higher numbers generally indicate more liquid, well-established markets
                       - Increasing trends suggest growing market interest

                    2. **Trader Count YoY % Change** (Top Right):
                       - Year-over-year percentage change in trader participation
                       - Positive values: Market is attracting new participants
                       - Negative values: Traders are leaving the market
                       - Volatility here may indicate market uncertainty or structural changes

                    3. **Avg Position per Trader** (Bottom Left):
                       - Average size of positions held by each trader (Open Interest ÷ Total Traders)
                       - Higher values suggest institutional participation or concentrated positions
                       - Lower values may indicate more retail participation
                       - Increasing trends can signal growing conviction or leverage

                    4. **Participation Score** (Bottom Right):
                       - Current trader count as % of historical maximum for each instrument
                       - 100% = At peak participation levels
                       - < 50% = Below median historical participation
                       - Helps identify if current activity is high or low relative to history

                    **How to Use This Analysis:**
                    - **Market Health**: Rising participation generally indicates healthy market interest
                    - **Contrarian Signals**: Extreme highs/lows in participation can signal potential reversals
                    - **Liquidity Assessment**: More traders typically means better liquidity
                    - **Cross-Market Comparison**: Compare participation trends across similar instruments
                    - **Market Maturity**: Stable, high participation suggests mature markets

                    **Example Interpretations:**
                    - Declining participation + rising prices = Potential weak rally
                    - Rising participation + sideways prices = Building energy for breakout
                    - Low participation score + high volatility = Thin market, risky conditions
                    - Increasing avg position size = Growing institutional interest
                    """)

                # Download button
                html_string = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="📥 Download Participation Chart",
                    data=html_string,
                    file_name=f"cftc_participation_{selected_category}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                    mime="text/html",
                    key="download_participation_chart"
                )
            else:
                st.error("Unable to generate participation comparison. Please check the data.")
        else:
            st.warning(f"No instruments found for category: {selected_category}")
    else:
        st.error(f"Category '{selected_category}' not found")


def display_dashboard(api_token):
    """Display the main dashboard overview"""
    st.header("Commodity Markets Overview")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Key metrics and positioning data for major commodity futures markets. Updated weekly with CFTC Commitments of Traders reports."
        "</p>",
        unsafe_allow_html=True
    )

    # Fetch dashboard data
    with st.spinner("Loading market data..."):
        df = fetch_dashboard_data(api_token)
    
    if df.empty:
        st.warning("No data available for dashboard instruments")
        return
    
    # Format the dataframe for display
    display_df = df.copy()
    
    # Format numeric columns
    numeric_cols = ['Open Interest', 'NC Net Position', 'Comm Long', 'Comm Short']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")

    # Format OI Change column with +/- signs and colors
    if 'OI Change' in display_df.columns:
        def format_oi_change(x):
            if pd.isna(x):
                return ""
            sign = "+" if x >= 0 else ""
            return f"{sign}{x:,.0f}"
        display_df['OI Change'] = display_df['OI Change'].apply(format_oi_change)
    
    # Format percentile columns
    percentile_cols = ['NC Net YTD %ile', 'Comm Long YTD %ile', 'Comm Short YTD %ile']
    for col in percentile_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")

    # Format enhanced correlation columns
    correlation_cols = [
        'NC Corr_P', 'NC Corr_S', 'NC L_L',
        'CL Corr_P', 'CL Corr_S', 'CL L_L',
        'CS Corr_P', 'CS Corr_S', 'CS L_L'
    ]
    for col in correlation_cols:
        if col in display_df.columns:
            def format_correlation(x):
                if pd.isna(x) or x is None:
                    return ""
                try:
                    return f"{float(x):.2f}"
                except (ValueError, TypeError):
                    return ""
            display_df[col] = display_df[col].apply(format_correlation)
    
    # Format concentration columns
    conc_cols = ['Conc Long 4T', 'Conc Short 4T']
    for col in conc_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    
    # Configure column display with LineChartColumn for sparklines
    column_config = {
        "Category": st.column_config.TextColumn("Category", width="small"),
        "Instrument": st.column_config.TextColumn("Instrument", width="medium"),
        "Open Interest": st.column_config.TextColumn("OI", help="Current Open Interest from Futures Data"),
        "OI Change": st.column_config.TextColumn("OI Δ", help="Daily Change in Open Interest"),
        "NC Net Position": st.column_config.TextColumn("NC Net", help="Non-Commercial Net Position"),
        "NC Net YTD %ile": st.column_config.TextColumn("YTD %", help="YTD Percentile"),
        "NC Corr_P": st.column_config.TextColumn("Corr_P", help="Pearson: NC Position Changes vs Weekly Returns", width="small"),
        "NC Corr_S": st.column_config.TextColumn("Corr_S", help="Spearman: NC Position Changes vs Weekly Returns", width="small"),
        "NC L_L": st.column_config.TextColumn("L_L", help="Lead-Lag: Best NC Position Changes vs Returns (±2wks)", width="small"),
        "Comm Long": st.column_config.TextColumn("Comm L", help="Commercial Long Positions"),
        "Comm Long YTD %ile": st.column_config.TextColumn("YTD %", help="YTD Percentile"),
        "CL Corr_P": st.column_config.TextColumn("Corr_P", help="Pearson: Comm Long Changes vs Weekly Returns", width="small"),
        "CL Corr_S": st.column_config.TextColumn("Corr_S", help="Spearman: Comm Long Changes vs Weekly Returns", width="small"),
        "CL L_L": st.column_config.TextColumn("L_L", help="Lead-Lag: Best Comm Long Changes vs Returns (±2wks)", width="small"),
        "Comm Short": st.column_config.TextColumn("Comm S", help="Commercial Short Positions"),
        "Comm Short YTD %ile": st.column_config.TextColumn("YTD %", help="YTD Percentile"),
        "CS Corr_P": st.column_config.TextColumn("Corr_P", help="Pearson: Comm Short Changes vs Weekly Returns", width="small"),
        "CS Corr_S": st.column_config.TextColumn("Corr_S", help="Spearman: Comm Short Changes vs Weekly Returns", width="small"),
        "CS L_L": st.column_config.TextColumn("L_L", help="Lead-Lag: Best Comm Short Changes vs Returns (±2wks)", width="small"),
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
    
    # Sort by custom category order: Index, Financial, Metals, Energy, Currency, Agricultural
    category_order = ['Index', 'Financial', 'Metals', 'Energy', 'Currency', 'Agricultural']
    display_df['category_sort'] = display_df['Category'].map({cat: i for i, cat in enumerate(category_order)})
    display_df = display_df.sort_values(['category_sort', 'Instrument'], ascending=[True, True])
    display_df = display_df.drop('category_sort', axis=1)
    
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

    # Add correlation explanation note below the table
    st.markdown(
        """
        <p style='color: #888888; font-size: 0.9em; margin-top: 10px;'>
        <b>Correlation Metrics:</b> Corr_P = Pearson correlation (linear relationship), Corr_S = Spearman correlation (monotonic relationship), L_L = Lead-Lag correlation (best timing across ±2 weeks).
        All correlations are calculated using a rolling 2-year window between weekly position changes and weekly futures returns.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Add Cross-Asset Z-Score Comparison below the table
    st.markdown("---")
    st.subheader("Cross-Asset Futures Positioning")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Z-scores show how current positioning compares to historical average. Values above +2 or below -2 indicate extreme positioning."
        "</p>",
        unsafe_allow_html=True
    )

    # Controls row
    st.markdown("Select trader category:")
    trader_category = st.selectbox(
        "Trader category",
        ["Non-Commercial Net", "Commercial Net", "Non-Reportable Net"],
        index=0,
        label_visibility="collapsed",
        key="dashboard_trader_category"
    )

    # Display both charts (Raw and % of OI)
    display_cross_asset_zscore(api_token, trader_category)

    # Add Week-over-Week Changes section
    st.markdown("---")
    st.subheader("Week-over-Week Position Changes")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Shows the weekly change in positioning as percentage of open interest. Positive values indicate increasing positions, negative indicate decreasing positions."
        "</p>",
        unsafe_allow_html=True
    )

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
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Percentile-based view of trader participation versus positioning concentration. Each instrument is ranked relative to its own 5-year history."
        "</p>",
        unsafe_allow_html=True
    )

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

    with col2_matrix:
        st.markdown("Filter by category:")
        # Get all categories dynamically from the instruments
        key_instruments = get_key_instruments()
        all_categories = ["All Categories"] + sorted(list(key_instruments.keys()))

        matrix_category_filter = st.selectbox(
            "Matrix category filter",
            options=all_categories,
            index=0,
            label_visibility="collapsed",
            key="matrix_category_filter"
        )

    # Display the Market Matrix chart
    matrix_fig = create_dashboard_market_matrix(api_token, matrix_concentration_metric, matrix_category_filter)
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

    # Add Positioning Concentration Analysis section
    display_positioning_concentration(api_token)

    # Add Strength Matrix section
    display_strength_matrix(api_token)

    # Add Participation Comparison section
    display_participation_comparison(api_token)

    # Add refresh button
    if st.button("🔄 Refresh Dashboard Data"):
        st.cache_data.clear()
        st.rerun()