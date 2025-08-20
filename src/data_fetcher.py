"""
Data fetching and processing module for CFTC COT data
"""
import streamlit as st
import pandas as pd
from sodapy import Socrata
import json
from config import CFTC_API_BASE, DATASET_CODE, DEFAULT_LIMIT, CFTC_COLUMNS
from historical_data_loader import get_historical_data_for_instrument, check_data_gap


@st.cache_data
def load_instruments_database():
    """Load the instruments JSON database"""
    try:
        # Try multiple locations for backward compatibility
        json_paths = [
            'data/instruments_LegacyF.json',  # New location
            'instruments_LegacyF.json',        # Legacy location
            '../data/instruments_LegacyF.json' # Alternative path
        ]
        
        for path in json_paths:
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                continue
                
        st.error("❌ instruments_LegacyF.json file not found. Please ensure it's in the data/ directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading instruments database: {e}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_cftc_data(instrument_name, api_token):
    """Fetch CFTC data for a specific instrument with historical data stitching"""
    try:
        # Strip the contract code if present (everything after the last space and parenthesis)
        # Format: "INSTRUMENT NAME (CODE)" -> "INSTRUMENT NAME"
        if ' (' in instrument_name and instrument_name.endswith(')'):
            # Find the last occurrence of ' (' which indicates the start of the code
            instrument_name_clean = instrument_name.rsplit(' (', 1)[0]
        else:
            instrument_name_clean = instrument_name
            
        client = Socrata(CFTC_API_BASE, api_token)

        # Special handling for WTI-PHYSICAL: merge with historical CRUDE OIL, LIGHT SWEET data
        if instrument_name_clean == "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE":
            # First, get historical data from CRUDE OIL, LIGHT SWEET (2000-2022)
            historical_results = client.get(
                DATASET_CODE,
                where="market_and_exchange_names='CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE'",
                select=",".join(CFTC_COLUMNS),
                order="report_date_as_yyyy_mm_dd ASC",
                limit=DEFAULT_LIMIT
            )
            
            # Then get current data from WTI-PHYSICAL (2022-present)
            current_results = client.get(
                DATASET_CODE,
                where="market_and_exchange_names='WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE'",
                select=",".join(CFTC_COLUMNS),
                order="report_date_as_yyyy_mm_dd ASC",
                limit=DEFAULT_LIMIT
            )
            
            # Merge the results
            results = historical_results + current_results
            
        else:
            # Standard fetch for all other instruments
            results = client.get(
                DATASET_CODE,
                where=f"market_and_exchange_names='{instrument_name_clean}'",
                select=",".join(CFTC_COLUMNS),
                order="report_date_as_yyyy_mm_dd ASC",
                limit=DEFAULT_LIMIT
            )

        client.close()

        if not results:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Convert date column
        df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
        
        # Remove duplicates based on date (in case of overlapping data at transition)
        df = df.drop_duplicates(subset=['report_date_as_yyyy_mm_dd'], keep='last')

        # Convert numeric columns
        numeric_columns = [col for col in CFTC_COLUMNS if
                           col != "report_date_as_yyyy_mm_dd" and col != "market_and_exchange_names"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate additional metrics
        df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
        df['net_comm_positions'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
        df['net_reportable_positions'] = df['tot_rept_positions_long_all'] - df['tot_rept_positions_short']
        
        # Now try to stitch historical data (pre-2000) if available
        # For WTI, we already handle 2000-2022, so check for pre-2000 data
        api_start_date = df['report_date_as_yyyy_mm_dd'].min()
        
        # Try to get historical data for this instrument
        # For WTI special case, try multiple historical names
        historical_df = None
        if instrument_name_clean == "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE":
            # Try the pre-2000 name first (with quotes around SWEET)
            historical_df = get_historical_data_for_instrument(
                "CRUDE OIL, LIGHT 'SWEET' - NEW YORK MERCANTILE EXCHANGE", 
                end_date=api_start_date
            )
            # If not found, try the 2000-2016 name (without quotes)
            if historical_df is None or historical_df.empty:
                historical_df = get_historical_data_for_instrument(
                    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
                    end_date=api_start_date
                )
        else:
            # For all other instruments, use the clean name
            historical_df = get_historical_data_for_instrument(
                instrument_name_clean, 
                end_date=api_start_date
            )
        
        if historical_df is not None and not historical_df.empty:
            # Check for gap between historical and API data
            historical_end_date = historical_df['report_date_as_yyyy_mm_dd'].max()
            has_gap, gap_message = check_data_gap(historical_end_date, api_start_date)
            
            if has_gap:
                # Show warning about data gap
                st.warning(gap_message)
            else:
                # Combine historical and API data
                # Ensure columns match
                common_columns = list(set(df.columns) & set(historical_df.columns))
                df = pd.concat([historical_df[common_columns], df[common_columns]], ignore_index=True)
                
                # Remove any duplicates that might occur at the boundary
                df = df.drop_duplicates(subset=['report_date_as_yyyy_mm_dd'], keep='last')
                
                # Recalculate metrics if needed
                if 'net_noncomm_positions' not in df.columns:
                    if 'noncomm_positions_long_all' in df.columns and 'noncomm_positions_short_all' in df.columns:
                        df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
                if 'net_comm_positions' not in df.columns:
                    if 'comm_positions_long_all' in df.columns and 'comm_positions_short_all' in df.columns:
                        df['net_comm_positions'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
                if 'net_reportable_positions' not in df.columns:
                    if 'tot_rept_positions_long_all' in df.columns and 'tot_rept_positions_short' in df.columns:
                        df['net_reportable_positions'] = df['tot_rept_positions_long_all'] - df['tot_rept_positions_short']

        return df.sort_values('report_date_as_yyyy_mm_dd')

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


