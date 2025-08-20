"""
Historical COT data loader for pre-2000 data
Handles loading and processing of FUT86_16.txt file
"""
import pandas as pd
import streamlit as st
from datetime import datetime
import os

# Cache the historical data loading
@st.cache_data
def load_historical_data():
    """Load and parse the historical COT data file (1986-2016)"""
    historical_file = 'instrument_management/FUT86_16.txt'
    
    if not os.path.exists(historical_file):
        return None
    
    try:
        # Read the CSV file with proper column mapping
        # The file has many columns but we only need the ones matching our API
        df = pd.read_csv(historical_file, low_memory=False)
        
        # Rename columns to match our API column names
        column_mapping = {
            'Market and Exchange Names': 'market_and_exchange_names',
            'As of Date in Form YYYY-MM-DD': 'report_date_as_yyyy_mm_dd',
            'Open Interest (All)': 'open_interest_all',
            'Noncommercial Positions-Long (All)': 'noncomm_positions_long_all',
            'Noncommercial Positions-Short (All)': 'noncomm_positions_short_all',
            'Noncommercial Positions-Spreading (All)': 'noncomm_postions_spread_all',
            'Commercial Positions-Long (All)': 'comm_positions_long_all',
            'Commercial Positions-Short (All)': 'comm_positions_short_all',
            'Total Reportable Positions-Long (All)': 'tot_rept_positions_long_all',
            'Total Reportable Positions-Short (All)': 'tot_rept_positions_short',
            'Nonreportable Positions-Long (All)': 'nonrept_positions_long_all',
            'Nonreportable Positions-Short (All)': 'nonrept_positions_short_all',
            # Trader counts
            'Traders-Total (All)': 'traders_tot_all',
            'Traders-Noncommercial-Long (All)': 'traders_noncomm_long_all',
            'Traders-Noncommercial-Short (All)': 'traders_noncomm_short_all',
            'Traders-Noncommercial-Spreading (All)': 'traders_noncomm_spread_all',
            'Traders-Commercial-Long (All)': 'traders_comm_long_all',
            'Traders-Commercial-Short (All)': 'traders_comm_short_all',
            'Traders-Total Reportable-Long (All)': 'traders_tot_rept_long_all',
            'Traders-Total Reportable-Short (All)': 'traders_tot_rept_short_all',
            # Concentration
            'Concentration-Gross LT = 4 TDR-Long (All)': 'conc_gross_le_4_tdr_long',
            'Concentration-Gross LT =4 TDR-Short (All)': 'conc_gross_le_4_tdr_short',
            'Concentration-Gross LT =8 TDR-Long (All)': 'conc_gross_le_8_tdr_long',
            'Concentration-Gross LT =8 TDR-Short (All)': 'conc_gross_le_8_tdr_short',
            'Concentration-Net LT =4 TDR-Long (All)': 'conc_net_le_4_tdr_long_all',
            'Concentration-Net LT =4 TDR-Short (All)': 'conc_net_le_4_tdr_short_all',
            'Concentration-Net LT =8 TDR-Long (All)': 'conc_net_le_8_tdr_long_all',
            'Concentration-Net LT =8 TDR-Short (All)': 'conc_net_le_8_tdr_short_all',
            # Percentage columns
            '% of OI-Noncommercial-Long (All)': 'pct_of_oi_noncomm_long_all',
            '% of OI-Noncommercial-Short (All)': 'pct_of_oi_noncomm_short_all',
            '% of OI-Noncommercial-Spreading (All)': 'pct_of_oi_noncomm_spread',
            '% of OI-Commercial-Long (All)': 'pct_of_oi_comm_long_all',
            '% of OI-Commercial-Short (All)': 'pct_of_oi_comm_short_all',
            '% of OI-Total Reportable-Long (All)': 'pct_of_oi_tot_rept_long_all',
            '% of OI-Total Reportable-Short (All)': 'pct_of_oi_tot_rept_short',
            '% of OI-Nonreportable-Long (All)': 'pct_of_oi_nonrept_long_all',
            '% of OI-Nonreportable-Short (All)': 'pct_of_oi_nonrept_short_all',
            # Change columns
            'Change in Open Interest (All)': 'change_in_open_interest_all',
            'Change in Noncommercial-Long (All)': 'change_in_noncomm_long_all',
            'Change in Noncommercial-Short (All)': 'change_in_noncomm_short_all',
            'Change in Noncommercial-Spreading (All)': 'change_in_noncomm_spead_all',
            'Change in Commercial-Long (All)': 'change_in_comm_long_all',
            'Change in Commercial-Short (All)': 'change_in_comm_short_all',
            'Change in Total Reportable-Long (All)': 'change_in_tot_rept_long_all',
            'Change in Total Reportable-Short (All)': 'change_in_tot_rept_short',
            'Change in Nonreportable-Long (All)': 'change_in_nonrept_long_all',
            'Change in Nonreportable-Short (All)': 'change_in_nonrept_short_all'
        }
        
        # Rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)
        
        # Convert date column
        df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
        
        # Clean numeric columns (remove spaces and convert)
        numeric_columns = [col for col in df.columns if col in column_mapping.values() and 
                          col not in ['market_and_exchange_names', 'report_date_as_yyyy_mm_dd']]
        
        for col in numeric_columns:
            if col in df.columns:
                # Remove spaces and convert to numeric
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
        
        return df
        
    except Exception as e:
        st.warning(f"Could not load historical data: {e}")
        return None


@st.cache_data
def get_historical_data_for_instrument(instrument_name, end_date=None):
    """
    Get historical data for a specific instrument up to a certain date
    
    Args:
        instrument_name: The instrument name (without contract code)
        end_date: The cutoff date (fetch data before this date)
    
    Returns:
        DataFrame with historical data or None if not found
    """
    # Load all historical data
    historical_df = load_historical_data()
    
    if historical_df is None:
        return None
    
    # Filter for the specific instrument
    instrument_data = historical_df[
        historical_df['market_and_exchange_names'] == instrument_name
    ].copy()
    
    if instrument_data.empty:
        return None
    
    # Filter by end date if provided
    if end_date:
        instrument_data = instrument_data[
            instrument_data['report_date_as_yyyy_mm_dd'] < pd.to_datetime(end_date)
        ]
    
    # Sort by date
    instrument_data = instrument_data.sort_values('report_date_as_yyyy_mm_dd')
    
    # Calculate net positions (same as in fetch_cftc_data)
    if 'noncomm_positions_long_all' in instrument_data.columns and 'noncomm_positions_short_all' in instrument_data.columns:
        instrument_data['net_noncomm_positions'] = (
            instrument_data['noncomm_positions_long_all'] - instrument_data['noncomm_positions_short_all']
        )
    
    if 'comm_positions_long_all' in instrument_data.columns and 'comm_positions_short_all' in instrument_data.columns:
        instrument_data['net_comm_positions'] = (
            instrument_data['comm_positions_long_all'] - instrument_data['comm_positions_short_all']
        )
    
    if 'tot_rept_positions_long_all' in instrument_data.columns and 'tot_rept_positions_short' in instrument_data.columns:
        instrument_data['net_reportable_positions'] = (
            instrument_data['tot_rept_positions_long_all'] - instrument_data['tot_rept_positions_short']
        )
    
    return instrument_data


def check_data_gap(historical_end_date, api_start_date, max_gap_years=2):
    """
    Check if there's a gap between historical data end and API data start
    
    Args:
        historical_end_date: Last date in historical data
        api_start_date: First date in API data
        max_gap_years: Maximum acceptable gap in years
    
    Returns:
        tuple: (has_gap, gap_message)
    """
    if historical_end_date is None or api_start_date is None:
        return False, None
    
    gap = (api_start_date - historical_end_date).days / 365.25
    
    if gap > max_gap_years:
        message = (
            f"⚠️ Data gap detected: Historical data ends {historical_end_date.strftime('%Y-%m-%d')}, "
            f"API data starts {api_start_date.strftime('%Y-%m-%d')} ({gap:.1f} year gap). "
            "Data may be incomplete."
        )
        return True, message
    
    return False, None