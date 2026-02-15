"""
Historical COT data loader for Disaggregated report (2006-2016)
Handles loading and processing of F_Disagg06_16.txt file
"""
import pandas as pd
import streamlit as st
import os


@st.cache_data
def load_disagg_historical_data():
    """Load and parse the disaggregated historical COT data file (2006-2016)"""
    historical_file = 'instrument_management/DisaggregatedF/F_Disagg06_16.txt'

    if not os.path.exists(historical_file):
        return None

    try:
        df = pd.read_csv(historical_file, low_memory=False)

        # Map file column names (Title_Case) to API column names (lowercase)
        column_mapping = {
            'Market_and_Exchange_Names': 'market_and_exchange_names',
            'Report_Date_as_YYYY-MM-DD': 'report_date_as_yyyy_mm_dd',
            # Core positions - All
            'Open_Interest_All': 'open_interest_all',
            'Prod_Merc_Positions_Long_All': 'prod_merc_positions_long',
            'Prod_Merc_Positions_Short_All': 'prod_merc_positions_short',
            'Swap_Positions_Long_All': 'swap_positions_long_all',
            'Swap__Positions_Short_All': 'swap__positions_short_all',
            'Swap__Positions_Spread_All': 'swap__positions_spread_all',
            'M_Money_Positions_Long_All': 'm_money_positions_long_all',
            'M_Money_Positions_Short_All': 'm_money_positions_short_all',
            'M_Money_Positions_Spread_All': 'm_money_positions_spread',
            'Other_Rept_Positions_Long_All': 'other_rept_positions_long',
            'Other_Rept_Positions_Short_All': 'other_rept_positions_short',
            'Other_Rept_Positions_Spread_All': 'other_rept_positions_spread',
            'Tot_Rept_Positions_Long_All': 'tot_rept_positions_long_all',
            'Tot_Rept_Positions_Short_All': 'tot_rept_positions_short',
            'NonRept_Positions_Long_All': 'nonrept_positions_long_all',
            'NonRept_Positions_Short_All': 'nonrept_positions_short_all',
            # Change columns
            'Change_in_Open_Interest_All': 'change_in_open_interest_all',
            'Change_in_Prod_Merc_Long_All': 'change_in_prod_merc_long',
            'Change_in_Prod_Merc_Short_All': 'change_in_prod_merc_short',
            'Change_in_Swap_Long_All': 'change_in_swap_long_all',
            'Change_in_Swap_Short_All': 'change_in_swap_short_all',
            'Change_in_Swap_Spread_All': 'change_in_swap_spread_all',
            'Change_in_M_Money_Long_All': 'change_in_m_money_long_all',
            'Change_in_M_Money_Short_All': 'change_in_m_money_short_all',
            'Change_in_M_Money_Spread_All': 'change_in_m_money_spread',
            'Change_in_Other_Rept_Long_All': 'change_in_other_rept_long',
            'Change_in_Other_Rept_Short_All': 'change_in_other_rept_short',
            'Change_in_Other_Rept_Spread_All': 'change_in_other_rept_spread',
            'Change_in_Tot_Rept_Long_All': 'change_in_tot_rept_long_all',
            'Change_in_Tot_Rept_Short_All': 'change_in_tot_rept_short',
            'Change_in_NonRept_Long_All': 'change_in_nonrept_long_all',
            'Change_in_NonRept_Short_All': 'change_in_nonrept_short_all',
            # Percentage of OI
            'Pct_of_OI_Prod_Merc_Long_All': 'pct_of_oi_prod_merc_long',
            'Pct_of_OI_Prod_Merc_Short_All': 'pct_of_oi_prod_merc_short',
            'Pct_of_OI_Swap_Long_All': 'pct_of_oi_swap_long_all',
            'Pct_of_OI_Swap_Short_All': 'pct_of_oi_swap_short_all',
            'Pct_of_OI_Swap_Spread_All': 'pct_of_oi_swap_spread_all',
            'Pct_of_OI_M_Money_Long_All': 'pct_of_oi_m_money_long_all',
            'Pct_of_OI_M_Money_Short_All': 'pct_of_oi_m_money_short_all',
            'Pct_of_OI_M_Money_Spread_All': 'pct_of_oi_m_money_spread',
            'Pct_of_OI_Other_Rept_Long_All': 'pct_of_oi_other_rept_long',
            'Pct_of_OI_Other_Rept_Short_All': 'pct_of_oi_other_rept_short',
            'Pct_of_OI_Other_Rept_Spread_All': 'pct_of_oi_other_rept_spread',
            'Pct_of_OI_Tot_Rept_Long_All': 'pct_of_oi_tot_rept_long_all',
            'Pct_of_OI_Tot_Rept_Short_All': 'pct_of_oi_tot_rept_short',
            'Pct_of_OI_NonRept_Long_All': 'pct_of_oi_nonrept_long_all',
            'Pct_of_OI_NonRept_Short_All': 'pct_of_oi_nonrept_short_all',
            # Trader counts
            'Traders_Tot_All': 'traders_tot_all',
            'Traders_Prod_Merc_Long_All': 'traders_prod_merc_long_all',
            'Traders_Prod_Merc_Short_All': 'traders_prod_merc_short_all',
            'Traders_Swap_Long_All': 'traders_swap_long_all',
            'Traders_Swap_Short_All': 'traders_swap_short_all',
            'Traders_Swap_Spread_All': 'traders_swap_spread_all',
            'Traders_M_Money_Long_All': 'traders_m_money_long_all',
            'Traders_M_Money_Short_All': 'traders_m_money_short_all',
            'Traders_M_Money_Spread_All': 'traders_m_money_spread_all',
            'Traders_Other_Rept_Long_All': 'traders_other_rept_long_all',
            'Traders_Other_Rept_Short_All': 'traders_other_rept_short',
            'Traders_Other_Rept_Spread_All': 'traders_other_rept_spread',
            'Traders_Tot_Rept_Long_All': 'traders_tot_rept_long_all',
            'Traders_Tot_Rept_Short_All': 'traders_tot_rept_short_all',
            # Concentration ratios
            'Conc_Gross_LE_4_TDR_Long_All': 'conc_gross_le_4_tdr_long',
            'Conc_Gross_LE_4_TDR_Short_All': 'conc_gross_le_4_tdr_short',
            'Conc_Gross_LE_8_TDR_Long_All': 'conc_gross_le_8_tdr_long',
            'Conc_Gross_LE_8_TDR_Short_All': 'conc_gross_le_8_tdr_short',
            'Conc_Net_LE_4_TDR_Long_All': 'conc_net_le_4_tdr_long_all',
            'Conc_Net_LE_4_TDR_Short_All': 'conc_net_le_4_tdr_short_all',
            'Conc_Net_LE_8_TDR_Long_All': 'conc_net_le_8_tdr_long_all',
            'Conc_Net_LE_8_TDR_Short_All': 'conc_net_le_8_tdr_short_all',
        }

        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)

        # Convert date
        df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])

        # Convert numeric columns
        numeric_columns = [col for col in df.columns if col in column_mapping.values()
                          and col not in ('market_and_exchange_names', 'report_date_as_yyyy_mm_dd')]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')

        return df

    except Exception as e:
        st.warning(f"Could not load disaggregated historical data: {e}")
        return None


@st.cache_data
def get_disagg_historical_data_for_instrument(instrument_name, end_date=None):
    """
    Get historical disaggregated data for a specific instrument up to a certain date.

    Args:
        instrument_name: The instrument name as it appears in F_Disagg06_16.txt
        end_date: The cutoff date (fetch data before this date)

    Returns:
        DataFrame with historical data or None if not found
    """
    historical_df = load_disagg_historical_data()

    if historical_df is None:
        return None

    instrument_data = historical_df[
        historical_df['market_and_exchange_names'] == instrument_name
    ].copy()

    if instrument_data.empty:
        return None

    if end_date:
        instrument_data = instrument_data[
            instrument_data['report_date_as_yyyy_mm_dd'] < pd.to_datetime(end_date)
        ]

    instrument_data = instrument_data.sort_values('report_date_as_yyyy_mm_dd')

    # Calculate derived net positions
    if 'm_money_positions_long_all' in instrument_data.columns and 'm_money_positions_short_all' in instrument_data.columns:
        instrument_data['net_mm_positions'] = (
            instrument_data['m_money_positions_long_all'] - instrument_data['m_money_positions_short_all']
        )
    if 'prod_merc_positions_long' in instrument_data.columns and 'prod_merc_positions_short' in instrument_data.columns:
        instrument_data['net_pm_positions'] = (
            instrument_data['prod_merc_positions_long'] - instrument_data['prod_merc_positions_short']
        )
    if 'swap_positions_long_all' in instrument_data.columns and 'swap__positions_short_all' in instrument_data.columns:
        instrument_data['net_swap_positions'] = (
            instrument_data['swap_positions_long_all'] - instrument_data['swap__positions_short_all']
        )
    if 'other_rept_positions_long' in instrument_data.columns and 'other_rept_positions_short' in instrument_data.columns:
        instrument_data['net_other_positions'] = (
            instrument_data['other_rept_positions_long'] - instrument_data['other_rept_positions_short']
        )

    return instrument_data