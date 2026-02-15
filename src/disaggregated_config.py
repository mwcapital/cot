"""
Configuration settings for Disaggregated COT Report
Dataset: 72hh-3qpy (CFTC Disaggregated Futures)
"""

# API settings
CFTC_API_BASE = "publicreporting.cftc.gov"
DISAGG_DATASET_CODE = "72hh-3qpy"
DEFAULT_LIMIT = 3000

# Data columns for Disaggregated Futures report
# NOTE: API has quirks - double underscores on swap short/spread, missing _all suffixes on some columns
DISAGG_COLUMNS = [
    "report_date_as_yyyy_mm_dd",
    "market_and_exchange_names",
    # Open Interest
    "open_interest_all",
    "change_in_open_interest_all",
    # Producer/Merchant positions (no _all suffix in API)
    "prod_merc_positions_long",
    "prod_merc_positions_short",
    # Swap Dealer positions (double underscore on short/spread!)
    "swap_positions_long_all",
    "swap__positions_short_all",
    "swap__positions_spread_all",
    # Managed Money positions
    "m_money_positions_long_all",
    "m_money_positions_short_all",
    "m_money_positions_spread",
    # Other Reportables (no _all suffix in API)
    "other_rept_positions_long",
    "other_rept_positions_short",
    "other_rept_positions_spread",
    # Total Reportable
    "tot_rept_positions_long_all",
    "tot_rept_positions_short",
    # Non-Reportable
    "nonrept_positions_long_all",
    "nonrept_positions_short_all",
    # Trader counts
    "traders_tot_all",
    "traders_prod_merc_long_all",
    "traders_prod_merc_short_all",
    "traders_swap_long_all",
    "traders_swap_short_all",
    "traders_swap_spread_all",
    "traders_m_money_long_all",
    "traders_m_money_short_all",
    "traders_m_money_spread_all",
    "traders_other_rept_long_all",
    "traders_other_rept_short",
    "traders_other_rept_spread",
    "traders_tot_rept_long_all",
    "traders_tot_rept_short_all",
    # Concentration
    "conc_gross_le_4_tdr_long",
    "conc_gross_le_4_tdr_short",
    "conc_gross_le_8_tdr_long",
    "conc_gross_le_8_tdr_short",
    "conc_net_le_4_tdr_long_all",
    "conc_net_le_4_tdr_short_all",
    "conc_net_le_8_tdr_long_all",
    "conc_net_le_8_tdr_short_all",
    # Percentage of OI
    "pct_of_oi_prod_merc_long",
    "pct_of_oi_prod_merc_short",
    "pct_of_oi_swap_long_all",
    "pct_of_oi_swap_short_all",
    "pct_of_oi_m_money_long_all",
    "pct_of_oi_m_money_short_all",
    "pct_of_oi_other_rept_long",
    "pct_of_oi_other_rept_short",
    # Change columns
    "change_in_prod_merc_long",
    "change_in_prod_merc_short",
    "change_in_swap_long_all",
    "change_in_swap_short_all",
    "change_in_m_money_long_all",
    "change_in_m_money_short_all",
    "change_in_other_rept_long",
    "change_in_other_rept_short",
    "change_in_tot_rept_long_all",
    "change_in_tot_rept_short",
    "change_in_nonrept_long_all",
    "change_in_nonrept_short_all",
]