"""
Configuration settings for CFTC COT Data Dashboard
"""

# Page configuration
PAGE_CONFIG = {
    "page_title": "CFTC COT Data Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# API settings
CFTC_API_BASE = "publicreporting.cftc.gov"
DATASET_CODE = "6dca-aqww"
DEFAULT_LIMIT = 3000

# Chart settings
CHART_HEIGHT = 600
PARTICIPATION_CHART_HEIGHT = 900

# Color schemes
CONCENTRATION_COLORS = {
    "low": "green",      # <15%
    "medium": "yellow",  # 15-25%
    "high": "red"        # >25%
}

# Data columns
CFTC_COLUMNS = [
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