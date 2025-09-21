# CFTC API Column Name Mapping Reference

## CRITICAL: Use these EXACT column names from the CFTC API

### Complete Column Mapping
This is the authoritative list of all CFTC API column names with their actual API field names (left) and display names (right).

```
id                                  ID
market_and_exchange_names           Market_and_Exchange_Names
report_date_as_yyyy_mm_dd          Report_Date_as_YYYY_MM_DD
yyyy_report_week_ww                YYYY Report Week WW
contract_market_name               CONTRACT_MARKET_NAME
cftc_contract_market_code          CFTC_Contract_Market_Code
cftc_market_code                   CFTC_Market_Code
cftc_region_code                   CFTC_Region_Code
cftc_commodity_code                CFTC_Commodity_Code
commodity_name                     Commodity Name
```

### Core Position Columns
```
open_interest_all                  Open_Interest_All
noncomm_positions_long_all         NonComm_Positions_Long_All
noncomm_positions_short_all        NonComm_Positions_Short_All
noncomm_postions_spread_all        NonComm_Postions_Spread_All        # Note: API has typo "postions"
comm_positions_long_all            Comm_Positions_Long_All
comm_positions_short_all           Comm_Positions_Short_All
tot_rept_positions_long_all        Tot_Rept_Positions_Long_All
tot_rept_positions_short            Tot_Rept_Positions_Short_All
nonrept_positions_long_all         NonRept_Positions_Long_All
nonrept_positions_short_all        NonRept_Positions_Short_All
```

### Old Positions
```
open_interest_old                  Open_Interest_Old
noncomm_positions_long_old         NonComm_Positions_Long_Old
noncomm_positions_short_old        NonComm_Positions_Short_Old
noncomm_positions_spread           NonComm_Positions_Spread_Old
comm_positions_long_old            Comm_Positions_Long_Old
comm_positions_short_old           Comm_Positions_Short_Old
tot_rept_positions_long_old        Tot_Rept_Positions_Long_Old
tot_rept_positions_short_1         Tot_Rept_Positions_Short_Old
nonrept_positions_long_old         NonRept_Positions_Long_Old
nonrept_positions_short_old        NonRept_Positions_Short_Old
```

### Other Positions
```
open_interest_other                Open_Interest_Other
noncomm_positions_long_other       NonComm_Positions_Long_Other
noncomm_positions_short_other      NonComm_Positions_Short_Other
noncomm_positions_spread_1         NonComm_Positions_Spread_Other
comm_positions_long_other          Comm_Positions_Long_Other
comm_positions_short_other         Comm_Positions_Short_Other
tot_rept_positions_long_other      Tot_Rept_Positions_Long_Other
tot_rept_positions_short_2         Tot_Rept_Positions_Short_Other
nonrept_positions_long_other       NonRept_Positions_Long_Other
nonrept_positions_short_other      NonRept_Positions_Short_Other
```

### Change Columns
```
change_in_open_interest_all        Change_in_Open_Interest_All
change_in_noncomm_long_all         Change_in_NonComm_Long_All
change_in_noncomm_short_all        Change_in_NonComm_Short_All
change_in_noncomm_spead_all        Change_in_NonComm_Spead_All        # Note: API has typo "spead"
change_in_comm_long_all            Change_in_Comm_Long_All
change_in_comm_short_all           Change_in_Comm_Short_All
change_in_tot_rept_long_all        Change_in_Tot_Rept_Long_All
change_in_tot_rept_short           Change_in_Tot_Rept_Short_All
change_in_nonrept_long_all         Change_in_NonRept_Long_All
change_in_nonrept_short_all        Change_in_NonRept_Short_All
```

### Percentage of Open Interest - All
```
pct_of_open_interest_all           Pct_of_Open_Interest_All
pct_of_oi_noncomm_long_all         Pct_of_OI_NonComm_Long_All
pct_of_oi_noncomm_short_all        Pct_of_OI_NonComm_Short_All
pct_of_oi_noncomm_spread           Pct_of_OI_NonComm_Spread_All
pct_of_oi_comm_long_all            Pct_of_OI_Comm_Long_All
pct_of_oi_comm_short_all           Pct_of_OI_Comm_Short_All
pct_of_oi_tot_rept_long_all        Pct_of_OI_Tot_Rept_Long_All
pct_of_oi_tot_rept_short           Pct_of_OI_Tot_Rept_Short_All
pct_of_oi_nonrept_long_all         Pct_of_OI_NonRept_Long_All
pct_of_oi_nonrept_short_all        Pct_of_OI_NonRept_Short_All
```

### Percentage of Open Interest - Old
```
pct_of_open_interest_old           Pct_of_Open_Interest_Old
pct_of_oi_noncomm_long_old         Pct_of_OI_NonComm_Long_Old
pct_of_oi_noncomm_short_old        Pct_of_OI_NonComm_Short_Old
pct_of_oi_noncomm_spread_1         Pct_of_OI_NonComm_Spread_Old
pct_of_oi_comm_long_old            Pct_of_OI_Comm_Long_Old
pct_of_oi_comm_short_old           Pct_of_OI_Comm_Short_Old
pct_of_oi_tot_rept_long_old        Pct_of_OI_Tot_Rept_Long_Old
pct_of_oi_tot_rept_short_1         Pct_of_OI_Tot_Rept_Short_Old
pct_of_oi_nonrept_long_old         Pct_of_OI_NonRept_Long_Old
pct_of_oi_nonrept_short_old        Pct_of_OI_NonRept_Short_Old
```

### Percentage of Open Interest - Other
```
pct_of_open_interest_other         Pct_of_Open_Interest_Other
pct_of_oi_noncomm_long_other       Pct_of_OI_NonComm_Long_Other
pct_of_oi_noncomm_short_other      Pct_of_OI_NonComm_Short_Other
pct_of_oi_noncomm_spread_2         Pct_of_OI_NonComm_Spread_Other
pct_of_oi_comm_long_other          Pct_of_OI_Comm_Long_Other
pct_of_oi_comm_short_other         Pct_of_OI_Comm_Short_Other
pct_of_oi_tot_rept_long_other      Pct_of_OI_Tot_Rept_Long_Other
pct_of_oi_tot_rept_short_2         Pct_of_OI_Tot_Rept_Short_Other
pct_of_oi_nonrept_long_other       Pct_of_OI_NonRept_Long_Other
pct_of_oi_nonrept_short_other      Pct_of_OI_NonRept_Short_Other
```

### Trader Counts - All
```
traders_tot_all                    Traders_Tot_All
traders_noncomm_long_all           Traders_NonComm_Long_All
traders_noncomm_short_all          Traders_NonComm_Short_All
traders_noncomm_spread_all         Traders_NonComm_Spread_All
traders_comm_long_all              Traders_Comm_Long_All
traders_comm_short_all             Traders_Comm_Short_All
traders_tot_rept_long_all          Traders_Tot_Rept_Long_All
traders_tot_rept_short_all         Traders_Tot_Rept_Short_All
```

### Trader Counts - Old
```
traders_tot_old                    Traders_Tot_Old
traders_noncomm_long_old           Traders_NonComm_Long_Old
traders_noncomm_short_old          Traders_NonComm_Short_Old
traders_noncomm_spead_old          Traders_NonComm_Spead_Old          # Note: API has typo "spead"
traders_comm_long_old              Traders_Comm_Long_Old
traders_comm_short_old             Traders_Comm_Short_Old
traders_tot_rept_long_old          Traders_Tot_Rept_Long_Old
traders_tot_rept_short_old         Traders_Tot_Rept_Short_Old
```

### Trader Counts - Other
```
traders_tot_other                  Traders_Tot_Other
traders_noncomm_long_other         Traders_NonComm_Long_Other
traders_noncomm_short_other        Traders_NonComm_Short_Other
traders_noncomm_spread_other       Traders_NonComm_Spread_Other
traders_comm_long_other            Traders_Comm_Long_Other
traders_comm_short_other           Traders_Comm_Short_Other
traders_tot_rept_long_other        Traders_Tot_Rept_Long_Other
traders_tot_rept_short_other       Traders_Tot_Rept_Short_Other
```

### Concentration Ratios - All
```
conc_gross_le_4_tdr_long           Conc_Gross_LE_4_TDR_Long_All
conc_gross_le_4_tdr_short          Conc_Gross_LE_4_TDR_Short_All
conc_gross_le_8_tdr_long           Conc_Gross_LE_8_TDR_Long_All
conc_gross_le_8_tdr_short          Conc_Gross_LE_8_TDR_Short_All
conc_net_le_4_tdr_long_all         Conc_Net_LE_4_TDR_Long_All
conc_net_le_4_tdr_short_all        Conc_Net_LE_4_TDR_Short_All
conc_net_le_8_tdr_long_all         Conc_Net_LE_8_TDR_Long_All
conc_net_le_8_tdr_short_all        Conc_Net_LE_8_TDR_Short_All
```

### Concentration Ratios - Old
```
conc_gross_le_4_tdr_long_1         Conc_Gross_LE_4_TDR_Long_Old
conc_gross_le_4_tdr_short_1        Conc_Gross_LE_4_TDR_Short_Old
conc_gross_le_8_tdr_long_1         Conc_Gross_LE_8_TDR_Long_Old
conc_gross_le_8_tdr_short_1        Conc_Gross_LE_8_TDR_Short_Old
conc_net_le_4_tdr_long_old         Conc_Net_LE_4_TDR_Long_Old
conc_net_le_4_tdr_short_old        Conc_Net_LE_4_TDR_Short_Old
conc_net_le_8_tdr_long_old         Conc_Net_LE_8_TDR_Long_Old
conc_net_le_8_tdr_short_old        Conc_Net_LE_8_TDR_Short_Old
```

### Concentration Ratios - Other
```
conc_gross_le_4_tdr_long_2         Conc_Gross_LE_4_TDR_Long_Other
conc_gross_le_4_tdr_short_2        Conc_Gross_LE_4_TDR_Short_Other
conc_gross_le_8_tdr_long_2         Conc_Gross_LE_8_TDR_Long_Other
conc_gross_le_8_tdr_short_2        Conc_Gross_LE_8_TDR_Short_Other
conc_net_le_4_tdr_long_other       Conc_Net_LE_4_TDR_Long_Other
conc_net_le_4_tdr_short_other      Conc_Net_LE_4_TDR_Short_Other
conc_net_le_8_tdr_long_other       Conc_Net_LE_8_TDR_Long_Other
conc_net_le_8_tdr_short_other      Conc_Net_LE_8_TDR_Short_Other
```

### Metadata Columns
```
contract_units                     Contract_Units
commodity                          COMMODITY_NAME
commodity_subgroup_name            COMMODITY_SUBGROUP_NAME
commodity_group_name               COMMODITY_GROUP_NAME
futonly_or_combined               FutOnly_or_Combined
```

## Important Notes

1. **API Typos**: The CFTC API contains several typos that must be used exactly as they appear:
   - `noncomm_postions_spread_all` (missing 'i' in positions)
   - `change_in_noncomm_spead_all` (typo: "spead" instead of "spread")
   - `traders_noncomm_spead_old` (typo: "spead" instead of "spread")

2. **Numeric Suffixes**: Some columns have numeric suffixes (_1, _2) instead of descriptive names for old/other categories

3. **Inconsistent Naming**: Some "short" columns are missing "_all" suffix (e.g., `tot_rept_positions_short` instead of `tot_rept_positions_short_all`)

4. **Always use the left column (API field name) in your code**, not the display name on the right