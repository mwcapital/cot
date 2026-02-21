# CFTC API Column Name Mapping Reference - Disaggregated Report

## Dataset Identifier: 72hh-3qpy

## CRITICAL: Use these EXACT column names from the CFTC API

### Identification Columns
```
id                                  ID
market_and_exchange_names           Market_and_Exchange_Names
report_date_as_yyyy_mm_dd          Report_Date_as_YYYY_MM_DD
yyyy_report_week_ww                YYYY Report Week WW
contract_market_name               Contract_Market_Name
cftc_contract_market_code          CFTC_Contract_Market_Code
cftc_market_code                   CFTC_Market_Code
cftc_region_code                   CFTC_Region_Code
cftc_commodity_code                CFTC_Commodity_Code
commodity_name                     Commodity Name
```

### Core Position Columns - All
```
open_interest_all                  Open_Interest_All
prod_merc_positions_long           Prod_Merc_Positions_Long_All
prod_merc_positions_short          Prod_Merc_Positions_Short_All
swap_positions_long_all            Swap_Positions_Long_All
swap__positions_short_all          Swap__Positions_Short_All              # Note: API has double underscore
swap__positions_spread_all         Swap__Positions_Spread_All             # Note: API has double underscore
m_money_positions_long_all         M_Money_Positions_Long_All
m_money_positions_short_all        M_Money_Positions_Short_All
m_money_positions_spread           M_Money_Positions_Spread_All
tot_rept_positions_long_all        Tot_Rept_Positions_Long_All
tot_rept_positions_short           Tot_Rept_Positions_Short_All
nonrept_positions_long_all         NonRept_Positions_Long_All
nonrept_positions_short_all        NonRept_Positions_Short_All
```

### Old Positions
```
open_interest_old                  Open_Interest_Old
prod_merc_positions_long_1         Prod_Merc_Positions_Long_Old
prod_merc_positions_short_1        Prod_Merc_Positions_Short_Old
swap_positions_long_old            Swap_Positions_Long_Old
swap__positions_short_old          Swap__Positions_Short_Old              # Note: API has double underscore
swap__positions_spread_old         Swap__Positions_Spread_Old             # Note: API has double underscore
m_money_positions_long_old         M_Money_Positions_Long_Old
m_money_positions_short_old        M_Money_Positions_Short_Old
m_money_positions_spread_1         M_Money_Positions_Spread_Old
other_rept_positions_long_1        Other_Rept_Positions_Long_Old
other_rept_positions_short_1       Other_Rept_Positions_Short_Old
other_rept_positions_spread_1      Other_Rept_Positions_Spread_Old
tot_rept_positions_long_old        Tot_Rept_Positions_Long_Old
tot_rept_positions_short_1         Tot_Rept_Positions_Short_Old
nonrept_positions_long_old         NonRept_Positions_Long_Old
nonrept_positions_short_old        NonRept_Positions_Short_Old
```

### Other Positions
```
open_interest_other                Open_Interest_Other
prod_merc_positions_long_2         Prod_Merc_Positions_Long_Other
prod_merc_positions_short_2        Prod_Merc_Positions_Short_Other
swap_positions_long_other          Swap_Positions_Long_Other
swap__positions_short_other        Swap__Positions_Short_Other            # Note: API has double underscore
swap__positions_spread_other       Swap__Positions_Spread_Other           # Note: API has double underscore
m_money_positions_long_other       M_Money_Positions_Long_Other
m_money_positions_short_other      M_Money_Positions_Short_Other
m_money_positions_spread_2         M_Money_Positions_Spread_Other
other_rept_positions_long_2        Other_Rept_Positions_Long_Other
other_rept_positions_short_2       Other_Rept_Positions_Short_Other
other_rept_positions_spread_2      Other_Rept_Positions_Spread_Other
tot_rept_positions_long_other      Tot_Rept_Positions_Long_Other
tot_rept_positions_short_2         Tot_Rept_Positions_Short_Other
nonrept_positions_long_other       NonRept_Positions_Long_Other
nonrept_positions_short_other      NonRept_Positions_Short_Other
```

### Change Columns
```
change_in_open_interest_all        Change_in_Open_Interest_All
change_in_prod_merc_long           Change_in_Prod_Merc_Long_All
change_in_prod_merc_short          Change_in_Prod_Merc_Short_All
change_in_swap_long_all            Change_in_Swap_Long_All
change_in_swap_short_all           Change_in_Swap_Short_All
change_in_swap_spread_all          Change_in_Swap_Spread_All
change_in_m_money_long_all         Change_in_M_Money_Long_All
change_in_m_money_short_all        Change_in_M_Money_Short_All
change_in_m_money_spread           Change_in_M_Money_Spread_All
change_in_other_rept_long          Change_in_Other_Rept_Long_All
change_in_other_rept_short         Change_in_Other_Rept_Short_All
change_in_other_rept_spread        Change_in_Other_Rept_Spread_All
change_in_tot_rept_long_all        Change_in_Tot_Rept_Long_All
change_in_tot_rept_short           Change_in_Tot_Rept_Short_All
change_in_nonrept_long_all         Change_in_NonRept_Long_All
change_in_nonrept_short_all        Change_in_NonRept_Short_All
```

### Percentage of Open Interest - All
```
pct_of_open_interest_all           Pct_of_Open_Interest_All
pct_of_oi_prod_merc_long           Pct_of_OI_Prod_Merc_Long_All
pct_of_oi_prod_merc_short          Pct_of_OI_Prod_Merc_Short_All
pct_of_oi_swap_long_all            Pct_of_OI_Swap_Long_All
pct_of_oi_swap_short_all           Pct_of_OI_Swap_Short_All
pct_of_oi_swap_spread_all          Pct_of_OI_Swap_Spread_All
pct_of_oi_m_money_long_all         Pct_of_OI_M_Money_Long_All
pct_of_oi_m_money_short_all        Pct_of_OI_M_Money_Short_All
pct_of_oi_m_money_spread           Pct_of_OI_M_Money_Spread_All
pct_of_oi_other_rept_long          Pct_of_OI_Other_Rept_Long_All
pct_of_oi_other_rept_short         Pct_of_OI_Other_Rept_Short_All
pct_of_oi_other_rept_spread        Pct_of_OI_Other_Rept_Spread_All
pct_of_oi_tot_rept_long_all        Pct_of_OI_Tot_Rept_Long_All
pct_of_oi_tot_rept_short           Pct_of_OI_Tot_Rept_Short_All
pct_of_oi_nonrept_long_all         Pct_of_OI_NonRept_Long_All
pct_of_oi_nonrept_short_all        Pct_of_OI_NonRept_Short_All
```

### Percentage of Open Interest - Old
```
pct_of_open_interest_old           Pct_of_Open_Interest_Old
pct_of_oi_prod_merc_long_1         Pct_of_OI_Prod_Merc_Long_Old
pct_of_oi_prod_merc_short_1        Pct_of_OI_Prod_Merc_Short_Old
pct_of_oi_swap_long_old            Pct_of_OI_Swap_Long_Old
pct_of_oi_swap_short_old           Pct_of_OI_Swap_Short_Old
pct_of_oi_swap_spread_old          Pct_of_OI_Swap_Spread_Old
pct_of_oi_m_money_long_old         Pct_of_OI_M_Money_Long_Old
pct_of_oi_m_money_short_old        Pct_of_OI_M_Money_Short_Old
pct_of_oi_m_money_spread_1         Pct_of_OI_M_Money_Spread_Old
pct_of_oi_other_rept_long_1        Pct_of_OI_Other_Rept_Long_Old
pct_of_oi_other_rept_short_1       Pct_of_OI_Other_Rept_Short_Old
pct_of_oi_other_rept_spread_1      Pct_of_OI_Other_Rept_Spread_Old
pct_of_oi_tot_rept_long_old        Pct_of_OI_Tot_Rept_Long_Old
pct_of_oi_tot_rept_short_1         Pct_of_OI_Tot_Rept_Short_Old
pct_of_oi_nonrept_long_old         Pct_of_OI_NonRept_Long_Old
pct_of_oi_nonrept_short_old        Pct_of_OI_NonRept_Short_Old
```

### Percentage of Open Interest - Other
```
pct_of_open_interest_other         Pct_of_Open_Interest_Other
pct_of_oi_prod_merc_long_2         Pct_of_OI_Prod_Merc_Long_Other
pct_of_oi_prod_merc_short_2        Pct_of_OI_Prod_Merc_Short_Other
pct_of_oi_swap_long_other          Pct_of_OI_Swap_Long_Other
pct_of_oi_swap_short_other         Pct_of_OI_Swap_Short_Other
pct_of_oi_swap_spread_other        Pct_of_OI_Swap_Spread_Other
pct_of_oi_m_money_long_other       Pct_of_OI_M_Money_Long_Other
pct_of_oi_m_money_short_other      Pct_of_OI_M_Money_Short_Other
pct_of_oi_m_money_spread_2         Pct_of_OI_M_Money_Spread_Other
pct_of_oi_other_rept_long_2        Pct_of_OI_Other_Rept_Long_Other
pct_of_oi_other_rept_short_2       Pct_of_OI_Other_Rept_Short_Other
pct_of_oi_other_rept_spread_2      Pct_of_OI_Other_Rept_Spread_Other
pct_of_oi_tot_rept_long_other      Pct_of_OI_Tot_Rept_Long_Other
pct_of_oi_tot_rept_short_2         Pct_of_OI_Tot_Rept_Short_Other
pct_of_oi_nonrept_long_other       Pct_of_OI_NonRept_Long_Other
pct_of_oi_nonrept_short_other      Pct_of_OI_NonRept_Short_Other
```

### Trader Counts - All
```
traders_tot_all                    Traders_Tot_All
traders_prod_merc_long_all         Traders_Prod_Merc_Long_All
traders_prod_merc_short_all        Traders_Prod_Merc_Short_All
traders_swap_long_all              Traders_Swap_Long_All
traders_swap_short_all             Traders_Swap_Short_All
traders_swap_spread_all            Traders_Swap_Spread_All
traders_m_money_long_all           Traders_M_Money_Long_All
traders_m_money_short_all          Traders_M_Money_Short_All
traders_m_money_spread_all         Traders_M_Money_Spread_All
traders_other_rept_long_all        Traders_Other_Rept_Long_All
traders_other_rept_short           Traders_Other_Rept_Short_All
traders_other_rept_spread          Traders_Other_Rept_Spread_All
traders_tot_rept_long_all          Traders_Tot_Rept_Long_All
traders_tot_rept_short_all         Traders_Tot_Rept_Short_All
```

### Trader Counts - Old
```
traders_tot_old                    Traders_Tot_Old
traders_prod_merc_long_old         Traders_Prod_Merc_Long_Old
traders_prod_merc_short_old        Traders_Prod_Merc_Short_Old
traders_swap_long_old              Traders_Swap_Long_Old
traders_swap_short_old             Traders_Swap_Short_Old
traders_swap_spread_old            Traders_Swap_Spread_Old
traders_m_money_long_old           Traders_M_Money_Long_Old
traders_m_money_short_old          Traders_M_Money_Short_Old
traders_m_money_spread_old         Traders_M_Money_Spread_Old
traders_other_rept_long_old        Traders_Other_Rept_Long_Old
traders_other_rept_short_1         Traders_Other_Rept_Short_Old
traders_other_rept_spread_1        Traders_Other_Rept_Spread_Old
traders_tot_rept_long_old          Traders_Tot_Rept_Long_Old
traders_tot_rept_short_old         Traders_Tot_Rept_Short_Old
```

### Trader Counts - Other
```
traders_tot_other                  Traders_Tot_Other
traders_prod_merc_long_other       Traders_Prod_Merc_Long_Other
traders_prod_merc_short_other      Traders_Prod_Merc_Short_Other
traders_swap_long_other            Traders_Swap_Long_Other
traders_swap_short_other           Traders_Swap_Short_Other
traders_swap_spread_other          Traders_Swap_Spread_Other
traders_m_money_long_other         Traders_M_Money_Long_Other
traders_m_money_short_other        Traders_M_Money_Short_Other
traders_m_money_spread_other       Traders_M_Money_Spread_Other
traders_other_rept_long_other      Traders_Other_Rept_Long_Other
traders_other_rept_short_2         Traders_Other_Rept_Short_Other
traders_other_rept_spread_2        Traders_Other_Rept_Spread_Other
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
cftc_subgroup_code                 CFTC_SubGroup_Code
commodity                          COMMODITY_NAME
commodity_subgroup_name            COMMODITY_SUBGROUP_NAME
commodity_group_name               COMMODITY_GROUP_NAME
```

## Trader Categories (Disaggregated vs Legacy)

### Disaggregated Report breaks traders into 4 reportable categories:
| Disaggregated Category | API Prefix    | Description |
|----------------------|---------------|-------------|
| Producer/Merchant    | prod_merc     | Entities that produce, process, or merchandise the commodity |
| Swap Dealers         | swap          | Entities dealing primarily in swaps |
| Managed Money        | m_money       | CTAs, CPOs, and hedge funds |
| Other Reportables    | other_rept    | Other reportable traders not in above categories |

### Plus non-reportable:
| Category             | API Prefix    | Description |
|----------------------|---------------|-------------|
| Non-Reportable       | nonrept       | Positions below reporting threshold |

### Mapping to Legacy Report:
| Legacy Category    | Disaggregated Equivalent |
|-------------------|--------------------------|
| Commercial        | Producer/Merchant + Swap Dealers |
| Non-Commercial    | Managed Money + Other Reportables |

## Data Suffixes

| Suffix | Meaning | Description |
|--------|---------|-------------|
| _all   | All     | Combined futures and options positions |
| _old   | Old     | Futures-only positions |
| _other | Other   | Options-only positions |

## Important Notes

1. **API Quirks**: The CFTC API contains inconsistencies that must be used exactly as they appear:
   - `swap__positions_short_all` (double underscore in swap short/spread columns)
   - `swap__positions_spread_all` (double underscore)
   - `m_money_positions_spread` missing `_all` suffix
   - `other_rept_positions_long/short/spread` missing `_all` suffix for core positions
   - `prod_merc_positions_long/short` missing `_all` suffix for core positions

2. **Numeric Suffixes**: Some columns use _1, _2 instead of _old, _other (same pattern as Legacy report)

3. **Always use the left column (API field name) in your code**, not the display name on the right

4. **Dataset Identifier**: `72hh-3qpy` (vs Legacy Futures: `6dca-aqe3`)
