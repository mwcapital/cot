# CFTC API Column Name Mapping Reference

## CRITICAL: Use these EXACT column names from the CFTC API

### Core Position Columns
```json
{
  "positions": {
    "noncommercial_long": "noncomm_positions_long_all",
    "noncommercial_short": "noncomm_positions_short_all",
    "noncommercial_spreading": "noncomm_positions_spread_all",
    "commercial_long": "comm_positions_long_all",
    "commercial_short": "comm_positions_short_all",
    "nonreportable_long": "nonrept_positions_long_all",
    "nonreportable_short": "nonrept_positions_short_all",
    "total_reportable_long": "tot_rept_positions_long_all",
    "total_reportable_short": "tot_rept_positions_short_all"
  }
}
```

### Open Interest
```json
{
  "open_interest": {
    "total": "open_interest_all",
    "old": "open_interest_old",
    "other": "open_interest_other"
  }
}
```

### Change Columns
```json
{
  "changes": {
    "open_interest": "change_in_open_interest_all",
    "noncomm_long": "change_in_noncomm_long_all",
    "noncomm_short": "change_in_noncomm_short_all",
    "noncomm_spread": "change_in_noncomm_spead_all",
    "comm_long": "change_in_comm_long_all",
    "comm_short": "change_in_comm_short_all",
    "tot_rept_long": "change_in_tot_rept_long_all",
    "tot_rept_short": "change_in_tot_rept_short",
    "nonrept_long": "change_in_nonrept_long_all",
    "nonrept_short": "change_in_nonrept_short_all"
  }
}
```

### Percentage of Open Interest
```json
{
  "pct_of_oi": {
    "noncomm_long": "pct_of_oi_noncomm_long_all",
    "noncomm_short": "pct_of_oi_noncomm_short_all",
    "noncomm_spread": "pct_of_oi_noncomm_spread",
    "comm_long": "pct_of_oi_comm_long_all",
    "comm_short": "pct_of_oi_comm_short_all",
    "tot_rept_long": "pct_of_oi_tot_rept_long_all",
    "tot_rept_short": "pct_of_oi_tot_rept_short",
    "nonrept_long": "pct_of_oi_nonrept_long_all",
    "nonrept_short": "pct_of_oi_nonrept_short_all"
  }
}
```

### Trader Counts
```json
{
  "traders": {
    "total_all": "traders_tot_all",
    "noncomm_long": "traders_noncomm_long_all",
    "noncomm_short": "traders_noncomm_short_all",
    "noncomm_spread": "traders_noncomm_spread_all",
    "comm_long": "traders_comm_long_all",
    "comm_short": "traders_comm_short_all",
    "tot_rept_long": "traders_tot_rept_long_all",
    "tot_rept_short": "traders_tot_rept_short_all"
  }
}
```

### Concentration Ratios
```json
{
  "concentration": {
    "gross_le_4_long": "conc_gross_le_4_tdr_long",
    "gross_le_4_short": "conc_gross_le_4_tdr_short",
    "gross_le_8_long": "conc_gross_le_8_tdr_long",
    "gross_le_8_short": "conc_gross_le_8_tdr_short",
    "net_le_4_long": "conc_net_le_4_tdr_long_all",
    "net_le_4_short": "conc_net_le_4_tdr_short_all",
    "net_le_8_long": "conc_net_le_8_tdr_long_all",
    "net_le_8_short": "conc_net_le_8_tdr_short_all"
  }
}
```

### Metadata Columns
```json
{
  "metadata": {
    "market_name": "market_and_exchange_names",
    "report_date": "report_date_as_yyyy_mm_dd",
    "report_week": "yyyy_report_week_ww",
    "contract_market": "contract_market_name",
    "contract_code": "cftc_contract_market_code",
    "market_code": "cftc_market_code",
    "region_code": "cftc_region_code",
    "commodity_code": "cftc_commodity_code",
    "commodity_name": "commodity_name",
    "commodity_subgroup": "commodity_subgroup_name",
    "commodity_group": "commodity_group_name",
    "contract_units": "contract_units",
    "futures_only_or_combined": "futonly_or_combined"
  }
}
```

## CALCULATED FIELDS (Not in API, must be computed)
```json
{
  "calculated": {
    "net_commercial": "comm_positions_long_all - comm_positions_short_all",
    "net_noncommercial": "noncomm_positions_long_all - noncomm_positions_short_all",
    "net_nonreportable": "nonrept_positions_long_all - nonrept_positions_short_all"
  }
}
```

## Common Mistakes to AVOID
❌ WRONG: `noncommercial_positions_long_all`
✅ CORRECT: `noncomm_positions_long_all`

❌ WRONG: `commercial_positions_long_all`
✅ CORRECT: `comm_positions_long_all`

❌ WRONG: `nonreportable_positions_long_all`
✅ CORRECT: `nonrept_positions_long_all`

❌ WRONG: `noncommercial_positions_spreading_all`
✅ CORRECT: `noncomm_positions_spread_all` (no 'ing')

❌ WRONG: `change_in_noncomm_spread_all`
✅ CORRECT: `change_in_noncomm_spead_all` (yes, 'spead' is the actual API typo!)

## Usage in Code
```python
# For checkbox selections in UI
selected_series = []
if st.checkbox("Non-Commercial Long"):
    selected_series.append("noncomm_positions_long_all")  # NOT "noncommercial_..."

if st.checkbox("Commercial Long"):
    selected_series.append("comm_positions_long_all")  # NOT "commercial_..."

# For calculated net positions
df['net_commercial'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
df['net_noncommercial'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
```

## Quick Reference for Display Functions
| UI Display Name | API Column Name | Type |
|-----------------|-----------------|------|
| Non-Commercial Long | noncomm_positions_long_all | Direct |
| Non-Commercial Short | noncomm_positions_short_all | Direct |
| Commercial Long | comm_positions_long_all | Direct |
| Commercial Short | comm_positions_short_all | Direct |
| Non-Commercial Net | net_noncommercial | Calculated |
| Commercial Net | net_commercial | Calculated |
| Open Interest (COT) | open_interest_all | Direct |
| Non-Reportable Long | nonrept_positions_long_all | Direct |
| Non-Reportable Short | nonrept_positions_short_all | Direct |
| Non-Commercial Spreading | noncomm_positions_spread_all | Direct |