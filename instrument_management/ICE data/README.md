# ICE Futures Europe COT Data

This directory contains COT (Commitments of Traders) data from ICE Futures Europe.

## Overview

ICE Futures Europe publishes weekly COT reports in the **Disaggregated Format**, which provides more granular trader classification than the traditional CFTC Legacy format.

### Key Differences from CFTC Legacy Data

| Aspect | CFTC Legacy | ICE Disaggregated |
|--------|-------------|-------------------|
| **Exchange** | US-based (CME, CBOT, etc.) | ICE Futures Europe |
| **Trader Categories** | 2 (Commercial, Non-Commercial) | 4 (Producer/Merchant, Swap, Managed Money, Other) |
| **Report Format** | Legacy | Disaggregated |
| **Geography** | US commodities | European commodities |
| **Instruments** | US futures contracts | ICE Europe contracts |

## Data Files

### Current Files
- `COTHist2025.csv` - 2025 COT data (Jan 7 - Nov 4, 44 weeks)
- `analyze_ice_cot.py` - Data analysis script
- `data_summary.json` - Automated data summary
- `ICE_COT_COLUMN_MAPPING.md` - Complete column reference
- `README.md` - This file

### Future Files
When you add historical data files, use the naming convention:
- `COTHist2024.csv`
- `COTHist2023.csv`
- etc.

## Instruments Covered

The data includes 7 unique markets (13 instruments total with Futures-only and Combined versions):

### Energy (3 markets)
| Instrument | Code | Contract Size |
|------------|------|---------------|
| ICE Brent Crude | B | 1,000 barrels |
| ICE Gasoil | G | 100 metric tonnes |
| ICE Dubai 1st Line | Dubai_1st | 1,000 barrels |

### Agricultural (4 markets)
| Instrument | Code | Contract Size |
|------------|------|---------------|
| ICE White Sugar | W | 50 tonnes |
| ICE Cocoa | Cocoa | 10 tonnes |
| ICE Robusta Coffee | RC | 10 tonnes |
| ICE Wheat | Wheat | 100 tonnes |

**Note:** Each market has two versions:
- **FutOnly** - Futures positions only
- **Combined** - Futures and Options combined

## Trader Categories

ICE uses the **Disaggregated Format** with 4 reportable trader types:

1. **Producer/Merchant (Prod_Merc)**
   - Commercial hedgers
   - Physical commodity handlers
   - Producers and processors

2. **Swap Dealers (Swap)**
   - Financial intermediaries
   - Dealers in commodity swaps
   - May hold spread positions

3. **Managed Money (M_Money)**
   - Hedge funds
   - Commodity Trading Advisors (CTAs)
   - Commodity pools
   - Speculators

4. **Other Reportable (Other_Rept)**
   - Large traders not fitting above categories
   - Mixed commercial/speculative

5. **Non-Reportable (NonRept)**
   - Small traders below reporting thresholds

## Data Structure

### Column Categories
- **191 total columns**
- Identification: 7 columns
- Position Data: 48 columns (across _All, _Old, _Other)
- Changes: 16 columns
- Percentages: 48 columns
- Trader Counts: 42 columns
- Concentration: 24 columns
- Metadata: 6 columns

### Suffix Meanings
- `_All` - All traders/crop years combined
- `_Old` - Old crop year (agricultural commodities)
- `_Other` - Other crop year (agricultural commodities)

## Usage

### Analyze Data
```bash
cd "instrument_management/ICE data"
source ../../venv/bin/activate
python analyze_ice_cot.py
```

### Key Columns for Analysis

**Managed Money Net Position:**
```python
mm_net = M_Money_Positions_Long_All - M_Money_Positions_Short_All
```

**Open Interest:**
```python
oi = Open_Interest_All
```

**Position as % of OI:**
```python
mm_long_pct = Pct_of_OI_M_Money_Long_All
mm_short_pct = Pct_of_OI_M_Money_Short_All
```

**Date:**
```python
date = As_of_Date_Form_MM/DD/YYYY  # Or As_of_Date_In_Form_YYMMDD
```

## Data Quality

Current data quality (as of 2025-01-11):
- ✓ No missing values in key columns
- ✓ No zero open interest records
- ✓ Complete weekly coverage
- ✓ All instruments present for all weeks

## Integration with Dashboard

### Steps for Integration

1. **Data Loading**
   - Create ICE-specific data fetcher module
   - Parse CSV files from this directory
   - Handle both FutOnly and Combined reports

2. **Instrument Mapping**
   - Create mapping similar to `futures_symbols_enhanced.json`
   - Map ICE commodity codes to price data sources
   - Define categories (Energy, Agricultural)

3. **Dashboard Display**
   - Add ICE section to overview table
   - Create ICE-specific charts
   - Show Managed Money positioning (equivalent to Non-Commercial in CFTC)

4. **Price Correlation**
   - Map to price data sources (if available)
   - Use appropriate futures symbols for ICE contracts
   - Calculate weekly returns for correlation

## Column Reference

For complete column documentation, see:
- **[ICE_COT_COLUMN_MAPPING.md](ICE_COT_COLUMN_MAPPING.md)** - Detailed column reference

Quick lookup for common columns:
- Market name: `Market_and_Exchange_Names`
- Date: `As_of_Date_Form_MM/DD/YYYY`
- Commodity code: `CFTC_Commodity_Code`
- Open interest: `Open_Interest_All`
- MM Long: `M_Money_Positions_Long_All`
- MM Short: `M_Money_Positions_Short_All`
- MM Net %: Calculate from long/short

## Data Source

- **Source**: ICE Futures Europe
- **Format**: Disaggregated COT Report (CSV)
- **Frequency**: Weekly
  - Positions as of: Tuesday market close
  - Released: Friday afternoon
- **URL**: Available from ICE Futures Europe website

## Notes

1. **Crop Years**: Agricultural commodities have _Old and _Other crop year breakdowns. Energy commodities typically don't use these.

2. **Spread Positions**: Unlike CFTC Legacy format, spread positions are tracked separately for Swap Dealers, Managed Money, and Other Reportable traders.

3. **Comparison to CFTC**:
   - Producer/Merchant + Swap Dealers ≈ CFTC Commercial
   - Managed Money ≈ CFTC Non-Commercial (speculative)
   - Be cautious with direct comparisons as classification methodologies differ

4. **Futures vs Combined**: Use Combined reports for total market positioning including options. Use FutOnly for futures-specific analysis.

5. **Concentration Ratios**: Top 4 and Top 8 trader concentration metrics are provided for both gross and net positions.

## Future Enhancements

- [ ] Add merge script for combining multiple years
- [ ] Create ICE-specific data fetcher for dashboard
- [ ] Map ICE instruments to price data sources
- [ ] Add ICE section to main dashboard
- [ ] Create ICE vs CFTC comparison charts
- [ ] Add automated data quality checks
- [ ] Create instrument mapping JSON file

## Contact & Resources

- **ICE Futures Europe COT Reports**: https://www.theice.com/marketdata/reports/
- **Dashboard Repository**: https://github.com/mwcapital/cot
- **Project Documentation**: `../../CLAUDE.md`

---

**Last Updated**: 2025-01-11
