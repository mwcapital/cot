# ICE COT Column Name Mapping Reference

## CRITICAL: ICE Disaggregated COT Report Format

This document provides the complete column mapping for ICE Futures Europe COT data files.

**Key Differences from CFTC Legacy Format:**
- Uses **Disaggregated Format** with 4 trader categories (vs 2 in CFTC Legacy)
- Trader categories: Producer/Merchant, Swap Dealers, Managed Money, Other Reportable
- ICE Europe exchange data (not CFTC)
- Data source: ICE Futures Europe weekly reports
- All column names use the exact format as they appear in the CSV files

---

## Column Categories Overview

| Category | Count | Suffix Types |
|----------|-------|--------------|
| Identification | 7 | N/A |
| Open Interest | 3 | _All, _Old, _Other |
| Positions (Per Category) | 48 | _All, _Old, _Other |
| Changes | 16 | _All only |
| Percentages | 48 | _All, _Old, _Other |
| Trader Counts | 42 | _All, _Old, _Other |
| Concentration | 24 | _All, _Old, _Other |
| Metadata | 6 | N/A |
| **Total** | **191** | |

---

## 1. Identification & Date Columns

```
Market_and_Exchange_Names           Market and Exchange Names
As_of_Date_In_Form_YYMMDD          As of Date (YYMMDD format)
As_of_Date_Form_MM/DD/YYYY         As of Date (MM/DD/YYYY format)
CFTC_Contract_Market_Code          CFTC Contract Market Code
CFTC_Market_Code                   CFTC Market Code (e.g., ICEU)
CFTC_Region_Code                   CFTC Region Code
CFTC_Commodity_Code                CFTC Commodity Code (e.g., B, G, W, RC)
```

**Notes:**
- Two date formats provided for convenience
- Market_Code = "ICEU" for ICE Futures Europe
- Commodity_Code is the primary instrument identifier

---

## 2. Open Interest

### All Traders
```
Open_Interest_All                  Total Open Interest (All Traders)
```

### Old Crop (for applicable commodities)
```
Open_Interest_Old                  Open Interest (Old Crop)
```

### Other Crop (for applicable commodities)
```
Open_Interest_Other                Open Interest (Other Crop)
```

**Notes:**
- _Old and _Other are used for agricultural commodities with crop year distinctions
- For non-agricultural commodities, these fields may be empty

---

## 3. Position Data by Trader Category

ICE uses the **Disaggregated Format** with 4 reportable trader categories:

1. **Producer/Merchant (Prod_Merc)** - Commercial hedgers, physical commodity handlers
2. **Swap Dealers (Swap)** - Financial intermediaries dealing in swaps
3. **Managed Money (M_Money)** - Hedge funds, CTAs, commodity pools
4. **Other Reportable (Other_Rept)** - Large traders not in above categories

Plus:
5. **Non-Reportable (NonRept)** - Small traders below reporting thresholds

### 3.1 Producer/Merchant Positions

#### All Traders
```
Prod_Merc_Positions_Long_All       Producer/Merchant Long (All)
Prod_Merc_Positions_Short_All      Producer/Merchant Short (All)
```

#### Old Crop
```
Prod_Merc_Positions_Long_Old       Producer/Merchant Long (Old)
Prod_Merc_Positions_Short_Old      Producer/Merchant Short (Old)
```

#### Other Crop
```
Prod_Merc_Positions_Long_Other     Producer/Merchant Long (Other)
Prod_Merc_Positions_Short_Other    Producer/Merchant Short (Other)
```

### 3.2 Swap Dealer Positions

#### All Traders
```
Swap_Positions_Long_All            Swap Dealer Long (All)
Swap_Positions_Short_All           Swap Dealer Short (All)
Swap_Positions_Spread_All          Swap Dealer Spread (All)
```

#### Old Crop
```
Swap_Positions_Long_Old            Swap Dealer Long (Old)
Swap_Positions_Short_Old           Swap Dealer Short (Old)
Swap_Positions_Spread_Old          Swap Dealer Spread (Old)
```

#### Other Crop
```
Swap_Positions_Long_Other          Swap Dealer Long (Other)
Swap_Positions_Short_Other         Swap Dealer Short (Other)
Swap_Positions_Spread_Other        Swap Dealer Spread (Other)
```

### 3.3 Managed Money Positions

#### All Traders
```
M_Money_Positions_Long_All         Managed Money Long (All)
M_Money_Positions_Short_All        Managed Money Short (All)
M_Money_Positions_Spread_All       Managed Money Spread (All)
```

#### Old Crop
```
M_Money_Positions_Long_Old         Managed Money Long (Old)
M_Money_Positions_Short_Old        Managed Money Short (Old)
M_Money_Positions_Spread_Old       Managed Money Spread (Old)
```

#### Other Crop
```
M_Money_Positions_Long_Other       Managed Money Long (Other)
M_Money_Positions_Short_Other      Managed Money Short (Other)
M_Money_Positions_Spread_Other     Managed Money Spread (Other)
```

### 3.4 Other Reportable Positions

#### All Traders
```
Other_Rept_Positions_Long_All      Other Reportable Long (All)
Other_Rept_Positions_Short_All     Other Reportable Short (All)
Other_Rept_Positions_Spread_All    Other Reportable Spread (All)
```

#### Old Crop
```
Other_Rept_Positions_Long_Old      Other Reportable Long (Old)
Other_Rept_Positions_Short_Old     Other Reportable Short (Old)
Other_Rept_Positions_Spread_Old    Other Reportable Spread (Old)
```

#### Other Crop
```
Other_Rept_Positions_Long_Other    Other Reportable Long (Other)
Other_Rept_Positions_Short_Other   Other Reportable Short (Other)
Other_Rept_Positions_Spread_Other  Other Reportable Spread (Other)
```

### 3.5 Total Reportable Positions

#### All Traders
```
Tot_Rept_Positions_Long_All        Total Reportable Long (All)
Tot_Rept_Positions_Short_All       Total Reportable Short (All)
```

#### Old Crop
```
Tot_Rept_Positions_Long_Old        Total Reportable Long (Old)
Tot_Rept_Positions_Short_Old       Total Reportable Short (Old)
```

#### Other Crop
```
Tot_Rept_Positions_Long_Other      Total Reportable Long (Other)
Tot_Rept_Positions_Short_Other     Total Reportable Short (Other)
```

### 3.6 Non-Reportable Positions

#### All Traders
```
NonRept_Positions_Long_All         Non-Reportable Long (All)
NonRept_Positions_Short_All        Non-Reportable Short (All)
```

#### Old Crop
```
NonRept_Positions_Long_Old         Non-Reportable Long (Old)
NonRept_Positions_Short_Old        Non-Reportable Short (Old)
```

#### Other Crop
```
NonRept_Positions_Long_Other       Non-Reportable Long (Other)
NonRept_Positions_Short_Other      Non-Reportable Short (Other)
```

---

## 4. Week-over-Week Changes

All change columns show the difference from the previous week's report.

```
Change_in_Open_Interest_All        Change in Open Interest
Change_in_Prod_Merc_Long_All       Change in Producer/Merchant Long
Change_in_Prod_Merc_Short_All      Change in Producer/Merchant Short
Change_in_Swap_Long_All            Change in Swap Dealer Long
Change_in_Swap_Short_All           Change in Swap Dealer Short
Change_in_Swap_Spread_All          Change in Swap Dealer Spread
Change_in_M_Money_Long_All         Change in Managed Money Long
Change_in_M_Money_Short_All        Change in Managed Money Short
Change_in_M_Money_Spread_All       Change in Managed Money Spread
Change_in_Other_Rept_Long_All      Change in Other Reportable Long
Change_in_Other_Rept_Short_All     Change in Other Reportable Short
Change_in_Other_Rept_Spread_All    Change in Other Reportable Spread
Change_in_Tot_Rept_Long_All        Change in Total Reportable Long
Change_in_Tot_Rept_Short_All       Change in Total Reportable Short
Change_in_NonRept_Long_All         Change in Non-Reportable Long
Change_in_NonRept_Short_All        Change in Non-Reportable Short
```

**Notes:**
- Changes are only provided for _All categories (not Old/Other)
- Values represent: Current Week Position - Previous Week Position

---

## 5. Percentage of Open Interest

Each position expressed as a percentage of the total open interest.

### 5.1 Open Interest Base (Always 100)

```
Pct_of_Open_Interest_All           % of Open Interest (All) = 100
Pct_of_Open_Interest_Old           % of Open Interest (Old) = 100
Pct_of_Open_Interest_Other         % of Open Interest (Other) = 100
```

### 5.2 Producer/Merchant Percentages

#### All Traders
```
Pct_of_OI_Prod_Merc_Long_All       % Prod/Merc Long (All)
Pct_of_OI_Prod_Merc_Short_All      % Prod/Merc Short (All)
```

#### Old Crop
```
Pct_of_OI_Prod_Merc_Long_Old       % Prod/Merc Long (Old)
Pct_of_OI_Prod_Merc_Short_Old      % Prod/Merc Short (Old)
```

#### Other Crop
```
Pct_of_OI_Prod_Merc_Long_Other     % Prod/Merc Long (Other)
Pct_of_OI_Prod_Merc_Short_Other    % Prod/Merc Short (Other)
```

### 5.3 Swap Dealer Percentages

#### All Traders
```
Pct_of_OI_Swap_Long_All            % Swap Dealer Long (All)
Pct_of_OI_Swap_Short_All           % Swap Dealer Short (All)
Pct_of_OI_Swap_Spread_All          % Swap Dealer Spread (All)
```

#### Old Crop
```
Pct_of_OI_Swap_Long_Old            % Swap Dealer Long (Old)
Pct_of_OI_Swap_Short_Old           % Swap Dealer Short (Old)
Pct_of_OI_Swap_Spread_Old          % Swap Dealer Spread (Old)
```

#### Other Crop
```
Pct_of_OI_Swap_Long_Other          % Swap Dealer Long (Other)
Pct_of_OI_Swap_Short_Other         % Swap Dealer Short (Other)
Pct_of_OI_Swap_Spread_Other        % Swap Dealer Spread (Other)
```

### 5.4 Managed Money Percentages

#### All Traders
```
Pct_of_OI_M_Money_Long_All         % Managed Money Long (All)
Pct_of_OI_M_Money_Short_All        % Managed Money Short (All)
Pct_of_OI_M_Money_Spread_All       % Managed Money Spread (All)
```

#### Old Crop
```
Pct_of_OI_M_Money_Long_Old         % Managed Money Long (Old)
Pct_of_OI_M_Money_Short_Old        % Managed Money Short (Old)
Pct_of_OI_M_Money_Spread_Old       % Managed Money Spread (Old)
```

#### Other Crop
```
Pct_of_OI_M_Money_Long_Other       % Managed Money Long (Other)
Pct_of_OI_M_Money_Short_Other      % Managed Money Short (Other)
Pct_of_OI_M_Money_Spread_Other     % Managed Money Spread (Other)
```

### 5.5 Other Reportable Percentages

#### All Traders
```
Pct_of_OI_Other_Rept_Long_All      % Other Reportable Long (All)
Pct_of_OI_Other_Rept_Short_All     % Other Reportable Short (All)
Pct_of_OI_Other_Rept_Spread_All    % Other Reportable Spread (All)
```

#### Old Crop
```
Pct_of_OI_Other_Rept_Long_Old      % Other Reportable Long (Old)
Pct_of_OI_Other_Rept_Short_Old     % Other Reportable Short (Old)
Pct_of_OI_Other_Rept_Spread_Old    % Other Reportable Spread (Old)
```

#### Other Crop
```
Pct_of_OI_Other_Rept_Long_Other    % Other Reportable Long (Other)
Pct_of_OI_Other_Rept_Short_Other   % Other Reportable Short (Other)
Pct_of_OI_Other_Rept_Spread_Other  % Other Reportable Spread (Other)
```

### 5.6 Total Reportable Percentages

#### All Traders
```
Pct_of_OI_Tot_Rept_Long_All        % Total Reportable Long (All)
Pct_of_OI_Tot_Rept_Short_All       % Total Reportable Short (All)
```

#### Old Crop
```
Pct_of_OI_Tot_Rept_Long_Old        % Total Reportable Long (Old)
Pct_of_OI_Tot_Rept_Short_Old       % Total Reportable Short (Old)
```

#### Other Crop
```
Pct_of_OI_Tot_Rept_Long_Other      % Total Reportable Long (Other)
Pct_of_OI_Tot_Rept_Short_Other     % Total Reportable Short (Other)
```

### 5.7 Non-Reportable Percentages

#### All Traders
```
Pct_of_OI_NonRept_Long_All         % Non-Reportable Long (All)
Pct_of_OI_NonRept_Short_All        % Non-Reportable Short (All)
```

#### Old Crop
```
Pct_of_OI_NonRept_Long_Old         % Non-Reportable Long (Old)
Pct_of_OI_NonRept_Short_Old        % Non-Reportable Short (Old)
```

#### Other Crop
```
Pct_of_OI_NonRept_Long_Other       % Non-Reportable Long (Other)
Pct_of_OI_NonRept_Short_Other      % Non-Reportable Short (Other)
```

---

## 6. Trader Counts

Number of traders holding positions in each category.

### 6.1 Total Traders

```
Traders_Tot_All                    Total Number of Traders (All)
Traders_Tot_Old                    Total Number of Traders (Old)
Traders_Tot_Other                  Total Number of Traders (Other)
```

### 6.2 Producer/Merchant Traders

#### All Traders
```
Traders_Prod_Merc_Long_All         # Prod/Merc Long Traders (All)
Traders_Prod_Merc_Short_All        # Prod/Merc Short Traders (All)
```

#### Old Crop
```
Traders_Prod_Merc_Long_Old         # Prod/Merc Long Traders (Old)
Traders_Prod_Merc_Short_Old        # Prod/Merc Short Traders (Old)
```

#### Other Crop
```
Traders_Prod_Merc_Long_Other       # Prod/Merc Long Traders (Other)
Traders_Prod_Merc_Short_Other      # Prod/Merc Short Traders (Other)
```

### 6.3 Swap Dealer Traders

#### All Traders
```
Traders_Swap_Long_All              # Swap Dealer Long Traders (All)
Traders_Swap_Short_All             # Swap Dealer Short Traders (All)
Traders_Swap_Spread_All            # Swap Dealer Spread Traders (All)
```

#### Old Crop
```
Traders_Swap_Long_Old              # Swap Dealer Long Traders (Old)
Traders_Swap_Short_Old             # Swap Dealer Short Traders (Old)
Traders_Swap_Spread_Old            # Swap Dealer Spread Traders (Old)
```

#### Other Crop
```
Traders_Swap_Long_Other            # Swap Dealer Long Traders (Other)
Traders_Swap_Short_Other           # Swap Dealer Short Traders (Other)
Traders_Swap_Spread_Other          # Swap Dealer Spread Traders (Other)
```

### 6.4 Managed Money Traders

#### All Traders
```
Traders_M_Money_Long_All           # Managed Money Long Traders (All)
Traders_M_Money_Short_All          # Managed Money Short Traders (All)
Traders_M_Money_Spread_All         # Managed Money Spread Traders (All)
```

#### Old Crop
```
Traders_M_Money_Long_Old           # Managed Money Long Traders (Old)
Traders_M_Money_Short_Old          # Managed Money Short Traders (Old)
Traders_M_Money_Spread_Old         # Managed Money Spread Traders (Old)
```

#### Other Crop
```
Traders_M_Money_Long_Other         # Managed Money Long Traders (Other)
Traders_M_Money_Short_Other        # Managed Money Short Traders (Other)
Traders_M_Money_Spread_Other       # Managed Money Spread Traders (Other)
```

### 6.5 Other Reportable Traders

#### All Traders
```
Traders_Other_Rept_Long_All        # Other Reportable Long Traders (All)
Traders_Other_Rept_Short_All       # Other Reportable Short Traders (All)
Traders_Other_Rept_Spread_All      # Other Reportable Spread Traders (All)
```

#### Old Crop
```
Traders_Other_Rept_Long_Old        # Other Reportable Long Traders (Old)
Traders_Other_Rept_Short_Old       # Other Reportable Short Traders (Old)
Traders_Other_Rept_Spread_Old      # Other Reportable Spread Traders (Old)
```

#### Other Crop
```
Traders_Other_Rept_Long_Other      # Other Reportable Long Traders (Other)
Traders_Other_Rept_Short_Other     # Other Reportable Short Traders (Other)
Traders_Other_Rept_Spread_Other    # Other Reportable Spread Traders (Other)
```

### 6.6 Total Reportable Traders

#### All Traders
```
Traders_Tot_Rept_Long_All          # Total Reportable Long Traders (All)
Traders_Tot_Rept_Short_All         # Total Reportable Short Traders (All)
```

#### Old Crop
```
Traders_Tot_Rept_Long_Old          # Total Reportable Long Traders (Old)
Traders_Tot_Rept_Short_Old         # Total Reportable Short Traders (Old)
```

#### Other Crop
```
Traders_Tot_Rept_Long_Other        # Total Reportable Long Traders (Other)
Traders_Tot_Rept_Short_Other       # Total Reportable Short Traders (Other)
```

---

## 7. Concentration Ratios

Percentage of open interest held by the top 4 and top 8 largest traders.

**LE = "Less than or Equal to"**
- LE_4_TDR = Top 4 traders
- LE_8_TDR = Top 8 traders

### 7.1 Gross Concentration (absolute long or short positions)

#### All Traders
```
Conc_Gross_LE_4_TDR_Long_All       Concentration: Top 4 Gross Long (All)
Conc_Gross_LE_4_TDR_Short_All      Concentration: Top 4 Gross Short (All)
Conc_Gross_LE_8_TDR_Long_All       Concentration: Top 8 Gross Long (All)
Conc_Gross_LE_8_TDR_Short_All      Concentration: Top 8 Gross Short (All)
```

#### Old Crop
```
Conc_Gross_LE_4_TDR_Long_Old       Concentration: Top 4 Gross Long (Old)
Conc_Gross_LE_4_TDR_Short_Old      Concentration: Top 4 Gross Short (Old)
Conc_Gross_LE_8_TDR_Long_Old       Concentration: Top 8 Gross Long (Old)
Conc_Gross_LE_8_TDR_Short_Old      Concentration: Top 8 Gross Short (Old)
```

#### Other Crop
```
Conc_Gross_LE_4_TDR_Long_Other     Concentration: Top 4 Gross Long (Other)
Conc_Gross_LE_4_TDR_Short_Other    Concentration: Top 4 Gross Short (Other)
Conc_Gross_LE_8_TDR_Long_Other     Concentration: Top 8 Gross Long (Other)
Conc_Gross_LE_8_TDR_Short_Other    Concentration: Top 8 Gross Short (Other)
```

### 7.2 Net Concentration (net long or net short positions)

#### All Traders
```
Conc_Net_LE_4_TDR_Long_All         Concentration: Top 4 Net Long (All)
Conc_Net_LE_4_TDR_Short_All        Concentration: Top 4 Net Short (All)
Conc_Net_LE_8_TDR_Long_All         Concentration: Top 8 Net Long (All)
Conc_Net_LE_8_TDR_Short_All        Concentration: Top 8 Net Short (All)
```

#### Old Crop
```
Conc_Net_LE_4_TDR_Long_Old         Concentration: Top 4 Net Long (Old)
Conc_Net_LE_4_TDR_Short_Old        Concentration: Top 4 Net Short (Old)
Conc_Net_LE_8_TDR_Long_Old         Concentration: Top 8 Net Long (Old)
Conc_Net_LE_8_TDR_Short_Old        Concentration: Top 8 Net Short (Old)
```

#### Other Crop
```
Conc_Net_LE_4_TDR_Long_Other       Concentration: Top 4 Net Long (Other)
Conc_Net_LE_4_TDR_Short_Other      Concentration: Top 4 Net Short (Other)
Conc_Net_LE_8_TDR_Long_Other       Concentration: Top 8 Net Long (Other)
Conc_Net_LE_8_TDR_Short_Other      Concentration: Top 8 Net Short (Other)
```

**Notes:**
- Values are percentages (0-100)
- Higher values indicate more concentrated positioning among large traders
- Gross concentration uses absolute positions (long + short separately)
- Net concentration uses net positions (long - short)

---

## 8. Metadata Columns

```
Contract_Units                     Contract Unit Specifications
CFTC_Contract_Market_Code_Quotes   Contract Market Code (for quotes)
CFTC_Market_Code_Quotes            Market Code (for quotes)
CFTC_Commodity_Code_Quotes         Commodity Code (for quotes)
CFTC_SubGroup_Code                 Commodity Subgroup Code
FutOnly_or_Combined                Report Type: FutOnly or Combined
```

**FutOnly_or_Combined values:**
- `FutOnly` = Futures-only positions
- `Combined` = Futures and Options combined

---

## Quick Reference: Key Column Patterns

### To Calculate Net Positions
```python
# Managed Money Net Position
net_mm = M_Money_Positions_Long_All - M_Money_Positions_Short_All

# Producer/Merchant Net Position
net_pm = Prod_Merc_Positions_Long_All - Prod_Merc_Positions_Short_All

# Swap Dealer Net Position (excluding spreads)
net_swap = Swap_Positions_Long_All - Swap_Positions_Short_All
```

### To Verify Data Integrity
```python
# Total Reportable should equal sum of all categories
total_reportable_long = (
    Prod_Merc_Positions_Long_All +
    Swap_Positions_Long_All +
    M_Money_Positions_Long_All +
    Other_Rept_Positions_Long_All
)

# Should equal Tot_Rept_Positions_Long_All

# Open Interest should equal all longs (or shorts)
open_interest = (
    Tot_Rept_Positions_Long_All +
    NonRept_Positions_Long_All
)

# Should equal Open_Interest_All
```

---

## Important Notes

1. **Disaggregated Format**: ICE COT uses the disaggregated trader classification system (4 reportable categories vs CFTC Legacy's 2 categories)

2. **Spread Positions**: Swap Dealers, Managed Money, and Other Reportable can hold spread positions. Producer/Merchant does not have spread positions in this format.

3. **Crop Year Distinctions**:
   - _All = Combined all crop years
   - _Old = Old crop year positions (for agricultural commodities)
   - _Other = Other crop year positions
   - For non-agricultural commodities, Old/Other may be empty

4. **ICE Europe Exchange**: This data is from ICE Futures Europe, not CFTC domestic exchanges

5. **Weekly Reporting**: Data is released weekly, typically on Fridays for positions as of Tuesday close

6. **Contract Units**: Each instrument has specific contract size specifications in the Contract_Units field

7. **Futures vs Combined**: Each instrument typically has both a FutOnly report and a Combined (futures+options) report

8. **Date Format**: Two date columns provided - use As_of_Date_Form_MM/DD/YYYY for easier parsing

9. **Percentage Validation**: All percentages for a given position type (Long, Short) should sum to approximately 100% of open interest

10. **Trader Confidentiality**: Some cells may show zero or be suppressed to protect trader confidentiality when there are fewer than 4 traders in a category

---

## Mapping to CFTC Legacy Format Concepts

For users familiar with CFTC Legacy reports, here's the conceptual mapping:

| CFTC Legacy | ICE Disaggregated |
|-------------|-------------------|
| Commercial Long/Short | ≈ Producer/Merchant + Swap Dealers |
| Non-Commercial Long/Short | ≈ Managed Money |
| Non-Reportable | Non-Reportable (same) |
| Spreading | Tracked separately for Swap, Managed Money, Other |

**Note:** This is conceptual only. The trader classification methodologies differ between CFTC Legacy and Disaggregated formats, so direct comparisons should be made carefully.

---

## Data Source

- **Source**: ICE Futures Europe
- **Format**: Disaggregated Commitments of Traders Report
- **Frequency**: Weekly (Tuesday positions, released Friday)
- **File Format**: CSV
- **Encoding**: UTF-8

---

## Version History

- **2025-01-11**: Initial documentation created for COT Analysis Dashboard project
