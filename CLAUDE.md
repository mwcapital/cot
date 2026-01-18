# CFTC COT Analysis Dashboard - Claude Instructions

## Project Overview

This is a Streamlit-based web application for analyzing CFTC (Commodity Futures Trading Commission) Commitments of Traders (COT) data. The application provides interactive visualizations and analytics for commodity futures positioning data.

**Live App**: Deployed on Streamlit Cloud
**Repository**: https://github.com/mwcapital/cot

## Tech Stack

- **Framework**: Streamlit
- **Database**: Supabase (PostgreSQL)
- **Data Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Price Data**: Stored in Supabase with multiple adjustment methods (NON, RAD, REV)
- **COT Data**: Fetched from CFTC API

## Key Architecture

### Main Entry Point
- `src/main.py` - Main Streamlit application entry point

### Core Modules
- `src/dashboard_overview.py` - Main dashboard with overview table, plots, and positioning analysis
- `src/data_fetcher.py` - Fetches COT data from CFTC API with instrument stitching
- `src/futures_price_fetcher.py` - Fetches futures price data from Supabase
- `src/multi_instrument_handler.py` - Handles multi-instrument analysis flows
- `src/historical_data_loader.py` - Loads pre-2000 historical data from FUT86_16.txt
- `src/display_functions_exact.py` - Single instrument analysis charts with time range controls
- `src/display_functions_futures_first.py` - Futures-first display functions
- `src/display_functions_clean.py` - Time series charts with Lightweight Charts integration

### Chart Modules
- `src/charts/cross_asset_analysis.py` - Cross-asset comparison charts
- `src/charts/market_state_clusters.py` - Market state cluster analysis
- `src/charts/percentile_charts.py` - Percentile-based visualizations
- `src/charts/trader_participation_analysis.py` - Trader participation metrics

### Configuration
- `instrument_management/futures/futures_symbols_enhanced.json` - Master mapping of futures symbols to COT instruments with categories

## Important Patterns

### 1. Supabase Configuration

**ALWAYS use this pattern for Supabase connections:**

```python
def get_supabase_client():
    """Initialize Supabase client using secure configuration"""
    try:
        # Try Streamlit secrets first (production)
        if hasattr(st, 'secrets') and 'SUPABASE_URL' in st.secrets:
            url = st.secrets['SUPABASE_URL']
            key = st.secrets['SUPABASE_KEY']
        else:
            # Fallback to environment variables
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            return None

        return create_client(url, key)
    except Exception as e:
        return None
```

**Local Development Setup:**
- Credentials stored in `.streamlit/secrets.toml` (gitignored)
- Format:
  ```toml
  SUPABASE_URL = "https://rkirpnpjuckcxqllbnxu.supabase.co"
  SUPABASE_KEY = "your_service_role_key"
  ```

**Production (Streamlit Cloud):**
- Credentials configured in Streamlit Cloud Secrets
- Uses same service_role key as local

### 2. Price Data Queries

**CRITICAL: Always filter by adjustment_method when querying prices**

```python
response = supabase.from_('futures_prices').select(
    'date, close'
).eq('symbol', futures_symbol).eq(
    'adjustment_method', 'NON'  # REQUIRED: NON, RAD, or REV
).gte(
    'date', start_date.strftime('%Y-%m-%d')
).order('date', desc=False).execute()
```

**Why:** The futures_prices table contains multiple adjustment methods. Always specify which one to use for consistent results.

### 3. Instrument Mapping

The `futures_symbols_enhanced.json` file contains the master mapping:

```json
{
  "futures_symbols": {
    "ZG": {
      "name": "Gold Electronic",
      "exchange": "COMEX",
      "category": "Metals",
      "cot_mapping": {
        "instruments": ["GOLD - COMMODITY EXCHANGE INC."],
        "codes": ["088691"],
        "matched": true
      }
    }
  }
}
```

**Categories used:**
- Metals
- Energy
- Financial
- Currency
- Agricultural
- Index

### 4. Correlation Calculations

**Use NON-adjusted prices for all correlation calculations:**

```python
response = supabase.from_('futures_prices').select(
    'date, close'
).eq('symbol', futures_symbol).eq(
    'adjustment_method', 'NON'
).gte('date', start_date.strftime('%Y-%m-%d')).order('date', desc=False).execute()
```

Weekly returns are calculated Tuesday-to-Tuesday to align with COT report dates.

### 5. Positioning Concentration Analysis

**Use absolute values for concentration metrics:**

```python
# Calculate net as % of open interest (using absolute value for concentration)
df['net_pct_oi'] = (abs(df[cols['net']]) / df['open_interest_all'] * 100).fillna(0)
```

**Why:** Concentration analysis focuses on magnitude of positioning, not direction (long/short).

### 6. Instrument Stitching (Historical Data)

Some instruments have been renamed over time in CFTC data. The app stitches multiple API queries + historical file data to create continuous time series.

**Historical Data File:**
- Location: `instrument_management/LegacyF/FUT86_16.txt`
- Contains: Legacy Futures COT data from 1986-2016
- Loaded by: `src/historical_data_loader.py`

**Stitched Instruments:**

| Current Name | Historical Names (API) | FUT86_16.txt Name |
|-------------|----------------------|-------------------|
| CRUDE OIL, LIGHT SWEET - NYMEX | CRUDE OIL, LIGHT SWEET - NYMEX | CRUDE OIL, LIGHT-NEW YORK MERCANTILE EXCHANGE |
| NAT GAS NYME | NATURAL GAS - NYMEX | NATURAL GAS- NEW YORK MERCANTILE EXCHANGE |
| GASOLINE RBOB - NYMEX | UNLEADED GASOLINE + GASOLINE BLENDSTOCK | NO.2 HEATING OIL, N.Y. HARBOR-NEW YORK MERCANTILE EXCHANGE |
| COPPER- #1 - COMEX | COPPER-GRADE #1 - COMEX | COPPER-COMMODITY EXCHANGE INC. |
| E-MINI S&P 500 - CME | S&P 500 STOCK INDEX - CME | STANDARD & POORS 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE |
| NASDAQ MINI - CME | NASDAQ-100 STOCK INDEX (MINI) + NASDAQ-100 | NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE |
| DJIA x $5 - CBOT | DJIA Consolidated (CME+CBOT) | DOW JONES INDUSTRIAL AVG- CBOT |
| RUSSELL E-MINI - CME | RUSSELL 2000 MINI (ICE) + E-MINI RUSSELL 2000 | RUSSELL 2000 MINI INDEX - ICE |

**Note:** Russell 2000 moved from ICE to CME in 2017, so both exchanges are stitched.

**Stitching Pattern in `data_fetcher.py`:**
```python
# Example: Multiple API queries merged, then historical file stitched
if instrument_name_clean == "GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE":
    historical_1_results = _fetch_with_retry(client, DATASET_CODE,
        "market_and_exchange_names='UNLEADED GASOLINE, N.Y. HARBOR - ...'", ...)
    historical_2_results = _fetch_with_retry(client, DATASET_CODE,
        "market_and_exchange_names='GASOLINE BLENDSTOCK (RBOB) - ...'", ...)
    current_results = _fetch_with_retry(client, DATASET_CODE,
        "market_and_exchange_names='GASOLINE RBOB - ...'", ...)
    results = historical_1_results + historical_2_results + current_results
```

### 7. Time Range Selectors

**Pattern for adding time range buttons to charts:**

```python
# Add time range selector buttons
st.markdown("#### Select Time Range")
col_buttons = st.columns(5)

# Initialize session state
if 'chart_name_range' not in st.session_state:
    st.session_state.chart_name_range = '2Y'  # Default

# Range buttons
with col_buttons[0]:
    if st.button("1Y", key="chart_range_1y", use_container_width=True,
                type="primary" if st.session_state.chart_name_range == '1Y' else "secondary"):
        st.session_state.chart_name_range = '1Y'
        st.rerun()
# ... repeat for 2Y, 5Y, 10Y, All

# Filter data based on selection
if st.session_state.chart_name_range == '1Y':
    start_date = latest_date - pd.DateOffset(years=1)
    df_filtered = df_sorted[df_sorted['report_date_as_yyyy_mm_dd'] >= start_date].copy()
```

### 8. Friendly Display Names

**Use `futures_symbols_enhanced.json` for friendly instrument names:**

```python
def get_short_instrument_name(instrument):
    """Get friendly display name from futures_symbols_enhanced.json mapping"""
    mapping_path = 'instrument_management/futures/futures_symbols_enhanced.json'
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    for symbol, symbol_data in mapping_data['futures_symbols'].items():
        cot_instruments = symbol_data.get('cot_mapping', {}).get('instruments', [])
        for cot_name in cot_instruments:
            if cot_name == instrument or instrument.startswith(cot_name.split(' - ')[0]):
                return symbol_data.get('name', instrument.split('-')[0].strip())
    return instrument.split('-')[0].strip()
```

This helper is in `src/charts/cross_asset_analysis.py` and maps COT names like "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE" to friendly names like "E-mini S&P 500".

## File Paths

### NEVER use absolute paths in code
**BAD:**
```python
config_path = '/Users/makson/Desktop/futures_supabase_sync/config.json'
```

**GOOD:**
```python
mapping_path = 'instrument_management/futures/futures_symbols_enhanced.json'
```

**GOOD (with relative path handling):**
```python
mapping_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'instrument_management', 'futures', 'futures_symbols_enhanced.json')
```

## Security Best Practices

### What NOT to commit:
- Hardcoded API keys or credentials
- Local file paths (e.g., `/Users/makson/...`)
- `.streamlit/secrets.toml` (already gitignored)
- Any files containing `supabase_key` or tokens

### Before committing:
1. Check for hardcoded credentials: `grep -r "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" src/`
2. Check for local paths: `grep -r "/Users/makson" src/`
3. Verify .gitignore includes `.streamlit/` and `*.toml`

### Git Commit Messages:
- Do NOT include AI attribution (no "Generated with Claude Code")
- Use clear, technical descriptions of changes
- Example: "Add positioning concentration analysis to main dashboard"

## Dashboard Structure

### Main Page Sections (in order):
1. **API Configuration** - Token input (optional)
2. **Commodity Markets Overview Table** - Key metrics for all instruments
3. **Cross-Asset Z-Score Analysis** - Positioning z-scores
4. **Market Structure Matrix** - Percentile-based scatter plot
5. **Positioning Concentration Analysis** - Category-based concentration charts

### Multi-Instrument Analysis:
- Cross-Asset
- Market Matrix
- WoW Changes
- Positioning Conc. (manual selection)
- Participation
- Strength Matrix

### Single Instrument Analysis - Trader Participation Analysis Tab:
- **Average Position Per Trader** - Concentration metrics using Net positions (4 or fewer traders)
- **Concentration Momentum** - Momentum analysis of concentration metrics
- **Participant Behavior Clusters** - Cluster analysis of trader behavior
- **Market Microstructure Analysis** - Detailed market structure metrics
- **Regime Detection** - Market regime classification based on positioning extremes and flow intensity

### Deprecated/Stashed Ideas (`old_ideas_stash/`):
- **Heterogeneity Index** - Attempted to measure commercial vs non-commercial disagreement
  - Removed due to questionable math in Component 3 (percentile subtraction)
  - Code preserved in `old_ideas_stash/heterogeneity_index_analysis.py`

## Common Tasks

### Adding a new chart to main dashboard:

1. Create chart function in appropriate module under `src/charts/`
2. Import in `dashboard_overview.py`
3. Create display function in `dashboard_overview.py`
4. Call display function in `display_dashboard()` at appropriate location
5. Use `st.markdown("---")` to separate sections
6. Use `st.subheader()` for section titles

### Adding a new instrument category:

1. Update `futures_symbols_enhanced.json` with new instruments
2. Set appropriate "category" field
3. Ensure "cot_mapping" is populated with matched: true and COT codes
4. The dashboard will automatically pick up new categories

### Debugging data issues:

1. Check Streamlit Cloud logs for errors
2. Look for 401 errors (authentication issues)
3. Verify Supabase credentials are correct in Streamlit Cloud secrets
4. Check if data exists in Supabase for the symbol
5. Verify adjustment_method filter is applied

## Recent Important Changes

### 2026-01-10 Changes:
1. **Renamed "Participation Density Dashboard" to "Average Position Per Trader"**:
   - More accurate description of what the analysis shows
   - Consistent naming throughout the UI

2. **Removed Concentration Divergence analysis**:
   - Was dividing position share by trader share which lacked clear economic meaning
   - API already provides % of OI directly - no need for complex calculations
   - Simplified the Trader Participation Analysis tab

3. **Removed Heterogeneity & Regime Analysis**:
   - Component 3 (Percentile Distance) was mathematically questionable
   - Only looked at LONG positions, ignored short side
   - Index didn't reliably correlate with price reversals
   - Code stashed in `old_ideas_stash/` for potential future improvements

4. **Created `old_ideas_stash/` folder**:
   - Gitignored folder for storing deprecated analysis ideas
   - Preserves code for potential future revisiting
   - Contains documentation of why ideas were removed

### 2026-01 Changes (earlier):
1. **Instrument stitching for extended history**:
   - Added stitching for Gasoline RBOB, Copper, and all Index instruments
   - Russell 2000 stitches ICE + CME data (contract moved exchanges in 2017)
   - Fixed FUT86_16.txt file path in historical_data_loader.py

2. **Changed YTD to 2-year percentiles**:
   - Overview table now shows 2Y percentiles instead of YTD
   - More consistent with other 2-year lookback calculations

3. **Friendly display names for Index instruments**:
   - Uses JSON mapping instead of splitting on "-"
   - Fixes "E, E, E" display issue for E-MINI instruments

### 2025-09 Changes:
1. **Fixed correlation discrepancies** between local and cloud by:
   - Filtering all price queries by `adjustment_method='NON'`
   - Using Streamlit secrets consistently in all modules

2. **Added Positioning Concentration Analysis** to main dashboard:
   - Category-based selection (Metals, Energy, etc.)
   - Absolute value calculations for concentration
   - Two charts: time series + bar chart
   - Removed average position per trader chart

3. **Removed all hardcoded paths**:
   - Eliminated local config.json references
   - All credentials via Streamlit secrets or env vars
   - Use relative paths for local files

## Data Sources

### COT Data:
- Source: CFTC API (https://publicreporting.cftc.gov/resource/)
- Update frequency: Weekly (Tuesday releases)
- Fields: positions_long_all, positions_short_all, open_interest_all, trader counts, concentration metrics

### Price Data:
- Source: Supabase (futures_prices table)
- Adjustment methods: NON (non-adjusted), RAD (ratio adjusted), REV (reverse adjusted)
- Update frequency: Daily via daily_futures_update.py
- Fields: date, open, high, low, close, volume, open_interest, symbol, adjustment_method

## Development Workflow

### Local Development:
1. Activate virtual environment: `source venv_new/bin/activate`
2. Ensure `.streamlit/secrets.toml` exists with credentials
3. Run: `streamlit run src/main.py`

### Deployment:
1. Make changes locally
2. Test thoroughly
3. Commit with clear message (no AI attribution)
4. Push to GitHub
5. Streamlit Cloud auto-deploys from main branch

### Authentication:
- GitHub uses SSH keys or Personal Access Tokens
- SSH key stored in GitHub settings
- Use `git@github.com:mwcapital/cot.git` format for SSH

## Code Style

- No emojis in code unless explicitly requested by user
- Use clear variable names
- Add comments for complex logic
- Use type hints where helpful
- Follow existing patterns in the codebase
- Keep functions focused and single-purpose

## Performance Considerations

- Use `@st.cache_data(ttl=3600)` for expensive data fetches
- Parallel processing with `ThreadPoolExecutor` for fetching multiple instruments
- Progress bars for long-running operations
- Limit data queries with date ranges where possible

## Troubleshooting

### "No data available" errors:
- Check if instrument has COT mapping in futures_symbols_enhanced.json
- Verify COT code is correct
- Check CFTC API availability

### Correlation columns empty:
- Verify futures symbol has price data in Supabase
- Check adjustment_method filter is applied
- Ensure Supabase credentials are correct

### Local vs Cloud differences:
- Compare Streamlit secrets configuration
- Check for hardcoded paths or credentials
- Verify same service_role key is used in both environments
- Check logs for specific errors

## Contact & Resources

- CFTC API Documentation: https://publicreporting.cftc.gov/
- Streamlit Documentation: https://docs.streamlit.io/
- Plotly Documentation: https://plotly.com/python/
- Supabase Documentation: https://supabase.com/docs
