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
- `src/data_fetcher.py` - Fetches COT data from CFTC API
- `src/futures_price_fetcher.py` - Fetches futures price data from Supabase
- `src/multi_instrument_handler.py` - Handles multi-instrument analysis flows
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
