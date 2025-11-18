# ICE COT Data - Supabase Migration Complete ✅

## Overview

The ICE COT data has been **successfully migrated to Supabase**. This eliminates the need for local file storage and enables seamless integration with Streamlit Cloud deployment.

## Migration Summary

### What Changed

**Before** (File-based):
- Data stored locally in `ICE_COT_Historical_Merged.csv` (5.5 MB)
- Weekly updates appended to local CSV file
- Not compatible with Streamlit Cloud (no persistent local storage)

**After** (Supabase-based):
- Data stored in Supabase cloud database (`ice_cot_data` table)
- Weekly updates append directly to Supabase
- Fully compatible with Streamlit Cloud
- Accessible from anywhere with credentials

### Migration Date

**Completed**: 2025-01-11

### Data Uploaded

- **Total Records**: 8,897
- **Date Range**: 2011-01-04 to 2025-11-04
- **Instruments**: 13 (7 markets × 2 report types)
- **Markets**: Brent Crude, Gasoil, Dubai 1st Line, White Sugar, Cocoa, Robusta Coffee, Wheat

---

## Database Schema

### Table: `ice_cot_data`

**Key Columns** (optimized for dashboard use):

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key (auto-increment) |
| `market_name` | TEXT | Market name (e.g., "ICE Brent Crude Futures - ICE Futures Europe") |
| `report_date` | DATE | Report date (YYYY-MM-DD) |
| `report_type` | TEXT | 'FutOnly' or 'Combined' |
| `commodity_code` | TEXT | ICE commodity code (e.g., 'B' for Brent) |
| `open_interest_all` | NUMERIC | Total open interest |
| **Managed Money (Speculative)** | | |
| `m_money_long_all` | NUMERIC | Managed money long positions |
| `m_money_short_all` | NUMERIC | Managed money short positions |
| `m_money_spread_all` | NUMERIC | Managed money spread positions |
| **Producer/Merchant (Hedgers)** | | |
| `prod_merc_long_all` | NUMERIC | Producer/merchant long positions |
| `prod_merc_short_all` | NUMERIC | Producer/merchant short positions |
| **Swap Dealers** | | |
| `swap_long_all` | NUMERIC | Swap dealer long positions |
| `swap_short_all` | NUMERIC | Swap dealer short positions |
| `swap_spread_all` | NUMERIC | Swap dealer spread positions |
| **Other Reportable** | | |
| `other_rept_long_all` | NUMERIC | Other reportable long positions |
| `other_rept_short_all` | NUMERIC | Other reportable short positions |
| **Percentages** | | |
| `pct_oi_m_money_long_all` | NUMERIC | Managed money long % of OI |
| `pct_oi_m_money_short_all` | NUMERIC | Managed money short % of OI |
| **Additional Data** | | |
| `additional_data` | JSONB | All remaining 197 columns stored as JSON |

**Indexes Created:**
- `idx_ice_cot_market_date` - Fast queries by market and date
- `idx_ice_cot_date` - Fast queries by date range
- `idx_ice_cot_market` - Fast queries by market
- `idx_ice_cot_report_type` - Fast filtering by report type
- `idx_ice_cot_commodity` - Fast queries by commodity code

**Unique Constraint:**
- `(market_name, report_date, report_type)` - Prevents duplicate records

---

## Updated Workflow

### 1. Weekly Automatic Updates

**New Script**: `update_ice_supabase.py`

**What it does:**
1. Connects to Supabase
2. Checks latest date in database
3. Downloads current year CSV from ICE website
4. Identifies only NEW records (after latest date)
5. Uploads new records to Supabase (batch size: 500)

**Run manually:**
```bash
cd "instrument_management/ICE data"
source ../../venv/bin/activate
python update_ice_supabase.py
```

**Run via shell script:**
```bash
cd "instrument_management/ICE data"
./weekly_update_supabase.sh
```

**Schedule via cron (Friday 6 PM):**
```bash
0 18 * * 5 /Users/makson/Desktop/COT-Analysis/instrument_management/ICE\ data/weekly_update_supabase.sh
```

### 2. Credentials Configuration

The update script requires Supabase credentials via environment variables:

**Option 1: Environment Variables**
```bash
export SUPABASE_URL="https://rkirpnpjuckcxqllbnxu.supabase.co"
export SUPABASE_KEY="your_service_role_key"
```

**Option 2: .streamlit/secrets.toml** (for local development)
```toml
SUPABASE_URL = "https://rkirpnpjuckcxqllbnxu.supabase.co"
SUPABASE_KEY = "your_service_role_key"
```

**Option 3: Streamlit Cloud Secrets** (for production)
- Settings → Secrets → Add credentials

---

## Files Structure

### Active Files (Supabase-based)

```
instrument_management/ICE data/
├── update_ice_supabase.py           ← NEW: Supabase updater
├── weekly_update_supabase.sh        ← NEW: Shell script for cron
├── upload_to_supabase.py            ← One-time migration script
├── create_ice_cot_table.sql         ← Table creation SQL
├── SUPABASE_MIGRATION.md            ← This file
├── README.md                        ← Updated documentation
└── update_logs/                     ← Update logs directory
```

### Legacy Files (kept for reference)

```
├── ICE_COT_Historical_Merged.csv    ← LEGACY: Local merged file (5.5 MB)
├── update_ice_data.py               ← LEGACY: File-based updater
├── weekly_update.sh                 ← LEGACY: File-based shell script
├── merge_historical_data.py         ← LEGACY: Merge script
├── COTHist2011-2025.csv             ← LEGACY: Individual year files
└── backups/                         ← LEGACY: Local backups
```

**Note**: Legacy files can be archived or deleted once you're confident in the Supabase migration.

---

## Querying ICE Data from Supabase

### Example Queries

**Get latest Brent Crude positioning:**
```sql
SELECT
    report_date,
    report_type,
    open_interest_all,
    m_money_long_all,
    m_money_short_all,
    (m_money_long_all - m_money_short_all) as mm_net,
    pct_oi_m_money_long_all,
    pct_oi_m_money_short_all
FROM ice_cot_data
WHERE market_name = 'ICE Brent Crude Futures - ICE Futures Europe'
  AND report_type = 'FutOnly'
ORDER BY report_date DESC
LIMIT 10;
```

**Get all markets latest positioning:**
```sql
SELECT
    market_name,
    report_date,
    report_type,
    (m_money_long_all - m_money_short_all) as mm_net_positions,
    ((m_money_long_all - m_money_short_all) / open_interest_all * 100) as mm_net_pct_oi
FROM ice_cot_data
WHERE report_date = (SELECT MAX(report_date) FROM ice_cot_data)
  AND report_type = 'FutOnly'
ORDER BY market_name;
```

**Python query example:**
```python
from supabase import create_client
import os

# Initialize client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Query latest Brent data
result = supabase.table('ice_cot_data').select(
    'report_date, m_money_long_all, m_money_short_all, open_interest_all'
).eq('market_name', 'ICE Brent Crude Futures - ICE Futures Europe').eq(
    'report_type', 'FutOnly'
).order('report_date', desc=True).limit(52).execute()

df = pd.DataFrame(result.data)
```

---

## Integration with Dashboard

### Next Steps for Dashboard Integration

1. **Create ICE data fetcher module** (`src/ice_data_fetcher.py`)
   - Query Supabase for ICE data
   - Similar structure to `data_fetcher.py` but for ICE
   - Cache results with `@st.cache_data`

2. **Create ICE instrument mapping** (`instrument_management/ice_instruments.json`)
   - Map ICE codes (B, G, W, etc.) to full names
   - Define categories (Energy, Agricultural)
   - Map to price data sources if available

3. **Add ICE section to dashboard** (`src/dashboard_overview.py`)
   - New tab or section for ICE instruments
   - Show Managed Money positioning (equivalent to CFTC Non-Commercial)
   - Cross-asset analysis for ICE markets

4. **Create ICE-specific charts**
   - Position time series (Managed Money vs Producers)
   - Net positioning as % of OI
   - Concentration metrics
   - Week-over-week changes

### Example Dashboard Code

```python
import streamlit as st
from supabase import create_client

@st.cache_data(ttl=3600)
def fetch_ice_cot_data(market_name, report_type='FutOnly', limit=500):
    """Fetch ICE COT data from Supabase"""
    supabase = create_client(
        st.secrets['SUPABASE_URL'],
        st.secrets['SUPABASE_KEY']
    )

    result = supabase.table('ice_cot_data').select('*').eq(
        'market_name', market_name
    ).eq('report_type', report_type).order(
        'report_date', desc=True
    ).limit(limit).execute()

    df = pd.DataFrame(result.data)
    df['report_date'] = pd.to_datetime(df['report_date'])

    # Calculate derived metrics
    df['mm_net'] = df['m_money_long_all'] - df['m_money_short_all']
    df['mm_net_pct_oi'] = (df['mm_net'] / df['open_interest_all'] * 100)

    return df
```

---

## Testing & Verification

### Verify Migration Success

```bash
# 1. Check record count
psql -d your_database -c "SELECT COUNT(*) FROM ice_cot_data;"
# Expected: 8,897

# 2. Check date range
psql -d your_database -c "SELECT MIN(report_date), MAX(report_date) FROM ice_cot_data;"
# Expected: 2011-01-04 to 2025-11-04

# 3. Check instruments
psql -d your_database -c "SELECT DISTINCT market_name FROM ice_cot_data ORDER BY market_name;"
# Expected: 13 instruments

# 4. Test update script (should show "Database is up to date")
python update_ice_supabase.py
```

### Monitor Updates

```bash
# Check update logs
ls -lh update_logs/
tail -50 update_logs/update_*.log

# Check latest update time
psql -d your_database -c "SELECT MAX(report_date) FROM ice_cot_data;"
```

---

## Advantages of Supabase Migration

✅ **Cloud-based**: No local file storage needed
✅ **Streamlit Cloud compatible**: Works seamlessly with deployment
✅ **Efficient updates**: Only downloads/processes new records
✅ **Scalable**: Can handle millions of records
✅ **Indexed queries**: Fast retrieval for dashboard
✅ **Backup & recovery**: Automatic Supabase backups
✅ **Concurrent access**: Multiple users can query simultaneously
✅ **Version control**: Queries are version-controlled, not data files

---

## Troubleshooting

### Issue: "Database is up to date" but I know there's new data

**Solution:**
1. Check the ICE website manually: https://www.ice.com/report/122
2. Verify the latest available date
3. Check Supabase latest date:
   ```bash
   psql -d your_database -c "SELECT MAX(report_date) FROM ice_cot_data;"
   ```
4. If there's a gap, manually run the update script

### Issue: "SUPABASE_URL or SUPABASE_KEY not found"

**Solution:**
1. Check environment variables: `echo $SUPABASE_URL`
2. Load from secrets.toml:
   ```bash
   export SUPABASE_URL=$(grep "SUPABASE_URL" .streamlit/secrets.toml | cut -d'"' -f2)
   export SUPABASE_KEY=$(grep "SUPABASE_KEY" .streamlit/secrets.toml | cut -d'"' -f2)
   ```
3. Or set them manually

### Issue: Upload fails with "duplicate key" error

**Solution:**
- This is expected behavior with `upsert`
- The unique constraint prevents duplicates
- Records with same (market_name, report_date, report_type) will be updated, not duplicated

---

## Maintenance

### Weekly Checks (Recommended)

1. **Monday morning**: Verify Friday's update ran successfully
   ```bash
   tail -50 update_logs/update_*.log | grep "✓ Update completed"
   ```

2. **Check record count** (should increase by ~13 weekly):
   ```bash
   psql -d your_database -c "SELECT COUNT(*) FROM ice_cot_data;"
   ```

3. **Verify latest date** (should be last Tuesday):
   ```bash
   psql -d your_database -c "SELECT MAX(report_date) FROM ice_cot_data;"
   ```

### Annual Maintenance

- Review and archive old update logs (keep last 12 months)
- Verify data integrity (no gaps in weekly reports)
- Update documentation if ICE changes report format

---

## Contact & Resources

- **ICE COT Reports**: https://www.ice.com/report/122
- **Supabase Dashboard**: https://supabase.com/dashboard
- **Project Repository**: https://github.com/mwcapital/cot
- **Project Documentation**: `../../CLAUDE.md`

---

**Migration Status**: ✅ COMPLETE

**Last Updated**: 2025-01-11
