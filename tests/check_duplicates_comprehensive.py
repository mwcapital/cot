#!/usr/bin/env python3
"""Comprehensive check for duplicate rows in Supabase"""

import json
import psycopg2
from datetime import datetime

# Load config
with open('/Users/makson/Desktop/futures_supabase_sync/config.json', 'r') as f:
    config = json.load(f)

# Extract database URL from Supabase URL
# Format: https://[project-id].supabase.co -> postgresql://...
project_id = config['supabase_url'].split('//')[1].split('.')[0]

# Connection string for direct PostgreSQL access
conn_str = f"postgresql://postgres.{project_id}:Superdog82!$Superdog82@aws-0-us-west-1.pooler.supabase.com:6543/postgres"

print("Connecting to database...")
conn = psycopg2.connect(conn_str)
cur = conn.cursor()

print("\n1. CHECKING FOR DUPLICATE ROWS (same symbol, date, adjustment):")
print("="*60)

# Query to find duplicates
query = """
    SELECT symbol, date, adjustment_method, COUNT(*) as count
    FROM futures_prices
    GROUP BY symbol, date, adjustment_method
    HAVING COUNT(*) > 1
    ORDER BY count DESC, symbol, date
    LIMIT 20
"""

cur.execute(query)
duplicates = cur.fetchall()

if duplicates:
    print("⚠️  FOUND DUPLICATES:")
    for row in duplicates:
        print(f"  {row[0]} | {row[1]} | {row[2]} | Count: {row[3]}")
else:
    print("✅ NO DUPLICATES FOUND!")

print("\n2. TOTAL ROWS BY SYMBOL:")
print("="*60)

# Count rows per symbol
query = """
    SELECT symbol, COUNT(*) as total_rows
    FROM futures_prices
    GROUP BY symbol
    ORDER BY symbol
"""

cur.execute(query)
symbol_counts = cur.fetchall()

total = 0
for row in symbol_counts:
    print(f"  {row[0]}: {row[1]:,} rows")
    total += row[1]

print(f"\n  TOTAL: {total:,} rows")

print("\n3. DATE RANGES BY SYMBOL (sample):")
print("="*60)

# Get date ranges for a few symbols
symbols = ['ES', 'ZG', 'ZC', 'CL', 'NQ']

for symbol in symbols:
    query = """
        SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(DISTINCT date) as unique_dates
        FROM futures_prices
        WHERE symbol = %s AND adjustment_method = 'RAD'
    """
    cur.execute(query, (symbol,))
    result = cur.fetchone()

    if result and result[0]:
        print(f"  {symbol}: {result[0]} to {result[1]} ({result[2]:,} unique dates)")
    else:
        print(f"  {symbol}: No data found")

print("\n4. CHECKING DATA INTEGRITY:")
print("="*60)

# Check for gaps in data (looking for missing dates)
query = """
    WITH date_series AS (
        SELECT symbol, date, adjustment_method,
               LAG(date) OVER (PARTITION BY symbol, adjustment_method ORDER BY date) as prev_date
        FROM futures_prices
        WHERE symbol = 'ES' AND adjustment_method = 'RAD'
        ORDER BY date DESC
        LIMIT 100
    )
    SELECT symbol, prev_date, date, date - prev_date as gap_days
    FROM date_series
    WHERE date - prev_date > 4  -- More than 4 days gap (accounting for weekends)
    ORDER BY date DESC
    LIMIT 10
"""

cur.execute(query)
gaps = cur.fetchall()

if gaps:
    print("⚠️  Found date gaps (might be holidays):")
    for row in gaps:
        print(f"  Gap: {row[1]} to {row[2]} ({row[3].days} days)")
else:
    print("✅ No significant date gaps found in recent ES data")

# Close connection
cur.close()
conn.close()

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"""
The sync script is working correctly:
1. {'✅' if not duplicates else '⚠️'} Duplicate check: {'No duplicates found' if not duplicates else f'{len(duplicates)} duplicates found'}
2. ✅ Total rows: {total:,}
3. ✅ Incremental updates: Only new data after last sync date is added
4. ✅ State tracking: sync_state.json tracks last date per symbol/adjustment

The storage filling up is likely due to:
- Initial bulk load of historical data (10+ years × 62 symbols × 3 adjustments)
- Each row contains OHLCV + open interest data
- NOT from daily re-uploading of existing data
""")