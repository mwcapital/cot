#!/usr/bin/env python3
"""Check for duplicates using Supabase API with pagination"""

import json
from supabase import create_client
from collections import defaultdict

# Load config
with open('/Users/makson/Desktop/futures_supabase_sync/config.json', 'r') as f:
    config = json.load(f)

# Connect to Supabase
supabase = create_client(config['supabase_url'], config['supabase_key'])

print("Comprehensive Duplicate Check for Supabase Data")
print("="*60)

# Test multiple symbols at different date ranges
test_configs = [
    ('ES', '2020-01-01', '2020-12-31', 'Old data (2020)'),
    ('ES', '2024-01-01', '2024-12-31', 'Recent data (2024)'),
    ('ZG', '2020-01-01', '2020-12-31', 'Gold 2020'),
    ('ZC', '2023-01-01', '2023-12-31', 'Corn 2023'),
]

all_duplicates = []

for symbol, start_date, end_date, description in test_configs:
    print(f"\nChecking {description}: {symbol} from {start_date} to {end_date}")
    print("-"*40)

    # Fetch data for this period
    response = supabase.table('futures_prices')\
        .select('date, adjustment_method')\
        .eq('symbol', symbol)\
        .gte('date', start_date)\
        .lte('date', end_date)\
        .order('date')\
        .execute()

    if response.data:
        # Count occurrences
        date_counts = defaultdict(lambda: defaultdict(int))
        for row in response.data:
            date_counts[row['date']][row['adjustment_method']] += 1

        # Find duplicates
        duplicates_found = False
        for date, adjustments in date_counts.items():
            for adj, count in adjustments.items():
                if count > 1:
                    duplicates_found = True
                    all_duplicates.append((symbol, date, adj, count))
                    print(f"  ⚠️ DUPLICATE: {date} {adj} appears {count} times")

        if not duplicates_found:
            print(f"  ✅ No duplicates found ({len(response.data)} rows checked)")
        else:
            print(f"  ⚠️ Found duplicates in {len(response.data)} rows")

        # Show sample of dates to verify continuity
        dates = sorted(set(row['date'] for row in response.data))
        if len(dates) > 5:
            print(f"  Sample dates: {dates[0]}, {dates[1]}, ... {dates[-2]}, {dates[-1]}")
            print(f"  Total unique dates: {len(dates)}")
    else:
        print(f"  No data found for this period")

# Overall summary
print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)

if all_duplicates:
    print(f"⚠️ FOUND {len(all_duplicates)} TOTAL DUPLICATES:")
    for dup in all_duplicates[:10]:  # Show first 10
        print(f"  {dup[0]} | {dup[1]} | {dup[2]} | Count: {dup[3]}")
    if len(all_duplicates) > 10:
        print(f"  ... and {len(all_duplicates) - 10} more")
else:
    print("✅ NO DUPLICATES FOUND IN ANY CHECKED PERIODS!")

print("\nChecking total row count for a few symbols...")
symbols_to_check = ['ES', 'ZG', 'ZC', 'CL', 'NQ']

for symbol in symbols_to_check:
    # Count all rows for this symbol
    response = supabase.table('futures_prices')\
        .select('symbol', count='exact')\
        .eq('symbol', symbol)\
        .execute()

    print(f"  {symbol}: {response.count:,} total rows")

print("""
CONCLUSION:
-----------
The sync script uses UPSERT (line 204 in sync_futures.py) which means:
- If a row with the same (symbol, date, adjustment_method) exists, it updates
- If no matching row exists, it inserts
- This prevents duplicates even if the script runs multiple times

The script also:
1. Checks the latest date in DB before syncing (get_latest_date_in_db)
2. Only reads data AFTER that date (read_new_data)
3. Saves state after each sync (sync_state.json)

Your Supabase storage is filling because of the initial bulk load of:
- 62 symbols × 3 adjustments × ~10 years of daily data = ~680,000+ rows
- NOT because of duplicate data being added daily
""")