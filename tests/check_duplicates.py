#!/usr/bin/env python3
"""Check for duplicate rows in Supabase futures_prices table"""

import json
from supabase import create_client

# Load config
with open('/Users/makson/Desktop/futures_supabase_sync/config.json', 'r') as f:
    config = json.load(f)

# Connect to Supabase
supabase = create_client(config['supabase_url'], config['supabase_key'])

# Check for duplicates by counting rows per symbol/date/adjustment
print("Checking for duplicates in futures_prices table...")

# Get a sample symbol to check
symbols = ['ES', 'ZG', 'ZC']
for symbol in symbols:
    print(f"\nChecking {symbol}...")

    # Get all data for this symbol
    response = supabase.table('futures_prices')\
        .select('date, adjustment_method')\
        .eq('symbol', symbol)\
        .order('date', desc=True)\
        .limit(100)\
        .execute()

    if response.data:
        # Count occurrences
        date_adjustment_pairs = {}
        for row in response.data:
            key = f"{row['date']}_{row['adjustment_method']}"
            date_adjustment_pairs[key] = date_adjustment_pairs.get(key, 0) + 1

        # Find duplicates
        duplicates = {k: v for k, v in date_adjustment_pairs.items() if v > 1}

        if duplicates:
            print(f"  ⚠️  Found duplicates:")
            for key, count in duplicates.items():
                print(f"    {key}: {count} rows")
        else:
            print(f"  ✓ No duplicates found in last 100 rows")

        # Show date range
        dates = [row['date'] for row in response.data]
        print(f"  Date range: {min(dates)} to {max(dates)}")

# Get total row count
print("\nGetting total row counts...")
response = supabase.rpc('count_rows', {}).execute()
print(f"Total rows in futures_prices: {response.data if response.data else 'Unable to count'}")