#!/usr/bin/env python3
"""Check what futures data is available in Supabase"""

import json
from supabase import create_client

# Load config
with open('/Users/makson/Desktop/futures_supabase_sync/config.json', 'r') as f:
    config = json.load(f)

# Connect to Supabase
supabase = create_client(config['supabase_url'], config['supabase_key'])

print("Checking available futures symbols in Supabase...")
print("="*60)

# Get unique symbols
response = supabase.table('futures_prices')\
    .select('symbol')\
    .execute()

if response.data:
    symbols = set(row['symbol'] for row in response.data)
    print(f"Found {len(symbols)} unique symbols:")
    for symbol in sorted(symbols):
        # Count rows for this symbol
        count_response = supabase.table('futures_prices')\
            .select('symbol', count='exact')\
            .eq('symbol', symbol)\
            .execute()
        print(f"  {symbol}: {count_response.count:,} rows")
else:
    print("No data found in futures_prices table")

print("\nChecking specifically for ZG data...")
response = supabase.table('futures_prices')\
    .select('*')\
    .eq('symbol', 'ZG')\
    .limit(5)\
    .execute()

if response.data:
    print(f"Found ZG data: {len(response.data)} sample rows")
    for row in response.data:
        print(f"  {row['date']} | {row['adjustment_method']} | Close: {row['close']}")
else:
    print("No ZG data found in Supabase")

print("\nChecking sync state file...")
with open('/Users/makson/Desktop/futures_supabase_sync/sync_state.json', 'r') as f:
    state = json.load(f)

zg_entries = [k for k in state.keys() if k.startswith('ZG_')]
if zg_entries:
    print(f"ZG sync state entries found:")
    for entry in zg_entries:
        print(f"  {entry}: {state[entry]}")
else:
    print("No ZG entries in sync state")