"""
Migrate selected futures price data to Supabase
Only migrates the 62 symbols with current data (2025-09-12)
Includes all three adjustment methods: NON, RAD, REV
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize Supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def create_tables_sql():
    """SQL to create tables in Supabase"""
    sql = """
    -- Create futures symbols table with metadata
    CREATE TABLE IF NOT EXISTS futures_symbols (
        symbol VARCHAR(10) PRIMARY KEY,
        name VARCHAR(100),
        exchange VARCHAR(50),
        category VARCHAR(30),
        last_updated DATE,
        created_at TIMESTAMP DEFAULT NOW()
    );

    -- Create futures prices table with adjustment method
    CREATE TABLE IF NOT EXISTS futures_prices (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        adjustment_method VARCHAR(3) NOT NULL, -- NON, RAD, or REV
        date DATE NOT NULL,
        open DECIMAL(12, 4),
        high DECIMAL(12, 4),
        low DECIMAL(12, 4),
        close DECIMAL(12, 4),
        volume BIGINT,
        open_interest BIGINT,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(symbol, adjustment_method, date)
    );

    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_futures_symbol_adj_date ON futures_prices(symbol, adjustment_method, date);
    CREATE INDEX IF NOT EXISTS idx_futures_symbol ON futures_prices(symbol);
    CREATE INDEX IF NOT EXISTS idx_futures_date ON futures_prices(date);
    CREATE INDEX IF NOT EXISTS idx_futures_adjustment ON futures_prices(adjustment_method);
    """

    print("=" * 80)
    print("IMPORTANT: Run this SQL in Supabase SQL editor first:")
    print("=" * 80)
    print(sql)
    print("=" * 80)
    return sql

def load_futures_symbols():
    """Load the JSON file with selected symbols"""
    json_path = '/instrument_management/futures_symbols_current.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['futures_symbols']

def process_futures_file(filepath, symbol, adjustment_method):
    """Process a single futures file and return dataframe"""
    try:
        # Read the file
        df = pd.read_csv(filepath, header=None,
                        names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])

        # Parse dates
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            except:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Handle price scaling for back-adjusted data
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            # Apply scaling for values > 10000
            mask_high = df[col] > 10000
            df.loc[mask_high, col] = df.loc[mask_high, col] / 10

        # Add metadata columns
        df['symbol'] = symbol
        df['adjustment_method'] = adjustment_method

        # Rename columns for database
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'OI': 'open_interest'
        })

        # Select columns and clean
        df = df[['symbol', 'adjustment_method', 'date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        # Convert date to string for JSON serialization
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        return df

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def upload_symbol_metadata(symbols_data):
    """Upload symbol metadata to futures_symbols table"""
    print("\nUploading symbol metadata...")

    metadata_list = []
    for symbol, info in symbols_data.items():
        metadata_list.append({
            'symbol': symbol,
            'name': info['name'],
            'exchange': info['exchange'],
            'category': info['category'],
            'last_updated': '2025-09-12'  # From our analysis
        })

    try:
        # Upsert metadata
        response = supabase.table('futures_symbols').upsert(metadata_list).execute()
        print(f"✓ Uploaded metadata for {len(metadata_list)} symbols")
        return True
    except Exception as e:
        print(f"✗ Error uploading metadata: {str(e)}")
        return False

def upload_price_data(df, batch_size=1000):
    """Upload price data in batches"""
    total_rows = len(df)
    uploaded = 0

    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_dict = batch.to_dict('records')

        try:
            response = supabase.table('futures_prices').upsert(batch_dict).execute()
            uploaded += len(batch)
            if (i + batch_size) % 5000 == 0:  # Progress update every 5000 rows
                print(f"    Uploaded {uploaded}/{total_rows} rows")
        except Exception as e:
            print(f"    Error in batch {i//batch_size + 1}: {str(e)[:100]}")

    return uploaded

def migrate_futures_data():
    """Main migration function"""
    print("\n" + "=" * 80)
    print("FUTURES DATA MIGRATION TO SUPABASE")
    print("=" * 80)

    # Show SQL for table creation
    create_tables_sql()

    print("\nAssuming tables have been created. Starting migration...")
    print("(If tables don't exist, the migration will fail)")

    # Load selected symbols
    symbols_data = load_futures_symbols()
    print(f"\nLoaded {len(symbols_data)} symbols from JSON")

    # Upload metadata first
    if not upload_symbol_metadata(symbols_data):
        print("Failed to upload metadata. Please check your Supabase configuration.")
        return

    # Define adjustment methods and their file suffixes
    adjustment_methods = {
        'NON': '_NON.CSV',  # Non-adjusted
        'RAD': '_RAD.CSV',  # ?-adjusted
        'REV': '_REV.CSV'   # Reverse-adjusted
    }

    data_dir = '/Users/makson/.wine/drive_c/DATA/CLCDATA/'

    # Process each symbol
    results = []
    total_symbols = len(symbols_data)

    for idx, (symbol, info) in enumerate(symbols_data.items(), 1):
        print(f"\n[{idx}/{total_symbols}] Processing {symbol} - {info['name']}")
        symbol_results = {'symbol': symbol, 'name': info['name']}

        # Process each adjustment method
        for adj_method, suffix in adjustment_methods.items():
            filepath = os.path.join(data_dir, f"{symbol}{suffix}")

            if not os.path.exists(filepath):
                print(f"  ✗ {adj_method}: File not found")
                symbol_results[adj_method] = 0
                continue

            print(f"  Processing {adj_method}...", end='')

            # Load and process the file
            df = process_futures_file(filepath, symbol, adj_method)

            if df is not None and not df.empty:
                # Upload to Supabase
                uploaded = upload_price_data(df)
                symbol_results[adj_method] = uploaded
                print(f" ✓ {uploaded} rows")
            else:
                symbol_results[adj_method] = 0
                print(f" ✗ Failed")

        results.append(symbol_results)

        # Small delay to avoid rate limiting
        if idx % 10 == 0:
            time.sleep(1)

    # Print summary
    print("\n" + "=" * 80)
    print("MIGRATION SUMMARY")
    print("=" * 80)

    total_rows = 0
    for result in results:
        total = result.get('NON', 0) + result.get('RAD', 0) + result.get('REV', 0)
        total_rows += total
        print(f"{result['symbol']:<6} {result['name']:<40} "
              f"NON: {result.get('NON', 0):>6} "
              f"RAD: {result.get('RAD', 0):>6} "
              f"REV: {result.get('REV', 0):>6} "
              f"Total: {total:>7}")

    print(f"\nTotal rows migrated: {total_rows:,}")

    # Test query
    print("\n" + "=" * 80)
    print("TESTING DATA RETRIEVAL")
    print("=" * 80)

    try:
        # Test fetching some data
        test_response = supabase.table('futures_prices').select("*").eq('symbol', 'ES').eq('adjustment_method', 'NON').limit(5).execute()
        print(f"✓ Successfully retrieved {len(test_response.data)} test rows for ES (NON)")

        # Test fetching symbol metadata
        meta_response = supabase.table('futures_symbols').select("*").eq('category', 'Index').limit(5).execute()
        print(f"✓ Successfully retrieved {len(meta_response.data)} Index symbols")

    except Exception as e:
        print(f"✗ Test query failed: {str(e)}")

    return results

if __name__ == "__main__":
    results = migrate_futures_data()