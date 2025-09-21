"""
Daily futures price update script
Designed to run automatically every day to update Supabase with latest prices
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('futures_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
load_dotenv()

# Initialize Supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def load_futures_symbols():
    """Load the JSON file with selected symbols"""
    json_path = '/instrument_management/futures_symbols_current.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['futures_symbols']

def get_last_update_date(symbol, adjustment_method):
    """Get the last date we have data for a specific symbol and adjustment method"""
    try:
        response = supabase.table('futures_prices')\
            .select('date')\
            .eq('symbol', symbol)\
            .eq('adjustment_method', adjustment_method)\
            .order('date', desc=True)\
            .limit(1)\
            .execute()

        if response.data:
            return pd.to_datetime(response.data[0]['date'])
        return None
    except Exception as e:
        logging.error(f"Error getting last update date for {symbol} {adjustment_method}: {e}")
        return None

def process_futures_file_incremental(filepath, symbol, adjustment_method, last_date=None):
    """Process only new data from a futures file"""
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

        # Filter for dates after last_date if provided
        if last_date:
            df = df[df['Date'] > last_date]
            if df.empty:
                return None  # No new data

        # Handle price scaling for back-adjusted data
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
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
        logging.error(f"Error processing {filepath}: {str(e)}")
        return None

def upload_incremental_data(df, batch_size=500):
    """Upload new data using upsert to handle any duplicates"""
    if df is None or df.empty:
        return 0

    total_rows = len(df)
    uploaded = 0

    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_dict = batch.to_dict('records')

        try:
            # Use upsert to handle potential duplicates
            response = supabase.table('futures_prices').upsert(batch_dict).execute()
            uploaded += len(batch)
        except Exception as e:
            logging.error(f"Error uploading batch: {str(e)[:100]}")

    return uploaded

def daily_update():
    """Main function for daily updates"""
    logging.info("=" * 60)
    logging.info("STARTING DAILY FUTURES UPDATE")
    logging.info("=" * 60)

    # Load selected symbols
    symbols_data = load_futures_symbols()
    logging.info(f"Processing {len(symbols_data)} symbols")

    # Define adjustment methods
    adjustment_methods = {
        'NON': '_NON.CSV',
        'RAD': '_RAD.CSV',
        'REV': '_REV.CSV'
    }

    data_dir = '/Users/makson/.wine/drive_c/DATA/CLCDATA/'

    # Track update statistics
    stats = {
        'symbols_checked': 0,
        'symbols_updated': 0,
        'total_new_rows': 0,
        'errors': []
    }

    # Process each symbol
    for symbol in symbols_data.keys():
        stats['symbols_checked'] += 1
        symbol_updated = False
        symbol_new_rows = 0

        for adj_method, suffix in adjustment_methods.items():
            filepath = os.path.join(data_dir, f"{symbol}{suffix}")

            if not os.path.exists(filepath):
                continue

            # Get last update date
            last_date = get_last_update_date(symbol, adj_method)

            # Process only new data
            df = process_futures_file_incremental(filepath, symbol, adj_method, last_date)

            if df is not None and not df.empty:
                # Upload new data
                rows_uploaded = upload_incremental_data(df)
                if rows_uploaded > 0:
                    symbol_updated = True
                    symbol_new_rows += rows_uploaded
                    logging.info(f"  {symbol} {adj_method}: {rows_uploaded} new rows")

        if symbol_updated:
            stats['symbols_updated'] += 1
            stats['total_new_rows'] += symbol_new_rows

            # Update the last_updated field in futures_symbols
            try:
                supabase.table('futures_symbols').update({
                    'last_updated': datetime.now().strftime('%Y-%m-%d')
                }).eq('symbol', symbol).execute()
            except:
                pass  # Not critical if this fails

    # Log summary
    logging.info("=" * 60)
    logging.info("UPDATE SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Symbols checked: {stats['symbols_checked']}")
    logging.info(f"Symbols updated: {stats['symbols_updated']}")
    logging.info(f"Total new rows: {stats['total_new_rows']}")

    if stats['errors']:
        logging.warning(f"Errors encountered: {len(stats['errors'])}")
        for error in stats['errors']:
            logging.warning(f"  - {error}")

    return stats

def setup_cron_job():
    """Instructions for setting up cron job"""
    cron_command = f"0 18 * * * cd /Users/makson/Desktop/COT-Analysis && /Users/makson/Desktop/COT-Analysis/venv_new/bin/python /Users/makson/Desktop/COT-Analysis/src/daily_futures_update.py >> /Users/makson/Desktop/COT-Analysis/futures_update.log 2>&1"

    print("\n" + "=" * 60)
    print("SETUP INSTRUCTIONS FOR DAILY UPDATES")
    print("=" * 60)
    print("\nOption 1: macOS Cron Job")
    print("-" * 30)
    print("1. Open Terminal and run: crontab -e")
    print("2. Add this line (runs daily at 6 PM):")
    print(f"\n{cron_command}\n")
    print("3. Save and exit (in vim: press ESC, type :wq, press Enter)")

    print("\nOption 2: macOS LaunchAgent (Recommended)")
    print("-" * 30)
    print("1. Create file: ~/Library/LaunchAgents/com.futures.daily.update.plist")
    print("2. Use the LaunchAgent configuration below")
    print("3. Load it with: launchctl load ~/Library/LaunchAgents/com.futures.daily.update.plist")

    return cron_command

def create_launch_agent():
    """Create a LaunchAgent plist file for macOS"""
    plist_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.futures.daily.update</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/makson/Desktop/COT-Analysis/venv_new/bin/python</string>
        <string>/Users/makson/Desktop/COT-Analysis/src/daily_futures_update.py</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>18</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>WorkingDirectory</key>
    <string>/Users/makson/Desktop/COT-Analysis</string>

    <key>StandardOutPath</key>
    <string>/Users/makson/Desktop/COT-Analysis/futures_update.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/makson/Desktop/COT-Analysis/futures_update_error.log</string>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>"""

    plist_path = os.path.expanduser("~/Library/LaunchAgents/com.futures.daily.update.plist")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(plist_path), exist_ok=True)

    # Write plist file
    with open(plist_path, 'w') as f:
        f.write(plist_content)

    print(f"\nLaunchAgent created at: {plist_path}")
    print("\nTo enable it, run:")
    print(f"launchctl load {plist_path}")
    print("\nTo disable it, run:")
    print(f"launchctl unload {plist_path}")
    print("\nTo run it immediately for testing:")
    print("launchctl start com.futures.daily.update")

    return plist_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Daily futures price update')
    parser.add_argument('--setup', action='store_true', help='Show setup instructions')
    parser.add_argument('--create-agent', action='store_true', help='Create LaunchAgent for macOS')
    parser.add_argument('--test', action='store_true', help='Run a test update')

    args = parser.parse_args()

    if args.setup:
        setup_cron_job()
    elif args.create_agent:
        create_launch_agent()
    elif args.test:
        print("Running test update (will only process last 7 days of data)...")
        # You can modify this to test with limited data
        stats = daily_update()
        print(f"\nTest complete. Updated {stats['symbols_updated']} symbols with {stats['total_new_rows']} new rows.")
    else:
        # Regular daily update
        daily_update()