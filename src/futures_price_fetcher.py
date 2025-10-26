#!/usr/bin/env python3
"""
Fetch futures price data from Supabase for COT analysis integration
"""

import os
import json
from datetime import datetime, timedelta
import pandas as pd
from supabase import create_client, Client
from typing import Optional, Dict, List, Tuple
try:
    import streamlit as st
except ImportError:
    st = None

class FuturesPriceFetcher:
    """Fetch and process futures price data from Supabase"""

    def __init__(self):
        """Initialize Supabase client"""
        # Try environment variables first
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")

        # If no credentials found, raise error
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY environment variables or use .streamlit/secrets.toml")

        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

        # Load futures to COT mapping
        mapping_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'instrument_management', 'futures', 'futures_symbols_enhanced.json')
        with open(mapping_path, 'r') as f:
            self.futures_mapping = json.load(f)

    def get_futures_symbol_for_cot(self, cot_code: str) -> Optional[str]:
        """Get futures symbol that matches a COT code"""
        for symbol, info in self.futures_mapping['futures_symbols'].items():
            if info['cot_mapping']['matched']:
                if cot_code in info['cot_mapping']['codes']:
                    return symbol
        return None

    def get_cot_codes_for_symbol(self, symbol: str) -> List[str]:
        """Get COT codes for a futures symbol"""
        if symbol in self.futures_mapping['futures_symbols']:
            info = self.futures_mapping['futures_symbols'][symbol]
            if info['cot_mapping']['matched']:
                return info['cot_mapping']['codes']
        return []

    def fetch_weekly_prices(self, symbol: str, start_date: str, end_date: str,
                          adjustment: str = 'RAD') -> pd.DataFrame:
        """
        Fetch weekly futures prices from Supabase

        Args:
            symbol: Futures symbol (e.g., 'ES', 'ZG')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            adjustment: Price adjustment method (NON, RAD, REV)

        Returns:
            DataFrame with weekly price data
        """
        try:
            # Query Supabase for price data (with pagination to handle large datasets)
            all_data = []
            offset = 0
            limit = 1000

            while True:
                response = self.supabase.table('futures_prices')\
                    .select('date, open, high, low, close, volume, open_interest')\
                    .eq('symbol', symbol)\
                    .eq('adjustment_method', adjustment)\
                    .gte('date', start_date)\
                    .lte('date', end_date)\
                    .order('date')\
                    .limit(limit)\
                    .offset(offset)\
                    .execute()

                if not response.data:
                    break

                all_data.extend(response.data)

                if len(response.data) < limit:
                    break

                offset += limit

            if not all_data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['date'])

            # Convert to weekly data (using Tuesday to match COT data)
            df.set_index('date', inplace=True)

            # Resample to weekly, ending on Tuesday
            weekly_df = df.resample('W-TUE').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'open_interest': 'last'
            }).dropna()

            weekly_df.reset_index(inplace=True)

            # Add symbol info
            if symbol in self.futures_mapping['futures_symbols']:
                info = self.futures_mapping['futures_symbols'][symbol]
                weekly_df['symbol'] = symbol
                weekly_df['name'] = info['name']
                weekly_df['exchange'] = info['exchange']

            return weekly_df

        except Exception as e:
            print(f"Error fetching price data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_prices_for_cot_code(self, cot_code: str, start_date: str,
                                  end_date: str) -> Optional[pd.DataFrame]:
        """Fetch futures prices for a COT instrument code"""
        symbol = self.get_futures_symbol_for_cot(cot_code)
        if symbol:
            return self.fetch_weekly_prices(symbol, start_date, end_date)
        return None

    def get_price_change_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate price change statistics"""
        if df.empty or len(df) < 2:
            return {}

        stats = {
            'latest_close': df.iloc[-1]['close'],
            'week_change': ((df.iloc[-1]['close'] - df.iloc[-2]['close']) /
                          df.iloc[-2]['close'] * 100) if len(df) > 1 else 0,
            'month_change': ((df.iloc[-1]['close'] - df.iloc[-4]['close']) /
                           df.iloc[-4]['close'] * 100) if len(df) > 4 else 0,
            'quarter_change': ((df.iloc[-1]['close'] - df.iloc[-13]['close']) /
                             df.iloc[-13]['close'] * 100) if len(df) > 13 else 0,
            'year_change': ((df.iloc[-1]['close'] - df.iloc[-52]['close']) /
                          df.iloc[-52]['close'] * 100) if len(df) > 52 else 0,
            'high_52w': df['high'].tail(52).max() if len(df) > 52 else df['high'].max(),
            'low_52w': df['low'].tail(52).min() if len(df) > 52 else df['low'].min(),
            'avg_volume': df['volume'].tail(52).mean() if len(df) > 52 else df['volume'].mean(),
            'latest_oi': df.iloc[-1]['open_interest']
        }

        return stats

    def calculate_correlation(self, price_df: pd.DataFrame, cot_df: pd.DataFrame,
                            cot_column: str = 'net_positions_long') -> float:
        """
        Calculate correlation between price and COT positioning

        Args:
            price_df: DataFrame with price data (must have 'date' and 'close' columns)
            cot_df: DataFrame with COT data (must have 'report_date' and specified column)
            cot_column: COT column to correlate with price

        Returns:
            Correlation coefficient
        """
        if price_df.empty or cot_df.empty:
            return 0.0

        # Merge on dates
        price_df = price_df.copy()
        cot_df = cot_df.copy()

        # Ensure date columns are datetime
        price_df['date'] = pd.to_datetime(price_df['date'])
        cot_df['report_date'] = pd.to_datetime(cot_df['report_date'])

        # Merge the dataframes
        merged = pd.merge_asof(
            price_df[['date', 'close']].sort_values('date'),
            cot_df[['report_date', cot_column]].sort_values('report_date'),
            left_on='date',
            right_on='report_date',
            direction='nearest',
            tolerance=pd.Timedelta('7 days')
        )

        # Calculate correlation
        if len(merged.dropna()) > 10:  # Need at least 10 points
            return merged['close'].corr(merged[cot_column])

        return 0.0