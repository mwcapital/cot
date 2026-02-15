"""
Data fetcher for Disaggregated COT Report (dataset 72hh-3qpy)
Also handles ICE instruments from Supabase ice_cot_data table
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
from sodapy import Socrata
from supabase import create_client, Client
from dotenv import load_dotenv
from disaggregated_config import CFTC_API_BASE, DISAGG_DATASET_CODE, DISAGG_COLUMNS
from disaggregated_historical_data_loader import get_disagg_historical_data_for_instrument

load_dotenv()


def _fetch_with_retry(client, dataset_code, where_clause, select_columns, order_by, limit, max_retries=3):
    """Helper function to fetch data with retry logic"""
    for attempt in range(max_retries):
        try:
            results = client.get(
                dataset_code,
                where=where_clause,
                select=select_columns,
                order=order_by,
                limit=limit
            )
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise e


def _get_supabase_client():
    """Initialize Supabase client"""
    try:
        if hasattr(st, 'secrets') and 'SUPABASE_URL' in st.secrets:
            url = st.secrets['SUPABASE_URL']
            key = st.secrets['SUPABASE_KEY']
        else:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _get_ice_instruments():
    """Load set of ICE instrument names from futures_symbols_enhanced.json"""
    ice_instruments = {}
    try:
        mapping_path = 'instrument_management/futures/futures_symbols_enhanced.json'
        with open(mapping_path, 'r') as f:
            data = json.load(f)
        for symbol, sdata in data['futures_symbols'].items():
            cot_map = sdata.get('cot_mapping', {})
            if cot_map.get('source') == 'ice_cot_data' and cot_map.get('matched'):
                for inst_name in cot_map.get('instruments', []):
                    ice_instruments[inst_name] = symbol
    except Exception:
        pass
    return ice_instruments


# Column mapping: ICE Supabase → CFTC Disaggregated standard names
ICE_TO_DISAGG_COLUMNS = {
    'report_date': 'report_date_as_yyyy_mm_dd',
    'market_name': 'market_and_exchange_names',
    'open_interest_all': 'open_interest_all',
    'change_oi_all': 'change_in_open_interest_all',
    'prod_merc_long_all': 'prod_merc_positions_long',
    'prod_merc_short_all': 'prod_merc_positions_short',
    'swap_long_all': 'swap_positions_long_all',
    'swap_short_all': 'swap__positions_short_all',
    'swap_spread_all': 'swap__positions_spread_all',
    'm_money_long_all': 'm_money_positions_long_all',
    'm_money_short_all': 'm_money_positions_short_all',
    'm_money_spread_all': 'm_money_positions_spread',
    'other_rept_long_all': 'other_rept_positions_long',
    'other_rept_short_all': 'other_rept_positions_short',
    'other_rept_spread_all': 'other_rept_positions_spread',
    'tot_rept_long_all': 'tot_rept_positions_long_all',
    'tot_rept_short_all': 'tot_rept_positions_short',
    'nonrept_long_all': 'nonrept_positions_long_all',
    'nonrept_short_all': 'nonrept_positions_short_all',
    'traders_tot_all': 'traders_tot_all',
    'conc_gross_le_4_long_all': 'conc_gross_le_4_tdr_long',
    'conc_gross_le_4_short_all': 'conc_gross_le_4_tdr_short',
    'conc_net_le_4_long_all': 'conc_net_le_4_tdr_long_all',
    'conc_net_le_4_short_all': 'conc_net_le_4_tdr_short_all',
    'change_prod_merc_long_all': 'change_in_prod_merc_long',
    'change_prod_merc_short_all': 'change_in_prod_merc_short',
    'change_swap_long_all': 'change_in_swap_long_all',
    'change_swap_short_all': 'change_in_swap_short_all',
    'change_m_money_long_all': 'change_in_m_money_long_all',
    'change_m_money_short_all': 'change_in_m_money_short_all',
    'pct_oi_prod_merc_long_all': 'pct_of_oi_prod_merc_long',
    'pct_oi_prod_merc_short_all': 'pct_of_oi_prod_merc_short',
    'pct_oi_swap_long_all': 'pct_of_oi_swap_long_all',
    'pct_oi_swap_short_all': 'pct_of_oi_swap_short_all',
    'pct_oi_m_money_long_all': 'pct_of_oi_m_money_long_all',
    'pct_oi_m_money_short_all': 'pct_of_oi_m_money_short_all',
    'pct_oi_other_rept_long_all': 'pct_of_oi_other_rept_long',
    'pct_oi_other_rept_short_all': 'pct_of_oi_other_rept_short',
}


@st.cache_data(ttl=3600)
def _fetch_ice_data_2year(instrument_name):
    """Fetch 2 years of ICE COT data from Supabase and normalize column names"""
    try:
        supabase = _get_supabase_client()
        if not supabase:
            return None

        from datetime import datetime, timedelta
        two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        # ICE instruments use market_name in Supabase
        response = supabase.from_('ice_cot_data').select(
            '*'
        ).eq(
            'market_name', instrument_name
        ).eq(
            'report_type', 'FutOnly'
        ).gte(
            'report_date', two_years_ago
        ).order('report_date', desc=False).execute()

        if not response.data:
            return None

        df = pd.DataFrame(response.data)

        # Rename columns to match CFTC disaggregated standard
        df = df.rename(columns=ICE_TO_DISAGG_COLUMNS)

        # Convert date
        df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])

        # Convert numeric columns
        for col in df.columns:
            if col not in ('report_date_as_yyyy_mm_dd', 'market_and_exchange_names',
                          'id', 'report_date_yymmdd', 'report_type', 'commodity_code',
                          'contract_units', 'cftc_region_code', 'additional_data',
                          'created_at', 'updated_at'):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate derived net positions
        df['net_mm_positions'] = df['m_money_positions_long_all'] - df['m_money_positions_short_all']
        df['net_pm_positions'] = df['prod_merc_positions_long'] - df['prod_merc_positions_short']
        df['net_swap_positions'] = df['swap_positions_long_all'] - df['swap__positions_short_all']
        df['net_other_positions'] = df['other_rept_positions_long'] - df['other_rept_positions_short']

        return df

    except Exception as e:
        print(f"Error fetching ICE data for {instrument_name}: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_disagg_data_2year(instrument_name, api_token):
    """Fetch 2 years of Disaggregated COT data for a specific instrument.
    Routes to Supabase for ICE instruments, CFTC API for everything else."""

    # Check if this is an ICE instrument
    ice_instruments = _get_ice_instruments()
    if instrument_name in ice_instruments:
        return _fetch_ice_data_2year(instrument_name)

    # Standard CFTC API fetch
    try:
        if ' (' in instrument_name and instrument_name.endswith(')'):
            instrument_name_clean = instrument_name.rsplit(' (', 1)[0]
        else:
            instrument_name_clean = instrument_name

        client = Socrata(CFTC_API_BASE, api_token, timeout=30)

        from datetime import datetime, timedelta
        two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        where_clause = f"market_and_exchange_names='{instrument_name_clean}' AND report_date_as_yyyy_mm_dd >= '{two_years_ago}'"

        results = _fetch_with_retry(
            client,
            DISAGG_DATASET_CODE,
            where_clause,
            ",".join(DISAGG_COLUMNS),
            "report_date_as_yyyy_mm_dd ASC",
            200
        )

        client.close()

        if not results:
            return None

        df = pd.DataFrame(results)

        # Convert date column
        df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])

        # Convert numeric columns
        numeric_columns = [col for col in DISAGG_COLUMNS
                          if col not in ("report_date_as_yyyy_mm_dd", "market_and_exchange_names")]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate derived net positions
        df['net_mm_positions'] = df['m_money_positions_long_all'] - df['m_money_positions_short_all']
        df['net_pm_positions'] = df['prod_merc_positions_long'] - df['prod_merc_positions_short']
        df['net_swap_positions'] = df['swap_positions_long_all'] - df['swap__positions_short_all']
        df['net_other_positions'] = df['other_rept_positions_long'] - df['other_rept_positions_short']

        return df

    except Exception as e:
        print(f"Error fetching disaggregated data for {instrument_name}: {e}")
        return None


# Instrument name mapping: current API name -> historical file name (F_Disagg06_16.txt)
# Only needed when the names differ between current API and historical file
DISAGG_HISTORICAL_NAME_MAP = {
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE":
        "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
    "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE":
        "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
    "GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE":
        "GASOLINE BLENDSTOCK (RBOB) - NEW YORK MERCANTILE EXCHANGE",
    "COPPER- #1 - COMMODITY EXCHANGE INC.":
        "COPPER-GRADE #1 - COMMODITY EXCHANGE INC.",
    "NY HARBOR ULSD - NEW YORK MERCANTILE EXCHANGE":
        "NO. 2 HEATING OIL, N.Y. HARBOR - NEW YORK MERCANTILE EXCHANGE",
    "WHEAT-HRSpring - MIAX FUTURES EXCHANGE":
        "WHEAT-HRSpring - MINNEAPOLIS GRAIN EXCHANGE",
}


@st.cache_data(ttl=3600)
def fetch_disagg_data_full(instrument_name, api_token):
    """Fetch full history of Disaggregated COT data for a specific instrument.
    Fetches all available API data (no date filter) and stitches historical
    data from F_Disagg06_16.txt for pre-API coverage.
    Routes to Supabase for ICE instruments (no historical stitching for ICE)."""

    # ICE instruments only have 2-year data from Supabase, no historical file
    ice_instruments = _get_ice_instruments()
    if instrument_name in ice_instruments:
        return _fetch_ice_data_2year(instrument_name)

    try:
        if ' (' in instrument_name and instrument_name.endswith(')'):
            instrument_name_clean = instrument_name.rsplit(' (', 1)[0]
        else:
            instrument_name_clean = instrument_name

        client = Socrata(CFTC_API_BASE, api_token, timeout=30)
        select_cols = ",".join(DISAGG_COLUMNS)
        order_by = "report_date_as_yyyy_mm_dd ASC"
        limit = 5000

        # API-level stitching for instruments that changed names
        if instrument_name_clean == "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE":
            historical_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            current_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            results = historical_results + current_results

        elif instrument_name_clean == "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE":
            historical_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='NATURAL GAS - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            current_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            results = historical_results + current_results

        elif instrument_name_clean == "GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE":
            # Name evolution: GASOLINE BLENDSTOCK (RBOB) (2006-2015)
            #   -> GASOLINE BLENDSTOCK (RBOB)  [extra space] (2015-2022)
            #   -> GASOLINE RBOB (2022+)
            historical_1_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='GASOLINE BLENDSTOCK (RBOB) - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            historical_2_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='GASOLINE BLENDSTOCK (RBOB)  - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            current_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            results = historical_1_results + historical_2_results + current_results

        elif instrument_name_clean == "COPPER- #1 - COMMODITY EXCHANGE INC.":
            historical_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='COPPER-GRADE #1 - COMMODITY EXCHANGE INC.'",
                select_cols, order_by, limit
            )
            current_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='COPPER- #1 - COMMODITY EXCHANGE INC.'",
                select_cols, order_by, limit
            )
            results = historical_results + current_results

        elif instrument_name_clean == "NY HARBOR ULSD - NEW YORK MERCANTILE EXCHANGE":
            # Name evolution: NO. 2 HEATING OIL (2006-2013) -> #2 HEATING OIL, (2013-2017)
            #   -> #2 HEATING OIL- (2017-2022) -> NY HARBOR ULSD (2022+)
            historical_1_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='NO. 2 HEATING OIL, N.Y. HARBOR - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            historical_2_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='#2 HEATING OIL, NY HARBOR-ULSD - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            historical_3_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='#2 HEATING OIL- NY HARBOR-ULSD - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            current_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='NY HARBOR ULSD - NEW YORK MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            results = historical_1_results + historical_2_results + historical_3_results + current_results

        elif instrument_name_clean == "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE":
            historical_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            current_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            results = historical_results + current_results

        elif instrument_name_clean == "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE":
            historical_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            current_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            results = historical_results + current_results

        elif instrument_name_clean == "DJIA x $5 - CHICAGO BOARD OF TRADE":
            historical_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='DOW JONES INDUSTRIAL AVG- x $5 - CHICAGO BOARD OF TRADE'",
                select_cols, order_by, limit
            )
            current_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='DJIA x $5 - CHICAGO BOARD OF TRADE'",
                select_cols, order_by, limit
            )
            results = historical_results + current_results

        elif instrument_name_clean == "RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE":
            ice_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='RUSSELL 2000 MINI INDEX FUTURE - ICE FUTURES U.S.'",
                select_cols, order_by, limit
            )
            cme_historical_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='E-MINI RUSSELL 2000 INDEX - CHICAGO MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            current_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE'",
                select_cols, order_by, limit
            )
            results = ice_results + cme_historical_results + current_results

        elif instrument_name_clean == "WHEAT-HRSpring - MIAX FUTURES EXCHANGE":
            historical_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='WHEAT-HRSpring - MINNEAPOLIS GRAIN EXCHANGE'",
                select_cols, order_by, limit
            )
            current_results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                "market_and_exchange_names='WHEAT-HRSpring - MIAX FUTURES EXCHANGE'",
                select_cols, order_by, limit
            )
            results = historical_results + current_results

        else:
            results = _fetch_with_retry(
                client, DISAGG_DATASET_CODE,
                f"market_and_exchange_names='{instrument_name_clean}'",
                select_cols, order_by, limit
            )

        client.close()

        if not results:
            return None

        df = pd.DataFrame(results)

        # Convert date column
        df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])

        # Remove duplicates at transition boundaries
        df = df.drop_duplicates(subset=['report_date_as_yyyy_mm_dd'], keep='last')

        # Convert numeric columns
        numeric_columns = [col for col in DISAGG_COLUMNS
                          if col not in ("report_date_as_yyyy_mm_dd", "market_and_exchange_names")]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate derived net positions
        df['net_mm_positions'] = df['m_money_positions_long_all'] - df['m_money_positions_short_all']
        df['net_pm_positions'] = df['prod_merc_positions_long'] - df['prod_merc_positions_short']
        df['net_swap_positions'] = df['swap_positions_long_all'] - df['swap__positions_short_all']
        df['net_other_positions'] = df['other_rept_positions_long'] - df['other_rept_positions_short']

        # Stitch historical data from F_Disagg06_16.txt
        api_start_date = df['report_date_as_yyyy_mm_dd'].min()

        # Determine historical file instrument name
        historical_name = DISAGG_HISTORICAL_NAME_MAP.get(
            instrument_name_clean, instrument_name_clean
        )

        historical_df = get_disagg_historical_data_for_instrument(
            historical_name, end_date=api_start_date
        )

        if historical_df is not None and not historical_df.empty:
            common_columns = list(set(df.columns) & set(historical_df.columns))
            df = pd.concat([historical_df[common_columns], df[common_columns]], ignore_index=True)
            df = df.drop_duplicates(subset=['report_date_as_yyyy_mm_dd'], keep='last')

            # Recalculate net positions after merge
            if 'm_money_positions_long_all' in df.columns and 'm_money_positions_short_all' in df.columns:
                df['net_mm_positions'] = df['m_money_positions_long_all'] - df['m_money_positions_short_all']
            if 'prod_merc_positions_long' in df.columns and 'prod_merc_positions_short' in df.columns:
                df['net_pm_positions'] = df['prod_merc_positions_long'] - df['prod_merc_positions_short']
            if 'swap_positions_long_all' in df.columns and 'swap__positions_short_all' in df.columns:
                df['net_swap_positions'] = df['swap_positions_long_all'] - df['swap__positions_short_all']
            if 'other_rept_positions_long' in df.columns and 'other_rept_positions_short' in df.columns:
                df['net_other_positions'] = df['other_rept_positions_long'] - df['other_rept_positions_short']

        return df.sort_values('report_date_as_yyyy_mm_dd')

    except Exception as e:
        print(f"Error fetching full disaggregated data for {instrument_name}: {e}")
        return None