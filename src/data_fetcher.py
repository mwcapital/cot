"""
Data fetching and processing module for CFTC COT data
"""
import streamlit as st
import pandas as pd
from sodapy import Socrata
import json
import time
from config import CFTC_API_BASE, DATASET_CODE, DEFAULT_LIMIT, CFTC_COLUMNS
from historical_data_loader import get_historical_data_for_instrument, check_data_gap


@st.cache_data
def load_instruments_database():
    """Load the instruments JSON database"""
    try:
        # Try multiple locations for backward compatibility
        json_paths = [
            'instrument_management/LegacyF/instruments_LegacyF.json',  # New organized location
            'instruments_LegacyF.json',        # Legacy location
            '../instrument_management/LegacyF/instruments_LegacyF.json' # Alternative path
        ]
        
        for path in json_paths:
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                continue
                
        st.error("❌ instruments_LegacyF.json file not found. Please ensure it's in the data/ directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading instruments database: {e}")
        return None


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
                # Wait before retrying (exponential backoff)
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)
            else:
                # Last attempt failed, raise the exception
                raise e


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_cftc_data(instrument_name, api_token):
    """Fetch CFTC data for a specific instrument with historical data stitching"""
    try:
        # Strip the contract code if present (everything after the last space and parenthesis)
        # Format: "INSTRUMENT NAME (CODE)" -> "INSTRUMENT NAME"
        if ' (' in instrument_name and instrument_name.endswith(')'):
            # Find the last occurrence of ' (' which indicates the start of the code
            instrument_name_clean = instrument_name.rsplit(' (', 1)[0]
        else:
            instrument_name_clean = instrument_name

        # Initialize Socrata client with increased timeout (30 seconds instead of default 10)
        client = Socrata(CFTC_API_BASE, api_token, timeout=30)

        # Special handling for WTI-PHYSICAL: merge with historical CRUDE OIL, LIGHT SWEET data
        if instrument_name_clean == "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE":
            # First, get historical data from CRUDE OIL, LIGHT SWEET (2000-2022)
            historical_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

            # Then get current data from WTI-PHYSICAL (2022-present)
            current_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

            # Merge the results
            results = historical_results + current_results

        # Special handling for NAT GAS NYME: merge with historical NATURAL GAS data
        elif instrument_name_clean == "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE":
            # First, get historical data from NATURAL GAS (up to 2016)
            historical_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='NATURAL GAS - NEW YORK MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

            # Then get current data from NAT GAS NYME (2022-present)
            current_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

            # Merge the results
            results = historical_results + current_results

        # Special handling for GASOLINE RBOB: merge with historical UNLEADED GASOLINE and GASOLINE BLENDSTOCK
        elif instrument_name_clean == "GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE":
            # Get historical data from UNLEADED GASOLINE (1986-2006)
            historical_1_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='UNLEADED GASOLINE, N.Y. HARBOR - NEW YORK MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

            # Get historical data from GASOLINE BLENDSTOCK (2006-2022)
            historical_2_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='GASOLINE BLENDSTOCK (RBOB) - NEW YORK MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

            # Get current data from GASOLINE RBOB (2022-present)
            current_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

            # Merge all results
            results = historical_1_results + historical_2_results + current_results

        # Special handling for COPPER: merge with historical COPPER and COPPER-GRADE #1
        elif instrument_name_clean == "COPPER- #1 - COMMODITY EXCHANGE INC.":
            # Get historical data from COPPER (1986-1989)
            historical_1_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='COPPER - COMMODITY EXCHANGE INC.'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

            # Get historical data from COPPER-GRADE #1 (1989-2022)
            historical_2_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='COPPER-GRADE #1 - COMMODITY EXCHANGE INC.'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

            # Get current data from COPPER- #1 (2022-present)
            current_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='COPPER- #1 - COMMODITY EXCHANGE INC.'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

            # Merge all results
            results = historical_1_results + historical_2_results + current_results

        # Special handling for E-MINI S&P 500: merge with historical E-MINI S&P 500 STOCK INDEX
        elif instrument_name_clean == "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE":
            # Get historical data (2000-2022)
            historical_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )
            # Get current data (2022-present)
            current_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )
            results = historical_results + current_results

        # Special handling for NASDAQ MINI: merge with historical NASDAQ-100 STOCK INDEX (MINI)
        elif instrument_name_clean == "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE":
            # Get historical data (2000-2022)
            historical_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )
            # Get current data (2022-present)
            current_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )
            results = historical_results + current_results

        # Special handling for DJIA x $5: merge with historical DOW JONES INDUSTRIAL AVG- x $5
        elif instrument_name_clean == "DJIA x $5 - CHICAGO BOARD OF TRADE":
            # Get historical data (2002-2022)
            historical_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='DOW JONES INDUSTRIAL AVG- x $5 - CHICAGO BOARD OF TRADE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )
            # Get current data (2022-present)
            current_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='DJIA x $5 - CHICAGO BOARD OF TRADE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )
            results = historical_results + current_results

        # Special handling for RUSSELL E-MINI: merge ICE (2008-2018) and CME (2017+) data
        elif instrument_name_clean == "RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE":
            # Get ICE historical data (2008-2018)
            ice_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='RUSSELL 2000 MINI INDEX FUTURE - ICE FUTURES U.S.'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )
            # Get CME historical data (2017-2022)
            cme_historical_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='E-MINI RUSSELL 2000 INDEX - CHICAGO MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )
            # Get current CME data (2022-present)
            current_results = _fetch_with_retry(
                client,
                DATASET_CODE,
                "market_and_exchange_names='RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )
            results = ice_results + cme_historical_results + current_results

        else:
            # Standard fetch for all other instruments
            results = _fetch_with_retry(
                client,
                DATASET_CODE,
                f"market_and_exchange_names='{instrument_name_clean}'",
                ",".join(CFTC_COLUMNS),
                "report_date_as_yyyy_mm_dd ASC",
                DEFAULT_LIMIT
            )

        client.close()

        if not results:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Convert date column
        df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
        
        # Remove duplicates based on date (in case of overlapping data at transition)
        df = df.drop_duplicates(subset=['report_date_as_yyyy_mm_dd'], keep='last')

        # Convert numeric columns
        numeric_columns = [col for col in CFTC_COLUMNS if
                           col != "report_date_as_yyyy_mm_dd" and col != "market_and_exchange_names"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate additional metrics
        df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
        df['net_comm_positions'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
        df['net_reportable_positions'] = df['tot_rept_positions_long_all'] - df['tot_rept_positions_short']
        
        # Now try to stitch historical data (pre-2000) if available
        # For WTI, we already handle 2000-2022, so check for pre-2000 data
        api_start_date = df['report_date_as_yyyy_mm_dd'].min()
        
        # Try to get historical data for this instrument
        # For WTI special case, try multiple historical names
        historical_df = None
        if instrument_name_clean == "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE":
            # Try the pre-2000 name first (with quotes around SWEET)
            historical_df = get_historical_data_for_instrument(
                "CRUDE OIL, LIGHT 'SWEET' - NEW YORK MERCANTILE EXCHANGE",
                end_date=api_start_date
            )
            # If not found, try the 2000-2016 name (without quotes)
            if historical_df is None or historical_df.empty:
                historical_df = get_historical_data_for_instrument(
                    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
                    end_date=api_start_date
                )
        elif instrument_name_clean == "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE":
            # For Natural Gas, try the historical name from FUT86_16.txt
            historical_df = get_historical_data_for_instrument(
                "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
                end_date=api_start_date
            )
        elif instrument_name_clean == "GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE":
            # For Gasoline RBOB, try the oldest historical name from FUT86_16.txt
            historical_df = get_historical_data_for_instrument(
                "UNLEADED GASOLINE, N.Y. HARBOR - NEW YORK MERCANTILE EXCHANGE",
                end_date=api_start_date
            )
        elif instrument_name_clean == "COPPER- #1 - COMMODITY EXCHANGE INC.":
            # For Copper, try the oldest historical name from FUT86_16.txt
            historical_df = get_historical_data_for_instrument(
                "COPPER - COMMODITY EXCHANGE INC.",
                end_date=api_start_date
            )
        elif instrument_name_clean == "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE":
            # For S&P 500 Mini, try the historical name from FUT86_16.txt
            historical_df = get_historical_data_for_instrument(
                "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
                end_date=api_start_date
            )
        elif instrument_name_clean == "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE":
            # For NASDAQ Mini, try the historical name from FUT86_16.txt
            historical_df = get_historical_data_for_instrument(
                "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE",
                end_date=api_start_date
            )
        elif instrument_name_clean == "DJIA x $5 - CHICAGO BOARD OF TRADE":
            # For Mini Dow Jones, try the historical name from FUT86_16.txt
            historical_df = get_historical_data_for_instrument(
                "DOW JONES INDUSTRIAL AVG- x $5 - CHICAGO BOARD OF TRADE",
                end_date=api_start_date
            )
        elif instrument_name_clean == "RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE":
            # For Russell 2000, try the ICE historical name from FUT86_16.txt
            historical_df = get_historical_data_for_instrument(
                "RUSSELL 2000 MINI INDEX FUTURE - ICE FUTURES U.S.",
                end_date=api_start_date
            )
        else:
            # For all other instruments, use the clean name
            historical_df = get_historical_data_for_instrument(
                instrument_name_clean,
                end_date=api_start_date
            )
        
        if historical_df is not None and not historical_df.empty:
            # Check for gap between historical and API data
            historical_end_date = historical_df['report_date_as_yyyy_mm_dd'].max()
            has_gap, gap_message = check_data_gap(historical_end_date, api_start_date)
            
            if has_gap:
                # Show warning about data gap
                st.warning(gap_message)
            else:
                # Combine historical and API data
                # Ensure columns match
                common_columns = list(set(df.columns) & set(historical_df.columns))
                df = pd.concat([historical_df[common_columns], df[common_columns]], ignore_index=True)
                
                # Remove any duplicates that might occur at the boundary
                df = df.drop_duplicates(subset=['report_date_as_yyyy_mm_dd'], keep='last')
                
                # Recalculate metrics if needed
                if 'net_noncomm_positions' not in df.columns:
                    if 'noncomm_positions_long_all' in df.columns and 'noncomm_positions_short_all' in df.columns:
                        df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
                if 'net_comm_positions' not in df.columns:
                    if 'comm_positions_long_all' in df.columns and 'comm_positions_short_all' in df.columns:
                        df['net_comm_positions'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
                if 'net_reportable_positions' not in df.columns:
                    if 'tot_rept_positions_long_all' in df.columns and 'tot_rept_positions_short' in df.columns:
                        df['net_reportable_positions'] = df['tot_rept_positions_long_all'] - df['tot_rept_positions_short']

        return df.sort_values('report_date_as_yyyy_mm_dd')

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_cftc_data_2year(instrument_name, api_token):
    """Fetch exactly 2 years of CFTC data for a specific instrument (optimized)"""
    try:
        # Strip the contract code if present
        if ' (' in instrument_name and instrument_name.endswith(')'):
            instrument_name_clean = instrument_name.rsplit(' (', 1)[0]
        else:
            instrument_name_clean = instrument_name

        # Initialize Socrata client with increased timeout
        client = Socrata(CFTC_API_BASE, api_token, timeout=30)

        # Calculate 2 years ago date
        from datetime import datetime, timedelta
        two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        # Fetch only 2 years of data using WHERE clause for date filtering
        where_clause = f"market_and_exchange_names='{instrument_name_clean}' AND report_date_as_yyyy_mm_dd >= '{two_years_ago}'"

        results = _fetch_with_retry(
            client,
            DATASET_CODE,
            where_clause,
            ",".join(CFTC_COLUMNS),
            "report_date_as_yyyy_mm_dd ASC",
            200  # ~104 weekly reports in 2 years, 200 is safe
        )

        client.close()

        if not results:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Convert date column
        df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])

        # Convert numeric columns
        numeric_columns = [col for col in CFTC_COLUMNS if
                           col != "report_date_as_yyyy_mm_dd" and col != "market_and_exchange_names"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate additional metrics (same as in other fetch functions)
        df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
        df['net_comm_positions'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
        df['net_reportable_positions'] = df['tot_rept_positions_long_all'] - df['tot_rept_positions_short']

        return df

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_cftc_data_ytd_only(instrument_name, api_token):
    """Optimized fetch for dashboard - only gets YTD data plus latest record"""
    try:
        from datetime import datetime

        # Strip the contract code if present
        if ' (' in instrument_name and instrument_name.endswith(')'):
            instrument_name_clean = instrument_name.rsplit(' (', 1)[0]
        else:
            instrument_name_clean = instrument_name

        # Initialize Socrata client with increased timeout
        client = Socrata(CFTC_API_BASE, api_token, timeout=30)
        
        # Get current year
        current_year = datetime.now().year
        year_start = f"{current_year}-01-01T00:00:00.000"
        
        # Only fetch the columns we need for dashboard
        # Note: net_noncomm_positions needs to be calculated
        dashboard_columns = [
            "report_date_as_yyyy_mm_dd",
            "open_interest_all",
            "change_in_open_interest_all",
            "noncomm_positions_long_all",
            "noncomm_positions_short_all",
            "comm_positions_long_all",
            "comm_positions_short_all",
            "conc_net_le_4_tdr_long_all",
            "conc_net_le_4_tdr_short_all"
        ]
        
        # Special handling for WTI-PHYSICAL
        if instrument_name_clean == "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE":
            # For WTI, just get YTD data from the current instrument
            results = _fetch_with_retry(
                client,
                DATASET_CODE,
                f"market_and_exchange_names='WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE' AND report_date_as_yyyy_mm_dd >= '{year_start}'",
                ",".join(dashboard_columns),
                "report_date_as_yyyy_mm_dd ASC",
                100  # YTD should be ~50 records max
            )
        else:
            # Standard fetch for YTD data only
            results = _fetch_with_retry(
                client,
                DATASET_CODE,
                f"market_and_exchange_names='{instrument_name_clean}' AND report_date_as_yyyy_mm_dd >= '{year_start}'",
                ",".join(dashboard_columns),
                "report_date_as_yyyy_mm_dd ASC",
                100  # YTD should be ~50 records max
            )
        
        if not results:
            return None
            
        df = pd.DataFrame.from_records(results)
        
        # Convert date column
        df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
        
        # Convert numeric columns
        numeric_columns = [col for col in dashboard_columns if col != 'report_date_as_yyyy_mm_dd']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate net positions
        if 'noncomm_positions_long_all' in df.columns and 'noncomm_positions_short_all' in df.columns:
            df['net_noncomm_positions'] = df['noncomm_positions_long_all'] - df['noncomm_positions_short_all']
        
        return df.sort_values('report_date_as_yyyy_mm_dd')
        
    except Exception as e:
        st.error(f"Error fetching YTD data: {e}")
        return None


