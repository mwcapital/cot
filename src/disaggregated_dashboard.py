"""
Dashboard overview for Disaggregated COT Report
Adapted from dashboard_overview.py for disaggregated trader categories
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
from supabase import create_client, Client
from dotenv import load_dotenv
from disaggregated_data_fetcher import fetch_disagg_data_2year, fetch_disagg_data_full
from charts.cross_asset_analysis import get_short_instrument_name

load_dotenv()

# Reuse shared helpers from legacy dashboard
from dashboard_overview import (
    load_futures_instruments,
    get_ticker_mapping,
    get_ticker_to_name_mapping,
    get_supabase_client,
    calculate_ytd_percentile,
    CATEGORY_COLORS,
)


def calculate_disagg_correlations_optimized(cot_data, futures_symbol, window_days=730):
    """
    Calculate Spearman correlations for 5 disaggregated position types in a single price fetch.

    Returns:
        dict with keys: 'mm', 'pml', 'pms', 'swl', 'sws'
        Each contains float (Spearman correlation) or np.nan
    """
    from scipy.stats import spearmanr

    result = {'mm': np.nan, 'pml': np.nan, 'pms': np.nan, 'swl': np.nan, 'sws': np.nan}

    try:
        if cot_data.empty or not futures_symbol:
            return result

        supabase = get_supabase_client()
        if not supabase:
            return result

        # Single price data fetch
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days + 100)

        response = supabase.from_('futures_prices').select(
            'date, close'
        ).eq('symbol', futures_symbol).eq(
            'adjustment_method', 'NON'
        ).gte(
            'date', start_date.strftime('%Y-%m-%d')
        ).order('date', desc=False).execute()

        if not response.data:
            return result

        price_df = pd.DataFrame(response.data)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['close'] = pd.to_numeric(price_df['close'], errors='coerce')
        price_df = price_df.dropna().sort_values('date')

        if len(price_df) < 100:
            return result

        # Weekly returns (Tuesday to Tuesday for COT alignment)
        price_df.set_index('date', inplace=True)
        weekly_df = price_df.resample('W-TUE').agg({'close': 'last'}).dropna()
        weekly_df['returns'] = weekly_df['close'].pct_change()
        weekly_df = weekly_df.dropna()

        if len(weekly_df) < 20:
            return result

        # Prepare COT data with position changes
        cot_df = cot_data.copy()
        cot_df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(cot_df['report_date_as_yyyy_mm_dd'])
        cot_df = cot_df.sort_values('report_date_as_yyyy_mm_dd')

        # Calculate all position changes
        change_cols = {
            'net_mm_positions': 'net_mm_positions_change',
            'prod_merc_positions_long': 'pml_change',
            'prod_merc_positions_short': 'pms_change',
            'swap_positions_long_all': 'swl_change',
            'swap__positions_short_all': 'sws_change',
        }
        for src, dst in change_cols.items():
            if src in cot_df.columns:
                cot_df[dst] = cot_df[src].diff()

        cot_df = cot_df.dropna()

        if len(cot_df) < 20:
            return result

        # Single merge for all position types
        weekly_df.reset_index(inplace=True)
        merge_cols = ['report_date_as_yyyy_mm_dd'] + list(change_cols.values())
        available_merge_cols = [c for c in merge_cols if c in cot_df.columns]

        merged = pd.merge_asof(
            weekly_df.sort_values('date'),
            cot_df[available_merge_cols].sort_values('report_date_as_yyyy_mm_dd'),
            left_on='date',
            right_on='report_date_as_yyyy_mm_dd',
            direction='nearest',
            tolerance=pd.Timedelta('7 days')
        ).dropna()

        if len(merged) < 15:
            return result

        returns = merged['returns'].values

        # Calculate Spearman for each position type
        col_key_map = {
            'net_mm_positions_change': 'mm',
            'pml_change': 'pml',
            'pms_change': 'pms',
            'swl_change': 'swl',
            'sws_change': 'sws',
        }

        for col, key in col_key_map.items():
            if col in merged.columns:
                try:
                    result[key] = spearmanr(merged[col].values, returns)[0]
                except Exception:
                    pass

        return result

    except Exception as e:
        print(f"Disagg correlation failed for {futures_symbol}: {e}")
        return result


def fetch_single_disagg_instrument(category, instrument, api_token):
    """Fetch data for a single instrument and build a table row"""
    try:
        df_2year = fetch_disagg_data_2year(instrument, api_token)

        if df_2year is None or df_2year.empty:
            return None

        latest = df_2year.iloc[-1]

        # Get display name from ticker mapping
        ticker_map = get_ticker_mapping()
        name_map = get_ticker_to_name_mapping()

        ticker = ticker_map.get(instrument)
        if not ticker:
            instrument_name = instrument.split(' - ')[0]
            ticker = ticker_map.get(instrument_name, instrument_name[:2].upper())

        display_name = name_map.get(ticker, instrument.split(' - ')[0])
        futures_symbol = ticker

        # Initialize correlation vars (Spearman only)
        mm_corr = pml_corr = pms_corr = swl_corr = sws_corr = np.nan

        # Calculate correlations
        if futures_symbol:
            try:
                all_corrs = calculate_disagg_correlations_optimized(df_2year, futures_symbol)
                mm_corr = all_corrs['mm']
                pml_corr = all_corrs['pml']
                pms_corr = all_corrs['pms']
                swl_corr = all_corrs['swl']
                sws_corr = all_corrs['sws']
            except Exception:
                pass

        row_data = {
            'Category': category,
            'Instrument': display_name,

            # Open Interest
            'Open Interest': latest.get('open_interest_all', np.nan),
            'OI Change': latest.get('change_in_open_interest_all', np.nan),

            # Money Managers Net
            'MM Net': latest.get('net_mm_positions', np.nan),
            'MM 2Y %': calculate_ytd_percentile(
                df_2year.get('net_mm_positions', pd.Series()),
                latest.get('net_mm_positions', np.nan)
            ),
            'MM Corr': mm_corr,

            # Prod/Merc Long
            'PM Long': latest.get('prod_merc_positions_long', np.nan),
            'PM L 2Y %': calculate_ytd_percentile(
                df_2year.get('prod_merc_positions_long', pd.Series()),
                latest.get('prod_merc_positions_long', np.nan)
            ),
            'PM L Corr': pml_corr,

            # Prod/Merc Short
            'PM Short': latest.get('prod_merc_positions_short', np.nan),
            'PM S 2Y %': calculate_ytd_percentile(
                df_2year.get('prod_merc_positions_short', pd.Series()),
                latest.get('prod_merc_positions_short', np.nan)
            ),
            'PM S Corr': pms_corr,

            # Swap Long
            'SW Long': latest.get('swap_positions_long_all', np.nan),
            'SW L 2Y %': calculate_ytd_percentile(
                df_2year.get('swap_positions_long_all', pd.Series()),
                latest.get('swap_positions_long_all', np.nan)
            ),
            'SW L Corr': swl_corr,

            # Swap Short
            'SW Short': latest.get('swap__positions_short_all', np.nan),
            'SW S 2Y %': calculate_ytd_percentile(
                df_2year.get('swap__positions_short_all', pd.Series()),
                latest.get('swap__positions_short_all', np.nan)
            ),
            'SW S Corr': sws_corr,

            # Concentration
            'Conc Long 4T': latest.get('conc_net_le_4_tdr_long_all', np.nan),
            'Conc Short 4T': latest.get('conc_net_le_4_tdr_short_all', np.nan),

            # Last update
            'Last Update': latest['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d'),
        }

        return row_data

    except Exception as e:
        print(f"Error fetching disagg data for {instrument}: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_disagg_dashboard_data(api_token):
    """Fetch disaggregated data for all dashboard instruments in parallel"""
    dashboard_data = []
    key_instruments = load_futures_instruments()

    # Flatten instrument list
    all_instruments = []
    for category, instruments in key_instruments.items():
        for ticker, cot_name in instruments.items():
            all_instruments.append((category, cot_name))

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Processing disaggregated data...")

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_instrument = {
            executor.submit(fetch_single_disagg_instrument, cat, inst, api_token): (cat, inst)
            for cat, inst in all_instruments
        }

        completed = 0
        for future in as_completed(future_to_instrument):
            completed += 1
            progress = completed / len(all_instruments)
            progress_bar.progress(progress)
            status_text.text(f"Processing... {completed}/{len(all_instruments)} ({progress:.0%})")

            try:
                row_data = future.result()
                if row_data:
                    dashboard_data.append(row_data)
            except Exception as e:
                category, instrument = future_to_instrument[future]
                st.warning(f"Could not fetch data for {instrument}: {e}")

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(dashboard_data)


def display_disagg_dashboard(api_token):
    """Display the disaggregated dashboard overview table"""

    with st.spinner("Loading disaggregated market data..."):
        df = fetch_disagg_dashboard_data(api_token)

    if df.empty:
        st.warning("No data available for disaggregated dashboard instruments")
        return

    # Get latest date
    latest_date_str = ""
    if 'Last Update' in df.columns and not df.empty:
        try:
            latest_date_str = f" - Latest COT Data: {pd.to_datetime(df['Last Update'].max()).strftime('%B %d, %Y')}"
        except:
            pass

    st.header(f"Disaggregated Markets Overview{latest_date_str}")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Disaggregated positioning data: Money Managers, Producer/Merchant, and Swap Dealer positions."
        "</p>",
        unsafe_allow_html=True
    )

    display_df = df.copy()

    # Format numeric position columns
    numeric_cols = ['Open Interest', 'MM Net', 'PM Long', 'PM Short', 'SW Long', 'SW Short']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")

    # Format OI Change
    if 'OI Change' in display_df.columns:
        def format_oi_change(x):
            if pd.isna(x):
                return ""
            sign = "+" if x >= 0 else ""
            return f"{sign}{x:,.0f}"
        display_df['OI Change'] = display_df['OI Change'].apply(format_oi_change)

    # Format percentile columns
    percentile_cols = ['MM 2Y %', 'PM L 2Y %', 'PM S 2Y %', 'SW L 2Y %', 'SW S 2Y %']
    for col in percentile_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")

    # Format correlation columns (Spearman only)
    correlation_cols = ['MM Corr', 'PM L Corr', 'PM S Corr', 'SW L Corr', 'SW S Corr']
    for col in correlation_cols:
        if col in display_df.columns:
            def format_corr(x):
                if pd.isna(x) or x is None:
                    return ""
                try:
                    return f"{float(x):.2f}"
                except (ValueError, TypeError):
                    return ""
            display_df[col] = display_df[col].apply(format_corr)

    # Format concentration columns
    conc_cols = ['Conc Long 4T', 'Conc Short 4T']
    for col in conc_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

    # Column config
    column_config = {
        "Category": st.column_config.TextColumn("Category", width="small"),
        "Instrument": st.column_config.TextColumn("Instrument", width="medium"),
        "Open Interest": st.column_config.TextColumn("OI", help="Current Open Interest"),
        "OI Change": st.column_config.TextColumn("OI Chg", help="Weekly Change in Open Interest"),
        # Money Managers
        "MM Net": st.column_config.TextColumn("MM Net", help="Managed Money Net Position"),
        "MM 2Y %": st.column_config.TextColumn("2Y %", help="2-Year Percentile of MM Net"),
        "MM Corr": st.column_config.TextColumn("Corr", help="Spearman: MM Net Changes vs Weekly Returns", width="small"),
        # Prod/Merc Long
        "PM Long": st.column_config.TextColumn("PM L", help="Producer/Merchant Long Positions"),
        "PM L 2Y %": st.column_config.TextColumn("2Y %", help="2-Year Percentile of PM Long"),
        "PM L Corr": st.column_config.TextColumn("Corr", help="Spearman: PM Long Changes vs Weekly Returns", width="small"),
        # Prod/Merc Short
        "PM Short": st.column_config.TextColumn("PM S", help="Producer/Merchant Short Positions"),
        "PM S 2Y %": st.column_config.TextColumn("2Y %", help="2-Year Percentile of PM Short"),
        "PM S Corr": st.column_config.TextColumn("Corr", help="Spearman: PM Short Changes vs Weekly Returns", width="small"),
        # Swap Long
        "SW Long": st.column_config.TextColumn("SW L", help="Swap Dealer Long Positions"),
        "SW L 2Y %": st.column_config.TextColumn("2Y %", help="2-Year Percentile of Swap Long"),
        "SW L Corr": st.column_config.TextColumn("Corr", help="Spearman: Swap Long Changes vs Weekly Returns", width="small"),
        # Swap Short
        "SW Short": st.column_config.TextColumn("SW S", help="Swap Dealer Short Positions"),
        "SW S 2Y %": st.column_config.TextColumn("2Y %", help="2-Year Percentile of Swap Short"),
        "SW S Corr": st.column_config.TextColumn("Corr", help="Spearman: Swap Short Changes vs Weekly Returns", width="small"),
        # Concentration & metadata
        "Conc Long 4T": st.column_config.TextColumn("4T L%", help="% held by 4 largest long traders"),
        "Conc Short 4T": st.column_config.TextColumn("4T S%", help="% held by 4 largest short traders"),
        "Last Update": st.column_config.TextColumn("Updated", width="small"),
    }

    # Sort by category order
    category_order = ['Index', 'Financial', 'Metals', 'Energy', 'Currency', 'Agricultural']
    display_df['category_sort'] = display_df['Category'].map({cat: i for i, cat in enumerate(category_order)})
    display_df = display_df.sort_values(['category_sort', 'Instrument'], ascending=[True, True])
    display_df = display_df.drop('category_sort', axis=1)

    # Row coloring by category
    def color_rows(row):
        category = row['Category']
        color = CATEGORY_COLORS.get(category, '#FFFFFF')
        return [f'background-color: {color}20' for _ in row]

    styled_df = display_df.style.apply(color_rows, axis=1)

    st.dataframe(
        styled_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=500
    )

    st.markdown(
        """
        <p style='color: #888888; font-size: 0.9em; margin-top: 10px;'>
        <b>Corr</b> = Spearman rank correlation between weekly position changes and weekly futures returns (2-year rolling window).
        <br><b>Categories:</b> MM = Managed Money, PM = Producer/Merchant, SW = Swap Dealers
        </p>
        """,
        unsafe_allow_html=True
    )

    # Cross-Asset Z-Score section
    st.markdown("---")
    st.subheader("Cross-Asset Futures Positioning")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Z-scores show how current positioning compares to historical average. Values above +2 or below -2 indicate extreme positioning."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("Select trader category:")
    trader_category = st.selectbox(
        "Trader category",
        ["Money Managers Net", "Prod/Merc Long", "Prod/Merc Short",
         "Swap Long", "Swap Short", "Other Reportables Net"],
        index=0,
        label_visibility="collapsed",
        key="disagg_trader_category"
    )

    display_disagg_cross_asset_zscore(api_token, trader_category)

    # Week-over-Week Changes section
    st.markdown("---")
    st.subheader("Week-over-Week Position Changes")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Shows the weekly change in positioning as percentage of open interest. "
        "Positive values indicate increasing positions, negative indicate decreasing positions."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("Select trader category:")
    wow_trader_category = st.selectbox(
        "WoW Trader category",
        ["Money Managers Net", "Prod/Merc Long", "Prod/Merc Short",
         "Swap Long", "Swap Short", "Other Reportables Net"],
        index=0,
        label_visibility="collapsed",
        key="disagg_wow_trader_category"
    )

    display_disagg_wow_changes(api_token, wow_trader_category)

    # Market Structure Matrix section
    st.markdown("---")
    st.subheader("Market Structure Matrix")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Percentile-based view of trader participation versus positioning concentration. "
        "Each instrument is ranked relative to its own 2-year history."
        "</p>",
        unsafe_allow_html=True
    )

    col1_matrix, col2_matrix, col3_matrix = st.columns([2, 2, 4])
    with col1_matrix:
        st.markdown("Select concentration metric:")
        concentration_options = {
            'conc_gross_le_4_tdr_long': 'Gross Top 4 Traders Long',
            'conc_gross_le_4_tdr_short': 'Gross Top 4 Traders Short',
            'conc_gross_le_8_tdr_long': 'Gross Top 8 Traders Long',
            'conc_gross_le_8_tdr_short': 'Gross Top 8 Traders Short',
            'conc_net_le_4_tdr_long_all': 'Net Top 4 Traders Long',
            'conc_net_le_4_tdr_short_all': 'Net Top 4 Traders Short',
            'conc_net_le_8_tdr_long_all': 'Net Top 8 Traders Long',
            'conc_net_le_8_tdr_short_all': 'Net Top 8 Traders Short',
        }
        matrix_conc_metric = st.selectbox(
            "Matrix concentration metric",
            options=list(concentration_options.keys()),
            format_func=lambda x: concentration_options[x],
            index=0,
            label_visibility="collapsed",
            key="disagg_matrix_concentration_metric"
        )

    with col2_matrix:
        st.markdown("Filter by category:")
        key_instr = load_futures_instruments()
        all_categories = ["All Categories"] + sorted(list(key_instr.keys()))
        matrix_cat_filter = st.selectbox(
            "Matrix category filter",
            options=all_categories,
            index=0,
            label_visibility="collapsed",
            key="disagg_matrix_category_filter"
        )

    matrix_fig = create_disagg_market_matrix(api_token, matrix_conc_metric, matrix_cat_filter)
    if matrix_fig:
        st.plotly_chart(matrix_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown(
        """
        <p style='color: #888; font-size: 0.9em; font-style: italic; margin-top: 10px;'>
        Market Structure Matrix shows where each instrument ranks relative to its own 2-year history.
        Bubble sizes are proportional to Open Interest (market activity level).
        Percentiles allow fair comparison across different markets with varying scales.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Positioning Concentration Analysis section
    display_disagg_positioning_concentration(api_token)


# -- Column mappings for disaggregated trader categories --

DISAGG_CATEGORY_COLUMNS = {
    "Money Managers": {
        "long": "m_money_positions_long_all",
        "short": "m_money_positions_short_all",
        "net": "net_mm_positions",
        "pct_long": "pct_of_oi_m_money_long_all",
        "pct_short": "pct_of_oi_m_money_short_all",
    },
    "Prod/Merc Long": {
        "col": "prod_merc_positions_long",
        "pct": "pct_of_oi_prod_merc_long",
    },
    "Prod/Merc Short": {
        "col": "prod_merc_positions_short",
        "pct": "pct_of_oi_prod_merc_short",
    },
    "Swap Long": {
        "col": "swap_positions_long_all",
        "pct": "pct_of_oi_swap_long_all",
    },
    "Swap Short": {
        "col": "swap__positions_short_all",
        "pct": "pct_of_oi_swap_short_all",
    },
    "Other Reportables": {
        "long": "other_rept_positions_long",
        "short": "other_rept_positions_short",
        "net": "net_other_positions",
        "pct_long": "pct_of_oi_other_rept_long",
        "pct_short": "pct_of_oi_other_rept_short",
    },
}


def _get_position_series(df, trader_category):
    """Return the position series for a given trader category.

    For 'net' categories (MM Net, Other Net) returns net = long - short.
    For single-leg categories (PM Long, SW Short etc.) returns that column directly.
    """
    cat = trader_category.replace(" Net", "")
    cols = DISAGG_CATEGORY_COLUMNS.get(cat)
    if cols is None:
        return None

    if "net" in cols:
        # Calculate net if not already present
        if cols["net"] not in df.columns:
            df[cols["net"]] = df[cols["long"]] - df[cols["short"]]
        return df[cols["net"]]
    else:
        col = cols["col"]
        if col in df.columns:
            return df[col]
    return None


def _get_pct_oi_series(df, trader_category):
    """Return the % of OI series.

    For single-leg categories: uses the API-provided pct_of_oi column directly.
    For net categories: calculates manually as net / open_interest_all * 100.
    """
    cat = trader_category.replace(" Net", "")
    cols = DISAGG_CATEGORY_COLUMNS.get(cat)
    if cols is None:
        return None

    if "pct" in cols:
        # Single-leg: use API column directly
        if cols["pct"] in df.columns:
            return pd.to_numeric(df[cols["pct"]], errors='coerce')
    elif "net" in cols:
        # Net category: calculate manually from net position / OI
        pos = _get_position_series(df, trader_category)
        if pos is not None and 'open_interest_all' in df.columns:
            oi = pd.to_numeric(df['open_interest_all'], errors='coerce')
            return (pos / oi * 100)
    return None


@st.cache_data(ttl=3600)
def fetch_disagg_zscore_data_parallel(api_token, trader_category):
    """Fetch and calculate Z-scores for all disaggregated instruments in parallel"""

    key_instruments = load_futures_instruments()
    ticker_map = get_ticker_mapping()
    name_map = get_ticker_to_name_mapping()

    all_instruments = []
    for category, instruments in key_instruments.items():
        for ticker, cot_name in instruments.items():
            all_instruments.append(cot_name)

    instrument_data = {}

    def process_single_instrument(instrument):
        try:
            df = fetch_disagg_data_2year(instrument, api_token)
            if df is None or df.empty:
                return None

            pos = _get_position_series(df, trader_category)
            if pos is None or pos.dropna().empty:
                return None

            pos = pos.astype(float)
            pos_mean = pos.mean()
            pos_std = pos.std()

            if pos_std <= 0 or len(pos) < 10:
                return None

            latest_val = pos.iloc[-1]
            z_score = (latest_val - pos_mean) / pos_std

            week_ago_z = None
            if len(pos) > 1:
                week_ago_val = pos.iloc[-2]
                week_ago_z = (week_ago_val - pos_mean) / pos_std

            # Z-score of position as % of OI (using API-provided pct_of_oi columns)
            z_score_pct = None
            week_ago_z_pct = None
            latest_pct = None

            pct_oi = _get_pct_oi_series(df, trader_category)
            if pct_oi is not None:
                pct_oi = pct_oi.fillna(0)
                mean_pct = pct_oi.mean()
                std_pct = pct_oi.std()

                if std_pct > 0:
                    latest_pct = pct_oi.iloc[-1]
                    z_score_pct = (latest_pct - mean_pct) / std_pct
                    if len(pct_oi) > 1:
                        week_ago_z_pct = (pct_oi.iloc[-2] - mean_pct) / std_pct

            # Resolve ticker
            ticker = ticker_map.get(instrument)
            if not ticker:
                instrument_short = instrument.split(' - ')[0]
                ticker = ticker_map.get(instrument_short, instrument_short[:2].upper())

            return {
                'ticker': ticker,
                'z_score': z_score,
                'z_score_pct': z_score_pct,
                'week_ago_z': week_ago_z,
                'week_ago_z_pct': week_ago_z_pct,
                'net_pct': latest_pct,
                'mean': pos_mean,
                'std': pos_std,
                'full_name': instrument,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_instrument, inst) for inst in all_instruments]
        for future in as_completed(futures):
            result = future.result()
            if result:
                instrument_data[result['ticker']] = result

    return instrument_data


def display_disagg_cross_asset_zscore(api_token, trader_category):
    """Display cross-asset Z-score charts for disaggregated categories"""

    with st.spinner("Loading cross-asset analysis..."):
        instrument_data = fetch_disagg_zscore_data_parallel(api_token, trader_category)

    if not instrument_data:
        st.warning("No valid data found for cross-asset comparison")
        return

    # Sort by raw Z-score
    sorted_instruments = sorted(
        instrument_data.items(),
        key=lambda x: x[1]['z_score'] if x[1]['z_score'] is not None else -999,
        reverse=True
    )

    name_map = get_ticker_to_name_mapping()

    # Build category color lookup
    key_instruments = load_futures_instruments()
    ticker_to_category = {}
    for category, instruments in key_instruments.items():
        for ticker in instruments:
            ticker_to_category[ticker] = category

    for display_mode in ["Raw", "as % of Open Interest"]:
        if display_mode == "Raw":
            y_values = [item[1]['z_score'] if item[1]['z_score'] is not None else 0
                        for item in sorted_instruments]
            week_ago_values = [item[1]['week_ago_z'] for item in sorted_instruments]
            y_title = "Z-Score"
        else:
            y_values = [item[1]['z_score_pct'] if item[1]['z_score_pct'] is not None else 0
                        for item in sorted_instruments]
            week_ago_values = [item[1]['week_ago_z_pct'] for item in sorted_instruments]
            y_title = "Z-Score (% of OI)"

        fig = go.Figure()

        tickers = [item[0] for item in sorted_instruments]
        display_names = [name_map.get(t, t) for t in tickers]

        # Color by category
        colors = []
        for t in tickers:
            cat = ticker_to_category.get(t, 'Agricultural')
            colors.append(CATEGORY_COLORS.get(cat, '#95E77E'))

        fig.add_trace(go.Bar(
            x=display_names,
            y=y_values,
            name='Current',
            marker=dict(color=colors),
            text=[f"{y:.2f}" for y in y_values],
            textposition='outside',
            hoverinfo='skip',
        ))

        # Week-ago diamond markers
        valid_week_ago = [(i, z) for i, z in enumerate(week_ago_values) if z is not None]
        if valid_week_ago:
            indices, week_z_vals = zip(*valid_week_ago)
            fig.add_trace(go.Scatter(
                x=[display_names[i] for i in indices],
                y=week_z_vals,
                mode='markers',
                name='Week Ago',
                marker=dict(symbol='diamond', size=10, color='purple',
                            line=dict(width=2, color='white')),
                hoverinfo='skip',
            ))

        title = (f"{trader_category} Positioning Z-Scores (2-year lookback)"
                 if display_mode == "Raw"
                 else f"{trader_category} Positioning Z-Scores (% of OI basis, 2-year lookback)")

        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title=y_title,
            height=500,
            showlegend=False,
            hovermode='x unified',
            yaxis=dict(
                zeroline=True, zerolinewidth=2, zerolinecolor='black',
                gridcolor='lightgray', range=[-3, 3],
            ),
            xaxis=dict(tickangle=-45),
            plot_bgcolor='white',
            bargap=0.2,
            margin=dict(l=50, r=50, t=80, b=150),
        )

        fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1)
        fig.add_hline(y=1, line_dash="dot", line_color="gray", line_width=1)
        fig.add_hline(y=-1, line_dash="dot", line_color="gray", line_width=1)

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


@st.cache_data(ttl=3600)
def fetch_disagg_wow_data(api_token, trader_category):
    """Fetch week-over-week changes for all disaggregated instruments in parallel"""

    key_instruments = load_futures_instruments()
    ticker_map = get_ticker_mapping()

    all_instruments = []
    for category, instruments in key_instruments.items():
        for ticker, cot_name in instruments.items():
            all_instruments.append(cot_name)

    instrument_data = {}

    def process_single_instrument(instrument):
        try:
            df = fetch_disagg_data_2year(instrument, api_token)
            if df is None or df.empty or len(df) < 2:
                return None

            df = df.sort_values('report_date_as_yyyy_mm_dd')

            pos = _get_position_series(df, trader_category)
            if pos is None or pos.dropna().empty or len(pos) < 2:
                return None

            pos = pos.astype(float)

            # Latest WoW raw change
            latest_val = pos.iloc[-1]
            prev_val = pos.iloc[-2]
            raw_change = latest_val - prev_val

            # Historical Z-score of raw WoW changes
            wow_changes = pos.diff().dropna()
            z_score = None
            if len(wow_changes) > 10:
                mean_chg = wow_changes.mean()
                std_chg = wow_changes.std()
                if std_chg > 0:
                    z_score = (raw_change - mean_chg) / std_chg

            # Z-score of WoW change as % of OI
            z_score_pct_oi = None
            pct_oi_series = _get_pct_oi_series(df, trader_category)
            if pct_oi_series is not None:
                pct_oi_series = pct_oi_series.astype(float)
                pct_change = pct_oi_series.diff().dropna()

                if len(pct_change) > 10:
                    latest_pct_chg = pct_oi_series.iloc[-1] - pct_oi_series.iloc[-2]
                    mean_pct_chg = pct_change.mean()
                    std_pct_chg = pct_change.std()
                    if std_pct_chg > 0:
                        z_score_pct_oi = (latest_pct_chg - mean_pct_chg) / std_pct_chg

            # Resolve ticker
            ticker = ticker_map.get(instrument)
            if not ticker:
                instrument_short = instrument.split(' - ')[0]
                ticker = ticker_map.get(instrument_short, instrument_short[:2].upper())

            return {
                'ticker': ticker,
                'raw_change': raw_change,
                'z_score': z_score,
                'z_score_pct_oi': z_score_pct_oi,
                'latest_val': latest_val,
                'prev_val': prev_val,
                'full_name': instrument,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_instrument, inst) for inst in all_instruments]
        for future in as_completed(futures):
            result = future.result()
            if result:
                instrument_data[result['ticker']] = result

    return instrument_data


def display_disagg_wow_changes(api_token, trader_category):
    """Display week-over-week changes for disaggregated categories"""

    with st.spinner("Loading week-over-week changes..."):
        instrument_data = fetch_disagg_wow_data(api_token, trader_category)

    if not instrument_data:
        st.warning("No valid data found for week-over-week changes")
        return

    # Sort by raw Z-score
    sorted_instruments = sorted(
        instrument_data.items(),
        key=lambda x: x[1]['z_score'] if x[1]['z_score'] is not None else 0,
        reverse=True
    )

    name_map = get_ticker_to_name_mapping()

    # Build category color lookup
    key_instruments = load_futures_instruments()
    ticker_to_category = {}
    for category, instruments in key_instruments.items():
        for ticker in instruments:
            ticker_to_category[ticker] = category

    for display_mode in ["Raw", "as % of Open Interest"]:
        if display_mode == "Raw":
            y_values = [item[1]['z_score'] if item[1]['z_score'] is not None else 0
                        for item in sorted_instruments]
            y_title = "WoW Change Z-Score (Historical)"
        else:
            y_values = [item[1]['z_score_pct_oi'] if item[1]['z_score_pct_oi'] is not None else 0
                        for item in sorted_instruments]
            y_title = "Change Z-Score (% of OI, Historical)"

        fig = go.Figure()

        tickers = [item[0] for item in sorted_instruments]
        display_names = [name_map.get(t, t) for t in tickers]

        colors = []
        for t in tickers:
            cat = ticker_to_category.get(t, 'Agricultural')
            colors.append(CATEGORY_COLORS.get(cat, '#95E77E'))

        fig.add_trace(go.Bar(
            x=display_names,
            y=y_values,
            name='WoW Change',
            marker=dict(color=colors),
            text=[f"{y:.2f}" for y in y_values],
            textposition='outside',
            hoverinfo='skip',
        ))

        title = (f"{trader_category} Week-over-Week Position Changes"
                 if display_mode == "Raw"
                 else f"{trader_category} Week-over-Week Position Changes (% of OI)")

        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title=y_title,
            height=700,
            showlegend=False,
            hovermode='x unified',
            yaxis=dict(
                zeroline=True, zerolinewidth=2, zerolinecolor='black',
                gridcolor='lightgray', range=[-3, 3],
            ),
            xaxis=dict(tickangle=-45, tickfont=dict(size=12)),
            plot_bgcolor='white',
            bargap=0.2,
            margin=dict(l=50, r=50, t=80, b=150),
        )

        fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1)
        fig.add_hline(y=1, line_dash="dot", line_color="gray", line_width=1)
        fig.add_hline(y=-1, line_dash="dot", line_color="gray", line_width=1)

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


@st.cache_data(ttl=3600)
def fetch_disagg_market_matrix_data(api_token):
    """Fetch 2-year disaggregated data for all instruments (for market matrix)"""

    key_instruments = load_futures_instruments()
    all_instruments = []
    for category, instruments in key_instruments.items():
        for ticker, cot_name in instruments.items():
            all_instruments.append(cot_name)

    all_data = {}

    def fetch_one(instrument):
        try:
            df = fetch_disagg_data_2year(instrument, api_token)
            if df is not None and not df.empty:
                return {instrument: df}
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_one, inst) for inst in all_instruments]
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_data.update(result)

    return all_data


def create_disagg_market_matrix(api_token, concentration_metric='conc_gross_le_4_tdr_long', category_filter='All Categories'):
    """Create market structure matrix for disaggregated dashboard instruments"""
    try:
        with st.spinner("Loading market structure data..."):
            all_data = fetch_disagg_market_matrix_data(api_token)

        if not all_data:
            st.warning("No data available for market matrix")
            return None

        scatter_data = []
        lookback_date = pd.Timestamp.now() - pd.DateOffset(years=2)

        key_instruments = load_futures_instruments()
        ticker_map = get_ticker_mapping()
        name_map = get_ticker_to_name_mapping()

        dashboard_instruments = []
        for category, instruments in key_instruments.items():
            for ticker, cot_name in instruments.items():
                dashboard_instruments.append(cot_name)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, instrument in enumerate(dashboard_instruments):
            status_text.text(f"Calculating percentiles for {instrument}...")
            progress_bar.progress((idx + 1) / len(dashboard_instruments))

            if instrument not in all_data:
                continue

            df = all_data[instrument]
            df_2yr = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()

            if len(df_2yr) < 10:
                if len(df) >= 10:
                    df_2yr = df.copy()
                else:
                    continue

            latest_idx = df['report_date_as_yyyy_mm_dd'].idxmax()
            latest_data = df.loc[latest_idx]

            trader_count = latest_data.get('traders_tot_all', 0)
            open_interest = latest_data.get('open_interest_all', 0)
            concentration = latest_data.get(concentration_metric, 0)

            if pd.isna(trader_count) or pd.isna(open_interest):
                continue

            trader_count = float(trader_count)
            open_interest = float(open_interest)
            concentration = float(concentration) if pd.notna(concentration) else 0

            # Percentiles
            traders_col = pd.to_numeric(df_2yr['traders_tot_all'], errors='coerce').dropna()
            trader_percentile = (traders_col <= trader_count).sum() / len(traders_col) * 100 if len(traders_col) > 0 else 50

            if concentration_metric in df_2yr.columns:
                conc_col = pd.to_numeric(df_2yr[concentration_metric], errors='coerce').dropna()
                conc_percentile = (conc_col <= concentration).sum() / len(conc_col) * 100 if len(conc_col) > 0 else 50
            else:
                conc_percentile = 50

            # Resolve category and ticker
            category = None
            ticker = None
            for cat, instruments in key_instruments.items():
                for tick, cot_name in instruments.items():
                    if cot_name == instrument:
                        category = cat
                        ticker = tick
                        break
                if category:
                    break

            if not ticker:
                ticker = ticker_map.get(instrument)
                if not ticker:
                    ticker = ticker_map.get(instrument.split(' - ')[0])

            commodity_name = name_map.get(ticker, instrument.split(' - ')[0])

            if category_filter == 'All Categories' or category == category_filter:
                scatter_data.append({
                    'instrument': instrument,
                    'trader_count': trader_count,
                    'concentration': concentration,
                    'trader_percentile': trader_percentile,
                    'conc_percentile': conc_percentile,
                    'open_interest': open_interest,
                    'short_name': commodity_name,
                    'category': category or 'Unknown',
                })

        progress_bar.empty()
        status_text.empty()

        if not scatter_data:
            st.warning(f"No data available for {category_filter if category_filter != 'All Categories' else 'selected instruments'}")
            return None

        total = len(dashboard_instruments)
        included = len(scatter_data)
        excluded = total - included
        cat_info = f" ({category_filter})" if category_filter != 'All Categories' else ""
        if excluded > 0:
            st.info(f"Market Matrix: Showing {included}/{total} instruments{cat_info} ({excluded} excluded due to data limitations)")
        elif category_filter != 'All Categories':
            st.info(f"Market Matrix: Showing {included} {category_filter} instruments")

        # Build chart
        fig = go.Figure()

        colors = [CATEGORY_COLORS.get(d['category'], '#95E77E') for d in scatter_data]

        # Smart text positioning
        def get_smart_text_position(x, y, index, data_points):
            if x < 15:
                if y > 85:
                    return 'bottom right'
                elif y < 15:
                    return 'top right'
                return 'middle right'
            elif x > 85:
                if y > 85:
                    return 'bottom left'
                elif y < 15:
                    return 'top left'
                return 'middle left'
            elif y > 85:
                return 'bottom center'
            elif y < 15:
                return 'top center'
            else:
                positions = ['top center', 'bottom center', 'middle left', 'middle right',
                             'top left', 'top right', 'bottom left', 'bottom right']
                min_overlap = float('inf')
                best = 'top center'
                for pos in positions:
                    overlap = 0
                    for i, other in enumerate(data_points):
                        if i != index:
                            dist = ((x - other['trader_percentile'])**2 + (y - other['conc_percentile'])**2)**0.5
                            if dist < 20:
                                overlap += 1
                    if overlap < min_overlap:
                        min_overlap = overlap
                        best = pos
                return best

        text_positions = [get_smart_text_position(d['trader_percentile'], d['conc_percentile'], i, scatter_data)
                          for i, d in enumerate(scatter_data)]

        fig.add_trace(go.Scatter(
            x=[d['trader_percentile'] for d in scatter_data],
            y=[d['conc_percentile'] for d in scatter_data],
            mode='markers+text',
            marker=dict(size=15, color=colors, line=dict(width=2, color='white')),
            text=[d['short_name'] for d in scatter_data],
            textposition=text_positions,
            hovertemplate='<b>%{customdata[0]}</b><br>'
                          'Category: %{customdata[1]}<br>'
                          'Trader Count Percentile: %{x:.1f}%<br>'
                          'Concentration Percentile: %{y:.1f}%<br>'
                          'Actual Traders: %{customdata[2]:,.0f}<br>'
                          'Actual Concentration: %{customdata[3]:.1f}%<br>'
                          'Open Interest: %{customdata[4]:,.0f}<extra></extra>',
            customdata=[[d['instrument'], d['category'], d['trader_count'],
                         d['concentration'], d['open_interest']] for d in scatter_data]
        ))

        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)

        fig.add_annotation(text="Below Median Concentration<br>Above Median Traders",
                           xref="paper", yref="paper", x=0.75, y=0.25,
                           showarrow=False, font=dict(size=10, color="green"), opacity=0.8)
        fig.add_annotation(text="Above Median Concentration<br>Above Median Traders",
                           xref="paper", yref="paper", x=0.75, y=0.75,
                           showarrow=False, font=dict(size=10, color="orange"), opacity=0.8)
        fig.add_annotation(text="Below Median Concentration<br>Below Median Traders",
                           xref="paper", yref="paper", x=0.25, y=0.25,
                           showarrow=False, font=dict(size=10, color="blue"), opacity=0.8)
        fig.add_annotation(text="Above Median Concentration<br>Below Median Traders",
                           xref="paper", yref="paper", x=0.25, y=0.75,
                           showarrow=False, font=dict(size=10, color="red"), opacity=0.8)

        metric_labels = {
            'conc_gross_le_4_tdr_long': 'Gross Top 4 Long',
            'conc_gross_le_4_tdr_short': 'Gross Top 4 Short',
            'conc_gross_le_8_tdr_long': 'Gross Top 8 Long',
            'conc_gross_le_8_tdr_short': 'Gross Top 8 Short',
            'conc_net_le_4_tdr_long_all': 'Net Top 4 Long',
            'conc_net_le_4_tdr_short_all': 'Net Top 4 Short',
            'conc_net_le_8_tdr_long_all': 'Net Top 8 Long',
            'conc_net_le_8_tdr_short_all': 'Net Top 8 Short',
        }
        metric_label = metric_labels.get(concentration_metric, concentration_metric)

        fig.update_layout(
            title=f"Market Structure Matrix - {metric_label} (2-Year Percentiles)",
            xaxis_title="Trader Count Percentile (2-Year)",
            yaxis_title=f"Concentration Percentile ({metric_label}, 2-Year)",
            height=700,
            showlegend=False,
            xaxis=dict(range=[0, 100], gridcolor='lightgray', zeroline=False, ticksuffix='%'),
            yaxis=dict(range=[0, 100], gridcolor='lightgray', zeroline=False, ticksuffix='%'),
        )

        fig.add_annotation(
            text="Bubble size = Open Interest (larger = more market activity)<br>Color = Asset Category",
            xref="paper", yref="paper", x=0.02, y=0.98,
            showarrow=False, font=dict(size=10), xanchor="left", yanchor="top"
        )

        return fig

    except Exception as e:
        st.error(f"Error creating market structure matrix: {str(e)}")
        return None


# -- Positioning Concentration Analysis --

# Map disaggregated trader categories to pct_of_oi columns
DISAGG_PCT_OI_COLUMNS = {
    "Money Managers Long": "pct_of_oi_m_money_long_all",
    "Money Managers Short": "pct_of_oi_m_money_short_all",
    "Prod/Merc Long": "pct_of_oi_prod_merc_long",
    "Prod/Merc Short": "pct_of_oi_prod_merc_short",
    "Swap Dealer Long": "pct_of_oi_swap_long_all",
    "Swap Dealer Short": "pct_of_oi_swap_short_all",
    "Other Reportables Long": "pct_of_oi_other_rept_long",
    "Other Reportables Short": "pct_of_oi_other_rept_short",
}


def create_disagg_positioning_concentration_charts(selected_instruments, trader_category, api_token):
    """Create time series and bar charts for disaggregated positioning concentration"""
    import time as _time
    try:
        all_data = {}
        failed_instruments = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        pct_oi_column = DISAGG_PCT_OI_COLUMNS[trader_category]

        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Fetching data for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))

            if idx > 0:
                _time.sleep(0.3)

            df = fetch_disagg_data_full(instrument, api_token)

            if df is not None and not df.empty:
                df = df.sort_values('report_date_as_yyyy_mm_dd')

                if pct_oi_column in df.columns:
                    df['position_pct_oi'] = pd.to_numeric(df[pct_oi_column], errors='coerce').fillna(0)

                    all_data[instrument] = {
                        'dates': df['report_date_as_yyyy_mm_dd'],
                        'position_pct_oi': df['position_pct_oi'],
                        'latest_pct': df['position_pct_oi'].iloc[-1] if len(df) > 0 else 0,
                    }
                else:
                    failed_instruments.append(instrument)
            else:
                failed_instruments.append(instrument)

        progress_bar.empty()
        status_text.empty()

        if failed_instruments:
            st.warning(f"Failed to fetch data for {len(failed_instruments)} instrument(s): "
                       f"{', '.join([get_short_instrument_name(i) for i in failed_instruments])}")

        if all_data:
            st.success(f"Successfully loaded data for {len(all_data)} instrument(s): "
                       f"{', '.join([get_short_instrument_name(i) for i in all_data.keys()])}")

        if not all_data:
            st.error("No valid data found for selected instruments")
            return None, None

        # Time series chart
        time_series_fig = go.Figure()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for idx, (instrument, data) in enumerate(all_data.items()):
            short_name = get_short_instrument_name(instrument)
            time_series_fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data['position_pct_oi'],
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate=f'<b>{short_name}</b><br>'
                              'Date: %{x}<br>'
                              'Position % of OI: %{y:.1f}%<extra></extra>'
            ))

        time_series_fig.update_layout(
            title=f"{trader_category} Positioning as % of Open Interest",
            xaxis_title="Date",
            yaxis_title="Positioning (% of OI)",
            height=500,
            hovermode='x unified',
            yaxis=dict(gridcolor='lightgray', ticksuffix='%'),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Bar chart for latest values
        bar_chart_fig = go.Figure()

        sorted_instruments = sorted(all_data.items(),
                                    key=lambda x: x[1]['latest_pct'],
                                    reverse=True)

        instruments_full = [item[0] for item in sorted_instruments]
        instruments_short = [get_short_instrument_name(name) for name in instruments_full]
        latest_values = [item[1]['latest_pct'] for item in sorted_instruments]

        bar_chart_fig.add_trace(go.Bar(
            x=instruments_short,
            y=latest_values,
            marker=dict(color='#3498db'),
            text=[f"{v:.1f}%" for v in latest_values],
            textposition='outside',
            hovertemplate='<b>%{customdata}</b><br>'
                          'Position % of OI: %{y:.1f}%<extra></extra>',
            customdata=instruments_full
        ))

        bar_chart_fig.update_layout(
            title=f"Latest {trader_category} Positioning (% of OI)",
            xaxis_title="",
            yaxis_title="Positioning (% of OI)",
            height=500,
            showlegend=False,
            yaxis=dict(gridcolor='lightgray', ticksuffix='%'),
            xaxis=dict(tickangle=-45),
            bargap=0.2
        )

        if sorted_instruments:
            latest_date = all_data[sorted_instruments[0][0]]['dates'].iloc[-1]
            bar_chart_fig.add_annotation(
                text=f"Data as of: {latest_date.strftime('%Y-%m-%d')}",
                xref="paper", yref="paper",
                x=0.99, y=0.99,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right", yanchor="top"
            )

        return time_series_fig, bar_chart_fig

    except Exception as e:
        st.error(f"Error creating positioning concentration charts: {str(e)}")
        return None, None


def display_disagg_positioning_concentration(api_token):
    """Display positioning concentration analysis for disaggregated categories"""

    st.markdown("---")
    st.subheader("Positioning Concentration Analysis")
    st.markdown(
        "<p style='color: #888; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px;'>"
        "Absolute long and short positions as percentage of open interest. "
        "Shows the magnitude of trader positions in each market."
        "</p>",
        unsafe_allow_html=True
    )

    key_instruments = load_futures_instruments()
    all_categories = sorted(list(key_instruments.keys()))

    col1_pos, col2_pos = st.columns([2, 6])
    with col1_pos:
        st.markdown("Select asset category:")
        selected_category = st.selectbox(
            "Category",
            options=all_categories,
            index=all_categories.index("Metals") if "Metals" in all_categories else 0,
            label_visibility="collapsed",
            key="disagg_positioning_category"
        )

    with col2_pos:
        st.markdown("Select trader category:")
        trader_category = st.selectbox(
            "Trader category",
            options=list(DISAGG_PCT_OI_COLUMNS.keys()),
            index=0,
            label_visibility="collapsed",
            key="disagg_positioning_trader_category"
        )

    if selected_category in key_instruments:
        category_instruments = list(key_instruments[selected_category].values())

        if category_instruments:
            with st.spinner(f"Calculating positioning concentration for {selected_category} instruments..."):
                fig_ts, fig_bar = create_disagg_positioning_concentration_charts(
                    category_instruments, trader_category, api_token
                )

            if fig_ts and fig_bar:
                st.plotly_chart(fig_ts, use_container_width=True)
                st.plotly_chart(fig_bar, use_container_width=True)

                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Download Time Series Chart", key="disagg_download_positioning_ts"):
                        html_string = fig_ts.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="Download Time Series",
                            data=html_string,
                            file_name=f"disagg_positioning_ts_{selected_category}_{trader_category}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                            mime="text/html",
                            key="disagg_download_ts_btn"
                        )
                with col2:
                    if st.button("Download Bar Chart", key="disagg_download_positioning_bar"):
                        html_string = fig_bar.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="Download Bar Chart",
                            data=html_string,
                            file_name=f"disagg_positioning_bar_{selected_category}_{trader_category}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                            mime="text/html",
                            key="disagg_download_bar_btn"
                        )
            else:
                st.error("Unable to generate positioning concentration charts. Please check the data.")
        else:
            st.warning(f"No instruments found for category: {selected_category}")
    else:
        st.error(f"Category '{selected_category}' not found")
