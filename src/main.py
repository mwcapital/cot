"""
Main application file that preserves the EXACT UI from legacyF.py
but uses modular components for implementation
"""
import streamlit as st
import pandas as pd

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="CFTC COT Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import data and chart functions
from data_fetcher import load_instruments_database, fetch_cftc_data
from display_functions_exact import (
    display_time_series_chart,
    display_trader_participation_chart
)
# Multi-instrument handler removed - all multi-instrument analyses moved to main dashboard
# from multi_instrument_handler import handle_multi_instrument_flow
from dashboard_overview import display_dashboard

def handle_single_instrument_flow(chart_type, instruments_db, api_token):
    """Handle single instrument selection and analysis - EXACT copy from legacyF.py"""
    st.header("🎯 Select Instrument")
    
    # Search method selection
    search_method = st.radio(
        "Choose search method:",
        ["Extensive Search", "Search by Commodity Subgroup", "Search by Commodity Type", "Free Text Search"],
        horizontal=True
    )

    selected_instrument = None

    if search_method == "Extensive Search":
        st.subheader("📂 Browse by Exchange Hierarchy")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            exchanges = list(instruments_db['exchanges'].keys())
            selected_exchange = st.selectbox("📍 Exchange:", exchanges)

        with col2:
            groups = list(instruments_db['exchanges'][selected_exchange].keys())
            selected_group = st.selectbox("📂 Group:", groups)

        with col3:
            subgroups = list(instruments_db['exchanges'][selected_exchange][selected_group].keys())
            selected_subgroup = st.selectbox("📁 Subgroup:", subgroups)

        with col4:
            commodities = list(instruments_db['exchanges'][selected_exchange][selected_group][selected_subgroup].keys())
            selected_commodity = st.selectbox("🔸 Commodity:", commodities)

        # Instrument selection
        instruments = instruments_db['exchanges'][selected_exchange][selected_group][selected_subgroup][
            selected_commodity]
        selected_instrument = st.selectbox("📊 Select Instrument:", instruments, key="hierarchy_instrument")

        # Show classification path
        st.info(
            f"📦 **{selected_exchange}** → **{selected_group}** → **{selected_subgroup}** → **{selected_commodity}**")

    elif search_method == "Search by Commodity Subgroup":
        st.subheader("📁 Search by Commodity Subgroup")
        col1, col2 = st.columns(2)

        with col1:
            subgroups = sorted(list(instruments_db['commodity_subgroups'].keys()))
            selected_subgroup = st.selectbox("📁 Select Commodity Subgroup:", subgroups)

        with col2:
            instruments = instruments_db['commodity_subgroups'][selected_subgroup]
            selected_instrument = st.selectbox("📊 Select Instrument:", sorted(instruments), key="subgroup_instrument")

        st.info(f"📁 **{selected_subgroup}** → {len(instruments)} available instruments")

    elif search_method == "Search by Commodity Type":
        st.subheader("🔸 Search by Commodity Type")
        col1, col2 = st.columns(2)

        with col1:
            commodities = sorted(list(instruments_db['commodities'].keys()))
            selected_commodity_type = st.selectbox("🔸 Select Commodity:", commodities)

        with col2:
            instruments = instruments_db['commodities'][selected_commodity_type]
            selected_instrument = st.selectbox("📊 Select Instrument:", sorted(instruments), key="commodity_instrument")

        st.info(f"🔸 **{selected_commodity_type}** → {len(instruments)} available instruments")

    else:  # Free Text Search
        st.subheader("🔍 Free Text Search")
        search_text = st.text_input(
            "Type instrument name or keyword:",
            placeholder="e.g., GOLD, CRUDE OIL, S&P 500, WHEAT..."
        )

        if search_text:
            # Filter instruments based on search text
            all_instruments = instruments_db['all_instruments']
            filtered_instruments = [
                inst for inst in all_instruments
                if search_text.upper() in inst.upper()
            ]

            if filtered_instruments:
                selected_instrument = st.selectbox(
                    f"📊 Select from {len(filtered_instruments)} matching instruments:",
                    sorted(filtered_instruments),
                    key="search_instrument"
                )
                st.success(f"✅ Found {len(filtered_instruments)} matching instruments")
            else:
                st.warning(f"⚠️ No instruments found matching '{search_text}'")
        else:
            st.info("💡 Start typing to search through all available instruments")

    # Fetch Data Button - Now positioned after instrument selection
    if selected_instrument:
        st.markdown("---")
        st.subheader(f"Selected: {selected_instrument}")

        # Check if we need to fetch new data
        need_new_fetch = (not st.session_state.data_fetched or
                          st.session_state.fetched_instrument != selected_instrument)

        if st.button("🚀 Fetch Data", type="primary", use_container_width=False, disabled=not need_new_fetch):
            with st.spinner(f"Fetching data for {selected_instrument}..."):
                df = fetch_cftc_data(selected_instrument, api_token)

            if df is not None and not df.empty:
                st.session_state.fetched_data = df
                st.session_state.data_fetched = True
                st.session_state.fetched_instrument = selected_instrument
                st.success(f"✅ Successfully fetched {len(df)} records")
            else:
                st.error("❌ No data available for the selected instrument")
                st.session_state.data_fetched = False

    # Display data if available
    if st.session_state.data_fetched and st.session_state.fetched_data is not None:
        df = st.session_state.fetched_data
        st.markdown("---")

        # Display chart based on pre-selected chart type
        if chart_type == "Time Series":
            display_time_series_chart(df, selected_instrument)

        elif chart_type == "Trader Participation":
            display_trader_participation_chart(df, selected_instrument)


def main():
    """Main function - EXACT copy of UI from legacyF.py"""
    # Initialize session state FIRST before anything else
    if 'data_fetched' not in st.session_state:
        st.session_state.data_fetched = False
    if 'fetched_data' not in st.session_state:
        st.session_state.fetched_data = None
    if 'fetched_instrument' not in st.session_state:
        st.session_state.fetched_instrument = None
    if 'fetched_instruments_multi' not in st.session_state:
        st.session_state.fetched_instruments_multi = []
    
    st.title("CFTC Commitments of Traders Dashboard")
    st.markdown("Interactive analysis of CFTC COT data")
    
    # Add Stephen Briese quote
    st.markdown("""
    <div style="
        background-color: #f0f2f6;
        padding: 20px;
        border-left: 4px solid #1f77b4;
        margin: 20px 0;
        border-radius: 5px;
    ">
        <p style="
            font-style: italic;
            font-size: 16px;
            line-height: 1.6;
            color: #333;
            margin: 0 0 10px 0;
        ">
            "In a letter to the Commission dated August 26, 2006, the International Swap and Derivatives Association objected 
            on behalf of its '725 member institutions,' noting that 'Market participants, including speculators, with such 
            information on non-traditional commercial trends, would gain a competitive advantage allowing them to trade ahead 
            of the swap dealers.' If the largest derivative traders in the world are afraid of your having access to this 
            intelligence, you had better take notice."
        </p>
        <p style="
            text-align: right;
            font-size: 14px;
            color: #666;
            margin: 0;
            font-weight: 600;
        ">
            — Stephen Briese, <em>The Commitments of Traders Bible</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load instruments database
    instruments_db = load_instruments_database()
    if not instruments_db:
        st.stop()

    # API Token Configuration
    # Use session state to persist the API token
    if 'api_token' not in st.session_state:
        st.session_state.api_token = "3CKjkFN6jIIHgSkIJH19i7VhK"  # Default token

    with st.expander("API Configuration", expanded=False):
        api_token = st.text_input(
            "API Token (optional):",
            value=st.session_state.api_token,
            type="password",
            help="Enter your CFTC API token for higher rate limits. Leave empty to use default limits.",
            key="api_token_input"
        )
        st.session_state.api_token = api_token

    st.markdown("---")

    # Display Dashboard Overview
    display_dashboard(api_token)

    st.markdown("---")

    # Chart Type Selection
    st.header("Select Analysis Type")

    single_chart_type = st.segmented_control(
        "Select chart type",
        ["Time Series", "Trader Participation"],
        selection_mode="single",
        default=None,
        key="single_chart_type"
    )

    st.markdown("---")

    # Determine which type of analysis was selected
    if single_chart_type:
        # Single Instrument Flow
        handle_single_instrument_flow(single_chart_type, instruments_db, api_token)
    else:
        # No chart type selected yet
        st.info("👆 Please select an analysis type from the options above to begin")


if __name__ == "__main__":
    main()