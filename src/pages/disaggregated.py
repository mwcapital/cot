"""
Disaggregated COT Report page
Uses the CFTC Disaggregated Futures report (dataset 72hh-3qpy)
4 reportable categories: Producer/Merchant, Swap Dealers, Managed Money, Other Reportables
"""
import streamlit as st
from disaggregated_dashboard import display_disagg_dashboard
from disaggregated_data_fetcher import load_disagg_instruments_database, fetch_disagg_data_full
from disaggregated_display_functions import display_disagg_time_series_chart

st.title("Disaggregated COT Report")
st.markdown("Disaggregated Futures positioning with 4 trader categories: "
            "Producer/Merchant, Swap Dealers, Managed Money, Other Reportables")

# API Token - shared via session state (initialized on Legacy page)
if 'api_token' not in st.session_state:
    st.session_state.api_token = "3CKjkFN6jIIHgSkIJH19i7VhK"

with st.expander("API Configuration", expanded=False):
    api_token = st.text_input(
        "API Token (optional):",
        value=st.session_state.api_token,
        type="password",
        help="Enter your CFTC API token for higher rate limits.",
        key="disagg_api_token_input"
    )
    st.session_state.api_token = api_token

st.markdown("---")

# Display the disaggregated overview table
display_disagg_dashboard(st.session_state.api_token)

st.markdown("---")

# --- Single Instrument Analysis ---

# Initialize session state
if 'disagg_data_fetched' not in st.session_state:
    st.session_state.disagg_data_fetched = False
if 'disagg_fetched_data' not in st.session_state:
    st.session_state.disagg_fetched_data = None
if 'disagg_fetched_instrument' not in st.session_state:
    st.session_state.disagg_fetched_instrument = None

# Chart Type Selection
st.header("Select Analysis Type")

disagg_chart_type = st.segmented_control(
    "Select chart type",
    ["Time Series", "Trader Participation"],
    selection_mode="single",
    default=None,
    key="disagg_chart_type"
)

st.markdown("---")

if disagg_chart_type:
    # Load instruments database
    instruments_db = load_disagg_instruments_database()
    if not instruments_db:
        st.stop()

    st.header("Select Instrument")

    search_method = st.radio(
        "Choose search method:",
        ["Extensive Search", "Search by Commodity Subgroup", "Search by Commodity Type", "Free Text Search"],
        horizontal=True,
        key="disagg_search_method"
    )

    selected_instrument = None

    if search_method == "Extensive Search":
        st.subheader("Browse by Exchange Hierarchy")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            exchanges = list(instruments_db['exchanges'].keys())
            selected_exchange = st.selectbox("Exchange:", exchanges, key="disagg_exchange")

        with col2:
            groups = list(instruments_db['exchanges'][selected_exchange].keys())
            selected_group = st.selectbox("Group:", groups, key="disagg_group")

        with col3:
            subgroups = list(instruments_db['exchanges'][selected_exchange][selected_group].keys())
            selected_subgroup = st.selectbox("Subgroup:", subgroups, key="disagg_subgroup")

        with col4:
            commodities = list(instruments_db['exchanges'][selected_exchange][selected_group][selected_subgroup].keys())
            selected_commodity = st.selectbox("Commodity:", commodities, key="disagg_commodity")

        instruments = instruments_db['exchanges'][selected_exchange][selected_group][selected_subgroup][selected_commodity]
        selected_instrument = st.selectbox("Select Instrument:", instruments, key="disagg_hierarchy_instrument")

        st.info(f"**{selected_exchange}** > **{selected_group}** > **{selected_subgroup}** > **{selected_commodity}**")

    elif search_method == "Search by Commodity Subgroup":
        st.subheader("Search by Commodity Subgroup")
        col1, col2 = st.columns(2)

        with col1:
            subgroups = sorted(list(instruments_db['commodity_subgroups'].keys()))
            selected_subgroup = st.selectbox("Select Commodity Subgroup:", subgroups, key="disagg_subgroup_select")

        with col2:
            instruments = instruments_db['commodity_subgroups'][selected_subgroup]
            selected_instrument = st.selectbox("Select Instrument:", sorted(instruments), key="disagg_subgroup_instrument")

        st.info(f"**{selected_subgroup}** - {len(instruments)} available instruments")

    elif search_method == "Search by Commodity Type":
        st.subheader("Search by Commodity Type")
        col1, col2 = st.columns(2)

        with col1:
            commodities = sorted(list(instruments_db['commodities'].keys()))
            selected_commodity_type = st.selectbox("Select Commodity:", commodities, key="disagg_commodity_type")

        with col2:
            instruments = instruments_db['commodities'][selected_commodity_type]
            selected_instrument = st.selectbox("Select Instrument:", sorted(instruments), key="disagg_commodity_instrument")

        st.info(f"**{selected_commodity_type}** - {len(instruments)} available instruments")

    else:  # Free Text Search
        st.subheader("Free Text Search")
        search_text = st.text_input(
            "Type instrument name or keyword:",
            placeholder="e.g., GOLD, CRUDE OIL, S&P 500, WHEAT...",
            key="disagg_search_text"
        )

        if search_text:
            all_instruments = instruments_db['all_instruments']
            filtered_instruments = [
                inst for inst in all_instruments
                if search_text.upper() in inst.upper()
            ]

            if filtered_instruments:
                selected_instrument = st.selectbox(
                    f"Select from {len(filtered_instruments)} matching instruments:",
                    sorted(filtered_instruments),
                    key="disagg_search_instrument"
                )
                st.success(f"Found {len(filtered_instruments)} matching instruments")
            else:
                st.warning(f"No instruments found matching '{search_text}'")
        else:
            st.info("Start typing to search through all available instruments")

    # Fetch Data Button
    if selected_instrument:
        st.markdown("---")
        st.subheader(f"Selected: {selected_instrument}")

        need_new_fetch = (not st.session_state.disagg_data_fetched or
                          st.session_state.disagg_fetched_instrument != selected_instrument)

        if st.button("Fetch Data", type="primary", use_container_width=False,
                     disabled=not need_new_fetch, key="disagg_fetch_btn"):
            with st.spinner(f"Fetching disaggregated data for {selected_instrument}..."):
                df = fetch_disagg_data_full(selected_instrument, api_token)

            if df is not None and not df.empty:
                st.session_state.disagg_fetched_data = df
                st.session_state.disagg_data_fetched = True
                st.session_state.disagg_fetched_instrument = selected_instrument
                st.success(f"Successfully fetched {len(df)} records")
            else:
                st.error("No data available for the selected instrument")
                st.session_state.disagg_data_fetched = False

    # Display data if available
    if st.session_state.disagg_data_fetched and st.session_state.disagg_fetched_data is not None:
        df = st.session_state.disagg_fetched_data
        st.markdown("---")

        if disagg_chart_type == "Time Series":
            display_disagg_time_series_chart(df, st.session_state.disagg_fetched_instrument)
        elif disagg_chart_type == "Trader Participation":
            st.info("Trader Participation analysis - coming soon")
else:
    st.info("Please select an analysis type from the options above to begin")
