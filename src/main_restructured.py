"""
Restructured main function for CFTC COT Dashboard
This will replace the existing main() function
"""

def main():
    st.title("📊 CFTC Commitments of Traders Dashboard")
    st.markdown("Interactive analysis of CFTC COT data")

    # Load instruments database
    instruments_db = load_instruments_database()
    if not instruments_db:
        st.stop()

    # API Token Configuration
    with st.expander("🔧 API Configuration", expanded=False):
        api_token = st.text_input(
            "API Token (optional):",
            type="password",
            help="Enter your CFTC API token for higher rate limits. Leave empty to use default limits."
        )

    st.markdown("---")

    # Chart Type Selection FIRST
    st.header("📈 Select Analysis Type")
    
    col_single, col_multi = st.columns(2)
    
    with col_single:
        st.markdown("### Single Instrument Analysis")
        single_chart_type = st.segmented_control(
            "",
            ["Time Series", "Seasonality", "Percentile", "Momentum", "Trader Participation"],
            selection_mode="single",
            default=None,
            key="single_chart_type"
        )
    
    with col_multi:
        st.markdown("### Multi-Instrument Analysis")
        multi_chart_type = st.segmented_control(
            "",
            ["Cross-Asset Comparison", "Market Structure Matrix"],
            selection_mode="single", 
            default=None,
            key="multi_chart_type"
        )
    
    # Clear the other selection when one is selected
    if single_chart_type and multi_chart_type:
        st.session_state.multi_chart_type = None
        multi_chart_type = None
    elif multi_chart_type and single_chart_type:
        st.session_state.single_chart_type = None
        single_chart_type = None
    
    st.markdown("---")
    
    # Determine which type of analysis was selected
    if single_chart_type:
        # Single Instrument Flow
        st.header("🎯 Select Instrument")
        
        # Instrument selection methods
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
            
            # Get instruments for selected commodity
            instruments = instruments_db['exchanges'][selected_exchange][selected_group][selected_subgroup][selected_commodity]
            
            if len(instruments) == 1:
                selected_instrument = instruments[0]
                st.success(f"✅ Selected: {selected_instrument}")
            else:
                selected_instrument = st.selectbox("📊 Select Instrument:", instruments)
        
        elif search_method == "Search by Commodity Subgroup":
            # Get all unique subgroups
            all_subgroups = set()
            for exchange in instruments_db['exchanges'].values():
                for group in exchange.values():
                    all_subgroups.update(group.keys())
            
            selected_subgroup = st.selectbox("Select Commodity Subgroup:", sorted(all_subgroups))
            
            # Get all instruments in this subgroup
            subgroup_instruments = []
            if selected_subgroup:
                for inst_list in instruments_db['commodity_subgroups'].get(selected_subgroup, []):
                    subgroup_instruments.append(inst_list)
                
                if subgroup_instruments:
                    selected_instrument = st.selectbox(
                        f"Select from {len(subgroup_instruments)} instruments in {selected_subgroup}:",
                        sorted(subgroup_instruments)
                    )
        
        elif search_method == "Search by Commodity Type":
            # Get all commodities
            all_commodities = list(instruments_db['commodities'].keys())
            selected_commodity = st.selectbox("Select Commodity Type:", sorted(all_commodities))
            
            if selected_commodity:
                commodity_instruments = instruments_db['commodities'][selected_commodity]
                selected_instrument = st.selectbox(
                    f"Select from {len(commodity_instruments)} instruments:",
                    sorted(commodity_instruments)
                )
        
        elif search_method == "Free Text Search":
            search_text = st.text_input("🔍 Search instruments:", placeholder="Type instrument name...")
            
            if search_text:
                # Filter instruments
                filtered_instruments = [
                    inst for inst in instruments_db['all_instruments']
                    if search_text.upper() in inst.upper()
                ]
                
                if filtered_instruments:
                    selected_instrument = st.selectbox(
                        f"📊 Select from {len(filtered_instruments)} matching instruments:",
                        sorted(filtered_instruments)
                    )
                else:
                    st.warning(f"⚠️ No instruments found matching '{search_text}'")
        
        # Fetch data for single instrument
        if selected_instrument:
            st.markdown("---")
            st.subheader(f"Selected: {selected_instrument}")
            
            if st.button("🚀 Fetch Data", type="primary"):
                with st.spinner(f"Fetching data for {selected_instrument}..."):
                    df = fetch_cftc_data(selected_instrument, api_token)
                
                if df is not None and not df.empty:
                    st.session_state.fetched_data = df
                    st.session_state.fetched_instrument = selected_instrument
                    st.success(f"✅ Successfully fetched {len(df)} records")
                    
                    # Display the selected chart type
                    display_single_instrument_chart(single_chart_type, df, selected_instrument)
                else:
                    st.error("❌ No data available for the selected instrument")
    
    elif multi_chart_type:
        # Multi-Instrument Flow
        st.header("🎯 Select Multiple Instruments")
        
        # Get all instruments
        instruments_list = []
        if instruments_db and 'exchanges' in instruments_db:
            for exchange, groups in instruments_db['exchanges'].items():
                for group, subgroups in groups.items():
                    for subgroup, instruments in subgroups.items():
                        instruments_list.extend(instruments)
        
        # Filter to major exchanges
        futures_only = [inst for inst in instruments_list if ' - CHICAGO' in inst or ' - NEW YORK' in inst or ' - ICE' in inst]
        
        selected_instruments = st.multiselect(
            "Choose instruments for comparison:",
            options=sorted(futures_only),
            max_selections=15,
            help="Select up to 15 instruments for comparison"
        )
        
        if selected_instruments and len(selected_instruments) >= 2:
            if st.button("🚀 Fetch Data for All Instruments", type="primary"):
                display_multi_instrument_chart(multi_chart_type, selected_instruments, api_token, instruments_db)
        elif selected_instruments:
            st.warning("Please select at least 2 instruments for comparison")
    
    else:
        # No chart type selected yet
        st.info("👆 Please select an analysis type from the options above to begin")


def display_single_instrument_chart(chart_type, df, instrument_name):
    """Display the appropriate single-instrument chart based on selection"""
    st.markdown("---")
    
    if chart_type == "Time Series":
        # Time series analysis code
        st.subheader("📈 Time Series Analysis")
        # Add time series specific code here
        
    elif chart_type == "Seasonality":
        # Seasonality analysis code  
        st.subheader("📅 Seasonality Analysis")
        # Add seasonality specific code here
        
    elif chart_type == "Percentile":
        # Percentile analysis code
        st.subheader("📊 Percentile Analysis") 
        # Add percentile specific code here
        
    elif chart_type == "Momentum":
        # Momentum dashboard code
        st.subheader("🚀 Momentum Dashboard")
        # Add momentum specific code here
        
    elif chart_type == "Trader Participation":
        # Trader participation code
        st.subheader("👥 Trader Participation Analysis")
        # Add trader participation specific code here


def display_multi_instrument_chart(chart_type, selected_instruments, api_token, instruments_db):
    """Display the appropriate multi-instrument chart based on selection"""
    st.markdown("---")
    
    if chart_type == "Cross-Asset Comparison":
        # Cross-asset comparison code
        st.subheader("🔄 Cross-Asset Comparison")
        # Add cross-asset specific code here
        
    elif chart_type == "Market Structure Matrix":
        # Market structure matrix code
        st.subheader("🎯 Market Structure Matrix")
        # Add market structure specific code here