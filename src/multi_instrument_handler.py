"""
Multi-instrument flow handler for the CFTC COT Dashboard
"""
import streamlit as st
import pandas as pd
from data_fetcher import fetch_cftc_data
from charts.cross_asset_analysis import (
    create_cross_asset_analysis,
    create_cross_asset_wow_changes,
    create_positioning_concentration_charts,
    create_cross_asset_participation_comparison,
    create_relative_strength_matrix,
    create_market_structure_matrix
)


def handle_multi_instrument_flow(chart_type, instruments_db, api_token):
    """Handle multi-instrument selection and analysis"""
    st.header("🎯 Select Multiple Instruments")
    
    # Initialize session state for multi-instrument selection
    if 'multi_selected_instruments' not in st.session_state:
        st.session_state.multi_selected_instruments = []
    
    # Search method selection
    search_method = st.radio(
        "Choose search method:",
        ["Search by Commodity Subgroup", "Search by Commodity Type", "Free Text Search"],
        horizontal=True
    )
    
    # Clear selections button
    if st.button("🗑️ Clear All Selections", type="secondary"):
        st.session_state.multi_selected_instruments = []
        st.rerun()
    
    selected_instruments = []
    
    if search_method == "Search by Commodity Subgroup":
        st.subheader("📁 Search by Commodity Subgroup")
        
        # Get all subgroups
        if instruments_db and 'commodity_subgroups' in instruments_db:
            subgroups = sorted(list(instruments_db['commodity_subgroups'].keys()))
            
            # Multi-select for subgroups
            selected_subgroups = st.multiselect(
                "Select commodity subgroups:",
                options=subgroups,
                help="Choose one or more subgroups to see their instruments"
            )
            
            if selected_subgroups:
                # Collect all instruments from selected subgroups
                filtered_instruments = []
                for subgroup in selected_subgroups:
                    filtered_instruments.extend(instruments_db['commodity_subgroups'][subgroup])
                
                filtered_instruments = sorted(set(filtered_instruments))
                st.success(f"✅ Found {len(filtered_instruments)} instruments in selected subgroups")
                
                # Combine with previously selected instruments
                combined_options = sorted(set(st.session_state.multi_selected_instruments + filtered_instruments))
                
                # Multi-select for instruments
                selected_instruments = st.multiselect(
                    f"📊 Select instruments (showing {len(filtered_instruments)} new matches):",
                    combined_options,
                    default=st.session_state.multi_selected_instruments,
                    max_selections=15,
                    help="Select up to 15 instruments for comparison",
                    key="subgroup_multiselect"
                )
                
                st.session_state.multi_selected_instruments = selected_instruments
            else:
                st.info("Select one or more subgroups to see available instruments")
                
    elif search_method == "Search by Commodity Type":
        st.subheader("🔸 Search by Commodity Type")
        
        # Get all commodities
        if instruments_db and 'commodities' in instruments_db:
            commodities = sorted(list(instruments_db['commodities'].keys()))
            
            # Multi-select for commodities
            selected_commodities = st.multiselect(
                "Select commodity types:",
                options=commodities,
                help="Choose one or more commodity types"
            )
            
            if selected_commodities:
                # Collect all instruments from selected commodities
                filtered_instruments = []
                for commodity in selected_commodities:
                    filtered_instruments.extend(instruments_db['commodities'][commodity])
                
                filtered_instruments = sorted(set(filtered_instruments))
                st.success(f"✅ Found {len(filtered_instruments)} instruments for selected commodities")
                
                # Combine with previously selected instruments
                combined_options = sorted(set(st.session_state.multi_selected_instruments + filtered_instruments))
                
                # Multi-select for instruments
                selected_instruments = st.multiselect(
                    f"📊 Select instruments (showing {len(filtered_instruments)} new matches):",
                    combined_options,
                    default=st.session_state.multi_selected_instruments,
                    max_selections=15,
                    help="Select up to 15 instruments for comparison",
                    key="commodity_multiselect"
                )
                
                st.session_state.multi_selected_instruments = selected_instruments
            else:
                st.info("Select one or more commodity types to see available instruments")
                
    else:  # Free Text Search
        st.subheader("🔍 Free Text Search")
        
        # Text input for search
        search_text = st.text_input(
            "Type keywords (comma-separated for multiple):",
            placeholder="e.g., gold, silver, crude oil",
            help="Enter instrument names or keywords. Use commas to search for multiple terms."
        )
        
        if search_text:
            # Get all instruments
            if instruments_db and 'all_instruments' in instruments_db:
                all_instruments = instruments_db['all_instruments']
            else:
                # Fallback: build all_instruments list
                all_instruments = []
                if instruments_db and 'exchanges' in instruments_db:
                    for exchange, groups in instruments_db['exchanges'].items():
                        for group, subgroups in groups.items():
                            for subgroup, commodities in subgroups.items():
                                for commodity, instruments in commodities.items():
                                    all_instruments.extend(instruments)
                all_instruments = sorted(list(set(all_instruments)))
            
            # Parse search terms
            search_terms = [term.strip().upper() for term in search_text.split(',') if term.strip()]
            
            # Filter instruments
            filtered_instruments = []
            for instrument in all_instruments:
                instrument_upper = instrument.upper()
                if any(term in instrument_upper for term in search_terms):
                    filtered_instruments.append(instrument)
            
            if filtered_instruments:
                st.success(f"✅ Found {len(filtered_instruments)} matching instruments")
                
                # Combine with previously selected instruments
                combined_options = sorted(set(st.session_state.multi_selected_instruments + filtered_instruments))
                
                # Multi-select for instruments
                selected_instruments = st.multiselect(
                    f"📊 Select instruments (showing {len(filtered_instruments)} new matches):",
                    combined_options,
                    default=st.session_state.multi_selected_instruments,
                    max_selections=15,
                    help="Select up to 15 instruments for comparison",
                    key="free_text_multiselect"
                )
                
                st.session_state.multi_selected_instruments = selected_instruments
            else:
                st.warning("No instruments found matching your search terms")
        else:
            # Show previously selected instruments if any
            if st.session_state.multi_selected_instruments:
                selected_instruments = st.multiselect(
                    "📊 Previously selected instruments:",
                    st.session_state.multi_selected_instruments,
                    default=st.session_state.multi_selected_instruments,
                    max_selections=15,
                    key="previous_multiselect"
                )
                st.session_state.multi_selected_instruments = selected_instruments
            else:
                st.info("Enter search terms to find instruments")
    
    # Show selected count
    if st.session_state.multi_selected_instruments:
        st.success(f"✅ {len(st.session_state.multi_selected_instruments)} instruments selected")
    
    selected_instruments = st.session_state.multi_selected_instruments
    
    if selected_instruments:
        # Initialize session state for analysis settings
        if 'analysis_data_fetched' not in st.session_state:
            st.session_state.analysis_data_fetched = False
        
        # UI elements outside the button to prevent reset
        if chart_type == "Cross-Asset":
            # Fetch data button
            if st.button("🚀 Fetch Data for All Instruments", type="primary", key="fetch_cross_asset"):
                st.session_state.analysis_data_fetched = True
            
            # Show analysis UI and chart if data has been fetched
            if st.session_state.analysis_data_fetched:
                st.markdown("---")
                st.subheader("🔄 Cross-Asset Comparison")
                
                # Trader category selection - as selectbox like in the screenshot
                trader_category = st.selectbox(
                    "Select trader category:",
                    ["Non-Commercial", "Commercial", "Non-Reportable"],
                    index=0,
                    key="cross_asset_trader_category"
                )
                
                # Lookback period for Z-score calculation - as selectbox
                lookback_period = st.selectbox(
                    "Lookback period for Z-score calculation:",
                    ["1 Year", "2 Years", "3 Years", "5 Years", "10 Years"],
                    index=1,
                    key="cross_asset_lookback"
                )
                
                # Map lookback period to start date
                lookback_map = {
                    "1 Year": pd.Timestamp.now() - pd.DateOffset(years=1),
                    "2 Years": pd.Timestamp.now() - pd.DateOffset(years=2), 
                    "3 Years": pd.Timestamp.now() - pd.DateOffset(years=3),
                    "5 Years": pd.Timestamp.now() - pd.DateOffset(years=5),
                    "10 Years": pd.Timestamp.now() - pd.DateOffset(years=10)
                }
                
                lookback_start = lookback_map[lookback_period]
                
                with st.spinner("Fetching data for selected instruments..."):
                    # Create cross-asset z-score analysis
                    fig = create_cross_asset_analysis(
                        selected_instruments,
                        trader_category,
                        api_token,
                        lookback_start,
                        True,  # Always show week ago values
                        instruments_db
                    )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        if st.button("💾 Download Cross-Asset Chart", key="download_cross"):
                            html_string = fig.to_html(include_plotlyjs='cdn')
                            st.download_button(
                                label="Download Chart",
                                data=html_string,
                                file_name=f"cftc_cross_asset_{trader_category}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                                mime="text/html"
                            )
        
        elif chart_type == "Market Matrix":
            # Initialize session state for Market Matrix
            if 'market_matrix_data' not in st.session_state:
                st.session_state.market_matrix_data = {}
                
            # For Market Matrix, show the concentration selector before the button
            if st.button("🚀 Fetch Data for All Instruments", type="primary"):
                # Fetch data for all instruments
                all_instruments_data = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, instrument in enumerate(selected_instruments):
                    status_text.text(f"Fetching data for {instrument}...")
                    progress_bar.progress((idx + 1) / len(selected_instruments))
                    
                    df = fetch_cftc_data(instrument, api_token)
                    if df is not None and not df.empty:
                        all_instruments_data[instrument] = df
                
                progress_bar.empty()
                status_text.empty()
                
                if all_instruments_data:
                    st.session_state.market_matrix_data = all_instruments_data
                    st.success("✅ Data fetched successfully!")
                else:
                    st.error("Failed to fetch data for the selected instruments")
            
            # Show the UI and chart if data exists
            if st.session_state.market_matrix_data:
                st.markdown("---")
                st.subheader("🎯 Market Structure Matrix")
                st.info("Visualizes market structure using 5-year percentiles for cross-market comparability. Each axis shows where instruments rank relative to their own history.")
                
                # Add concentration metric selector - will persist after data is fetched
                concentration_options = {
                    'conc_gross_le_4_tdr_long': 'Gross Top 4 Traders Long',
                    'conc_gross_le_4_tdr_short': 'Gross Top 4 Traders Short',
                    'conc_gross_le_8_tdr_long': 'Gross Top 8 Traders Long',
                    'conc_gross_le_8_tdr_short': 'Gross Top 8 Traders Short',
                    'conc_net_le_4_tdr_long_all': 'Net Top 4 Traders Long',
                    'conc_net_le_4_tdr_short_all': 'Net Top 4 Traders Short',
                    'conc_net_le_8_tdr_long_all': 'Net Top 8 Traders Long',
                    'conc_net_le_8_tdr_short_all': 'Net Top 8 Traders Short'
                }
                
                concentration_metric = st.selectbox(
                    "Select concentration metric for Y-axis:",
                    options=list(concentration_options.keys()),
                    format_func=lambda x: concentration_options[x],
                    index=0,
                    help="Choose which concentration metric to compare across markets",
                    key="market_matrix_concentration_metric"
                )
                
                # Create and show the chart
                with st.spinner("Calculating 5-year percentiles..."):
                    fig = create_market_structure_matrix(
                        st.session_state.market_matrix_data, 
                        selected_instruments, 
                        concentration_metric
                    )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    with st.expander("📊 Understanding the Percentile-Based Matrix", expanded=False):
                        st.markdown("""
                        **Why Percentiles?**
                        - Raw trader counts and concentration ratios vary significantly across markets
                        - Percentiles allow fair comparison by showing where each market stands relative to its own 5-year history
                        - 50th percentile (median) divides the quadrants
                        
                        **How to Read:**
                        - **X-axis**: Percentile of 'traders_tot_all' variable from API (0% = historically low, 100% = historically high)
                        - **Y-axis**: Concentration percentile based on selected metric
                        - **Best Quadrant** (Green): Above median traders + Below median concentration
                        - **Worst Quadrant** (Red): Below median traders + Above median concentration
                        
                        **Hover for Details:**
                        - See both percentile rankings and actual values
                        - Compare how different markets rank on the same scale
                        """)
        
        elif chart_type == "WoW Changes":
            # Initialize session state for WoW Changes
            if 'wow_changes_data_fetched' not in st.session_state:
                st.session_state.wow_changes_data_fetched = False
            
            # Fetch data button
            if st.button("🚀 Fetch Data for All Instruments", type="primary", key="fetch_wow_changes"):
                st.session_state.wow_changes_data_fetched = True
            
            # Show analysis UI and chart if data has been fetched
            if st.session_state.wow_changes_data_fetched:
                st.markdown("---")
                st.subheader("📊 Week-over-Week Changes")
                
                # Trader category selection - appears only after data is fetched
                trader_category = st.selectbox(
                    "Select trader category:",
                    ["Non-Commercial", "Commercial", "Non-Reportable"],
                    index=0,
                    key="wow_changes_trader_category"
                )
                
                with st.spinner("Calculating week-over-week changes..."):
                    # Create WoW changes chart
                    fig = create_cross_asset_wow_changes(
                        selected_instruments,
                        trader_category,
                        api_token,
                        instruments_db
                    )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        if st.button("💾 Download WoW Changes Chart", key="download_wow"):
                            html_string = fig.to_html(include_plotlyjs='cdn')
                            st.download_button(
                                label="Download Chart",
                                data=html_string,
                                file_name=f"cftc_wow_changes_{trader_category}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                                mime="text/html"
                            )
        
        elif chart_type == "Positioning Conc.":
            # Initialize session state for Positioning Concentration
            if 'positioning_conc_data_fetched' not in st.session_state:
                st.session_state.positioning_conc_data_fetched = False
            
            # Fetch data button
            if st.button("🚀 Fetch Data for All Instruments", type="primary", key="fetch_positioning_conc"):
                st.session_state.positioning_conc_data_fetched = True
            
            # Show analysis UI and charts if data has been fetched
            if st.session_state.positioning_conc_data_fetched:
                st.markdown("---")
                st.subheader("📈 Positioning Concentration Analysis")
                st.info("Compares positioning as % of open interest across instruments")
                
                # Trader category selection - appears only after data is fetched
                trader_category = st.selectbox(
                    "Select trader category:",
                    ["Non-Commercial", "Commercial", "Non-Reportable"],
                    index=0,
                    key="positioning_conc_trader_category"
                )
                
                with st.spinner("Calculating positioning concentration..."):
                    # Create positioning concentration charts
                    fig_ts, fig_bar, fig_avg = create_positioning_concentration_charts(
                        selected_instruments,
                        trader_category,
                        api_token,
                        instruments_db
                    )
                
                if fig_ts and fig_bar and fig_avg:
                    st.plotly_chart(fig_ts, use_container_width=True)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.plotly_chart(fig_avg, use_container_width=True)
                    
                    # Download buttons
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        if st.button("💾 Download Time Series Chart", key="download_positioning_ts"):
                            html_string = fig_ts.to_html(include_plotlyjs='cdn')
                            st.download_button(
                                label="Download Time Series",
                                data=html_string,
                                file_name=f"cftc_positioning_timeseries_{trader_category}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                                mime="text/html"
                            )
                    with col2:
                        if st.button("💾 Download Bar Chart", key="download_positioning_bar"):
                            html_string = fig_bar.to_html(include_plotlyjs='cdn')
                            st.download_button(
                                label="Download Bar Chart",
                                data=html_string,
                                file_name=f"cftc_positioning_current_{trader_category}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                                mime="text/html"
                            )
                    with col3:
                        if st.button("💾 Download Average Positions Chart", key="download_positioning_avg"):
                            html_string = fig_avg.to_html(include_plotlyjs='cdn')
                            st.download_button(
                                label="Download Average Positions",
                                data=html_string,
                                file_name=f"cftc_positioning_average_{trader_category}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                                mime="text/html"
                            )
        
        elif chart_type == "Participation":
            # Initialize session state for Participation
            if 'participation_data_fetched' not in st.session_state:
                st.session_state.participation_data_fetched = False
            
            # Fetch data button
            if st.button("🚀 Fetch Data for All Instruments", type="primary", key="fetch_participation"):
                st.session_state.participation_data_fetched = True
            
            # Show analysis UI and chart if data has been fetched
            if st.session_state.participation_data_fetched:
                st.markdown("---")
                st.subheader("👥 Trader Participation Comparison")
                st.info("Analyzes trader count trends, year-over-year changes, average positions per trader, and participation scores across instruments")
                
                with st.spinner("Calculating participation metrics..."):
                    # Create participation comparison
                    fig = create_cross_asset_participation_comparison(
                        selected_instruments,
                        api_token,
                        instruments_db
                    )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        if st.button("💾 Download Participation Chart", key="download_participation"):
                            html_string = fig.to_html(include_plotlyjs='cdn')
                            st.download_button(
                                label="Download Chart",
                                data=html_string,
                                file_name=f"cftc_participation_comparison_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                                mime="text/html"
                            )
        
        elif chart_type == "Strength Matrix":
            # Initialize session state for Strength Matrix
            if 'strength_matrix_data_fetched' not in st.session_state:
                st.session_state.strength_matrix_data_fetched = False
            
            # Fetch data button
            if st.button("🚀 Fetch Data for All Instruments", type="primary", key="fetch_strength_matrix"):
                st.session_state.strength_matrix_data_fetched = True
            
            # Show analysis UI and chart if data has been fetched
            if st.session_state.strength_matrix_data_fetched:
                st.markdown("---")
                st.subheader("💪 Relative Strength Matrix")
                st.info("Heatmap showing positioning correlations between instruments over the selected time period")
                
                # Time period selector - appears only after data is fetched
                time_period = st.selectbox(
                    "Select time period:",
                    ["6 Months", "1 Year", "2 Years", "5 Years", "10 Years"],
                    index=1,
                    key="strength_matrix_time_period"
                )
                
                with st.spinner("Calculating positioning correlations..."):
                    # Create relative strength matrix
                    fig = create_relative_strength_matrix(
                        selected_instruments,
                        api_token,
                        time_period,
                        instruments_db
                    )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explainer
                    with st.expander("📊 Understanding the Strength Matrix", expanded=False):
                        st.markdown("""
                        **What This Matrix Shows:**
                        - Correlation between Non-Commercial net positioning (long - short) across different instruments
                        - Values range from -1 to +1, where:
                          - **+1** (dark blue): Perfect positive correlation - instruments move together
                          - **0** (white): No correlation - instruments move independently
                          - **-1** (dark red): Perfect negative correlation - instruments move opposite
                        
                        **How to Use It:**
                        - **Portfolio Diversification**: Look for instruments with low or negative correlations
                        - **Risk Management**: High correlations mean similar market exposure
                        - **Trading Opportunities**: Divergence from typical correlations may signal opportunities
                        - **Market Regime**: Changing correlations can indicate shifts in market dynamics
                        
                        **Example Interpretations:**
                        - If Gold and Silver show +0.8: They tend to move in the same direction
                        - If Oil and Bonds show -0.5: They often move in opposite directions
                        - If Wheat and Gold show 0.1: They have little relationship
                        
                        **Time Period Impact:**
                        - Shorter periods (6M-1Y): Capture recent market dynamics and short-term relationships
                        - Medium periods (2Y): Balance between recent trends and historical patterns
                        - Longer periods (5Y-10Y): Show stable, long-term relationships and structural correlations
                        """)
                    
                    # Download button
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        if st.button("💾 Download Strength Matrix", key="download_strength"):
                            html_string = fig.to_html(include_plotlyjs='cdn')
                            st.download_button(
                                label="Download Chart",
                                data=html_string,
                                file_name=f"cftc_strength_matrix_{time_period.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                                mime="text/html"
                            )
                        
                        
                        
    else:
        st.warning("Please select at least one instrument for analysis")