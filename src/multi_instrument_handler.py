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

        # ===== NOTE: Some chart types moved to main dashboard =====
        # Cross-Asset, Market Matrix, WoW Changes, and Positioning Conc.
        # have been moved to the main dashboard page.
        # The code for these is preserved in git history (commit aaba921)
        # To restore: revert changes to this file and add back to main.py
        # ============================================================

        # UI elements outside the button to prevent reset
        if chart_type == "Cross-Asset":
            st.info("⬅️ This analysis has been moved to the main dashboard page.")

        elif chart_type == "Market Matrix":
            st.info("⬅️ This analysis has been moved to the main dashboard page.")

        elif chart_type == "WoW Changes":
            st.info("⬅️ This analysis has been moved to the main dashboard page.")

        elif chart_type == "Positioning Conc.":
            st.info("⬅️ This analysis has been moved to the main dashboard page.")

        elif chart_type == "Strength Matrix":
            st.info("⬅️ This analysis has been moved to the main dashboard page.")

        elif chart_type == "Participation":
            st.info("⬅️ This analysis has been moved to the main dashboard page.")

    else:
        st.warning("Please select at least one instrument for analysis")