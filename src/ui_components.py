"""
UI components and layouts for CFTC COT Dashboard
"""
import streamlit as st
import pandas as pd


def render_sidebar(instruments_db):
    """Render the sidebar with instrument selection"""
    st.sidebar.header("🎯 Instrument Selection")
    
    # Exchange selection
    exchanges = list(instruments_db['exchanges'].keys())
    selected_exchange = st.sidebar.selectbox(
        "Select Exchange:",
        exchanges,
        index=exchanges.index("NYMEX") if "NYMEX" in exchanges else 0
    )
    
    # Get commodity groups for selected exchange
    if selected_exchange:
        commodity_groups = list(instruments_db['exchanges'][selected_exchange].keys())
        selected_group = st.sidebar.selectbox(
            "Select Commodity Group:",
            commodity_groups,
            index=0
        )
        
        # Get subgroups
        if selected_group:
            subgroups = list(instruments_db['exchanges'][selected_exchange][selected_group].keys())
            selected_subgroup = st.sidebar.selectbox(
                "Select Subgroup:",
                subgroups,
                index=0
            )
            
            # Get commodities
            if selected_subgroup:
                commodities = list(
                    instruments_db['exchanges'][selected_exchange][selected_group][selected_subgroup].keys()
                )
                selected_commodity = st.sidebar.selectbox(
                    "Select Commodity:",
                    commodities,
                    index=0
                )
                
                # Get instruments
                if selected_commodity:
                    instruments = instruments_db['exchanges'][selected_exchange][selected_group][
                        selected_subgroup][selected_commodity]
                    
                    if len(instruments) > 1:
                        selected_instrument = st.sidebar.selectbox(
                            "Select Instrument:",
                            instruments,
                            index=0
                        )
                    else:
                        selected_instrument = instruments[0]
                        st.sidebar.info(f"📊 Instrument: {selected_instrument}")
                    
                    return selected_instrument
    
    return None


def render_key_metrics(df, col1, col2, col3, col4):
    """Render key metrics in columns"""
    latest_date = df['report_date_as_yyyy_mm_dd'].max()
    latest_data = df[df['report_date_as_yyyy_mm_dd'] == latest_date].iloc[0]
    
    # Calculate week-over-week changes
    week_ago = latest_date - pd.Timedelta(days=7)
    week_ago_data = df[df['report_date_as_yyyy_mm_dd'] <= week_ago].iloc[-1] if len(
        df[df['report_date_as_yyyy_mm_dd'] <= week_ago]) > 0 else None
    
    with col1:
        oi_value = latest_data['open_interest_all']
        oi_change = ((oi_value - week_ago_data['open_interest_all']) / week_ago_data[
            'open_interest_all'] * 100) if week_ago_data is not None else 0
        st.metric(
            "Open Interest",
            f"{oi_value:,.0f}",
            f"{oi_change:+.1f}%" if week_ago_data is not None else None
        )
    
    with col2:
        net_position = latest_data.get('net_noncomm_positions', 0)
        net_change = (net_position - week_ago_data.get('net_noncomm_positions', 0)) if week_ago_data is not None else 0
        st.metric(
            "Net Non-Comm Position",
            f"{net_position:,.0f}",
            f"{net_change:+,.0f}" if week_ago_data is not None else None
        )
    
    with col3:
        if 'traders_tot_all' in latest_data:
            traders = latest_data['traders_tot_all']
            traders_change = ((traders - week_ago_data['traders_tot_all']) / week_ago_data[
                'traders_tot_all'] * 100) if week_ago_data is not None else 0
            st.metric(
                "Total Traders",
                f"{traders:,.0f}",
                f"{traders_change:+.1f}%" if week_ago_data is not None else None
            )
        else:
            st.metric("Total Traders", "N/A")
    
    with col4:
        if 'conc_net_le_4_tdr_long_all' in latest_data:
            concentration = (latest_data['conc_net_le_4_tdr_long_all'] + 
                           latest_data.get('conc_net_le_4_tdr_short_all', 0)) / 2
            st.metric(
                "Top 4 Concentration",
                f"{concentration:.1f}%",
                delta_color="inverse"
            )
        else:
            st.metric("Top 4 Concentration", "N/A")


def render_data_table(df):
    """Render the raw data table"""
    st.subheader("📊 Raw Data (Last 10 Rows)")
    
    # Select columns to display
    display_columns = [
        'report_date_as_yyyy_mm_dd',
        'open_interest_all',
        'noncomm_positions_long_all',
        'noncomm_positions_short_all',
        'comm_positions_long_all',
        'comm_positions_short_all',
        'traders_tot_all'
    ]
    
    # Filter columns that exist
    available_columns = [col for col in display_columns if col in df.columns]
    
    # Format the dataframe
    df_display = df[available_columns].tail(10).copy()
    df_display['report_date_as_yyyy_mm_dd'] = df_display['report_date_as_yyyy_mm_dd'].dt.strftime('%Y-%m-%d')
    
    # Rename columns for better display
    column_names = {
        'report_date_as_yyyy_mm_dd': 'Date',
        'open_interest_all': 'Open Interest',
        'noncomm_positions_long_all': 'Non-Comm Long',
        'noncomm_positions_short_all': 'Non-Comm Short',
        'comm_positions_long_all': 'Comm Long',
        'comm_positions_short_all': 'Comm Short',
        'traders_tot_all': 'Total Traders'
    }
    
    df_display = df_display.rename(columns=column_names)
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset as CSV",
            data=csv,
            file_name=f"cot_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create Excel download
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='COT Data')
            excel_data = buffer.getvalue()
            
            st.download_button(
                label="📥 Download Full Dataset as Excel",
                data=excel_data,
                file_name=f"cot_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.info("Install openpyxl to enable Excel downloads")


def render_chart_selector():
    """Render chart type selector"""
    return st.selectbox(
        "Select Chart Type:",
        [
            "Time Series Analysis",
            "Seasonality Analysis",
            "Trader Participation Analysis",
            "Positioning & Concentration",
            "Percentile Analysis",
            "Cross-Asset Comparison"
        ],
        index=0
    )