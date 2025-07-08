#!/usr/bin/env python3
"""Test Market Microstructure Analysis implementation"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('src')

from charts.market_microstructure import create_market_microstructure_analysis

# Create test data
dates = pd.date_range(end=datetime.now(), periods=52, freq='W')
test_data = []

for i, date in enumerate(dates):
    # Create realistic test data
    oi = 100000 + np.random.normal(0, 5000)
    
    # Concentration percentages (of total OI)
    conc_4_long = 25 + np.random.normal(0, 3)
    conc_8_long = 35 + np.random.normal(0, 3)
    conc_4_short = 23 + np.random.normal(0, 3)
    conc_8_short = 33 + np.random.normal(0, 3)
    
    # Positions
    comm_long = oi * 0.3
    comm_short = oi * 0.32
    noncomm_long = oi * 0.25
    noncomm_short = oi * 0.23
    
    # Traders
    comm_traders_long = 150 + np.random.randint(-20, 20)
    comm_traders_short = 160 + np.random.randint(-20, 20)
    noncomm_traders_long = 80 + np.random.randint(-10, 10)
    noncomm_traders_short = 75 + np.random.randint(-10, 10)
    
    test_data.append({
        'report_date_as_yyyy_mm_dd': date,
        'open_interest_all': oi,
        'conc_gross_le_4_tdr_long': conc_4_long,
        'conc_gross_le_8_tdr_long': conc_8_long,
        'conc_gross_le_4_tdr_short': conc_4_short,
        'conc_gross_le_8_tdr_short': conc_8_short,
        'comm_positions_long_all': comm_long,
        'comm_positions_short_all': comm_short,
        'noncomm_positions_long_all': noncomm_long,
        'noncomm_positions_short_all': noncomm_short,
        'traders_comm_long_all': comm_traders_long,
        'traders_comm_short_all': comm_traders_short,
        'traders_noncomm_long_all': noncomm_traders_long,
        'traders_noncomm_short_all': noncomm_traders_short,
        'traders_tot_all': comm_traders_long + comm_traders_short + noncomm_traders_long + noncomm_traders_short,
        'tot_rept_positions_long_all': comm_long + noncomm_long,
        'tot_rept_positions_short': comm_short + noncomm_short
    })

df = pd.DataFrame(test_data)

print("Test data created successfully")
print(f"Data shape: {df.shape}")
print(f"Date range: {df['report_date_as_yyyy_mm_dd'].min()} to {df['report_date_as_yyyy_mm_dd'].max()}")

# Test the function
try:
    import streamlit as st
    # Mock streamlit components for testing
    st.set_page_config = lambda **kwargs: None
    
    # Run the analysis
    print("\nTesting Market Microstructure Analysis...")
    result = create_market_microstructure_analysis(df, "Test Instrument")
    
    if result is not None:
        print("✓ Market Microstructure Analysis executed successfully")
        print(f"✓ Returned figure object: {type(result)}")
    else:
        print("✗ Market Microstructure Analysis returned None")
        
except Exception as e:
    print(f"✗ Error in Market Microstructure Analysis: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nTest complete")