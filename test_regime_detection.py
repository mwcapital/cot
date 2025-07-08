import sys
sys.path.append('src')

import pandas as pd
from charts.regime_detection import create_regime_detection_dashboard

# Create dummy data to test
dates = pd.date_range('2020-01-01', '2024-01-01', freq='W')
df = pd.DataFrame({
    'report_date_as_yyyy_mm_dd': dates,
    'comm_positions_long_all': [100000 + i*1000 for i in range(len(dates))],
    'comm_positions_short_all': [80000 + i*800 for i in range(len(dates))],
    'noncomm_positions_long_all': [120000 + i*1200 for i in range(len(dates))],
    'noncomm_positions_short_all': [90000 + i*900 for i in range(len(dates))],
    'open_interest_all': [400000 + i*4000 for i in range(len(dates))],
    'conc_gross_le_4_tdr_long': [30 + (i%20) for i in range(len(dates))],
    'conc_gross_le_4_tdr_short': [35 + (i%15) for i in range(len(dates))],
    'conc_net_le_4_tdr_long_all': [25 + (i%25) for i in range(len(dates))],
    'conc_net_le_4_tdr_short_all': [28 + (i%22) for i in range(len(dates))],
    'traders_tot_all': [500 + i%50 for i in range(len(dates))]
})

print("Creating regime detection dashboard...")
fig, df_regime, latest = create_regime_detection_dashboard(df, "TEST INSTRUMENT")

if fig:
    print("Dashboard created successfully!")
    print(f"Title: {fig.layout.title.text}")
    print(f"Height: {fig.layout.height}")
    print(f"Number of subplots: {len(fig._grid_ref)}")
    
    if latest is not None:
        print("\nLatest regime metrics:")
        print(f"  Regime: {latest.get('regime', 'N/A')}")
        print(f"  Extremity Score: {latest.get('regime_extremity', 0):.1f}")
        print(f"  Long Concentration Percentile: {latest.get('long_conc_pct', 0):.1f}")
else:
    print("Failed to create dashboard")