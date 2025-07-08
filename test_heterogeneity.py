import sys
sys.path.append('src')

import pandas as pd
from charts.trader_participation_analysis import create_heterogeneity_index

# Create dummy data to test
dates = pd.date_range('2020-01-01', '2024-01-01', freq='W')
df = pd.DataFrame({
    'report_date_as_yyyy_mm_dd': dates,
    'comm_positions_long_all': [100000 + i*1000 for i in range(len(dates))],
    'comm_positions_short_all': [80000 + i*800 for i in range(len(dates))],
    'noncomm_positions_long_all': [120000 + i*1200 for i in range(len(dates))],
    'noncomm_positions_short_all': [90000 + i*900 for i in range(len(dates))],
    'open_interest_all': [400000 + i*4000 for i in range(len(dates))],
    'traders_comm_long_all': [50 + i%10 for i in range(len(dates))],
    'traders_comm_short_all': [45 + i%8 for i in range(len(dates))],
    'traders_noncomm_long_all': [100 + i%15 for i in range(len(dates))],
    'traders_noncomm_short_all': [85 + i%12 for i in range(len(dates))]
})

print("Creating heterogeneity index chart...")
fig = create_heterogeneity_index(df, "TEST INSTRUMENT")

if fig:
    print("Chart created successfully!")
    print(f"Title: {fig.layout.title.text}")
    print(f"Height: {fig.layout.height}")
    print(f"Right margin: {fig.layout.margin.r}")
    print(f"Number of traces: {len(fig.data)}")
    print(f"Number of annotations: {len(fig.layout.annotations)}")
else:
    print("Failed to create chart")