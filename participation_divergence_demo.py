"""
Demonstration of Participation Divergence Calculation
Using CFTC COT Data with Step-by-Step Calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Create sample data structure similar to CFTC COT data
# This represents actual trader counts from a COT report
sample_data = {
    'report_date': ['2024-12-17', '2024-12-10', '2024-12-03', '2024-11-26', '2024-11-19'],
    'traders_comm_long_all': [157, 155, 153, 151, 149],  # Commercial traders with long positions
    'traders_comm_short_all': [142, 140, 138, 136, 134], # Commercial traders with short positions
    'traders_noncomm_long_all': [89, 87, 85, 83, 81],    # Non-commercial traders with long positions
    'traders_noncomm_short_all': [76, 74, 72, 70, 68],   # Non-commercial traders with short positions
    'traders_tot_all': [464, 456, 448, 440, 432],        # Total number of traders
    'open_interest_all': [285000, 280000, 275000, 270000, 265000]  # Total open interest
}

# Convert to DataFrame
df = pd.DataFrame(sample_data)
df['report_date'] = pd.to_datetime(df['report_date'])
df = df.sort_values('report_date')

print("=" * 80)
print("PARTICIPATION DIVERGENCE CALCULATION DEMONSTRATION")
print("=" * 80)
print(f"\nUsing data from the latest COT report date: {df['report_date'].iloc[-1].strftime('%Y-%m-%d')}")
print("=" * 80)

# STEP 1: Calculate unique traders for each category
# Note: Some traders may have both long and short positions, so we take the maximum
print("\nSTEP 1: Calculate Unique Traders per Category")
print("-" * 50)

# Get latest values
latest_idx = -1
comm_long = df['traders_comm_long_all'].iloc[latest_idx]
comm_short = df['traders_comm_short_all'].iloc[latest_idx]
noncomm_long = df['traders_noncomm_long_all'].iloc[latest_idx]
noncomm_short = df['traders_noncomm_short_all'].iloc[latest_idx]
total_traders = df['traders_tot_all'].iloc[latest_idx]

# Calculate unique traders (taking max of long/short as approximation)
df['traders_comm_unique'] = df[['traders_comm_long_all', 'traders_comm_short_all']].max(axis=1)
df['traders_noncomm_unique'] = df[['traders_noncomm_long_all', 'traders_noncomm_short_all']].max(axis=1)

comm_unique = df['traders_comm_unique'].iloc[latest_idx]
noncomm_unique = df['traders_noncomm_unique'].iloc[latest_idx]

print(f"Commercial Traders:")
print(f"  - Long positions: {comm_long}")
print(f"  - Short positions: {comm_short}")
print(f"  - Unique traders (max): {comm_unique}")
print(f"\nNon-Commercial Traders:")
print(f"  - Long positions: {noncomm_long}")
print(f"  - Short positions: {noncomm_short}")
print(f"  - Unique traders (max): {noncomm_unique}")
print(f"\nTotal Traders: {total_traders}")

# STEP 2: Calculate trader shares (percentages)
print("\n\nSTEP 2: Calculate Trader Shares (Percentages)")
print("-" * 50)

df['comm_share'] = (df['traders_comm_unique'] / df['traders_tot_all']) * 100
df['noncomm_share'] = (df['traders_noncomm_unique'] / df['traders_tot_all']) * 100

comm_share = df['comm_share'].iloc[latest_idx]
noncomm_share = df['noncomm_share'].iloc[latest_idx]

print(f"Commercial share: {comm_unique} / {total_traders} × 100 = {comm_share:.2f}%")
print(f"Non-Commercial share: {noncomm_unique} / {total_traders} × 100 = {noncomm_share:.2f}%")
print(f"Total accounted for: {comm_share + noncomm_share:.2f}%")

# STEP 3: Calculate historical statistics (52-week period)
print("\n\nSTEP 3: Calculate Historical Statistics (52-week period)")
print("-" * 50)

# For demonstration, we'll calculate statistics over our sample period
# In real application, this would be over 52 weeks
lookback_periods = len(df)

comm_mean = df['comm_share'].mean()
comm_std = df['comm_share'].std()
noncomm_mean = df['noncomm_share'].mean()
noncomm_std = df['noncomm_share'].std()

print(f"Commercial Share Statistics:")
print(f"  - Mean: {comm_mean:.2f}%")
print(f"  - Std Dev: {comm_std:.2f}%")
print(f"  - Current: {comm_share:.2f}%")

print(f"\nNon-Commercial Share Statistics:")
print(f"  - Mean: {noncomm_mean:.2f}%")
print(f"  - Std Dev: {noncomm_std:.2f}%")
print(f"  - Current: {noncomm_share:.2f}%")

# STEP 4: Calculate Z-scores
print("\n\nSTEP 4: Calculate Z-Scores")
print("-" * 50)

comm_zscore = (comm_share - comm_mean) / comm_std if comm_std > 0 else 0
noncomm_zscore = (noncomm_share - noncomm_mean) / noncomm_std if noncomm_std > 0 else 0

print(f"Commercial Z-score: ({comm_share:.2f} - {comm_mean:.2f}) / {comm_std:.2f} = {comm_zscore:.3f}")
print(f"Non-Commercial Z-score: ({noncomm_share:.2f} - {noncomm_mean:.2f}) / {noncomm_std:.2f} = {noncomm_zscore:.3f}")

# STEP 5: Calculate Participation Divergence
print("\n\nSTEP 5: Calculate Participation Divergence Score")
print("-" * 50)

# Participation divergence is the difference between the z-scores
# Positive values indicate commercials are more active than usual relative to non-commercials
participation_divergence = comm_zscore - noncomm_zscore

print(f"Participation Divergence = Commercial Z-score - Non-Commercial Z-score")
print(f"Participation Divergence = {comm_zscore:.3f} - {noncomm_zscore:.3f}")
print(f"Participation Divergence = {participation_divergence:.3f}")

# STEP 6: Interpretation
print("\n\nSTEP 6: Interpretation")
print("-" * 50)

if abs(participation_divergence) < 0.5:
    interpretation = "NEUTRAL: Both trader groups are participating at normal levels"
elif participation_divergence > 1.5:
    interpretation = "STRONG COMMERCIAL BIAS: Commercials are unusually active relative to speculators"
elif participation_divergence > 0.5:
    interpretation = "MODERATE COMMERCIAL BIAS: Commercials are somewhat more active than usual"
elif participation_divergence < -1.5:
    interpretation = "STRONG SPECULATIVE BIAS: Speculators are unusually active relative to commercials"
else:
    interpretation = "MODERATE SPECULATIVE BIAS: Speculators are somewhat more active than usual"

print(f"Score: {participation_divergence:.3f}")
print(f"Interpretation: {interpretation}")

# STEP 7: Additional Context - Show the time series
print("\n\nSTEP 7: Time Series of Participation Shares")
print("-" * 50)
print("\nDate          | Comm Share | NonComm Share | Divergence")
print("-" * 60)

# Calculate divergence for all periods
for i in range(len(df)):
    if i >= lookback_periods - 1:  # Need enough data for statistics
        # Calculate rolling statistics
        comm_roll_mean = df['comm_share'].iloc[:i+1].mean()
        comm_roll_std = df['comm_share'].iloc[:i+1].std()
        noncomm_roll_mean = df['noncomm_share'].iloc[:i+1].mean()
        noncomm_roll_std = df['noncomm_share'].iloc[:i+1].std()
        
        # Calculate z-scores
        comm_z = (df['comm_share'].iloc[i] - comm_roll_mean) / comm_roll_std if comm_roll_std > 0 else 0
        noncomm_z = (df['noncomm_share'].iloc[i] - noncomm_roll_mean) / noncomm_roll_std if noncomm_roll_std > 0 else 0
        
        # Calculate divergence
        div = comm_z - noncomm_z
        
        print(f"{df['report_date'].iloc[i].strftime('%Y-%m-%d')} | "
              f"{df['comm_share'].iloc[i]:10.2f}% | "
              f"{df['noncomm_share'].iloc[i]:13.2f}% | "
              f"{div:10.3f}")

# STEP 8: Summary Statistics
print("\n\nSTEP 8: Summary of Key Metrics")
print("-" * 50)
print(f"Latest Report Date: {df['report_date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"Total Traders: {total_traders:,}")
print(f"Open Interest: {df['open_interest_all'].iloc[-1]:,}")
print(f"Average Position per Trader: {df['open_interest_all'].iloc[-1] / total_traders:,.0f} contracts")
print(f"\nTrader Participation:")
print(f"  Commercial: {comm_share:.1f}% ({comm_unique} traders)")
print(f"  Non-Commercial: {noncomm_share:.1f}% ({noncomm_unique} traders)")
print(f"\nParticipation Divergence Score: {participation_divergence:.3f}")
print(f"Interpretation: {interpretation}")

print("\n" + "=" * 80)
print("END OF DEMONSTRATION")
print("=" * 80)