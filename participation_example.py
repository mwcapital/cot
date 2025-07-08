"""
Participation Divergence Calculation - Numerical Example
Using realistic CFTC COT data for Gold
"""

# Step 1: Latest data point values (example from Gold futures)
comm_long_traders = 45
comm_short_traders = 38
noncomm_long_traders = 142
noncomm_short_traders = 98

# Step 2: Calculate total traders for each group
# Note: Some traders might be both long and short, so we don't simply add
comm_traders = comm_long_traders + comm_short_traders  # 45 + 38 = 83
noncomm_traders = noncomm_long_traders + noncomm_short_traders  # 142 + 98 = 240
total_traders = comm_traders + noncomm_traders  # 83 + 240 = 323

print("=== STEP 1: Current Trader Counts ===")
print(f"Commercial traders: {comm_traders} (Long: {comm_long_traders}, Short: {comm_short_traders})")
print(f"Non-Commercial traders: {noncomm_traders} (Long: {noncomm_long_traders}, Short: {noncomm_short_traders})")
print(f"Total traders: {total_traders}")

# Step 3: Calculate current participation shares
comm_share = (comm_traders / total_traders) * 100  # (83 / 323) * 100 = 25.70%
noncomm_share = (noncomm_traders / total_traders) * 100  # (240 / 323) * 100 = 74.30%

print("\n=== STEP 2: Current Participation Shares ===")
print(f"Commercial share: {comm_share:.2f}%")
print(f"Non-Commercial share: {noncomm_share:.2f}%")

# Step 4: Historical statistics (52-week rolling window)
# These would be calculated from historical data
comm_share_mean_52w = 28.5  # Historical average commercial share
comm_share_std_52w = 3.2    # Historical standard deviation
noncomm_share_mean_52w = 71.5  # Historical average non-commercial share  
noncomm_share_std_52w = 3.2    # Historical standard deviation

print("\n=== STEP 3: Historical Statistics (52-week) ===")
print(f"Commercial share - Mean: {comm_share_mean_52w}%, Std: {comm_share_std_52w}%")
print(f"Non-Commercial share - Mean: {noncomm_share_mean_52w}%, Std: {noncomm_share_std_52w}%")

# Step 5: Calculate Z-scores
comm_z_score = (comm_share - comm_share_mean_52w) / comm_share_std_52w
# (25.70 - 28.5) / 3.2 = -2.8 / 3.2 = -0.875

noncomm_z_score = (noncomm_share - noncomm_share_mean_52w) / noncomm_share_std_52w
# (74.30 - 71.5) / 3.2 = 2.8 / 3.2 = 0.875

print("\n=== STEP 4: Z-Score Calculations ===")
print(f"Commercial Z-score: ({comm_share:.2f} - {comm_share_mean_52w}) / {comm_share_std_52w} = {comm_z_score:.3f}")
print(f"Non-Commercial Z-score: ({noncomm_share:.2f} - {noncomm_share_mean_52w}) / {noncomm_share_std_52w} = {noncomm_z_score:.3f}")

# Step 6: Calculate Participation Divergence
participation_divergence = (abs(comm_z_score) + abs(noncomm_z_score)) * 10
# (|-0.875| + |0.875|) * 10 = (0.875 + 0.875) * 10 = 1.75 * 10 = 17.5

print("\n=== STEP 5: Final Participation Divergence ===")
print(f"Participation Divergence = (|{comm_z_score:.3f}| + |{noncomm_z_score:.3f}|) * 10")
print(f"Participation Divergence = ({abs(comm_z_score):.3f} + {abs(noncomm_z_score):.3f}) * 10")
print(f"Participation Divergence = {participation_divergence:.1f}")

# Interpretation
print("\n=== INTERPRETATION ===")
print(f"Score of {participation_divergence:.1f} indicates:")
if participation_divergence < 10:
    print("- Normal trader distribution")
elif participation_divergence < 20:
    print("- Slightly unusual trader distribution")
    print("- Commercial participation is below average (-0.875 std)")
    print("- Non-Commercial participation is above average (+0.875 std)")
elif participation_divergence < 30:
    print("- Moderately unusual trader distribution")
else:
    print("- Highly unusual trader distribution")

print("\nIn this example:")
print("- Fewer commercials than usual are participating (25.7% vs 28.5% average)")
print("- More non-commercials than usual are participating (74.3% vs 71.5% average)")
print("- This shift in participation balance contributes 17.5 points to the heterogeneity index")