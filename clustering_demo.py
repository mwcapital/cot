"""
Demo: Participant Behavior Clusters - Feature Calculations
Shows exactly how each feature is calculated with numerical examples
"""

import numpy as np
from scipy import stats

print("PARTICIPANT BEHAVIOR CLUSTERS - FEATURE CALCULATION DEMO")
print("="*60)

# Example data for a single point in time
print("\n📊 SAMPLE DATA (Week 52)")
print("-"*40)
print("Commercial Long:")
print("  Positions: 120,000 contracts")
print("  Traders: 40")
print("\nCommercial Short:")
print("  Positions: 80,000 contracts") 
print("  Traders: 35")
print("\nNon-Commercial Long:")
print("  Positions: 150,000 contracts")
print("  Traders: 60")
print("\nNon-Commercial Short:")
print("  Positions: 130,000 contracts")
print("  Traders: 55")
print("\nOpen Interest: 500,000 contracts")

# Let's calculate features for Commercial Long
print("\n" + "="*60)
print("CALCULATING FEATURES FOR: Commercial Long")
print("="*60)

# Given values
positions = 120000
traders = 40
open_interest = 500000

# Feature 1: Average Position Size (Normalized)
print("\n1️⃣ AVERAGE POSITION SIZE (NORMALIZED)")
print("-"*40)
avg_position = positions / traders
avg_position_norm = avg_position / open_interest * 100

print(f"Step 1: avg_position = {positions:,} / {traders} = {avg_position:,.0f} contracts/trader")
print(f"Step 2: avg_position_norm = {avg_position:,.0f} / {open_interest:,} × 100 = {avg_position_norm:.2f}%")
print(f"\n✅ Result: Each trader holds {avg_position_norm:.2f}% of total market")

# Feature 2: Directional Bias
print("\n2️⃣ DIRECTIONAL BIAS")
print("-"*40)
direction = 1  # Because it's "Long"
print(f"Category: Commercial Long")
print(f"Direction = +1 (bullish)")
print(f"\n✅ Result: {direction}")

# Feature 3: Activity Level
print("\n3️⃣ ACTIVITY LEVEL")
print("-"*40)
print("Previous week (Week 51):")
print("  Commercial Long positions: 115,000")
prev_positions = 115000
activity = abs(positions - prev_positions) / open_interest * 100

print(f"\nCalculation:")
print(f"activity = |{positions:,} - {prev_positions:,}| / {open_interest:,} × 100")
print(f"activity = {abs(positions - prev_positions):,} / {open_interest:,} × 100 = {activity:.2f}%")
print(f"\n✅ Result: Position changed by {activity:.2f}% of market")

# Feature 4: Concentration Score
print("\n4️⃣ CONCENTRATION SCORE")
print("-"*40)
print("Historical trader counts (52 weeks):")
historical_traders = [38, 42, 45, 41, 39, 44, 46, 43, 40, 37, 
                     35, 43, 47, 44, 42, 38, 41, 45, 39, 36,
                     40, 44, 42, 38, 43, 45, 41, 37, 39, 42,
                     46, 43, 40, 38, 41, 44, 42, 39, 37, 41,
                     43, 45, 42, 38, 40, 44, 41, 39, 36, 38, 42, 40]

trader_pct = stats.percentileofscore(historical_traders, traders)
concentration = 100 - trader_pct

print(f"Current traders: {traders}")
print(f"Historical range: {min(historical_traders)} to {max(historical_traders)}")
print(f"Percentile rank: {trader_pct:.1f}th percentile")
print(f"Concentration score = 100 - {trader_pct:.1f} = {concentration:.1f}")
print(f"\n✅ Result: {concentration:.1f} (medium concentration)")

# Feature 5: Relative Strength
print("\n5️⃣ RELATIVE STRENGTH")
print("-"*40)
comm_long = 120000
comm_short = 80000
total_comm = comm_long + comm_short
relative_strength = positions / total_comm * 100

print(f"Commercial Long: {comm_long:,}")
print(f"Commercial Short: {comm_short:,}")
print(f"Total Commercial: {total_comm:,}")
print(f"\nrelative_strength = {positions:,} / {total_comm:,} × 100 = {relative_strength:.1f}%")
print(f"\n✅ Result: Controls {relative_strength:.1f}% of all commercial positions")

# Summary of all features
print("\n" + "="*60)
print("FEATURE VECTOR SUMMARY")
print("="*60)
print(f"1. Average Position (normalized): {avg_position_norm:.2f}%")
print(f"2. Directional Bias: {direction}")
print(f"3. Activity Level: {activity:.2f}%") 
print(f"4. Concentration Score: {concentration:.1f}")
print(f"5. Relative Strength: {relative_strength:.1f}%")

# How these might cluster
print("\n" + "="*60)
print("LIKELY CLUSTER ASSIGNMENT")
print("="*60)
print("Based on these features:")
print(f"- Medium-sized positions ({avg_position_norm:.2f}%)")
print(f"- Bullish bias (direction = {direction})")
print(f"- Low activity ({activity:.2f}%)")
print(f"- Medium concentration ({concentration:.1f})")
print(f"- Dominant in category ({relative_strength:.1f}%)")
print("\n🛡️ Likely cluster: HEDGERS")
print("(Stable positions, moderate size, low activity)")

# Show how different participants might cluster differently
print("\n" + "="*60)
print("EXAMPLE CLUSTER PROFILES")
print("="*60)

profiles = [
    {
        "name": "🚀 Aggressive Traders",
        "avg_position_norm": 0.8,
        "direction": 0.9,
        "activity": 6.5,
        "concentration": 70,
        "relative_strength": 65
    },
    {
        "name": "⚡ Market Makers",
        "avg_position_norm": 0.2,
        "direction": 0.1,
        "activity": 8.2,
        "concentration": 20,
        "relative_strength": 45
    },
    {
        "name": "🐋 Large Speculators",
        "avg_position_norm": 2.5,
        "direction": -0.7,
        "activity": 1.8,
        "concentration": 85,
        "relative_strength": 75
    },
    {
        "name": "🛡️ Hedgers",
        "avg_position_norm": 0.6,
        "direction": -0.8,
        "activity": 0.8,
        "concentration": 50,
        "relative_strength": 55
    }
]

for profile in profiles:
    print(f"\n{profile['name']}:")
    print(f"  Avg Position: {profile['avg_position_norm']:.1f}%")
    print(f"  Direction: {profile['direction']:+.1f}")
    print(f"  Activity: {profile['activity']:.1f}%")
    print(f"  Concentration: {profile['concentration']:.0f}")
    print(f"  Relative Strength: {profile['relative_strength']:.0f}%")