"""
Participation Divergence Implementation
This code shows how to calculate participation divergence from COT data
"""

import pandas as pd
import numpy as np

def calculate_participation_divergence(df, lookback_weeks=52):
    """
    Calculate participation divergence score from COT data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COT data with columns for trader counts
    lookback_weeks : int
        Number of weeks for rolling statistics (default 52)
    
    Returns:
    --------
    pandas.DataFrame
        Original dataframe with added participation metrics
    """
    
    # Make a copy to avoid modifying original
    df_calc = df.copy()
    
    # STEP 1: Calculate unique traders for each category
    # Using max of long/short as approximation for unique traders
    df_calc['traders_comm_unique'] = df_calc[['traders_comm_long_all', 
                                               'traders_comm_short_all']].max(axis=1)
    df_calc['traders_noncomm_unique'] = df_calc[['traders_noncomm_long_all', 
                                                  'traders_noncomm_short_all']].max(axis=1)
    
    # STEP 2: Calculate trader shares (percentages)
    df_calc['comm_share'] = (df_calc['traders_comm_unique'] / 
                             df_calc['traders_tot_all']) * 100
    df_calc['noncomm_share'] = (df_calc['traders_noncomm_unique'] / 
                                df_calc['traders_tot_all']) * 100
    
    # STEP 3: Calculate rolling statistics
    # Using expanding window for early periods, then fixed window
    min_periods = min(20, lookback_weeks // 2)  # Need some data for meaningful stats
    
    # Commercial statistics
    df_calc['comm_mean'] = df_calc['comm_share'].rolling(
        window=lookback_weeks, min_periods=min_periods).mean()
    df_calc['comm_std'] = df_calc['comm_share'].rolling(
        window=lookback_weeks, min_periods=min_periods).std()
    
    # Non-commercial statistics
    df_calc['noncomm_mean'] = df_calc['noncomm_share'].rolling(
        window=lookback_weeks, min_periods=min_periods).mean()
    df_calc['noncomm_std'] = df_calc['noncomm_share'].rolling(
        window=lookback_weeks, min_periods=min_periods).std()
    
    # STEP 4: Calculate Z-scores
    df_calc['comm_zscore'] = np.where(
        df_calc['comm_std'] > 0,
        (df_calc['comm_share'] - df_calc['comm_mean']) / df_calc['comm_std'],
        0
    )
    
    df_calc['noncomm_zscore'] = np.where(
        df_calc['noncomm_std'] > 0,
        (df_calc['noncomm_share'] - df_calc['noncomm_mean']) / df_calc['noncomm_std'],
        0
    )
    
    # STEP 5: Calculate Participation Divergence
    df_calc['participation_divergence'] = df_calc['comm_zscore'] - df_calc['noncomm_zscore']
    
    # STEP 6: Add interpretation
    df_calc['divergence_signal'] = pd.cut(
        df_calc['participation_divergence'],
        bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
        labels=['Strong Spec Bias', 'Moderate Spec Bias', 'Neutral', 
                'Moderate Comm Bias', 'Strong Comm Bias']
    )
    
    return df_calc


# Example usage with actual COT data structure
def demonstrate_with_real_data():
    """
    Demonstrate the calculation with a real example
    """
    
    # Sample data representing actual COT report structure
    dates = pd.date_range('2024-01-01', periods=52, freq='W-TUE')
    
    # Simulate realistic trader counts
    np.random.seed(42)
    base_comm_long = 150
    base_comm_short = 140
    base_noncomm_long = 85
    base_noncomm_short = 75
    base_total = 450
    
    data = {
        'report_date_as_yyyy_mm_dd': dates,
        'traders_comm_long_all': base_comm_long + np.random.randint(-10, 10, 52),
        'traders_comm_short_all': base_comm_short + np.random.randint(-10, 10, 52),
        'traders_noncomm_long_all': base_noncomm_long + np.random.randint(-8, 8, 52),
        'traders_noncomm_short_all': base_noncomm_short + np.random.randint(-8, 8, 52),
        'traders_tot_all': base_total + np.random.randint(-20, 20, 52),
        'open_interest_all': 280000 + np.random.randint(-20000, 20000, 52)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate participation divergence
    df_result = calculate_participation_divergence(df)
    
    # Display results for the latest 5 periods
    print("Participation Divergence Analysis - Latest 5 Weeks")
    print("=" * 100)
    
    cols_to_display = [
        'report_date_as_yyyy_mm_dd',
        'traders_tot_all',
        'comm_share',
        'noncomm_share',
        'comm_zscore',
        'noncomm_zscore',
        'participation_divergence',
        'divergence_signal'
    ]
    
    print(df_result[cols_to_display].tail().to_string(index=False))
    
    # Detailed breakdown for the latest period
    latest = df_result.iloc[-1]
    
    print("\n" + "=" * 100)
    print("DETAILED BREAKDOWN FOR LATEST PERIOD")
    print("=" * 100)
    print(f"Date: {latest['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d')}")
    print(f"\nTrader Counts:")
    print(f"  Commercial Long: {int(latest['traders_comm_long_all'])}")
    print(f"  Commercial Short: {int(latest['traders_comm_short_all'])}")
    print(f"  Non-Commercial Long: {int(latest['traders_noncomm_long_all'])}")
    print(f"  Non-Commercial Short: {int(latest['traders_noncomm_short_all'])}")
    print(f"  Total Traders: {int(latest['traders_tot_all'])}")
    
    print(f"\nParticipation Shares:")
    print(f"  Commercial: {latest['comm_share']:.2f}% (Historical Avg: {latest['comm_mean']:.2f}%)")
    print(f"  Non-Commercial: {latest['noncomm_share']:.2f}% (Historical Avg: {latest['noncomm_mean']:.2f}%)")
    
    print(f"\nZ-Scores:")
    print(f"  Commercial: {latest['comm_zscore']:.3f}")
    print(f"  Non-Commercial: {latest['noncomm_zscore']:.3f}")
    
    print(f"\nParticipation Divergence Score: {latest['participation_divergence']:.3f}")
    print(f"Signal: {latest['divergence_signal']}")
    
    # Show extreme readings in the dataset
    print("\n" + "=" * 100)
    print("EXTREME READINGS IN DATASET")
    print("=" * 100)
    
    # Find max and min divergence
    max_div_idx = df_result['participation_divergence'].idxmax()
    min_div_idx = df_result['participation_divergence'].idxmin()
    
    print(f"\nMaximum Commercial Bias:")
    print(f"  Date: {df_result.loc[max_div_idx, 'report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d')}")
    print(f"  Score: {df_result.loc[max_div_idx, 'participation_divergence']:.3f}")
    print(f"  Commercial Share: {df_result.loc[max_div_idx, 'comm_share']:.2f}%")
    print(f"  Non-Commercial Share: {df_result.loc[max_div_idx, 'noncomm_share']:.2f}%")
    
    print(f"\nMaximum Speculative Bias:")
    print(f"  Date: {df_result.loc[min_div_idx, 'report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d')}")
    print(f"  Score: {df_result.loc[min_div_idx, 'participation_divergence']:.3f}")
    print(f"  Commercial Share: {df_result.loc[min_div_idx, 'comm_share']:.2f}%")
    print(f"  Non-Commercial Share: {df_result.loc[min_div_idx, 'noncomm_share']:.2f}%")
    
    return df_result


# Function to create a visual representation of the calculation
def create_calculation_summary(df, date_index=-1):
    """
    Create a formatted summary of the calculation for a specific date
    """
    
    row = df.iloc[date_index]
    
    summary = f"""
PARTICIPATION DIVERGENCE CALCULATION SUMMARY
==========================================
Date: {row['report_date_as_yyyy_mm_dd'].strftime('%Y-%m-%d')}

1. RAW TRADER COUNTS:
   Commercial Long: {int(row['traders_comm_long_all'])}
   Commercial Short: {int(row['traders_comm_short_all'])}
   Non-Commercial Long: {int(row['traders_noncomm_long_all'])}
   Non-Commercial Short: {int(row['traders_noncomm_short_all'])}
   Total Traders: {int(row['traders_tot_all'])}

2. UNIQUE TRADERS (MAX OF LONG/SHORT):
   Commercial: max({int(row['traders_comm_long_all'])}, {int(row['traders_comm_short_all'])}) = {int(row['traders_comm_unique'])}
   Non-Commercial: max({int(row['traders_noncomm_long_all'])}, {int(row['traders_noncomm_short_all'])}) = {int(row['traders_noncomm_unique'])}

3. PARTICIPATION SHARES:
   Commercial: {int(row['traders_comm_unique'])} / {int(row['traders_tot_all'])} × 100 = {row['comm_share']:.2f}%
   Non-Commercial: {int(row['traders_noncomm_unique'])} / {int(row['traders_tot_all'])} × 100 = {row['noncomm_share']:.2f}%

4. HISTORICAL STATISTICS (52-week):
   Commercial - Mean: {row['comm_mean']:.2f}%, Std: {row['comm_std']:.2f}%
   Non-Commercial - Mean: {row['noncomm_mean']:.2f}%, Std: {row['noncomm_std']:.2f}%

5. Z-SCORES:
   Commercial: ({row['comm_share']:.2f} - {row['comm_mean']:.2f}) / {row['comm_std']:.2f} = {row['comm_zscore']:.3f}
   Non-Commercial: ({row['noncomm_share']:.2f} - {row['noncomm_mean']:.2f}) / {row['noncomm_std']:.2f} = {row['noncomm_zscore']:.3f}

6. PARTICIPATION DIVERGENCE:
   Score = {row['comm_zscore']:.3f} - {row['noncomm_zscore']:.3f} = {row['participation_divergence']:.3f}
   
7. INTERPRETATION: {row['divergence_signal']}
"""
    
    return summary


if __name__ == "__main__":
    # Run the demonstration
    df_results = demonstrate_with_real_data()
    
    # Show calculation summary for latest date
    print("\n" + "=" * 100)
    print(create_calculation_summary(df_results, -1))