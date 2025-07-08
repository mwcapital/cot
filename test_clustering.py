"""
Test script for Participant Behavior Clusters visualization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Create sample data similar to COT structure
def create_sample_data():
    """Create sample COT data for testing clustering"""
    
    # Generate 52 weeks of data
    dates = pd.date_range(end=datetime.now(), periods=52, freq='W-TUE')
    
    data = []
    for i, date in enumerate(dates):
        # Add some trends and randomness
        trend = i / 52  # Gradual trend over the year
        
        # Commercial data
        comm_long_pos = 100000 + int(20000 * np.sin(i/8)) + np.random.randint(-5000, 5000)
        comm_short_pos = 80000 + int(15000 * np.cos(i/8)) + np.random.randint(-5000, 5000)
        comm_long_traders = 40 + int(10 * np.sin(i/12)) + np.random.randint(-5, 5)
        comm_short_traders = 35 + int(8 * np.cos(i/12)) + np.random.randint(-5, 5)
        
        # Non-commercial data
        noncomm_long_pos = 120000 + int(25000 * np.cos(i/10)) + np.random.randint(-8000, 8000)
        noncomm_short_pos = 110000 + int(20000 * np.sin(i/10)) + np.random.randint(-8000, 8000)
        noncomm_long_traders = 80 + int(20 * np.cos(i/8)) + np.random.randint(-10, 10)
        noncomm_short_traders = 75 + int(15 * np.sin(i/8)) + np.random.randint(-10, 10)
        
        # Open interest
        open_interest = comm_long_pos + comm_short_pos + noncomm_long_pos + noncomm_short_pos + np.random.randint(0, 50000)
        
        data.append({
            'report_date_as_yyyy_mm_dd': date,
            'comm_positions_long_all': comm_long_pos,
            'comm_positions_short_all': comm_short_pos,
            'traders_comm_long_all': comm_long_traders,
            'traders_comm_short_all': comm_short_traders,
            'noncomm_positions_long_all': noncomm_long_pos,
            'noncomm_positions_short_all': noncomm_short_pos,
            'traders_noncomm_long_all': noncomm_long_traders,
            'traders_noncomm_short_all': noncomm_short_traders,
            'open_interest_all': open_interest
        })
    
    return pd.DataFrame(data)

def calculate_clustering_features(df, lookback_weeks=26):
    """Calculate the 5 features for clustering"""
    
    # Define categories
    categories = [
        ('Commercial Long', 'comm_positions_long_all', 'traders_comm_long_all'),
        ('Commercial Short', 'comm_positions_short_all', 'traders_comm_short_all'),
        ('Non-Commercial Long', 'noncomm_positions_long_all', 'traders_noncomm_long_all'),
        ('Non-Commercial Short', 'noncomm_positions_short_all', 'traders_noncomm_short_all')
    ]
    
    # Get latest data window
    latest_date = df['report_date_as_yyyy_mm_dd'].max()
    start_date = latest_date - timedelta(weeks=lookback_weeks)
    df_window = df[df['report_date_as_yyyy_mm_dd'] >= start_date].copy()
    
    clustering_data = []
    
    for date in df_window['report_date_as_yyyy_mm_dd'].unique():
        date_data = df_window[df_window['report_date_as_yyyy_mm_dd'] == date]
        
        for cat_name, pos_col, trader_col in categories:
            row = date_data.iloc[0]
            
            # Feature 1: Average position size (normalized)
            traders = float(row[trader_col]) if float(row[trader_col]) > 0 else 0.001
            positions = float(row[pos_col])
            avg_position = positions / traders
            avg_position_norm = avg_position / float(row['open_interest_all']) * 100
            
            # Feature 2: Directional bias
            direction = 1 if 'Long' in cat_name else -1
            
            # Feature 3: Activity level
            prev_week = date - timedelta(days=7)
            prev_data = df_window[df_window['report_date_as_yyyy_mm_dd'] <= prev_week].tail(1)
            if len(prev_data) > 0:
                prev_positions = float(prev_data.iloc[0][pos_col])
                activity = abs(positions - prev_positions) / float(row['open_interest_all']) * 100
            else:
                activity = 0
            
            # Feature 4: Concentration (using percentile)
            hist_data = df[df['report_date_as_yyyy_mm_dd'] >= (latest_date - timedelta(weeks=52))]
            trader_values = hist_data[trader_col].values
            trader_pct = stats.percentileofscore(trader_values, traders)
            concentration = 100 - trader_pct
            
            # Feature 5: Relative strength
            if 'comm' in pos_col:
                total_positions = float(row['comm_positions_long_all']) + float(row['comm_positions_short_all'])
            else:
                total_positions = float(row['noncomm_positions_long_all']) + float(row['noncomm_positions_short_all'])
            relative_strength = positions / total_positions * 100 if total_positions > 0 else 0
            
            clustering_data.append({
                'date': date,
                'category': cat_name,
                'avg_position_norm': avg_position_norm,
                'direction': direction,
                'activity': activity,
                'concentration': concentration,
                'relative_strength': relative_strength,
                'traders': traders,
                'positions': positions
            })
            
            print(f"\n{cat_name} on {date.strftime('%Y-%m-%d')}:")
            print(f"  Avg Position (norm): {avg_position_norm:.2f}%")
            print(f"  Direction: {direction}")
            print(f"  Activity: {activity:.2f}%")
            print(f"  Concentration: {concentration:.1f}")
            print(f"  Relative Strength: {relative_strength:.1f}%")
    
    return pd.DataFrame(clustering_data)

def perform_clustering(cluster_df, n_clusters=4):
    """Perform K-means clustering"""
    
    feature_cols = ['avg_position_norm', 'direction', 'activity', 'concentration', 'relative_strength']
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cluster_df[feature_cols])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_df['cluster'] = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters
    print("\n" + "="*50)
    print("CLUSTER ANALYSIS")
    print("="*50)
    
    for i in range(n_clusters):
        cluster_data = cluster_df[cluster_df['cluster'] == i]
        print(f"\nCluster {i} ({len(cluster_data)} members):")
        print(f"  Avg Position: {cluster_data['avg_position_norm'].mean():.2f}%")
        print(f"  Direction Bias: {cluster_data['direction'].mean():.2f}")
        print(f"  Activity: {cluster_data['activity'].mean():.2f}%")
        print(f"  Concentration: {cluster_data['concentration'].mean():.1f}")
        print(f"  Relative Strength: {cluster_data['relative_strength'].mean():.1f}%")
        
        # Classify cluster
        if cluster_data['activity'].mean() > cluster_df['activity'].quantile(0.75):
            if abs(cluster_data['direction'].mean()) > 0.5:
                cluster_type = "🚀 Aggressive Traders"
            else:
                cluster_type = "⚡ Market Makers"
        elif cluster_data['concentration'].mean() > cluster_df['concentration'].quantile(0.75):
            cluster_type = "🐋 Large Speculators"
        elif abs(cluster_data['direction'].mean()) < 0.2 and cluster_data['activity'].mean() < cluster_df['activity'].quantile(0.25):
            cluster_type = "🛡️ Hedgers"
        else:
            cluster_type = "📊 Trend Followers"
        
        print(f"  Type: {cluster_type}")
    
    # Show distribution by category
    print("\n" + "="*50)
    print("CLUSTER DISTRIBUTION BY CATEGORY")
    print("="*50)
    
    distribution = cluster_df.groupby(['category', 'cluster']).size().unstack(fill_value=0)
    print(distribution)
    
    return cluster_df, kmeans

def visualize_clusters_text(cluster_df):
    """Simple text visualization of clustering results"""
    
    print("\n" + "="*50)
    print("TEMPORAL EVOLUTION")
    print("="*50)
    
    # Show how clusters change over time
    temporal = cluster_df.groupby(['date', 'cluster']).size().unstack(fill_value=0)
    print("\nCluster populations over time (last 5 weeks):")
    print(temporal.tail())
    
    # Check for transitions
    print("\n" + "="*50)
    print("CLUSTER TRANSITIONS")
    print("="*50)
    
    latest_date = cluster_df['date'].max()
    prev_date = latest_date - timedelta(days=7)
    
    latest_clusters = cluster_df[cluster_df['date'] == latest_date][['category', 'cluster']]
    prev_clusters = cluster_df[cluster_df['date'] == prev_date][['category', 'cluster']]
    
    transitions = latest_clusters.merge(prev_clusters, on='category', suffixes=('_current', '_previous'))
    transitions = transitions[transitions['cluster_current'] != transitions['cluster_previous']]
    
    if len(transitions) > 0:
        print("\nDetected transitions:")
        for _, trans in transitions.iterrows():
            print(f"  {trans['category']}: Cluster {trans['cluster_previous']} → Cluster {trans['cluster_current']}")
    else:
        print("\nNo cluster transitions detected in the last week")

# Main execution
if __name__ == "__main__":
    print("Testing Participant Behavior Clusters Implementation")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample COT data...")
    df = create_sample_data()
    print(f"   Generated {len(df)} weeks of data")
    
    # Calculate features
    print("\n2. Calculating clustering features...")
    cluster_df = calculate_clustering_features(df, lookback_weeks=26)
    print(f"   Calculated features for {len(cluster_df)} data points")
    
    # Perform clustering
    print("\n3. Performing K-means clustering...")
    cluster_df, kmeans = perform_clustering(cluster_df, n_clusters=4)
    
    # Visualize results
    print("\n4. Analyzing results...")
    visualize_clusters_text(cluster_df)
    
    # Feature importance (using PCA)
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE (PCA)")
    print("="*50)
    
    feature_cols = ['avg_position_norm', 'direction', 'activity', 'concentration', 'relative_strength']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cluster_df[feature_cols])
    
    pca = PCA(n_components=2)
    pca.fit(features_scaled)
    
    print("\nPrincipal Component loadings:")
    for i, pc in enumerate(pca.components_):
        print(f"\nPC{i+1} (explains {pca.explained_variance_ratio_[i]*100:.1f}% variance):")
        for j, feature in enumerate(feature_cols):
            print(f"  {feature}: {pc[j]:.3f}")
    
    print("\n" + "="*60)
    print("Test completed successfully!")