"""
Market State Clustering Analysis for CFTC COT Dashboard
Clusters weeks based on complete market state (all 4 categories together)
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats


def create_market_state_clusters(df, instrument_name):
    """Create market state clustering analysis"""

    # Grey subtitle explaining the analysis
    st.caption(
        "Clusters weekly market snapshots to identify distinct regimes using 20 features "
        "(5 per category: Position Size, Activity, Traders, Avg Position, Trend Alignment). "
        "Analysis period: 3 years. Backward-looking historical analysis only - not for prediction. "
        "Use the PCA 2D plot to verify clustering quality - clusters should appear well-separated. "
        "If clusters overlap heavily in PCA space, the clustering may not be meaningful."
    )

    # Cluster count selector (default to 3)
    cluster_count_manual = st.selectbox(
        "Number of Clusters:",
        options=[2, 3, 4],
        index=1,  # Default to 3
        key="market_state_cluster_count"
    )

    # Fixed 3-year analysis window
    lookback_weeks = 156

    # Try to fetch price data for trend alignment
    price_returns = {}
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from futures_price_fetcher import FuturesPriceFetcher
        import json

        # Get futures symbol for this instrument
        def get_futures_symbol_for_cot_instrument(inst_name):
            clean_name = inst_name
            if ' (' in inst_name and inst_name.endswith(')'):
                clean_name = inst_name.rsplit(' (', 1)[0]

            possible_paths = [
                'instrument_management/futures/futures_symbols_enhanced.json',
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            'instrument_management', 'futures', 'futures_symbols_enhanced.json'),
            ]

            mapping_data = None
            for mapping_path in possible_paths:
                try:
                    with open(mapping_path, 'r') as f:
                        mapping_data = json.load(f)
                    break
                except FileNotFoundError:
                    continue

            if not mapping_data:
                return None

            for symbol, symbol_data in mapping_data['futures_symbols'].items():
                if 'cot_mapping' in symbol_data and symbol_data['cot_mapping'].get('matched', False):
                    for cot_instrument in symbol_data['cot_mapping'].get('instruments', []):
                        if clean_name in cot_instrument or cot_instrument in clean_name:
                            return symbol
                        if inst_name in cot_instrument or cot_instrument in inst_name:
                            return symbol
            return None

        futures_symbol = get_futures_symbol_for_cot_instrument(instrument_name)

        if futures_symbol:
            fetcher = FuturesPriceFetcher()
            df_sorted = df.sort_values('report_date_as_yyyy_mm_dd')
            start_date_str = df_sorted['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
            end_date_str = df_sorted['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')

            price_df = fetcher.fetch_weekly_prices(futures_symbol, start_date_str, end_date_str, adjustment='NON')

            if not price_df.empty:
                price_df = price_df.sort_values('date')
                price_df['weekly_return'] = price_df['close'].pct_change()
                price_df['price_direction'] = np.sign(price_df['weekly_return'])

                for _, row in price_df.iterrows():
                    price_returns[row['date'].date()] = {
                        'return': row['weekly_return'],
                        'direction': row['price_direction']
                    }
    except Exception:
        pass  # If price fetch fails, trend alignment will be 0

    has_price_data = len(price_returns) > 0

    # Check for sklearn availability
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA
        sklearn_available = True
    except ImportError:
        sklearn_available = False
        st.warning("scikit-learn not installed. Install with: pip install scikit-learn")
        return None

    # Prepare data
    df_cluster = df.copy()
    df_cluster = df_cluster.sort_values('report_date_as_yyyy_mm_dd')

    # Get data for specified window
    latest_date = df_cluster['report_date_as_yyyy_mm_dd'].max()
    start_date = latest_date - pd.Timedelta(weeks=lookback_weeks)
    df_window = df_cluster[df_cluster['report_date_as_yyyy_mm_dd'] >= start_date].copy()

    if len(df_window) < 52:
        st.warning(f"Insufficient data for market state clustering (found {len(df_window)} weeks, need at least 52)")
        return None
    
    
    # Build feature matrix for each week
    # 5 features per category: Position %, Activity (signed), Traders, Avg Position, Trend Alignment
    # Total: 20 features (4 categories × 5 features)
    weekly_features = []

    for date in df_window['report_date_as_yyyy_mm_dd'].unique():
        week_data = df_window[df_window['report_date_as_yyyy_mm_dd'] == date].iloc[0]
        oi = float(week_data['open_interest_all']) if week_data['open_interest_all'] > 0 else 1

        # Get price direction for this week
        date_key = date.date() if hasattr(date, 'date') else date
        price_info = price_returns.get(date_key, {})
        weekly_price_direction = price_info.get('direction', 0)

        features = []

        # Commercial Long features (5 features)
        comm_long_change = float(week_data['change_in_comm_long_all']) if 'change_in_comm_long_all' in week_data else 0
        comm_long_change_dir = np.sign(comm_long_change)
        comm_long_trend = comm_long_change_dir * weekly_price_direction if comm_long_change_dir != 0 and weekly_price_direction != 0 else 0
        comm_long_features = [
            float(week_data['pct_of_oi_comm_long_all']) if 'pct_of_oi_comm_long_all' in week_data else 0,
            comm_long_change / oi * 100,  # Signed activity
            float(week_data['traders_comm_long_all']) if 'traders_comm_long_all' in week_data else 0,
            float(week_data['comm_positions_long_all']) / float(week_data['traders_comm_long_all']) if week_data['traders_comm_long_all'] > 0 else 0,
            comm_long_trend  # Trend alignment
        ]
        features.extend(comm_long_features)

        # Commercial Short features (5 features)
        comm_short_change = float(week_data['change_in_comm_short_all']) if 'change_in_comm_short_all' in week_data else 0
        comm_short_change_dir = np.sign(comm_short_change)
        comm_short_trend = comm_short_change_dir * weekly_price_direction if comm_short_change_dir != 0 and weekly_price_direction != 0 else 0
        comm_short_features = [
            float(week_data['pct_of_oi_comm_short_all']) if 'pct_of_oi_comm_short_all' in week_data else 0,
            comm_short_change / oi * 100,  # Signed activity
            float(week_data['traders_comm_short_all']) if 'traders_comm_short_all' in week_data else 0,
            float(week_data['comm_positions_short_all']) / float(week_data['traders_comm_short_all']) if week_data['traders_comm_short_all'] > 0 else 0,
            comm_short_trend  # Trend alignment
        ]
        features.extend(comm_short_features)

        # Non-Commercial Long features (5 features)
        noncomm_long_change = float(week_data['change_in_noncomm_long_all']) if 'change_in_noncomm_long_all' in week_data else 0
        noncomm_long_change_dir = np.sign(noncomm_long_change)
        noncomm_long_trend = noncomm_long_change_dir * weekly_price_direction if noncomm_long_change_dir != 0 and weekly_price_direction != 0 else 0
        noncomm_long_features = [
            float(week_data['pct_of_oi_noncomm_long_all']) if 'pct_of_oi_noncomm_long_all' in week_data else 0,
            noncomm_long_change / oi * 100,  # Signed activity
            float(week_data['traders_noncomm_long_all']) if 'traders_noncomm_long_all' in week_data else 0,
            float(week_data['noncomm_positions_long_all']) / float(week_data['traders_noncomm_long_all']) if week_data['traders_noncomm_long_all'] > 0 else 0,
            noncomm_long_trend  # Trend alignment
        ]
        features.extend(noncomm_long_features)

        # Non-Commercial Short features (5 features)
        noncomm_short_change = float(week_data['change_in_noncomm_short_all']) if 'change_in_noncomm_short_all' in week_data else 0
        noncomm_short_change_dir = np.sign(noncomm_short_change)
        noncomm_short_trend = noncomm_short_change_dir * weekly_price_direction if noncomm_short_change_dir != 0 and weekly_price_direction != 0 else 0
        noncomm_short_features = [
            float(week_data['pct_of_oi_noncomm_short_all']) if 'pct_of_oi_noncomm_short_all' in week_data else 0,
            noncomm_short_change / oi * 100,  # Signed activity
            float(week_data['traders_noncomm_short_all']) if 'traders_noncomm_short_all' in week_data else 0,
            float(week_data['noncomm_positions_short_all']) / float(week_data['traders_noncomm_short_all']) if week_data['traders_noncomm_short_all'] > 0 else 0,
            noncomm_short_trend  # Trend alignment
        ]
        features.extend(noncomm_short_features)

        # Store concentration for display purposes (not used in clustering)
        top4_long_net = float(week_data['conc_net_le_4_tdr_long_all']) if 'conc_net_le_4_tdr_long_all' in week_data else 0
        top4_short_net = float(week_data['conc_net_le_4_tdr_short_all']) if 'conc_net_le_4_tdr_short_all' in week_data else 0

        weekly_features.append({
            'date': date,
            'features': features,
            'top4_long_net': top4_long_net,
            'top4_short_net': top4_short_net
        })
    
    # Create feature matrix
    feature_matrix = np.array([w['features'] for w in weekly_features])
    dates = [w['date'] for w in weekly_features]

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)

    # Use manually selected cluster count
    cluster_count = cluster_count_manual

    # Calculate metrics for all K values (for expander)
    K_range = range(2, 5)  # 2, 3, 4 clusters
    silhouette_scores_list = []
    inertias = []
    total_inertia = np.sum((features_scaled - features_scaled.mean(axis=0))**2)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(features_scaled)
        inertias.append(km.inertia_)
        if len(set(km.labels_)) > 1:
            silhouette_scores_list.append(silhouette_score(features_scaled, km.labels_))
        else:
            silhouette_scores_list.append(0)

    # Calculate variance explained
    variance_explained = [(1 - inertia / total_inertia) * 100 for inertia in inertias]

    # Get metrics for selected K
    selected_idx = cluster_count - 2  # K_range starts at 2
    best_silhouette = silhouette_scores_list[selected_idx]
    best_variance = variance_explained[selected_idx]

    # Perform clustering with selected K
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)

    # Create results dataframe with additional metrics
    results_df = pd.DataFrame({
        'date': dates,
        'cluster': clusters,
        'top4_long_net': [w['top4_long_net'] for w in weekly_features],
        'top4_short_net': [w['top4_short_net'] for w in weekly_features]
    })
    
    # Add additional metrics to results_df for box plot options
    # Feature indices: 5 features per category [pos%, activity, traders, avg_pos, trend_alignment]
    # Comm Long: 0-4, Comm Short: 5-9, NonComm Long: 10-14, NonComm Short: 15-19
    activity_indices = [1, 6, 11, 16]  # Activity for each category
    avg_pos_indices = [3, 8, 13, 18]  # Avg position for each category
    trend_indices = [4, 9, 14, 19]  # Trend alignment for each category

    for idx, date in enumerate(dates):
        week_data = df_window[df_window['report_date_as_yyyy_mm_dd'] == date].iloc[0]
        week_features = feature_matrix[idx]

        # Average activity across all categories
        results_df.loc[idx, 'avg_activity_all'] = np.mean([week_features[i] for i in activity_indices])

        # Average position size across all categories
        results_df.loc[idx, 'avg_position_size_all'] = np.mean([week_features[i] for i in avg_pos_indices])

        # Average trend alignment across all categories
        results_df.loc[idx, 'avg_trend_alignment'] = np.mean([week_features[i] for i in trend_indices])

        # Store individual category metrics
        results_df.loc[idx, 'comm_long_pct_oi'] = float(week_data['pct_of_oi_comm_long_all']) if 'pct_of_oi_comm_long_all' in week_data else 0
        results_df.loc[idx, 'comm_short_pct_oi'] = float(week_data['pct_of_oi_comm_short_all']) if 'pct_of_oi_comm_short_all' in week_data else 0
        results_df.loc[idx, 'noncomm_long_pct_oi'] = float(week_data['pct_of_oi_noncomm_long_all']) if 'pct_of_oi_noncomm_long_all' in week_data else 0
        results_df.loc[idx, 'noncomm_short_pct_oi'] = float(week_data['pct_of_oi_noncomm_short_all']) if 'pct_of_oi_noncomm_short_all' in week_data else 0

        # Derived metrics (NOT used in clustering - for display only)
        results_df.loc[idx, 'comm_net_position'] = (
            float(week_data['comm_positions_long_all']) - float(week_data['comm_positions_short_all'])
        ) / float(week_data['open_interest_all']) * 100 if week_data['open_interest_all'] > 0 else 0

        results_df.loc[idx, 'noncomm_net_position'] = (
            float(week_data['noncomm_positions_long_all']) - float(week_data['noncomm_positions_short_all'])
        ) / float(week_data['open_interest_all']) * 100 if week_data['open_interest_all'] > 0 else 0

        results_df.loc[idx, 'total_traders'] = float(week_data['traders_tot_all']) if 'traders_tot_all' in week_data else 0

        # Calculate spread between commercial and non-commercial
        results_df.loc[idx, 'comm_vs_noncomm_net'] = results_df.loc[idx, 'comm_net_position'] - results_df.loc[idx, 'noncomm_net_position']
    
    # Calculate cluster profiles
    cluster_profiles = []
    for i in range(cluster_count):
        cluster_mask = results_df['cluster'] == i
        cluster_weeks = results_df[cluster_mask]
        cluster_features = feature_matrix[clusters == i]
        
        profile = {
            'cluster': i,
            'size': len(cluster_weeks),
            'avg_concentration_long': cluster_weeks['top4_long_net'].mean(),
            'avg_concentration_short': cluster_weeks['top4_short_net'].mean(),
            'date_range': f"{cluster_weeks['date'].min().strftime('%Y-%m')} to {cluster_weeks['date'].max().strftime('%Y-%m')}"
        }
        
        # Calculate detailed metrics for each category
        # % of OI for each category
        profile['comm_long_pct'] = cluster_weeks['comm_long_pct_oi'].mean()
        profile['comm_short_pct'] = cluster_weeks['comm_short_pct_oi'].mean()
        profile['noncomm_long_pct'] = cluster_weeks['noncomm_long_pct_oi'].mean()
        profile['noncomm_short_pct'] = cluster_weeks['noncomm_short_pct_oi'].mean()
        
        # Calculate average positions and trader counts for each category
        comm_long_positions = []
        comm_short_positions = []
        noncomm_long_positions = []
        noncomm_short_positions = []
        
        comm_long_traders = []
        comm_short_traders = []
        noncomm_long_traders = []
        noncomm_short_traders = []
        
        for idx, week in cluster_weeks.iterrows():
            week_data = df_window[df_window['report_date_as_yyyy_mm_dd'] == week['date']].iloc[0]
            
            # Average positions
            if week_data['traders_comm_long_all'] > 0:
                comm_long_positions.append(week_data['comm_positions_long_all'] / week_data['traders_comm_long_all'])
            if week_data['traders_comm_short_all'] > 0:
                comm_short_positions.append(week_data['comm_positions_short_all'] / week_data['traders_comm_short_all'])
            if week_data['traders_noncomm_long_all'] > 0:
                noncomm_long_positions.append(week_data['noncomm_positions_long_all'] / week_data['traders_noncomm_long_all'])
            if week_data['traders_noncomm_short_all'] > 0:
                noncomm_short_positions.append(week_data['noncomm_positions_short_all'] / week_data['traders_noncomm_short_all'])
            
            # Trader counts
            comm_long_traders.append(week_data['traders_comm_long_all'])
            comm_short_traders.append(week_data['traders_comm_short_all'])
            noncomm_long_traders.append(week_data['traders_noncomm_long_all'])
            noncomm_short_traders.append(week_data['traders_noncomm_short_all'])
        
        # Store averages
        profile['comm_long_avg_pos'] = np.mean(comm_long_positions) if comm_long_positions else 0
        profile['comm_short_avg_pos'] = np.mean(comm_short_positions) if comm_short_positions else 0
        profile['noncomm_long_avg_pos'] = np.mean(noncomm_long_positions) if noncomm_long_positions else 0
        profile['noncomm_short_avg_pos'] = np.mean(noncomm_short_positions) if noncomm_short_positions else 0
        
        profile['comm_long_traders'] = np.mean(comm_long_traders)
        profile['comm_short_traders'] = np.mean(comm_short_traders)
        profile['noncomm_long_traders'] = np.mean(noncomm_long_traders)
        profile['noncomm_short_traders'] = np.mean(noncomm_short_traders)
        
        # Analyze dominant behaviors from features
        avg_features = cluster_features.mean(axis=0)

        # Extract activity metrics (index 1, 6, 11, 16 for each category)
        comm_long_activity = avg_features[1]
        comm_short_activity = avg_features[6]
        noncomm_long_activity = avg_features[11]
        noncomm_short_activity = avg_features[16]

        # Calculate average activity
        profile['avg_activity'] = (comm_long_activity + comm_short_activity +
                                  noncomm_long_activity + noncomm_short_activity) / 4

        # Extract trend alignment metrics (index 4, 9, 14, 19 for each category)
        comm_long_trend = avg_features[4]
        comm_short_trend = avg_features[9]
        noncomm_long_trend = avg_features[14]
        noncomm_short_trend = avg_features[19]

        # Calculate average trend alignment (-1 = contrarian, +1 = trend-following)
        profile['avg_trend_alignment'] = (comm_long_trend + comm_short_trend +
                                          noncomm_long_trend + noncomm_short_trend) / 4

        cluster_profiles.append(profile)
    
    # Calculate regime persistence and show at the top
    regime_lengths = []
    current_regime = results_df.iloc[0]['cluster']
    current_length = 1
    
    for i in range(1, len(results_df)):
        if results_df.iloc[i]['cluster'] == current_regime:
            current_length += 1
        else:
            regime_lengths.append(current_length)
            current_regime = results_df.iloc[i]['cluster']
            current_length = 1
    regime_lengths.append(current_length)
    
    avg_regime_length = np.mean(regime_lengths)
    st.caption(f"Optimal clusters: {cluster_count} (silhouette: {best_silhouette:.2f}). Average regime duration: {avg_regime_length:.1f} weeks.")

    # Create visualizations - Timeline and Cluster Statistics only
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Cluster Timeline', 'Cluster Statistics'],
        specs=[[{"type": "scatter"}, {"type": "table"}]],
        horizontal_spacing=0.1,
        column_widths=[0.6, 0.4]
    )

    # 1. Timeline with regime colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i in range(cluster_count):
        mask = results_df['cluster'] == i
        profile = cluster_profiles[i]

        fig.add_trace(
            go.Scatter(
                x=results_df.loc[mask, 'date'],
                y=[i] * mask.sum(),
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors[i % len(colors)],
                    symbol='square'
                ),
                name=f"C{i}",
                legendgroup=f"cluster{i}",
                showlegend=True,
                hovertemplate='%{x}<br>Cluster ' + str(i) + '<extra></extra>'
            ),
            row=1, col=1
        )

    # 2. Regime profiles table
    profile_data = []
    for profile in cluster_profiles:
        profile_data.append([
            f"C{profile['cluster']}",
            f"{profile['size']}",
            f"{profile['comm_long_pct']:.1f}",
            f"{profile['comm_short_pct']:.1f}",
            f"{profile['noncomm_long_pct']:.1f}",
            f"{profile['noncomm_short_pct']:.1f}",
            f"{profile['comm_long_avg_pos']:.0f}",
            f"{profile['comm_short_avg_pos']:.0f}",
            f"{profile['noncomm_long_avg_pos']:.0f}",
            f"{profile['noncomm_short_avg_pos']:.0f}"
        ])

    headers = [
        'Cluster', 'Weeks',
        'CL%', 'CS%', 'SL%', 'SS%',
        'CL Avg', 'CS Avg', 'SL Avg', 'SS Avg'
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=headers,
                fill_color='paleturquoise',
                align='center',
                font=dict(size=10),
                height=28
            ),
            cells=dict(
                values=list(zip(*profile_data)) if profile_data else [[]]*10,
                fill_color='lavender',
                align='center',
                font=dict(size=10),
                height=24
            )
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"Market State Clustering - {instrument_name} ({lookback_weeks} weeks)",
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.3
        )
    )

    # Update axes
    fig.update_yaxes(title_text="Cluster", ticktext=[f"C{i}" for i in range(cluster_count)],
                     tickvals=list(range(cluster_count)), row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Table legend
    st.caption(
        "CL/CS = Commercial Long/Short, SL/SS = Speculative Long/Short. "
        "% = position as % of OI, Avg = avg contracts per trader."
    )

    # PCA 2D Plot for cluster validation
    st.subheader("Cluster Separation (PCA 2D)")

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    # Calculate variance explained
    var_explained = pca.explained_variance_ratio_ * 100

    fig_pca = go.Figure()

    for i in range(cluster_count):
        mask = clusters == i
        fig_pca.add_trace(
            go.Scatter(
                x=features_pca[mask, 0],
                y=features_pca[mask, 1],
                mode='markers',
                marker=dict(size=8, color=colors[i % len(colors)]),
                name=f"C{i} ({sum(mask)} weeks)",
                hovertemplate=f'Cluster {i}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
            )
        )

    fig_pca.update_layout(
        title=f"PCA Projection (PC1: {var_explained[0]:.1f}%, PC2: {var_explained[1]:.1f}% variance)",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig_pca, use_container_width=True)

    st.caption(
        "Well-separated clusters in PCA space indicate meaningful groupings. "
        "Overlapping clusters suggest the regimes may not be distinct."
    )
    
    # Add Market Regime Summary Table
    st.subheader("Market Regime Summary")
    
    # Calculate percentage of time in each cluster
    cluster_time_pct = results_df['cluster'].value_counts(normalize=True).sort_index() * 100
    
    # Calculate average positions for each cluster
    regime_summary_data = []
    
    for i in range(cluster_count):
        cluster_mask = results_df['cluster'] == i
        cluster_weeks = results_df[cluster_mask]
        cluster_features = feature_matrix[clusters == i]
        
        # Calculate average features
        avg_features = cluster_features.mean(axis=0)

        # Extract key percentages (% of OI) - indices 0, 5, 10, 15
        comm_long_pct = avg_features[0]
        comm_short_pct = avg_features[5]
        noncomm_long_pct = avg_features[10]
        noncomm_short_pct = avg_features[15]
        
        # Calculate episode duration
        episode_lengths = []
        in_cluster = False
        current_length = 0
        
        for _, row in results_df.iterrows():
            if row['cluster'] == i:
                if not in_cluster:
                    in_cluster = True
                    current_length = 1
                else:
                    current_length += 1
            else:
                if in_cluster:
                    episode_lengths.append(current_length)
                    in_cluster = False
                    current_length = 0
        
        if in_cluster:
            episode_lengths.append(current_length)
        
        avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0
        
        regime_summary_data.append({
            'Cluster': f"Cluster {i}",
            '% of Time': f"{cluster_time_pct.get(i, 0):.1f}%",
            'Comm Long %': f"{comm_long_pct:.1f}%",
            'Comm Short %': f"{comm_short_pct:.1f}%",
            'Spec Long %': f"{noncomm_long_pct:.1f}%",
            'Spec Short %': f"{noncomm_short_pct:.1f}%",
            'Avg Duration': f"{avg_episode_length:.1f} weeks"
        })
    
    # Create summary table
    summary_df = pd.DataFrame(regime_summary_data)
    
    # Display as a nice formatted table
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cluster": st.column_config.TextColumn("Regime", width="small"),
                "% of Time": st.column_config.TextColumn("Time Share", width="small"),
                "Comm Long %": st.column_config.TextColumn("Comm Long", width="small"),
                "Comm Short %": st.column_config.TextColumn("Comm Short", width="small"),
                "Spec Long %": st.column_config.TextColumn("Spec Long", width="small"),
                "Spec Short %": st.column_config.TextColumn("Spec Short", width="small"),
                "Avg Duration": st.column_config.TextColumn("Avg Episode", width="small")
            }
        )
    
    with col2:
        # Add a pie chart showing time distribution
        fig_pie = go.Figure(data=[go.Pie(
            labels=[f"Cluster {i}" for i in range(cluster_count)],
            values=[cluster_time_pct.get(i, 0) for i in range(cluster_count)],
            hole=0.3,
            marker_colors=colors[:cluster_count]
        )])
        
        fig_pie.update_layout(
            title="Time Distribution",
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Add interpretation guide
    st.caption(
        "Time Share = % of analysis period in this regime. "
        "Comm/Spec % = average positions as % of OI. "
        "Avg Episode = typical regime duration."
    )

    # Variable analysis by cluster (time series)
    st.subheader("Feature Analysis by Cluster")

    feature_options = {
        'top4_long_net': 'Top 4 Long Concentration',
        'top4_short_net': 'Top 4 Short Concentration',
        'comm_net_position': 'Commercial Net Position (% of OI)',
        'noncomm_net_position': 'Non-Commercial Net Position (% of OI)',
        'comm_long_activity': 'Commercial Long Activity',
        'comm_short_activity': 'Commercial Short Activity',
        'noncomm_long_activity': 'Non-Commercial Long Activity',
        'noncomm_short_activity': 'Non-Commercial Short Activity',
        'comm_long_trend': 'Commercial Long Trend Alignment',
        'comm_short_trend': 'Commercial Short Trend Alignment',
        'noncomm_long_trend': 'Non-Commercial Long Trend Alignment',
        'noncomm_short_trend': 'Non-Commercial Short Trend Alignment'
    }

    selected_feature = st.selectbox(
        "Select feature to analyze:",
        options=list(feature_options.keys()),
        format_func=lambda x: feature_options[x],
        index=0
    )

    # Calculate the selected feature for visualization
    feature_data = []
    for idx, row in results_df.iterrows():
        date = row['date']
        cluster = row['cluster']
        week_features = feature_matrix[idx]

        if selected_feature == 'top4_long_net':
            value = row['top4_long_net']
        elif selected_feature == 'top4_short_net':
            value = row['top4_short_net']
        elif selected_feature == 'comm_net_position':
            value = row['comm_net_position']
        elif selected_feature == 'noncomm_net_position':
            value = row['noncomm_net_position']
        elif selected_feature == 'comm_long_activity':
            value = week_features[1]  # Index 1
        elif selected_feature == 'comm_short_activity':
            value = week_features[6]  # Index 6
        elif selected_feature == 'noncomm_long_activity':
            value = week_features[11]  # Index 11
        elif selected_feature == 'noncomm_short_activity':
            value = week_features[16]  # Index 16
        elif selected_feature == 'comm_long_trend':
            value = week_features[4]  # Index 4
        elif selected_feature == 'comm_short_trend':
            value = week_features[9]  # Index 9
        elif selected_feature == 'noncomm_long_trend':
            value = week_features[14]  # Index 14
        else:  # noncomm_short_trend
            value = week_features[19]  # Index 19

        feature_data.append({
            'date': date,
            'cluster': cluster,
            'value': value
        })

    feature_df = pd.DataFrame(feature_data)

    # Create time series plot
    fig_feature = go.Figure()

    for i in range(cluster_count):
        mask = feature_df['cluster'] == i
        fig_feature.add_trace(
            go.Scatter(
                x=feature_df.loc[mask, 'date'],
                y=feature_df.loc[mask, 'value'],
                mode='markers',
                marker=dict(size=8, color=colors[i % len(colors)]),
                name=f"C{i}"
            )
        )

    fig_feature.update_layout(
        title=f"{feature_options[selected_feature]} by Cluster",
        xaxis_title="Date",
        yaxis_title=feature_options[selected_feature],
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig_feature, use_container_width=True)

    # Cluster selection analysis in expander at bottom
    with st.expander("Cluster Selection Analysis", expanded=False):
        # Create elbow plot
        fig_elbow = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Variance Explained (%)', 'Silhouette Score']
        )

        fig_elbow.add_trace(
            go.Scatter(x=list(K_range), y=variance_explained, mode='lines+markers',
                      name='Variance', line=dict(color='#2ca02c')),
            row=1, col=1
        )
        fig_elbow.add_trace(
            go.Scatter(x=list(K_range), y=silhouette_scores_list, mode='lines+markers',
                      name='Silhouette', line=dict(color='#1f77b4')),
            row=1, col=2
        )

        # Mark selected K
        fig_elbow.add_trace(
            go.Scatter(x=[cluster_count], y=[best_variance], mode='markers',
                      marker=dict(symbol='star', size=15, color='red'), showlegend=False),
            row=1, col=1
        )
        fig_elbow.add_trace(
            go.Scatter(x=[cluster_count], y=[best_silhouette], mode='markers',
                      marker=dict(symbol='star', size=15, color='red'), showlegend=False),
            row=1, col=2
        )

        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
        fig_elbow.update_yaxes(title_text="Variance Explained (%)", row=1, col=1)
        fig_elbow.update_yaxes(title_text="Silhouette Score", row=1, col=2)

        fig_elbow.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_elbow, use_container_width=True)

        # Scores table
        scores_df = pd.DataFrame({
            'K': list(K_range),
            'Variance (%)': [f"{v:.1f}" for v in variance_explained],
            'Silhouette': [f"{s:.3f}" for s in silhouette_scores_list],
            'Selected': ['*' if k == cluster_count else '' for k in K_range]
        })
        st.dataframe(scores_df, use_container_width=True, hide_index=True)

        st.caption(
            "Optimal K selected by highest silhouette score. "
            "Silhouette measures cluster separation (-1 to 1, higher is better)."
        )

    return fig