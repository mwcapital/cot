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
    
    # Explanation section
    with st.expander("📖 Understanding Market State Clusters", expanded=False):
        st.markdown("""
        ## What are Market State Clusters?
        
        Unlike behavioral clustering (which treats each category-week independently), 
        this analysis treats each **week as a complete market state** with all 4 categories together.
        
        ---
        
        ## Key Differences from Behavioral Clustering
        
        **Behavioral Clustering**: 
        - Each row = one category on one date
        - Finds when different traders act similarly
        - 104 data points for 26 weeks
        - Assumes trader behavior is consistent over time
        
        **Market State Clustering**:
        - Each row = complete market snapshot (all 4 categories)
        - Finds similar market regimes/conditions
        - 26 data points for 26 weeks
        - Recognizes that market regimes naturally evolve
        
        ---
        
        ## Why This Approach is Superior
        
        ### 🎯 Practical Actionability
        You can now identify **"what kind of market we're in"** rather than just **"what kind of traders exist"**. 
        This is immediately useful for decision-making.
        
        ### ⏰ Solves Temporal Issues
        Instead of assuming trader behavior is consistent over 10 years, you're identifying market regimes 
        that naturally evolve. Markets change - this approach embraces that reality.
        
        ### 🌐 Holistic Market View
        Each observation captures the complete market ecosystem at one moment. You see how ALL participants 
        interact together, not just individual behaviors in isolation.
        
        ---
        
        ## Features for Each Week
        
        For each of the 4 categories:
        1. Position size (% of OI) - from CFTC API
        2. Activity (week-over-week change as % of OI)
        3. Number of traders
        4. Average position per trader
        5. Direction (+1 long, -1 short) - optional
        6. Concentration (gross concentration from CFTC)
        
        Plus 2 market-wide features (same for all):
        - Top 4 Net Long Concentration
        - Top 4 Net Short Concentration
        
        **Total**: 22 features (without direction) or 26 features (with direction)
        
        ---
        
        ## The Power of Shared Concentration Features
        
        By including the same concentration values for each week, clustering reveals:
        - Which trader behaviors correlate with high/low concentration
        - How market structure influences participant behavior
        - Hidden relationships between concentration and trading patterns
        
        ---
        
        ## What This Reveals
        
        1. **Market Regimes**: Distinct market conditions (trending, ranging, crisis)
        2. **Structural Patterns**: How concentration levels affect behavior
        3. **Behavioral Correlations**: Which groups act together in different regimes
        4. **Regime Transitions**: When and how market states change
        
        ---
        
        ## The Power of Blind Clustering
        
        The key insight is that the algorithm doesn't know what categories or weeks it's clustering - 
        it only sees patterns in the data. By clustering "blindly" based on behavior alone, 
        we discover:
        
        - **True Market Regimes**: Natural groupings that emerge from participant behavior
        - **Hidden Relationships**: Which seemingly different periods share similar dynamics
        - **Behavioral Validation**: If similar behaviors cluster together, it validates our features
        
        This blind approach reveals the true structure of market behavior without preconceptions.
        """)
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cluster_count = st.selectbox(
            "Number of Market Regimes:",
            [3, 4, 5, 6],
            index=1,  # Default to 4
            help="More clusters = finer regime distinctions"
        )
        
        # Add option to show elbow plot
        show_elbow = st.checkbox(
            "Show Elbow Plot",
            value=False,
            help="Helps determine optimal number of clusters",
            key="market_state_elbow"
        )
    
    with col2:
        window_options = {
            52: "52 weeks (1 year)",
            104: "104 weeks (2 years)",
            156: "156 weeks (3 years)",
            260: "260 weeks (5 years)",
            520: "520 weeks (10 years)"
        }
        lookback_weeks = st.selectbox(
            "Analysis Window:",
            options=list(window_options.keys()),
            format_func=lambda x: window_options[x],
            index=0,  # Default to 1 year
            help="Longer windows reveal more regime types"
        )
    
    with col3:
        # Add experimental options
        st.markdown("**Experimental Options:**")
        exclude_direction = st.checkbox(
            "Exclude Direction Features",
            value=False,
            help="Remove +1/-1 direction from all categories to focus on behavior patterns",
            key="market_state_exclude_direction"
        )
        
        show_transitions = st.checkbox(
            "Show Regime Transitions",
            value=False,
            help="Track week-over-week movement between market regimes",
            key="market_state_transitions"
        )
    
    # Check for sklearn availability
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        sklearn_available = True
    except ImportError:
        sklearn_available = False
        st.warning("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")
        return None
    
    # Prepare data
    df_cluster = df.copy()
    df_cluster = df_cluster.sort_values('report_date_as_yyyy_mm_dd')
    
    # Get data for specified window
    latest_date = df_cluster['report_date_as_yyyy_mm_dd'].max()
    start_date = latest_date - pd.Timedelta(weeks=lookback_weeks)
    df_window = df_cluster[df_cluster['report_date_as_yyyy_mm_dd'] >= start_date].copy()
    
    if len(df_window) < 10:
        st.warning("Insufficient data for market state clustering (need at least 10 weeks)")
        return None
    
    # Show elbow plot if requested
    if show_elbow:
        st.subheader("📊 Elbow Plot - Finding Optimal Clusters")
        
        # Prepare data for elbow plot
        inertias = []
        silhouette_scores = []
        K = range(2, 9)
        
        # Build temporary feature matrix
        temp_features = []
        for date in df_window['report_date_as_yyyy_mm_dd'].unique():
            week_data = df_window[df_window['report_date_as_yyyy_mm_dd'] == date].iloc[0]
            features = []
            
            # Same feature extraction logic but simplified
            for category in ['comm_long', 'comm_short', 'noncomm_long', 'noncomm_short']:
                if 'long' in category:
                    features.extend([1.0 if not exclude_direction else 0])  # Direction
                else:
                    features.extend([-1.0 if not exclude_direction else 0])  # Direction
            
            temp_features.append(features)
        
        temp_matrix = np.array(temp_features)
        temp_scaler = StandardScaler()
        temp_scaled = temp_scaler.fit_transform(temp_matrix)
        
        # Calculate metrics for each K
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(temp_scaled)
            inertias.append(kmeans.inertia_)
            
            if k < len(temp_scaled):  # Ensure we have enough samples
                score = silhouette_score(temp_scaled, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        # Create elbow plot
        fig_elbow = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Elbow Method', 'Silhouette Score')
        )
        
        # Elbow plot
        fig_elbow.add_trace(
            go.Scatter(x=list(K), y=inertias, mode='lines+markers', name='Inertia'),
            row=1, col=1
        )
        
        # Silhouette score plot
        fig_elbow.add_trace(
            go.Scatter(x=list(K), y=silhouette_scores, mode='lines+markers', 
                      name='Silhouette Score', line=dict(color='orange')),
            row=1, col=2
        )
        
        # Mark selected K
        fig_elbow.add_vline(x=cluster_count, line_dash="dash", line_color="red", row=1, col=1)
        fig_elbow.add_vline(x=cluster_count, line_dash="dash", line_color="red", row=1, col=2)
        
        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
        fig_elbow.update_yaxes(title_text="Within-Cluster Sum of Squares", row=1, col=1)
        fig_elbow.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        fig_elbow.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        st.info(f"Selected K={cluster_count}. Look for the 'elbow' in the left plot and peak in the right plot.")
    
    # Build feature matrix for each week
    weekly_features = []
    
    for date in df_window['report_date_as_yyyy_mm_dd'].unique():
        week_data = df_window[df_window['report_date_as_yyyy_mm_dd'] == date].iloc[0]
        
        features = []
        
        # Commercial Long features
        comm_long_features = [
            float(week_data['pct_of_oi_comm_long_all']) if 'pct_of_oi_comm_long_all' in week_data else 0,
            abs(float(week_data['change_in_comm_long_all'])) / float(week_data['open_interest_all']) * 100 if 'change_in_comm_long_all' in week_data else 0,
            float(week_data['traders_comm_long_all']) if 'traders_comm_long_all' in week_data else 0,
            float(week_data['comm_positions_long_all']) / float(week_data['traders_comm_long_all']) if week_data['traders_comm_long_all'] > 0 else 0,
            float(week_data['conc_gross_le_4_tdr_long']) if 'conc_gross_le_4_tdr_long' in week_data else 0
        ]
        if not exclude_direction:
            comm_long_features.insert(4, 1.0)  # Direction for long
        features.extend(comm_long_features)
        
        # Commercial Short features
        comm_short_features = [
            float(week_data['pct_of_oi_comm_short_all']) if 'pct_of_oi_comm_short_all' in week_data else 0,
            abs(float(week_data['change_in_comm_short_all'])) / float(week_data['open_interest_all']) * 100 if 'change_in_comm_short_all' in week_data else 0,
            float(week_data['traders_comm_short_all']) if 'traders_comm_short_all' in week_data else 0,
            float(week_data['comm_positions_short_all']) / float(week_data['traders_comm_short_all']) if week_data['traders_comm_short_all'] > 0 else 0,
            float(week_data['conc_gross_le_4_tdr_short']) if 'conc_gross_le_4_tdr_short' in week_data else 0
        ]
        if not exclude_direction:
            comm_short_features.insert(4, -1.0)  # Direction for short
        features.extend(comm_short_features)
        
        # Non-Commercial Long features
        noncomm_long_features = [
            float(week_data['pct_of_oi_noncomm_long_all']) if 'pct_of_oi_noncomm_long_all' in week_data else 0,
            abs(float(week_data['change_in_noncomm_long_all'])) / float(week_data['open_interest_all']) * 100 if 'change_in_noncomm_long_all' in week_data else 0,
            float(week_data['traders_noncomm_long_all']) if 'traders_noncomm_long_all' in week_data else 0,
            float(week_data['noncomm_positions_long_all']) / float(week_data['traders_noncomm_long_all']) if week_data['traders_noncomm_long_all'] > 0 else 0,
            float(week_data['conc_gross_le_8_tdr_long']) if 'conc_gross_le_8_tdr_long' in week_data else 0
        ]
        if not exclude_direction:
            noncomm_long_features.insert(4, 1.0)  # Direction for long
        features.extend(noncomm_long_features)
        
        # Non-Commercial Short features
        noncomm_short_features = [
            float(week_data['pct_of_oi_noncomm_short_all']) if 'pct_of_oi_noncomm_short_all' in week_data else 0,
            abs(float(week_data['change_in_noncomm_short_all'])) / float(week_data['open_interest_all']) * 100 if 'change_in_noncomm_short_all' in week_data else 0,
            float(week_data['traders_noncomm_short_all']) if 'traders_noncomm_short_all' in week_data else 0,
            float(week_data['noncomm_positions_short_all']) / float(week_data['traders_noncomm_short_all']) if week_data['traders_noncomm_short_all'] > 0 else 0,
            float(week_data['conc_gross_le_8_tdr_short']) if 'conc_gross_le_8_tdr_short' in week_data else 0
        ]
        if not exclude_direction:
            noncomm_short_features.insert(4, -1.0)  # Direction for short
        features.extend(noncomm_short_features)
        
        # Market-wide concentration features (SAME for all categories in this week)
        top4_long_net = float(week_data['conc_net_le_4_tdr_long_all']) if 'conc_net_le_4_tdr_long_all' in week_data else 0
        top4_short_net = float(week_data['conc_net_le_4_tdr_short_all']) if 'conc_net_le_4_tdr_short_all' in week_data else 0
        
        features.extend([top4_long_net, top4_short_net])
        
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
    
    # Perform clustering
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
    for idx, date in enumerate(dates):
        week_data = df_window[df_window['report_date_as_yyyy_mm_dd'] == date].iloc[0]
        
        # Store the actual clustering features (these are aggregated across all 4 categories)
        week_features = feature_matrix[idx]
        
        # Calculate average activity across all categories (indices depend on exclude_direction)
        if exclude_direction:
            # Without direction: [pos%, activity, traders, avg_pos, conc] for each category
            activity_indices = [1, 6, 11, 16]  # Activity for each category
            avg_pos_indices = [3, 8, 13, 18]   # Avg position for each category
        else:
            # With direction: [pos%, activity, traders, avg_pos, dir, conc] for each category
            activity_indices = [1, 7, 13, 19]  # Activity for each category
            avg_pos_indices = [3, 9, 15, 21]   # Avg position for each category
        
        # Average activity across all categories
        results_df.loc[idx, 'avg_activity_all'] = np.mean([week_features[i] for i in activity_indices])
        
        # Average position size across all categories
        results_df.loc[idx, 'avg_position_size_all'] = np.mean([week_features[i] for i in avg_pos_indices])
        
        # Store individual category metrics (clustering features)
        results_df.loc[idx, 'comm_long_pct_oi'] = float(week_data['pct_of_oi_comm_long_all']) if 'pct_of_oi_comm_long_all' in week_data else 0
        results_df.loc[idx, 'comm_short_pct_oi'] = float(week_data['pct_of_oi_comm_short_all']) if 'pct_of_oi_comm_short_all' in week_data else 0
        results_df.loc[idx, 'noncomm_long_pct_oi'] = float(week_data['pct_of_oi_noncomm_long_all']) if 'pct_of_oi_noncomm_long_all' in week_data else 0
        results_df.loc[idx, 'noncomm_short_pct_oi'] = float(week_data['pct_of_oi_noncomm_short_all']) if 'pct_of_oi_noncomm_short_all' in week_data else 0
        
        # Concentration metrics (these are clustering features)
        results_df.loc[idx, 'conc_gross_le_4_long'] = float(week_data['conc_gross_le_4_tdr_long']) if 'conc_gross_le_4_tdr_long' in week_data else 0
        results_df.loc[idx, 'conc_gross_le_4_short'] = float(week_data['conc_gross_le_4_tdr_short']) if 'conc_gross_le_4_tdr_short' in week_data else 0
        
        # Derived metrics (not used in clustering)
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
        
        # Extract key metrics
        # Adjust indices based on whether direction is excluded
        if exclude_direction:
            # Without direction features, indices shift
            comm_long_activity = avg_features[1]
            comm_short_activity = avg_features[6]
            noncomm_long_activity = avg_features[11]
            noncomm_short_activity = avg_features[16]
        else:
            # With direction features
            comm_long_activity = avg_features[1]
            comm_short_activity = avg_features[7]
            noncomm_long_activity = avg_features[13]
            noncomm_short_activity = avg_features[19]
        
        # Calculate average activity
        profile['avg_activity'] = (comm_long_activity + comm_short_activity + 
                                  noncomm_long_activity + noncomm_short_activity) / 4
        
        # Just use cluster number
        profile['type'] = f"Cluster {i}"
        
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
    if avg_regime_length > 8:
        st.info(f"📊 Market regimes are persistent (avg {avg_regime_length:.1f} weeks)")
    else:
        st.info(f"🔄 Market regimes change frequently (avg {avg_regime_length:.1f} weeks)")
    
    # Add variable selector for box plot
    st.subheader("📊 Cluster Analysis Options")
    
    # Only show derived metrics that weren't directly used in clustering
    box_plot_options = {
        'comm_net_position': 'Commercial Net Position (% of OI)',
        'noncomm_net_position': 'Non-Commercial Net Position (% of OI)',
        'comm_vs_noncomm_net': 'Commercial vs Non-Commercial Spread',
        'total_traders': 'Total Number of Traders',
        'top4_long_net': 'Top 4 Net Long Concentration %',
        'top4_short_net': 'Top 4 Net Short Concentration %'
    }
    
    selected_variable = st.selectbox(
        "Select variable to analyze by cluster:",
        options=list(box_plot_options.keys()),
        format_func=lambda x: box_plot_options[x],
        index=0,  # Default to comm_net_position
        help="These metrics show how different variables behave across the identified market regimes"
    )
    
    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Cluster Timeline',
            'Cluster Distribution (PCA 2D Projection)',
            f'{box_plot_options[selected_variable]} by Cluster',
            'Cluster Statistics'
        ],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "box"}, {"type": "table"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
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
                name=f"Cluster {i}",
                legendgroup=f"cluster{i}",
                showlegend=True,  # Show in legend
                hovertemplate='%{x}<br>Cluster ' + str(i) + '<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. 2D projection using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    
    for i in range(cluster_count):
        mask = clusters == i
        profile = cluster_profiles[i]
        cluster_dates = results_df[results_df['cluster'] == i]['date']
        
        fig.add_trace(
            go.Scatter(
                x=features_2d[mask, 0],
                y=features_2d[mask, 1],
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors[i % len(colors)],
                    line=dict(width=1, color='black')
                ),
                name=f"Cluster {i}",
                legendgroup=f"cluster{i}",
                showlegend=False,  # Don't duplicate in legend
                text=[date.strftime('%Y-%m-%d') for date in cluster_dates],
                hovertemplate='%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # 3. Box plot of selected variable by cluster
    for i in range(cluster_count):
        mask = results_df['cluster'] == i
        profile = cluster_profiles[i]
        
        fig.add_trace(
            go.Box(
                y=results_df.loc[mask, selected_variable],
                name=f"Cluster {i}",
                marker_color=colors[i % len(colors)],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Regime profiles table - enhanced with detailed category breakdown
    profile_data = []
    for profile in cluster_profiles:
        profile_data.append([
            f"C{profile['cluster']}",
            f"{profile['size']}",
            # % of OI
            f"{profile['comm_long_pct']:.1f}",
            f"{profile['comm_short_pct']:.1f}",
            f"{profile['noncomm_long_pct']:.1f}",
            f"{profile['noncomm_short_pct']:.1f}",
            # Average positions
            f"{profile['comm_long_avg_pos']:.0f}",
            f"{profile['comm_short_avg_pos']:.0f}",
            f"{profile['noncomm_long_avg_pos']:.0f}",
            f"{profile['noncomm_short_avg_pos']:.0f}",
            # Trader counts
            f"{profile['comm_long_traders']:.0f}",
            f"{profile['comm_short_traders']:.0f}",
            f"{profile['noncomm_long_traders']:.0f}",
            f"{profile['noncomm_short_traders']:.0f}"
        ])
    
    # Create multi-level headers
    headers = [
        'Cluster', 'Weeks',
        'CL%', 'CS%', 'SL%', 'SS%',  # % of OI
        'CL Avg', 'CS Avg', 'SL Avg', 'SS Avg',  # Avg positions
        'CL #', 'CS #', 'SL #', 'SS #'  # Trader counts
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=headers,
                fill_color='paleturquoise',
                align='center',
                font=dict(size=9),
                height=25
            ),
            cells=dict(
                values=list(zip(*profile_data)) if profile_data else [[]]*14,
                fill_color='lavender',
                align='center',
                font=dict(size=9),
                height=20
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Market State Clustering - {instrument_name} ({lookback_weeks} weeks)",
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update axes
    fig.update_yaxes(title_text="Cluster", ticktext=[f"C{i}" for i in range(cluster_count)], 
                     tickvals=list(range(cluster_count)), row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Component 1", row=1, col=2)
    fig.update_yaxes(title_text="Component 2", row=1, col=2)
    fig.update_yaxes(title_text=box_plot_options[selected_variable], row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add legend for the enhanced table
    with st.expander("📊 Table Legend", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **% of OI Columns:**
            - CL% = Commercial Long % of OI
            - CS% = Commercial Short % of OI
            - SL% = Speculative Long % of OI
            - SS% = Speculative Short % of OI
            """)
        with col2:
            st.markdown("""
            **Average Position Columns:**
            - CL Avg = Avg contracts per Commercial Long trader
            - CS Avg = Avg contracts per Commercial Short trader
            - SL Avg = Avg contracts per Speculative Long trader
            - SS Avg = Avg contracts per Speculative Short trader
            """)
        with col3:
            st.markdown("""
            **Trader Count Columns:**
            - CL # = Number of Commercial Long traders
            - CS # = Number of Commercial Short traders
            - SL # = Number of Speculative Long traders
            - SS # = Number of Speculative Short traders
            """)
    
    # Add Market Regime Summary Table
    st.subheader("📊 Market Regime Summary")
    
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
        
        # Extract key percentages (% of OI)
        if exclude_direction:
            comm_long_pct = avg_features[0]
            comm_short_pct = avg_features[5]
            noncomm_long_pct = avg_features[10]
            noncomm_short_pct = avg_features[15]
        else:
            comm_long_pct = avg_features[0]
            comm_short_pct = avg_features[6]
            noncomm_long_pct = avg_features[12]
            noncomm_short_pct = avg_features[18]
        
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
    st.info("""
    **How to interpret:**
    - **Time Share**: What % of the analysis period was spent in this regime
    - **Comm/Spec %**: Average positions as % of open interest
    - **Avg Episode**: How long the market typically stays in this regime
    
    Look for regimes with extreme positioning (>60% or <20%) or unusual commercial/speculative alignment.
    """)
    
    # Show transitions if requested
    if show_transitions is not None:
        st.subheader("🔄 Regime Transitions")
        
        # Calculate transitions
        transitions = []
        for i in range(1, len(results_df)):
            prev_regime = results_df.iloc[i-1]['cluster']
            curr_regime = results_df.iloc[i]['cluster']
            if prev_regime != curr_regime:
                transitions.append({
                    'date': results_df.iloc[i]['date'],
                    'from': prev_regime,
                    'to': curr_regime
                })
        
        if transitions:
            # Create transitions dataframe
            trans_df = pd.DataFrame(transitions)
            
            # Show recent transitions
            st.write("**Recent Regime Changes:**")
            for _, trans in trans_df.tail(5).iterrows():
                st.write(f"📅 {trans['date'].strftime('%Y-%m-%d')}: Cluster {trans['from']} → Cluster {trans['to']}")
            
            # Transition matrix
            st.write("\n**Transition Frequency Matrix:**")
            
            # Create transition count matrix
            trans_matrix = np.zeros((cluster_count, cluster_count))
            for i in range(1, len(results_df)):
                prev = results_df.iloc[i-1]['cluster']
                curr = results_df.iloc[i]['cluster']
                trans_matrix[prev, curr] += 1
            
            # Normalize rows to get probabilities
            row_sums = trans_matrix.sum(axis=1)
            trans_prob = trans_matrix / row_sums[:, np.newaxis]
            trans_prob = np.nan_to_num(trans_prob, 0)
            
            # Create heatmap
            fig_trans = go.Figure(data=go.Heatmap(
                z=trans_prob,
                x=[f"To Cluster {i}" for i in range(cluster_count)],
                y=[f"From Cluster {i}" for i in range(cluster_count)],
                colorscale='Blues',
                text=[[f"{val:.1%}" for val in row] for row in trans_prob],
                texttemplate="%{text}",
                textfont={"size": 12}
            ))
            
            fig_trans.update_layout(
                title="Cluster Transition Probabilities",
                height=400,
                xaxis_title="To Cluster",
                yaxis_title="From Cluster"
            )
            
            st.plotly_chart(fig_trans, use_container_width=True)
        else:
            st.info("No regime transitions found in the selected period.")
    
    # Feature importance analysis
    st.subheader("🔍 Regime Characteristics")
    
    # Create heatmap of average features by regime
    if exclude_direction:
        feature_names = [
            'Comm Long %OI', 'Comm Long Activity', 'Comm Long Traders', 'Comm Long Avg Pos', 
            'Comm Long Conc',
            'Comm Short %OI', 'Comm Short Activity', 'Comm Short Traders', 'Comm Short Avg Pos',
            'Comm Short Conc',
            'NonComm Long %OI', 'NonComm Long Activity', 'NonComm Long Traders', 'NonComm Long Avg Pos',
            'NonComm Long Conc',
            'NonComm Short %OI', 'NonComm Short Activity', 'NonComm Short Traders', 'NonComm Short Avg Pos',
            'NonComm Short Conc',
            'Top 4 Long Net', 'Top 4 Short Net'
        ]
    else:
        feature_names = [
            'Comm Long %OI', 'Comm Long Activity', 'Comm Long Traders', 'Comm Long Avg Pos', 
            'Comm Long Dir', 'Comm Long Conc',
            'Comm Short %OI', 'Comm Short Activity', 'Comm Short Traders', 'Comm Short Avg Pos',
            'Comm Short Dir', 'Comm Short Conc',
            'NonComm Long %OI', 'NonComm Long Activity', 'NonComm Long Traders', 'NonComm Long Avg Pos',
            'NonComm Long Dir', 'NonComm Long Conc',
            'NonComm Short %OI', 'NonComm Short Activity', 'NonComm Short Traders', 'NonComm Short Avg Pos',
            'NonComm Short Dir', 'NonComm Short Conc',
            'Top 4 Long Net', 'Top 4 Short Net'
        ]
    
    # Calculate average SCALED features for each cluster
    cluster_feature_avgs_scaled = []
    for i in range(cluster_count):
        cluster_features_scaled = features_scaled[clusters == i]
        cluster_feature_avgs_scaled.append(cluster_features_scaled.mean(axis=0))
    
    # Create heatmap with standardized values
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=np.array(cluster_feature_avgs_scaled).T,
        x=[f"Cluster {i}" for i in range(cluster_count)],
        y=feature_names,
        colorscale='RdBu',
        zmid=0,
        zmin=-2,  # Typical range for standardized data
        zmax=2,
        colorbar=dict(title="Std. Deviations")
    ))
    
    fig_heatmap.update_layout(
        title="Standardized Feature Values by Cluster (in Standard Deviations)",
        height=800,
        xaxis_title="Cluster",
        yaxis_title="Features"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Add PCA explanation
    with st.expander("Understanding the 2D Projection", expanded=False):
        st.info("""
        **Why the 2D plot looks different from clustering:**
        
        • **Clustering happens in high dimensions** (26-30 features per week)
        • **PCA compresses to 2D** for visualization only
        • **Colors = cluster membership** (assigned in high-D space)
        • **Position = similar features** (weeks close together have similar patterns)
        
        **What validates good clustering:**
        • Distinct colored regions (same cluster weeks group together)
        • Clear separation between different colors
        • Minimal color mixing in the same area
        
        Think of it like projecting a 3D globe onto a 2D map - some distortion occurs, 
        but the essential relationships remain visible.
        """)
    
    # Stationarity Check
    st.subheader("📊 Stationarity Analysis")
    
    # Feature selector for stationarity check
    feature_options = {
        'top4_long_net': 'Top 4 Long Concentration',
        'top4_short_net': 'Top 4 Short Concentration',
        'total_activity': 'Total Market Activity',
        'comm_vs_noncomm': 'Commercial vs Non-Commercial Balance',
        'avg_position_size': 'Average Position Size (All Categories)',
        'concentration_spread': 'Long vs Short Concentration Spread',
        'regime_volatility': 'Regime Transition Frequency (12-week rolling)'
    }
    
    col1, col2 = st.columns([2, 1])
    with col1:
        stationarity_feature = st.selectbox(
            "Select Feature to Analyze:",
            options=list(feature_options.keys()),
            format_func=lambda x: feature_options[x],
            index=0,
            help="Choose which market feature to check for stability across clusters"
        )
    
    # Calculate the selected feature for visualization
    stationarity_data = []
    
    for idx, row in results_df.iterrows():
        date = row['date']
        cluster = row['cluster']
        
        if stationarity_feature == 'top4_long_net':
            value = row['top4_long_net']
        elif stationarity_feature == 'top4_short_net':
            value = row['top4_short_net']
        elif stationarity_feature == 'total_activity':
            # Calculate total activity from the feature matrix
            week_features = feature_matrix[idx]
            # Activity is at indices 1, 7, 13, 19 (or adjusted if direction excluded)
            if exclude_direction:
                activity_indices = [1, 6, 11, 16]
            else:
                activity_indices = [1, 7, 13, 19]
            value = sum(week_features[i] for i in activity_indices)
        elif stationarity_feature == 'comm_vs_noncomm':
            # Calculate commercial vs non-commercial balance
            week_features = feature_matrix[idx]
            comm_long = week_features[0]  # Comm Long %OI
            comm_short = week_features[5 if exclude_direction else 6]  # Comm Short %OI
            noncomm_long = week_features[10 if exclude_direction else 12]  # NonComm Long %OI
            noncomm_short = week_features[15 if exclude_direction else 18]  # NonComm Short %OI
            value = (comm_long + comm_short) - (noncomm_long + noncomm_short)
        elif stationarity_feature == 'avg_position_size':
            # Calculate average position size across all categories
            week_features = feature_matrix[idx]
            # Average position indices: 3, 9, 15, 21 (or adjusted if direction excluded)
            if exclude_direction:
                pos_indices = [3, 8, 13, 18]
            else:
                pos_indices = [3, 9, 15, 21]
            value = np.mean([week_features[i] for i in pos_indices])
        elif stationarity_feature == 'concentration_spread':
            # Long concentration minus short concentration
            value = row['top4_long_net'] - row['top4_short_net']
        else:  # regime_volatility
            # Calculate rolling transition frequency (how often regimes change)
            # Use a 12-week rolling window
            window_size = 12
            
            # Find the current position in the results dataframe
            current_pos = idx
            
            # Get the window of data around this point
            start_pos = max(0, current_pos - window_size // 2)
            end_pos = min(len(results_df), current_pos + window_size // 2)
            
            # Count transitions in this window
            transitions_count = 0
            window_data = results_df.iloc[start_pos:end_pos]
            
            for j in range(1, len(window_data)):
                if window_data.iloc[j]['cluster'] != window_data.iloc[j-1]['cluster']:
                    transitions_count += 1
            
            # Normalize by window size to get transitions per week
            # Multiply by 100 to make it percentage-like for better visualization
            value = (transitions_count / len(window_data)) * 100 if len(window_data) > 0 else 0
            
        stationarity_data.append({
            'date': date,
            'cluster': cluster,
            'value': value
        })
    
    stat_df = pd.DataFrame(stationarity_data)
    
    # Create stationarity plot
    fig_stat = go.Figure()
    
    for i in range(cluster_count):
        mask = stat_df['cluster'] == i
        fig_stat.add_trace(
            go.Scatter(
                x=stat_df.loc[mask, 'date'],
                y=stat_df.loc[mask, 'value'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)]
                ),
                name=f"Cluster {i}"
            )
        )
    
    fig_stat.update_layout(
        title=f"Stationarity Check: {feature_options[stationarity_feature]} by Cluster",
        xaxis_title="Date",
        yaxis_title=feature_options[stationarity_feature],
        height=400
    )
    
    st.plotly_chart(fig_stat, use_container_width=True)
    
    # Calculate stationarity statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Mean values by cluster
        st.write("**Average Values by Cluster:**")
        for i in range(cluster_count):
            cluster_mean = stat_df[stat_df['cluster'] == i]['value'].mean()
            st.metric(f"Cluster {i}", f"{cluster_mean:.2f}")
    
    with col2:
        # Variance analysis
        st.write("**Within-Cluster Variance:**")
        for i in range(cluster_count):
            cluster_std = stat_df[stat_df['cluster'] == i]['value'].std()
            st.metric(f"Cluster {i} Std Dev", f"{cluster_std:.2f}")
    
    # Show appropriate interpretation based on selected feature
    if stationarity_feature == 'regime_volatility':
        st.info("""
        **How to interpret Regime Transition Frequency:**
        - **0%** = No regime changes in 12-week window (very stable market)
        - **8-10%** = About 1 transition per 10-12 weeks (normal volatility)
        - **20%+** = Frequent regime changes (unstable/transitional market)
        - **Sudden spikes** = May signal upcoming volatility or major market shifts
        - Look for whether certain clusters tend to have higher transition rates
        """)
    else:
        st.info("""
        **How to interpret:**
        - If clusters show distinct, non-overlapping value ranges → Feature is stable and defines regimes
        - If values overlap significantly between clusters → Feature may not be stationary
        - High within-cluster variance → Regime characteristics are changing over time
        """)
    
    # Don't show key insights section anymore
    
    # Add download button for pre-clustered data at the bottom
    st.subheader("📥 Download Pre-Clustered Data")
    
    # Create download dataframe
    download_df = results_df.copy()
    
    # Add feature columns to the download
    feature_names_for_download = []
    if exclude_direction:
        feature_names_for_download = [
            'Comm_Long_%OI', 'Comm_Long_Activity', 'Comm_Long_Traders', 'Comm_Long_Avg_Pos', 
            'Comm_Long_Conc',
            'Comm_Short_%OI', 'Comm_Short_Activity', 'Comm_Short_Traders', 'Comm_Short_Avg_Pos',
            'Comm_Short_Conc',
            'NonComm_Long_%OI', 'NonComm_Long_Activity', 'NonComm_Long_Traders', 'NonComm_Long_Avg_Pos',
            'NonComm_Long_Conc',
            'NonComm_Short_%OI', 'NonComm_Short_Activity', 'NonComm_Short_Traders', 'NonComm_Short_Avg_Pos',
            'NonComm_Short_Conc',
            'Top_4_Long_Net', 'Top_4_Short_Net'
        ]
    else:
        feature_names_for_download = [
            'Comm_Long_%OI', 'Comm_Long_Activity', 'Comm_Long_Traders', 'Comm_Long_Avg_Pos', 
            'Comm_Long_Dir', 'Comm_Long_Conc',
            'Comm_Short_%OI', 'Comm_Short_Activity', 'Comm_Short_Traders', 'Comm_Short_Avg_Pos',
            'Comm_Short_Dir', 'Comm_Short_Conc',
            'NonComm_Long_%OI', 'NonComm_Long_Activity', 'NonComm_Long_Traders', 'NonComm_Long_Avg_Pos',
            'NonComm_Long_Dir', 'NonComm_Long_Conc',
            'NonComm_Short_%OI', 'NonComm_Short_Activity', 'NonComm_Short_Traders', 'NonComm_Short_Avg_Pos',
            'NonComm_Short_Dir', 'NonComm_Short_Conc',
            'Top_4_Long_Net', 'Top_4_Short_Net'
        ]
    
    # Add feature values to download dataframe
    for i, fname in enumerate(feature_names_for_download):
        download_df[fname] = feature_matrix[:, i]
    
    # Add scaled features too
    for i, fname in enumerate(feature_names_for_download):
        download_df[f"{fname}_scaled"] = features_scaled[:, i]
    
    # Convert to CSV
    csv = download_df.to_csv(index=False)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="Download Pre-Clustered Data (CSV)",
            data=csv,
            file_name=f"market_state_clusters_{instrument_name}_{lookback_weeks}weeks.csv",
            mime="text/csv",
            help="Download the raw feature matrix used for clustering along with cluster assignments"
        )
    
    st.info("""
    **Download includes:**
    - Date and cluster assignment
    - All raw feature values used in clustering
    - Scaled feature values (after standardization)
    - Top 4 concentration metrics
    
    This allows you to see exactly what data the algorithm clustered on.
    """)
    
    return fig