"""
Participant Behavior Clusters Analysis for CFTC COT Dashboard
Complete implementation from legacyF.py
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats


def create_participant_behavior_clusters(df, instrument_name):
    """Create participant behavior clustering analysis"""

    # Add tabs for different clustering approaches
    tab1, tab2 = st.tabs(["Market State Clustering", "Trend-Aware Clustering"])

    with tab1:
        from .market_state_clusters import create_market_state_clusters
        create_market_state_clusters(df, instrument_name)

    with tab2:
        create_trend_aware_clustering(df, instrument_name)


def create_trend_aware_clustering(df, instrument_name):
    """
    Trend-Aware Clustering: Adds price trend alignment to detect when
    commercials behave like speculators (trend-following vs counter-trend)
    """

    # Concise explanation
    st.caption(
        "This ML algorithm uncovers behavioral similarities between commercials and non-commercials. "
        "**When commercials show positive trend alignment, they're behaving like speculators.** "
        "It uses 4 features (Position Size, Activity, Market Share, Trend Alignment) "
        "to cluster categories blindly—without seeing if they're Commercial or Non-Commercial—based purely on behavior. "
        "Clusters are not interpretable labels—they simply group similar data points together. "
        "Analysis period: 5 years."
    )

    # Try to fetch price data
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from futures_price_fetcher import FuturesPriceFetcher

        # Get futures symbol for this instrument
        def get_futures_symbol_for_cot_instrument(inst_name):
            """Map COT instrument name to futures symbol"""
            try:
                # Strip the code in parentheses if present: "GOLD - EXCHANGE (088691)" -> "GOLD - EXCHANGE"
                clean_name = inst_name
                if ' (' in inst_name and inst_name.endswith(')'):
                    clean_name = inst_name.rsplit(' (', 1)[0]

                # Try multiple path options
                possible_paths = [
                    'instrument_management/futures/futures_symbols_enhanced.json',  # From project root
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                'instrument_management', 'futures', 'futures_symbols_enhanced.json'),  # Relative to file
                ]

                import json
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
                            # Check if either name contains the other (handles partial matches)
                            if clean_name in cot_instrument or cot_instrument in clean_name:
                                return symbol
                            # Also check the original name
                            if inst_name in cot_instrument or cot_instrument in inst_name:
                                return symbol
                return None
            except Exception as e:
                st.error(f"Mapping error: {e}")
                return None

        futures_symbol = get_futures_symbol_for_cot_instrument(instrument_name)

        if not futures_symbol:
            st.warning(f"⚠️ Could not find futures symbol mapping for '{instrument_name}'. Price data unavailable.")
            st.info("This analysis requires price data to calculate trend alignment. Please use the standard Behavioral Clustering tab instead.")
            # Debug: show what we tried to match
            clean_name = instrument_name
            if ' (' in instrument_name and instrument_name.endswith(')'):
                clean_name = instrument_name.rsplit(' (', 1)[0]
            st.caption(f"Debug: Tried to match '{clean_name}'")
            return None

        # Symbol matched (no message shown to keep UI clean)

        # Initialize price fetcher
        try:
            fetcher = FuturesPriceFetcher()
        except Exception as e:
            st.error(f"❌ Could not initialize price fetcher: {e}")
            st.info("Please ensure Supabase credentials are configured.")
            return None

        # Get date range from COT data
        df_sorted = df.sort_values('report_date_as_yyyy_mm_dd')
        start_date = df_sorted['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')
        end_date = df_sorted['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')

        # Fetch weekly price data (aligned to Tuesday)
        with st.spinner(f"Fetching price data for {futures_symbol}..."):
            price_df = fetcher.fetch_weekly_prices(futures_symbol, start_date, end_date, adjustment='NON')

        if price_df.empty:
            st.warning(f"⚠️ No price data available for {futures_symbol}")
            return None

        # Price data loaded (no message shown to keep UI clean)

    except ImportError as e:
        st.error(f"❌ Could not import price fetcher: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error fetching price data: {e}")
        return None

    # Check for sklearn
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except ImportError:
        st.warning("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")
        return None

    # Fixed 5-year analysis window
    df_cluster = df.copy()
    df_cluster = df_cluster.sort_values('report_date_as_yyyy_mm_dd')
    latest_date = df_cluster['report_date_as_yyyy_mm_dd'].max()
    start_date_filter = latest_date - pd.Timedelta(weeks=260)

    df_window = df_cluster[df_cluster['report_date_as_yyyy_mm_dd'] >= start_date_filter].copy()

    if len(df_window) < 10:
        st.warning("Insufficient data for clustering analysis")
        return None

    # Analysis window set (no message shown to keep UI clean)

    # Calculate weekly price returns
    price_df = price_df.sort_values('date')
    price_df['weekly_return'] = price_df['close'].pct_change()
    price_df['price_direction'] = np.sign(price_df['weekly_return'])

    # Create date lookup for price returns
    price_returns = {}
    for _, row in price_df.iterrows():
        price_returns[row['date'].date()] = {
            'return': row['weekly_return'],
            'direction': row['price_direction']
        }

    # Define categories
    categories = [
        ('Commercial Long', 'comm_positions_long_all', 'traders_comm_long_all', 'change_in_comm_long_all'),
        ('Commercial Short', 'comm_positions_short_all', 'traders_comm_short_all', 'change_in_comm_short_all'),
        ('Non-Commercial Long', 'noncomm_positions_long_all', 'traders_noncomm_long_all', 'change_in_noncomm_long_all'),
        ('Non-Commercial Short', 'noncomm_positions_short_all', 'traders_noncomm_short_all', 'change_in_noncomm_short_all')
    ]

    # Build feature data
    cluster_data = []
    price_match_count = 0
    price_miss_count = 0
    missing_dates = []  # Track which dates have no price data

    for idx, row in df_window.iterrows():
        date = row['report_date_as_yyyy_mm_dd']

        # Find matching price data (COT Tuesday → same week's price)
        # Try exact match first, then nearby dates
        price_info = None
        date_key = date.date() if hasattr(date, 'date') else date

        # Try exact match
        if date_key in price_returns:
            price_info = price_returns[date_key]
        else:
            # Try +/- 1-3 days to handle date misalignment
            for offset in [1, -1, 2, -2, 3, -3]:
                try:
                    nearby_date = (pd.Timestamp(date) + pd.Timedelta(days=offset)).date()
                    if nearby_date in price_returns:
                        price_info = price_returns[nearby_date]
                        break
                except:
                    continue

        if price_info:
            price_match_count += 1
            weekly_price_direction = price_info['direction']
        else:
            price_miss_count += 1
            missing_dates.append(date)  # Track this missing date
            weekly_price_direction = 0  # No price data

        for cat_name, pos_col, trader_col, change_col in categories:
            if pd.isna(row[pos_col]) or pd.isna(row[trader_col]) or row[trader_col] == 0:
                continue

            # Feature 1: Average position size (normalized by OI)
            avg_position = row[pos_col] / row[trader_col]
            avg_position_norm = avg_position / float(row['open_interest_all']) * 100

            # Feature 2: Direction (category label)
            direction = 1 if 'Long' in cat_name else -1

            # Feature 3: Activity level
            if change_col in row.index and not pd.isna(row[change_col]):
                change_value = float(row[change_col])
                activity = (abs(change_value) / float(row['open_interest_all'])) * 100
                position_change_direction = np.sign(change_value)
            else:
                activity = 0
                position_change_direction = 0

            # Feature 4: Market share (% of OI)
            pct_oi_map = {
                'Commercial Long': 'pct_of_oi_comm_long_all',
                'Commercial Short': 'pct_of_oi_comm_short_all',
                'Non-Commercial Long': 'pct_of_oi_noncomm_long_all',
                'Non-Commercial Short': 'pct_of_oi_noncomm_short_all'
            }
            pct_oi_col = pct_oi_map.get(cat_name)
            if pct_oi_col and pct_oi_col in row.index:
                position_pct_of_oi = float(row[pct_oi_col])
            else:
                position_pct_of_oi = (row[pos_col] / row['open_interest_all']) * 100

            # Feature 6: TREND ALIGNMENT (NEW!)
            # +1 = trend-following (speculative), -1 = counter-trend (hedging)
            if position_change_direction != 0 and weekly_price_direction != 0:
                trend_alignment = position_change_direction * weekly_price_direction
            else:
                trend_alignment = 0

            cluster_data.append({
                'date': date,
                'category': cat_name,
                'avg_position_norm': avg_position_norm,
                'direction': direction,
                'activity': activity,
                'position_pct_of_oi': position_pct_of_oi,
                'trend_alignment': trend_alignment,
                'position_change_dir': position_change_direction,
                'price_direction': weekly_price_direction
            })

    if len(cluster_data) == 0:
        st.warning("No valid data for clustering")
        return None

    # Price match stats tracked internally (no message shown to keep UI clean)

    # Create DataFrame
    cluster_df = pd.DataFrame(cluster_data)

    # Features for clustering (4 category-specific features)
    # trend_alignment is the KEY feature that captures hedger vs speculator behavior
    feature_cols = ['avg_position_norm', 'activity', 'position_pct_of_oi', 'trend_alignment']

    # Cluster count selector
    cluster_count = st.selectbox(
        "Number of Clusters:",
        options=[2, 3, 4],
        index=1,  # Default to 3
        key="trend_cluster_count"
    )

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cluster_df[feature_cols])

    # Calculate variance/silhouette for all K values (for analysis expander)
    kmeans_k1 = KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans_k1.fit(features_scaled)
    total_inertia = kmeans_k1.inertia_

    K_range = range(2, 5)  # Test K from 2 to 4
    inertias = []
    variance_explained = []
    silhouette_scores = []

    for k in K_range:
        temp_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        temp_kmeans.fit(features_scaled)
        inertias.append(temp_kmeans.inertia_)

        var_exp = (1 - temp_kmeans.inertia_ / total_inertia) * 100
        variance_explained.append(var_exp)

        n_unique_labels = len(set(temp_kmeans.labels_))
        if k < len(features_scaled) and n_unique_labels > 1:
            score = silhouette_score(features_scaled, temp_kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)

    # Get stats for selected K
    selected_idx = cluster_count - 2  # K=2 is index 0
    best_variance = variance_explained[selected_idx]
    best_silhouette = silhouette_scores[selected_idx]

    # Perform clustering with auto-selected K
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    cluster_df['cluster'] = kmeans.fit_predict(features_scaled)

    # Analyze cluster profiles
    cluster_profiles = []
    for i in range(cluster_count):
        cluster_data_i = cluster_df[cluster_df['cluster'] == i]
        profile = {
            'cluster': i,
            'size': len(cluster_data_i),
            'avg_position': cluster_data_i['avg_position_norm'].mean(),
            'direction_bias': cluster_data_i['direction'].mean(),
            'activity': cluster_data_i['activity'].mean(),
            'position_pct_of_oi': cluster_data_i['position_pct_of_oi'].mean(),
            'trend_alignment': cluster_data_i['trend_alignment'].mean()
        }

        # Type classification (kept for internal use but not displayed)
        if profile['trend_alignment'] > 0.3:
            profile['type'] = "Trend-Following"
        elif profile['trend_alignment'] < -0.3:
            profile['type'] = "Counter-Trend"
        else:
            profile['type'] = "Neutral"

        cluster_profiles.append(profile)

    # Visualizations
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Cluster Distribution by Category',
            'Trend Alignment by Category (Key Insight!)',
            '',
            'Cluster Profiles'
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar", "colspan": 1}, {"type": "table"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 1. Cluster distribution by category
    cluster_counts = cluster_df.groupby(['category', 'cluster']).size().reset_index(name='count')
    for i in range(cluster_count):
        cluster_i = cluster_counts[cluster_counts['cluster'] == i]
        profile = next(p for p in cluster_profiles if p['cluster'] == i)
        fig.add_trace(
            go.Bar(
                x=cluster_i['category'],
                y=cluster_i['count'],
                name=f"C{i}",
                marker_color=colors[i % len(colors)]
            ),
            row=1, col=1
        )

    # 2. Trend alignment by category (THE KEY CHART!)
    trend_by_category = cluster_df.groupby('category')['trend_alignment'].mean().reset_index()
    bar_colors = ['#2ca02c' if x > 0 else '#d62728' for x in trend_by_category['trend_alignment']]
    fig.add_trace(
        go.Bar(
            x=trend_by_category['category'],
            y=trend_by_category['trend_alignment'],
            marker_color=bar_colors,
            showlegend=False,
            text=[f"{x:.2f}" for x in trend_by_category['trend_alignment']],
            textposition='outside'
        ),
        row=1, col=2
    )

    # Add reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    # 3. Trend alignment over time (more useful than scatter)
    trend_over_time = cluster_df.groupby(['date', 'category'])['trend_alignment'].mean().reset_index()
    # Show just Commercial Long vs Non-Commercial Long for clarity
    for cat, color, dash in [('Commercial Long', '#1f77b4', 'solid'), ('Non-Commercial Long', '#2ca02c', 'dash')]:
        cat_data = trend_over_time[trend_over_time['category'] == cat].sort_values('date')
        if len(cat_data) > 0:
            # Rolling average for smoother line
            cat_data['trend_smooth'] = cat_data['trend_alignment'].rolling(window=8, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=cat_data['date'],
                    y=cat_data['trend_smooth'],
                    mode='lines',
                    name=cat,
                    line=dict(color=color, dash=dash),
                    showlegend=True
                ),
                row=2, col=1
            )

    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Trend Alignment (8-wk avg)", row=2, col=1)

    # 4. Cluster Profiles table
    profile_data = []
    for profile in cluster_profiles:
        profile_data.append([
            f"C{profile['cluster']}",
            f"{profile['size']}",
            f"{profile['activity']:.1f}",
            f"{profile['trend_alignment']:.2f}",
            f"{profile['position_pct_of_oi']:.1f}%"
        ])

    fig.add_trace(
        go.Table(
            header=dict(
                values=['Cluster', 'N', 'Activity', 'Trend Align', '% OI'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=list(zip(*profile_data)) if profile_data else [[]]*5,
                fill_color='lavender',
                align='left'
            )
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"Trend-Aware Clustering: {instrument_name}",
        height=800,
        showlegend=True
    )

    fig.update_xaxes(title_text="Category", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Category", row=1, col=2)
    fig.update_yaxes(title_text="Avg Trend Alignment", row=1, col=2)
    fig.update_xaxes(title_text="Activity Level", row=2, col=1)
    fig.update_yaxes(title_text="Trend Alignment", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Category Mix table (right below Cluster Profiles)
    st.caption("**Category Mix per Cluster** - If Commercials and Non-Commercials appear in the same cluster, they're showing similar behavior.")
    category_dist = cluster_df.groupby(['cluster', 'category']).size().unstack(fill_value=0)
    category_pct = category_dist.div(category_dist.sum(axis=1), axis=0) * 100

    dist_data = []
    for cluster_id in range(cluster_count):
        if cluster_id in category_pct.index:
            row = {'Cluster': f"C{cluster_id}"}
            for cat in ['Commercial Long', 'Commercial Short', 'Non-Commercial Long', 'Non-Commercial Short']:
                if cat in category_pct.columns:
                    pct = category_pct.loc[cluster_id, cat]
                    row[cat.replace('Commercial', 'Comm').replace('Non-Comm', 'NC')] = f"{pct:.0f}%"
                else:
                    row[cat.replace('Commercial', 'Comm').replace('Non-Comm', 'NC')] = "0%"
            dist_data.append(row)

    if dist_data:
        dist_df = pd.DataFrame(dist_data)
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # Show cluster selection analysis (elbow and silhouette plots)
    st.markdown("---")
    with st.expander("📊 Cluster Selection Analysis", expanded=False):
        st.markdown(f"""
        **K = {cluster_count}** selected. Variance Explained: {best_variance:.0f}%, Silhouette Score: {best_silhouette:.2f}

        - **Variance Explained** (left): Higher = more data variance captured.
        - **Silhouette Score** (right): Higher = better cluster separation.
        """)

        from plotly.subplots import make_subplots as make_subplots_elbow
        fig_elbow = make_subplots_elbow(
            rows=1, cols=2,
            subplot_titles=['Variance Explained (%)', 'Silhouette Score']
        )

        # Variance explained plot (more intuitive than raw inertia)
        fig_elbow.add_trace(
            go.Scatter(x=list(K_range), y=variance_explained, mode='lines+markers',
                      name='Variance', line=dict(color='#2ca02c')),
            row=1, col=1
        )
        # Silhouette plot
        fig_elbow.add_trace(
            go.Scatter(x=list(K_range), y=silhouette_scores, mode='lines+markers',
                      name='Silhouette', line=dict(color='#1f77b4')),
            row=1, col=2
        )

        # Mark selected K with a star marker
        fig_elbow.add_trace(
            go.Scatter(x=[cluster_count], y=[best_variance], mode='markers',
                      marker=dict(symbol='star', size=15, color='red'),
                      name='Selected', showlegend=False),
            row=1, col=1
        )
        fig_elbow.add_trace(
            go.Scatter(x=[cluster_count], y=[best_silhouette], mode='markers',
                      marker=dict(symbol='star', size=15, color='red'),
                      name='Selected', showlegend=False),
            row=1, col=2
        )

        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
        fig_elbow.update_yaxes(title_text="Variance Explained (%)", row=1, col=1)
        fig_elbow.update_yaxes(title_text="Silhouette Score", row=1, col=2)

        fig_elbow.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_elbow, use_container_width=True)

        # Show scores table
        scores_df = pd.DataFrame({
            'K': list(K_range),
            'Variance (%)': [f"{v:.1f}" for v in variance_explained],
            'Silhouette': [f"{s:.3f}" for s in silhouette_scores],
            'Selected': ['⭐' if k == cluster_count else '' for k in K_range]
        })
        st.dataframe(scores_df, use_container_width=True, hide_index=True)

        # Show missing price data dates
        st.markdown("---")
        total_weeks = price_match_count + price_miss_count
        st.markdown(f"**Price Data Coverage:** {price_match_count}/{total_weeks} weeks matched")
        if missing_dates:
            st.markdown(f"**Missing dates ({len(missing_dates)}):**")
            missing_str = ", ".join([d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in sorted(missing_dates)])
            st.caption(missing_str)
        else:
            st.success("All weeks have price data")

    return fig