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
    tab1, tab2 = st.tabs(["Behavioral Clustering", "Market State Clustering"])
    
    with tab1:
        create_behavioral_clustering(df, instrument_name)
    
    with tab2:
        from .market_state_clusters import create_market_state_clusters
        create_market_state_clusters(df, instrument_name)


def create_behavioral_clustering(df, instrument_name):
    """Original behavioral clustering approach"""
    
    # Explanation section
    with st.expander("📖 Understanding Participant Behavior Clusters", expanded=False):
        st.markdown("""
        ## What are Participant Behavior Clusters?
        
        This analysis uses K-means machine learning to group market participants based on their trading behavior. 
        Think of it like sorting traders into "personality types" based on how they act in the market.
        
        ---
        
        ## The 5 Key Behaviors Measured
        
        ### 1. **Position Size (Average per Trader)**
        - **What**: Average contracts per trader, normalized by total open interest
        - **Calculation**: `(Total Positions / Number of Traders) / Open Interest × 100`
        - **Example**: 50,000 contracts ÷ 100 traders ÷ 500,000 OI × 100 = 0.1%
        - **Insight**: Identifies whether few large traders or many small traders dominate
        
        ### 2. **Direction (+1 Long, -1 Short)**
        - **What**: Simply labels whether this is a long or short position
        - **Values**: Long positions = +1, Short positions = -1
        - **Purpose**: Helps algorithm separate bullish from bearish behaviors
        - **In Clusters**: Average near 0 = mixed, near +1 = mostly longs, near -1 = mostly shorts
        
        ### 3. **Activity Level (Week-over-Week Changes)**
        - **What**: How much positions changed from last week (from CFTC API)
        - **Calculation**: `|change_in_positions| / Open Interest × 100`
        - **Example**: 15,000 contract change ÷ 200,000 OI × 100 = 7.5% activity
        - **Interpretation**:
          - 0-2%: Holding steady
          - 2-5%: Normal adjustments
          - 5-10%: Active trading
          - >10%: Major repositioning
        
        ### 4. **Concentration (Based on Trader Count)**
        - **What**: How concentrated the market is among few vs many traders
        - **Calculation**: `100 - percentile_rank(trader_count)`
        - **Example**: If 30 traders is 10th percentile historically → 90% concentration
        - **Interpretation**:
          - 80-100: Very few traders (oligopoly)
          - 60-80: Concentrated
          - 40-60: Normal
          - 0-40: Many traders (democratic)
        
        ### 5. **Market Share (% of Total Open Interest)**
        - **What**: What percentage of the ENTIRE market this position represents
        - **Source**: Direct from CFTC API (pct_of_oi fields)
        - **Example**: Commercial Longs = 12% of total open interest
        - **Why it matters**: Shows actual market impact, not just relative size
        
        ---
        
        ## How Clustering Works
        
        1. **Data Collection**: Takes all 4 categories × N weeks of data
           - 26 weeks = 104 data points
           - 1 year = 208 data points
           - 2 years = 416 data points
           - 5 years = 1,040 data points
           - 10 years = 2,080 data points
        2. **Feature Matrix**: Each row = [size, direction, activity, concentration, market_share]
        3. **Standardization**: Scales all features to comparable ranges
        4. **K-means**: Groups similar rows together into K clusters (usually 4)
        5. **Analysis**: Labels clusters based on their average characteristics
        
        ### Sample Data Structure (Before Clustering)
        
        | Date | Category | Avg Position | Direction | Activity | Concentration | % of OI |
        |------|----------|--------------|-----------|----------|---------------|---------|
        | 2024-01-01 | Commercial Long | 0.8 | +1 | 2.5 | 45 | 12.3 |
        | 2024-01-01 | Commercial Short | 0.6 | -1 | 1.2 | 52 | 18.7 |
        | 2024-01-01 | Non-Comm Long | 1.2 | +1 | 5.8 | 23 | 15.4 |
        | 2024-01-01 | Non-Comm Short | 0.9 | -1 | 6.1 | 28 | 14.2 |
        | 2024-01-08 | Commercial Long | 0.7 | +1 | 3.1 | 43 | 11.9 |
        | 2024-01-08 | Commercial Short | 0.6 | -1 | 0.8 | 54 | 19.1 |
        | ... | ... | ... | ... | ... | ... | ... |
        
        **Total rows**: 4 categories × N weeks (varies by window selected)
        
        ### After Clustering (Same Data + Cluster Assignment)
        
        | Date | Category | Avg Position | Direction | Activity | Concentration | % of OI | **Cluster** |
        |------|----------|--------------|-----------|----------|---------------|---------|-------------|
        | 2024-01-01 | Commercial Long | 0.8 | +1 | 2.5 | 45 | 12.3 | **2** |
        | 2024-01-01 | Commercial Short | 0.6 | -1 | 1.2 | 52 | 18.7 | **1** |
        | 2024-01-01 | Non-Comm Long | 1.2 | +1 | 5.8 | 23 | 15.4 | **0** |
        | 2024-01-01 | Non-Comm Short | 0.9 | -1 | 6.1 | 28 | 14.2 | **0** |
        | 2024-01-08 | Commercial Long | 0.7 | +1 | 3.1 | 43 | 11.9 | **2** |
        | 2024-01-08 | Commercial Short | 0.6 | -1 | 0.8 | 54 | 19.1 | **1** |
        
        **The Magic**: The algorithm finds patterns across TIME and CATEGORIES. Notice how:
        - Commercial Shorts from different weeks (rows 2 & 6) both got Cluster 1 (similar behavior)
        - Non-Comm Long & Short from same week got Cluster 0 (both acting aggressively)
        - The algorithm doesn't "know" these are different categories - it just sees similar numbers!
        
        ### Why Each Category-Week is a Separate Row
        
        **The Logic**: The algorithm treats each category independently to find behavioral patterns that 
        transcend time. It asks: "When do Commercial Longs behave like Non-Commercial Shorts?"
        
        **Why This Approach Makes Sense**:
        
        1. **Behavioral Patterns**: We want to know when different trader types exhibit similar behavior
           - Example: When do commercials act like speculators?
           - When do hedgers become aggressive?
        
        2. **Cross-Category Insights**: If CommShort and NonCommLong cluster together, they're showing 
           similar behavior despite being theoretical opposites
           - This often signals major market shifts
           - Natural opponents acting alike = market stress or transition
        
        3. **Temporal Patterns**: We can see if CommLong behavior in week 1 matches CommLong behavior in week 20
           - Identifies regime changes over time
           - Shows evolution of trading strategies
        
        **Alternative Approach**: We could analyze "market states" with one row per week containing all 
        4 categories together. That would show different market regimes but wouldn't reveal when 
        different trader types act similarly.
        
        **The Power**: This approach reveals hidden patterns - like when natural opponents start acting 
        the same way, which often precedes major market moves!
        
        ---
        
        ## Reading the Charts
        
        ### Cluster Distribution (Top Left)
        - Shows how many data points from each category fall into each cluster
        - High bars = this category often exhibits this behavior
        - Reveals typical behavior patterns for each trader type
        
        ### 2D Projection (Top Right)
        - Visualizes all data points compressed to 2D using PCA
        - Close dots = similar behavior
        - Separated groups = distinct trading styles
        
        ### Temporal Evolution (Bottom Left)
        - Shows cluster sizes changing over time
        - Rising lines = this behavior becoming more common
        - Reveals regime shifts and behavioral trends
        
        ### Cluster Profiles (Bottom Right)
        - Statistical summary of each cluster
        - Members: Total data points in cluster
        - Other columns: Average values for each feature
        - Helps understand what makes each cluster unique
        
        ---
        
        ## Trader Types (Auto-Identified)
        
        **Primary Types (Assigned First)**:
        
        **🐋 Large Speculators**
        - Highest concentration + >60% concentration
        - Few traders controlling large positions
        
        **🚀 Aggressive Traders**
        - Highest activity + directional bias > 0.5
        - Making big, fast moves in one direction
        
        **⚡ Market Makers**
        - Highest activity + balanced direction
        - Trading both sides actively
        
        **🛡️ Hedgers**
        - Lowest activity + balanced direction
        - Stable, protective positions
        
        **Secondary Types (For Remaining Clusters)**:
        
        **📈 Directional Momentum**
        - Above median activity + strong direction (>0.7)
        - Trending positions but not most aggressive
        
        **💰 Major Position Holders**
        - High % of open interest (>70th percentile)
        - Significant market share but not highest concentration
        
        **⚖️ Balanced Traders**
        - Low directional bias (<0.3)
        - Neither aggressive nor defensive
        
        **📊 Trend Followers**
        - Default for clusters with no standout features
        - Moderate activity and direction
        
        ---
        
        ## Why This Matters
        
        1. **Regime Detection**: When behaviors shift between clusters → market dynamics changing
        2. **Risk Assessment**: Many aggressive traders → expect volatility
        3. **Crowding**: Low diversity → everyone doing the same thing (risky)
        4. **Opportunities**: Behavioral extremes often precede reversals
        
        **Key Insight**: This reveals WHO is in control of the market and HOW they're behaving!
        
        ---
        
        ## Understanding Cluster Count Selection
        
        ### What Changing Clusters Does
        
        Think of it like sorting a deck of cards:
        - **3 clusters**: Sort into "High, Medium, Low" (broad categories)
        - **4 clusters**: Sort into suits (balanced detail)
        - **5 clusters**: Add wildcards (capture outliers)
        
        ### When to Use Each Setting
        
        **3 Clusters (Low Resolution)**
        - **Use when**: You want the big picture
        - **Shows**: Only very different behaviors
        - **Example**: Aggressive (>10% activity), Defensive (<1%), Everything else
        - **Best for**: Crisis periods when behaviors are extreme
        
        **4 Clusters (Balanced - Default)**
        - **Use when**: Normal market analysis
        - **Shows**: Nuanced behavior types
        - **Example**: Aggressive, Market Makers, Hedgers, Trend Followers
        - **Best for**: Most situations - good detail without overcomplication
        
        **5 Clusters (High Resolution)**
        - **Use when**: You see one huge cluster that needs splitting
        - **Shows**: Fine distinctions and outliers
        - **Example**: Splits "Aggressive" into "Long Aggressive" vs "Short Aggressive"
        - **Best for**: Calm markets with subtle strategy differences
        
        ### How to Choose
        
        1. **Start with 4**, then adjust based on:
           - **2D Chart**: Overlapping groups → use fewer; Scattered groups → use more
           - **Cluster Sizes**: Tiny cluster (<5%) → use fewer; Huge cluster (>40%) → use more
           - **Profiles Table**: Similar profiles → use fewer; High variance → use more
        
        2. **Red Flags**:
           - Empty clusters = too many
           - One cluster has 60%+ = too few
           - Identical profiles = too many
        
        ### Pro Tip: Multi-Pass Analysis
        
        1. Run with 3 to see major groups
        2. If one group is huge, rerun with 5 to split it
        3. Compare what the split reveals
        
        **Example**: 3 clusters shows 60% are "Trend Followers". With 5 clusters, this splits into:
        - Momentum chasers (25%)
        - Slow trend followers (20%)  
        - Mean reversion traders (15%)
        
        This reveals three different strategies hidden in one group!
        
        **The Power**: Comparing different cluster counts shows how behaviors split or merge, 
        revealing hidden sub-strategies and market complexity.
        
        ---
        
        ## 🎯 End Goal: Current Behavior Identification
        
        **The ultimate purpose** is to answer: **"What behavior is each trader category exhibiting RIGHT NOW?"**
        
        For example, today's analysis might reveal:
        - **Commercial Long**: Currently in Cluster 2 → "🛡️ Hedgers" (low activity, balanced)
        - **Commercial Short**: Currently in Cluster 1 → "⚡ Market Makers" (high activity)
        - **Non-Commercial Long**: Currently in Cluster 0 → "🐋 Large Speculators" (concentrated)
        - **Non-Commercial Short**: Currently in Cluster 3 → "📈 Trend Followers"
        
        This tells you the **current market dynamics** and helps predict likely behaviors.
        
        ---
        
        ## 💡 The Key Insight: Blind Clustering Reveals True Behaviors
        
        ### Traditional Analysis vs. Clustering
        
        **Traditional:** "Let's analyze Commercial Longs"
        - Assumes all Commercial Longs behave the same
        - Might miss that some act like speculators
        
        **Clustering:** "Let's find similar behaviors, THEN see who's doing them"
        - Algorithm is BLIND to categories - just sees feature patterns
        - Groups similar behaviors together
        - THEN reveals which categories are in each group
        
        ### Why This Matters
        
        The clustering might reveal that "Hedgers" cluster contains:
        - 80% Commercial Longs (expected)
        - 15% Commercial Shorts (expected)
        - 5% Non-Commercial Shorts (surprising!)
        
        **The insight:** Those Non-Commercial Shorts are behaving like hedgers, not speculators!
        
        ### Real-World Example
        
        You might discover Commercial Longs split across 3 clusters:
        - Some in "Hedgers" → Traditional commodity hedging
        - Some in "Speculators" → Acting like hedge funds
        - Some in "Market Makers" → Providing liquidity
        
        **The power:** Reveals that regulatory categories don't always predict trading behavior. The algorithm finds TRUE behavioral groups based on how they actually trade, not their label.
        
        ---
        
        ## 📋 Recommended Analysis Workflow
        
        ### Step 1: Check Temporal Stability (Compare All Windows)
        - Start with **"Compare All Windows"** to see if behaviors are consistent over time
        - Look for stable vs shifting patterns
        - If stable → current analysis is reliable
        - If changing → market regime may be transitioning
        
        ### Step 2: Analyze Current Period (Last 3 Years)
        - Switch to **"Last 3 years (T-3 to T)"** for current analysis
        - Check the **Cluster Distribution** to see which categories are in which clusters
        - Review **Cluster Profiles** table to understand what each cluster represents
        
        ### Step 3: Identify Current Behaviors
        - Look at **Week-over-Week Cluster Assignments** (enable checkbox)
        - This shows you EXACTLY which cluster each category is in RIGHT NOW
        - Match cluster numbers to their behavioral profiles
        
        ### Step 4: Validate and Refine
        - Use **Stationarity Analysis** to ensure features are reliable
        - Try different **Comparison Views** to understand relationships
        - Check **Show Elbow Plot** to confirm K=4 is optimal
        
        ### Example Interpretation:
        If Commercial Long is in the "Hedgers" cluster with low activity and Commercial Short is in "Market Makers" with high activity, this suggests:
        - Longs are holding defensive positions (typical hedging)
        - Shorts are actively trading (providing liquidity)
        - Market may be in a risk-off environment
        
        ---
        
        ## 🎨 Understanding the 2D Projection Plot
        
        **Important: PCA components and K-means clusters are two separate things!**
        
        ### What's Actually Happening:
        
        1. **K-means clustering** (happens first):
           - Works in 5D space using all features
           - Assigns each data point to a cluster (determines the colors)
           
        2. **PCA projection** (happens second):
           - Takes the 5D data and creates a 2D visualization
           - Preserves as much information as possible
           - This is ONLY for visualization
        
        3. **The colors** show which cluster each data point belongs to (from K-means)
           - The position shows where it lands in PCA 2D space
        
        ### Why Do Colors Form Distinct Regions?
        
        This is actually **validation that clustering worked well!**
        
        - K-means found 4 groups in 5D space
        - When PCA projects to 2D, those groups STILL appear separated
        - This means the clusters are genuinely different in behavior
        
        **If clustering was bad**, you'd see all colors mixed together randomly.
        
        ### Think of it Like This:
        
        1. You have ~600 data points (4 categories × ~150 weeks)
        2. K-means looks at all 5 features and says "these 150 points are similar" → Color them blue
        3. PCA then says "let me show all 600 points on a 2D map"
        4. The fact that blue points end up near each other on the map confirms they really are similar
        
        **The key insight**: The colors were assigned in 5D space, but they remain grouped in 2D space, which validates that the behavioral clusters are real and meaningful!
        
        **What the plot shows:**
        - **Each dot** = One category in one week (e.g., "Commercial Long on 2024-01-15")
        - **Colors** = Which cluster that data point belongs to
        - **Distance between dots** = How similar their behaviors are
        - **Tight clusters** = Consistent, well-defined behaviors
        - **Spread clusters** = More variable behaviors
        
        **Limitation:** You can't directly interpret what Component 1 or 2 mean - use Cluster Profiles table for that!
        
        ---
        
        ## 🧪 Advanced Considerations
        
        ### DTW vs Euclidean Distance
        
        **Current Implementation**: Uses Euclidean distance (K-means default)
        - Measures straight-line distance between points in feature space
        - Fast and works well for behavioral similarity at each point in time
        
        **DTW (Dynamic Time Warping)** would be useful if:
        - You want to match similar patterns that occur at different times
        - Example: "Did commercials show similar behavior patterns but 2 weeks later?"
        - More computationally expensive
        - Better for time-series pattern matching
        
        **For this analysis**: Euclidean is appropriate because we're comparing behaviors at the same point in time, not looking for time-shifted patterns.
        
        ### Excluding Direction Feature
        
        **Why experiment without direction?**
        - See if Long and Short positions of the same category type show similar behaviors
        - Example: Do Commercial Longs and Commercial Shorts both act as "Hedgers"?
        - Reveals behavior patterns independent of market bias
        
        **What to expect:**
        - More mixing of Long/Short in same clusters
        - Clusters defined by activity, concentration, and size rather than direction
        - May reveal that some Longs and Shorts behave identically (just opposite sides)
        
        ---
        
        ## Key Formulas (Copyable)
        """)
        
        st.code("""
# Feature Calculations:
1. Average Position (normalized) = (Positions / Traders) / Open_Interest × 100
2. Direction = +1 for Long, -1 for Short  
3. Activity = |change_in_positions| / Open_Interest × 100
4. Concentration = 100 - percentile_rank(trader_count)
5. Market Share = pct_of_oi from CFTC API

# Clustering Process:
- Total data points = 4 categories × N weeks
- Features are standardized: (value - mean) / std_dev
- K-means minimizes within-cluster sum of squares
        """, language='python')
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Add option to show elbow plot
        show_elbow = st.checkbox(
            "Show Elbow Plot",
            value=False,
            help="Helps determine optimal number of clusters"
        )
        
        cluster_count = st.selectbox(
            "Number of Clusters:",
            [3, 4, 5],
            index=1,  # Default to 4
            help="More clusters = finer behavioral distinctions"
        )
    
    with col2:
        # Create rolling 3-year windows
        window_options = {
            "last_52": "Last 52 weeks (1 year)",
            "0_to_3": "Last 3 years (T-3 to T)",
            "3_to_6": "Previous 3 years (T-6 to T-3)",
            "6_to_9": "Earlier 3 years (T-9 to T-6)",
            "9_to_12": "Historical 3 years (T-12 to T-9)",
            "compare_all": "Compare All Windows"
        }
        window_selection = st.selectbox(
            "Analysis Window:",
            options=list(window_options.keys()),
            format_func=lambda x: window_options[x],
            index=0,
            help="""Compare clustering results across different 3-year periods to test stationarity.
            If clusters are consistent across periods, patterns are stable.
            If clusters change significantly, market regimes may be shifting."""
        )
    
    with col3:
        # Feature selector for stationarity check
        feature_options = {
            'avg_position_norm': 'Average Position (Normalized)',
            'activity': 'Activity Level',
            'concentration': 'Concentration Score',
            'position_pct_of_oi': 'Position % of OI'
        }
        stationarity_feature = st.selectbox(
            "Stationarity Check Feature:",
            options=list(feature_options.keys()),
            format_func=lambda x: feature_options[x],
            index=1,  # Default to activity
            help="Select which feature to check for stationarity over time"
        )
        
        # Add option to exclude direction feature
        st.markdown("**Experimental Options:**")
        exclude_direction = st.checkbox(
            "Exclude Direction Feature",
            value=False,
            help="Remove +1/-1 direction from clustering to focus on behavior regardless of long/short"
        )
        
        # Add comparison plot feature selector
        if window_selection == "compare_all":
            comparison_options = {
                'activity_vs_position': 'Activity vs Position Size (Default)',
                'concentration_vs_activity': 'Concentration vs Activity',
                'concentration_vs_market_share': 'Concentration vs Market Share',
                'position_vs_market_share': 'Position Size vs Market Share',
                'activity_vs_market_share': 'Activity vs Market Share'
            }
            comparison_view = st.selectbox(
                "Comparison View:",
                options=list(comparison_options.keys()),
                format_func=lambda x: comparison_options[x],
                index=0,
                help="Select which features to compare across windows"
            )
        else:
            comparison_view = None
        
        show_transitions = st.checkbox(
            "Show Cluster Transitions",
            value=False,
            help="Track week-over-week movement between clusters"
        )
    
    # Check for sklearn availability
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        sklearn_available = True
    except ImportError:
        sklearn_available = False
        st.warning("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")
        return None
    
    # Define categories to analyze (moved up to be available for all code paths)
    categories = [
        ('Commercial Long', 'comm_positions_long_all', 'traders_comm_long_all'),
        ('Commercial Short', 'comm_positions_short_all', 'traders_comm_short_all'),
        ('Non-Commercial Long', 'noncomm_positions_long_all', 'traders_noncomm_long_all'),
        ('Non-Commercial Short', 'noncomm_positions_short_all', 'traders_noncomm_short_all')
    ]
    
    # Prepare data
    df_cluster = df.copy()
    df_cluster = df_cluster.sort_values('report_date_as_yyyy_mm_dd')
    
    # Get data for specified window
    latest_date = df_cluster['report_date_as_yyyy_mm_dd'].max()
    
    # Handle compare_all option separately
    if window_selection == "compare_all":
        # Run clustering for all windows and compare results
        st.subheader("🔄 Comparing Clustering Results Across Time Windows")
        
        window_results = []
        windows = [
            ("last_52", "Last 52 weeks", 0, 52),
            ("0_to_3", "Last 3 years", 0, 156),
            ("3_to_6", "Previous 3 years", 156, 312),
            ("6_to_9", "Earlier 3 years", 312, 468),
            ("9_to_12", "Historical 3 years", 468, 624)
        ]
        
        for window_key, window_name, start_offset, end_offset in windows:
            window_end = latest_date - pd.Timedelta(weeks=start_offset)
            window_start = latest_date - pd.Timedelta(weeks=end_offset)
            
            df_test = df_cluster[
                (df_cluster['report_date_as_yyyy_mm_dd'] >= window_start) & 
                (df_cluster['report_date_as_yyyy_mm_dd'] <= window_end)
            ].copy()
            
            if len(df_test) >= 10:  # Need minimum data
                window_results.append({
                    'window': window_name,
                    'start': df_test['report_date_as_yyyy_mm_dd'].min(),
                    'end': df_test['report_date_as_yyyy_mm_dd'].max(),
                    'weeks': len(df_test),
                    'data': df_test
                })
        
        if len(window_results) < 2:
            st.warning("Not enough historical data to compare multiple windows")
            return None
        
        # Store comparison summary for later display
        comparison_df = pd.DataFrame([
            {
                'Window': r['window'],
                'Period': f"{r['start'].strftime('%Y-%m')} to {r['end'].strftime('%Y-%m')}",
                'Weeks': r['weeks']
            }
            for r in window_results
        ])
        
        # Run clustering for each window and compare
        st.info(f"Running cluster analysis across {len(window_results)} time windows...")
        
        # Process each window separately
        all_window_clusters = []
        
        for window_result in window_results:
            window_name = window_result['window']
            df_window_temp = window_result['data']
            
            # Build features for this window
            window_cluster_data = []
            
            for idx, row in df_window_temp.iterrows():
                date = row['report_date_as_yyyy_mm_dd']
                
                for cat_name, pos_col, trader_col in categories:
                    if pd.isna(row[pos_col]) or pd.isna(row[trader_col]) or row[trader_col] == 0:
                        continue
                    
                    # Calculate all features (same as main clustering)
                    avg_position = row[pos_col] / row[trader_col]
                    avg_position_norm = avg_position / float(row['open_interest_all']) * 100
                    
                    direction = 1 if 'Long' in cat_name else -1
                    
                    change_map = {
                        'Commercial Long': 'change_in_comm_long_all',
                        'Commercial Short': 'change_in_comm_short_all',
                        'Non-Commercial Long': 'change_in_noncomm_long_all',
                        'Non-Commercial Short': 'change_in_noncomm_short_all'
                    }
                    
                    change_col = change_map.get(cat_name)
                    if change_col and change_col in row.index and not pd.isna(row[change_col]):
                        change_value = abs(float(row[change_col]))
                        activity = (change_value / float(row['open_interest_all'])) * 100
                    else:
                        activity = 0
                    
                    hist_data = df_cluster[df_cluster['report_date_as_yyyy_mm_dd'] >= pd.Timestamp('2010-01-01')]
                    if len(hist_data[trader_col].dropna()) > 0:
                        trader_pct = stats.percentileofscore(hist_data[trader_col].dropna(), row[trader_col])
                        concentration = 100 - trader_pct
                    else:
                        concentration = 50
                    
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
                        position_pct_of_oi = (row[pos_col] / row['open_interest_all']) * 100 if row['open_interest_all'] > 0 else 0
                    
                    window_cluster_data.append({
                        'date': date,
                        'category': cat_name,
                        'avg_position_norm': avg_position_norm,
                        'direction': direction,
                        'activity': activity,
                        'concentration': concentration,
                        'position_pct_of_oi': position_pct_of_oi
                    })
            
            if len(window_cluster_data) > 0:
                # Create DataFrame and perform clustering
                window_df = pd.DataFrame(window_cluster_data)
                
                # Prepare features (respect exclude_direction setting)
                if exclude_direction:
                    feature_cols = ['avg_position_norm', 'activity', 'concentration', 'position_pct_of_oi']
                else:
                    feature_cols = ['avg_position_norm', 'direction', 'activity', 'concentration', 'position_pct_of_oi']
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(window_df[feature_cols])
                
                # Perform clustering
                kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
                window_df['cluster'] = kmeans.fit_predict(features_scaled)
                
                # Calculate cluster profiles
                cluster_profiles = []
                for i in range(cluster_count):
                    cluster_data = window_df[window_df['cluster'] == i]
                    profile = {
                        'cluster': i,
                        'size': len(cluster_data),
                        'avg_position': cluster_data['avg_position_norm'].mean(),
                        'direction_bias': cluster_data['direction'].mean(),
                        'activity': cluster_data['activity'].mean(),
                        'concentration': cluster_data['concentration'].mean(),
                        'position_pct_of_oi': cluster_data['position_pct_of_oi'].mean()
                    }
                    cluster_profiles.append(profile)
                
                all_window_clusters.append({
                    'window': window_name,
                    'period': f"{window_result['start'].strftime('%Y-%m')} to {window_result['end'].strftime('%Y-%m')}",
                    'cluster_data': window_df,
                    'profiles': cluster_profiles,
                    'centers': kmeans.cluster_centers_
                })
        
        # Store comparison data for later display at the bottom
        comparison_data = {
            'all_window_clusters': all_window_clusters,
            'comparison_view': comparison_view,
            'cluster_count': cluster_count,
            'feature_cols': feature_cols,
            'exclude_direction': exclude_direction
        }
        
        # If comparison mode, use most recent window for main visualization
        if window_selection == "compare_all" and window_results:
            df_window = window_results[0]['data']
        
    else:
        # Parse single window selection
        if window_selection == "last_52":
            end_date = latest_date
            start_date = latest_date - pd.Timedelta(weeks=52)  # 1 year
        elif window_selection == "0_to_3":
            end_date = latest_date
            start_date = latest_date - pd.Timedelta(weeks=156)  # 3 years
        elif window_selection == "3_to_6":
            end_date = latest_date - pd.Timedelta(weeks=156)   # T-3
            start_date = latest_date - pd.Timedelta(weeks=312)  # T-6
        elif window_selection == "6_to_9":
            end_date = latest_date - pd.Timedelta(weeks=312)   # T-6
            start_date = latest_date - pd.Timedelta(weeks=468)  # T-9
        else:  # "9_to_12"
            end_date = latest_date - pd.Timedelta(weeks=468)   # T-9
            start_date = latest_date - pd.Timedelta(weeks=624)  # T-12
        
        df_window = df_cluster[
            (df_cluster['report_date_as_yyyy_mm_dd'] >= start_date) & 
            (df_cluster['report_date_as_yyyy_mm_dd'] <= end_date)
        ].copy()
    
    if len(df_window) < 2:
        if window_selection != "compare_all":
            st.warning(f"Insufficient data for clustering analysis in the selected period ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        return None
    
    # Show the actual date range being analyzed (for single window only)
    if window_selection != "compare_all":
        st.info(f"🗓️ Analyzing data from **{df_window['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')}** to **{df_window['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')}** ({len(df_window)} weeks)")
    
    # Calculate features for each category and week
    cluster_data = []
    
    for idx, row in df_window.iterrows():
        date = row['report_date_as_yyyy_mm_dd']
        
        for cat_name, pos_col, trader_col in categories:
            if pd.isna(row[pos_col]) or pd.isna(row[trader_col]) or row[trader_col] == 0:
                continue
                
            # Feature 1: Average position size (normalized by OI)
            avg_position = row[pos_col] / row[trader_col]
            avg_position_norm = avg_position / float(row['open_interest_all']) * 100
            
            # Feature 2: Direction
            if 'Long' in cat_name:
                direction = 1
            else:
                direction = -1
            
            # Feature 3: Activity level (week-over-week change from API)
            # Map category to appropriate change column
            change_map = {
                'Commercial Long': 'change_in_comm_long_all',
                'Commercial Short': 'change_in_comm_short_all',
                'Non-Commercial Long': 'change_in_noncomm_long_all',
                'Non-Commercial Short': 'change_in_noncomm_short_all'
            }
            
            change_col = change_map.get(cat_name)
            if change_col and change_col in row.index and not pd.isna(row[change_col]):
                # API provides the change directly, normalize by open interest
                change_value = abs(float(row[change_col]))
                activity = (change_value / float(row['open_interest_all'])) * 100
            else:
                activity = 0
            
            # Feature 4: Concentration (based on trader count percentile)
            # Get historical data since 2010 for percentile calculation
            hist_data = df_cluster[df_cluster['report_date_as_yyyy_mm_dd'] >= pd.Timestamp('2010-01-01')]
            if len(hist_data[trader_col].dropna()) > 0:
                trader_pct = stats.percentileofscore(hist_data[trader_col].dropna(), row[trader_col])
                concentration = 100 - trader_pct  # Invert: high percentile = many traders = low concentration
            else:
                concentration = 50
            
            # Feature 5: Position as % of total open interest (from API)
            # Map category to appropriate pct_of_oi column
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
                # Fallback calculation if API field not available
                position_pct_of_oi = (row[pos_col] / row['open_interest_all']) * 100 if row['open_interest_all'] > 0 else 0
            
            cluster_data.append({
                'date': date,
                'category': cat_name,
                'avg_position_norm': avg_position_norm,
                'direction': direction,
                'activity': activity,
                'concentration': concentration,
                'position_pct_of_oi': position_pct_of_oi,
                'positions': row[pos_col],
                'traders': row[trader_col]
            })
    
    if len(cluster_data) == 0:
        st.warning("No valid data for clustering")
        return None
    
    # Create DataFrame
    cluster_df = pd.DataFrame(cluster_data)
    
    # Prepare features for clustering
    if exclude_direction:
        feature_cols = ['avg_position_norm', 'activity', 'concentration', 'position_pct_of_oi']
        st.info("🔬 Experimental Mode: Clustering without direction feature - Long and Short positions may cluster together based on behavior similarity")
    else:
        feature_cols = ['avg_position_norm', 'direction', 'activity', 'concentration', 'position_pct_of_oi']
    
    # Standardize features on ALL data points
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cluster_df[feature_cols])
    
    # Show elbow plot if requested
    if show_elbow:
        st.subheader("📊 Clustering Quality Analysis")
        
        # Import silhouette_score
        from sklearn.metrics import silhouette_score
        
        # Calculate metrics for different K values
        K_range = range(2, 11)  # Test from 2 to 10 clusters
        inertias = []
        silhouette_scores = []
        
        for k in K_range:
            kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans_test.fit_predict(features_scaled)
            
            # Inertia (within-cluster sum of squares)
            inertias.append(kmeans_test.inertia_)
            
            # Silhouette score (how similar objects are to their own cluster vs other clusters)
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Calculate the percentage of variance explained
        total_variance = np.var(features_scaled, axis=0).sum() * len(features_scaled)
        variance_explained = [(1 - inertia/total_variance) * 100 for inertia in inertias]
        
        # Create quality metrics plot
        fig_elbow = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Variance Explained', 'Silhouette Score (closer to 1 is better)'],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # Plot 1: Variance Explained
        fig_elbow.add_trace(
            go.Scatter(
                x=list(K_range),
                y=variance_explained,
                mode='lines+markers',
                name='Variance Explained',
                line=dict(color='green', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Add marker for current selection
        if cluster_count in K_range:
            idx = list(K_range).index(cluster_count)
            fig_elbow.add_trace(
                go.Scatter(
                    x=[cluster_count],
                    y=[variance_explained[idx]],
                    mode='markers',
                    name='Current Selection',
                    marker=dict(size=15, color='red', symbol='star'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Plot 2: Silhouette Score
        fig_elbow.add_trace(
            go.Scatter(
                x=list(K_range),
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score',
                line=dict(color='purple', width=2),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Add marker for current selection
        if cluster_count in K_range:
            idx = list(K_range).index(cluster_count)
            fig_elbow.add_trace(
                go.Scatter(
                    x=[cluster_count],
                    y=[silhouette_scores[idx]],
                    mode='markers',
                    name='Current Selection',
                    marker=dict(size=15, color='red', symbol='star'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Update layout
        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
        
        fig_elbow.update_yaxes(title_text="Variance Explained (%)", row=1, col=1)
        fig_elbow.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        fig_elbow.update_layout(
            height=300,
            title="Cluster Selection Analysis",
            showlegend=True
        )
        
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        # Add interpretation
        col1_elbow, col2_elbow = st.columns(2)
        
        with col1_elbow:
            # Calculate elbow point using the "knee" method
            if len(inertias) > 2:
                # Calculate the angle for each point
                x = np.array(list(K_range))
                y = np.array(inertias)
                
                # Normalize
                x_norm = (x - x.min()) / (x.max() - x.min())
                y_norm = (y - y.min()) / (y.max() - y.min())
                
                # Calculate distances
                distances = []
                for i in range(1, len(x_norm)-1):
                    # Distance from line connecting first and last point
                    p1 = np.array([x_norm[0], y_norm[0]])
                    p2 = np.array([x_norm[-1], y_norm[-1]])
                    p3 = np.array([x_norm[i], y_norm[i]])
                    
                    distance = np.abs(np.cross(p2-p1, p3-p1)) / np.linalg.norm(p2-p1)
                    distances.append(distance)
                
                elbow_idx = np.argmax(distances) + 1
                elbow_k = list(K_range)[elbow_idx]
                
                st.info(f"📐 **Suggested K**: {elbow_k} clusters (elbow point)")
            
        with col2_elbow:
            if cluster_count in K_range:
                idx = list(K_range).index(cluster_count)
                st.info(f"📊 **Current K={cluster_count}**: Explains {variance_explained[idx]:.1f}% of variance | Silhouette: {silhouette_scores[idx]:.3f}")
        
        st.markdown("---")
    
    # Store download data for later
    download_df = cluster_df[['date', 'category'] + feature_cols].copy()
    
    # Add scaled features to show what goes into clustering
    scaled_feature_names = [f"{col}_scaled" for col in feature_cols]
    for i, col in enumerate(scaled_feature_names):
        download_df[col] = features_scaled[:, i]
    
    # Perform clustering on ALL data points
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    cluster_df['cluster'] = kmeans.fit_predict(features_scaled)
    
    # Get latest week's data for analysis
    latest_week = cluster_df['date'].max()
    latest_data = cluster_df[cluster_df['date'] == latest_week]
    
    # Analyze and classify clusters based on ALL data
    cluster_profiles = []
    
    # First, calculate all profiles
    for i in range(cluster_count):
        cluster_data = cluster_df[cluster_df['cluster'] == i]
        profile = {
            'cluster': i,
            'size': len(cluster_data),
            'avg_position': cluster_data['avg_position_norm'].mean(),
            'direction_bias': cluster_data['direction'].mean(),
            'activity': cluster_data['activity'].mean(),
            'concentration': cluster_data['concentration'].mean(),
            'position_pct_of_oi': cluster_data['position_pct_of_oi'].mean()
        }
        cluster_profiles.append(profile)
    
    # Sort profiles by various metrics to rank them
    activity_rank = sorted(cluster_profiles, key=lambda x: x['activity'], reverse=True)
    concentration_rank = sorted(cluster_profiles, key=lambda x: x['concentration'], reverse=True)
    directional_rank = sorted(cluster_profiles, key=lambda x: abs(x['direction_bias']), reverse=True)
    
    # Assign types based on relative rankings
    assigned_types = set()
    
    for profile in cluster_profiles:
        # Reset type
        profile['type'] = None
        
        # Check if highest concentration (and not yet assigned)
        if profile == concentration_rank[0] and profile['concentration'] > 60:
            profile['type'] = "🐋 Large Speculators"
            assigned_types.add(profile['cluster'])
        
        # Check if highest activity with direction
        elif profile == activity_rank[0] and profile['cluster'] not in assigned_types:
            if abs(profile['direction_bias']) > 0.5:
                profile['type'] = "🚀 Aggressive Traders"
            else:
                profile['type'] = "⚡ Market Makers"
            assigned_types.add(profile['cluster'])
        
        # Check if lowest activity with balanced direction
        elif profile == activity_rank[-1] and abs(profile['direction_bias']) < 0.3 and profile['cluster'] not in assigned_types:
            profile['type'] = "🛡️ Hedgers"
            assigned_types.add(profile['cluster'])
    
    # Assign remaining clusters more intelligently
    for profile in cluster_profiles:
        if profile['type'] is None:
            # Look at distinguishing features
            if profile['activity'] > cluster_df['activity'].median() and abs(profile['direction_bias']) > 0.7:
                profile['type'] = "📈 Directional Momentum"
            elif profile['position_pct_of_oi'] > cluster_df['position_pct_of_oi'].quantile(0.7):
                profile['type'] = "💰 Major Position Holders"
            elif abs(profile['direction_bias']) < 0.3:
                profile['type'] = "⚖️ Balanced Traders"
            else:
                profile['type'] = "📊 Trend Followers"
    
    # Create cluster characteristics dictionary for backward compatibility
    cluster_characteristics = {p['cluster']: p['type'] for p in cluster_profiles}
    
    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Cluster Distribution by Category',
            'Cluster Characteristics (2D Projection)',
            'Feature Stationarity Check',
            'Cluster Profiles'
        ],
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Cluster distribution by category
    cluster_counts = cluster_df.groupby(['category', 'cluster']).size().reset_index(name='count')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(cluster_count):
        cluster_i = cluster_counts[cluster_counts['cluster'] == i]
        profile = next(p for p in cluster_profiles if p['cluster'] == i)
        fig.add_trace(
            go.Bar(
                x=cluster_i['category'],
                y=cluster_i['count'],
                name=f"Cluster {i}: {profile['type']}",
                marker_color=colors[i % len(colors)],
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. 2D projection using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    
    for i in range(cluster_count):
        mask = cluster_df['cluster'] == i
        profile = next(p for p in cluster_profiles if p['cluster'] == i)
        fig.add_trace(
            go.Scatter(
                x=features_2d[mask, 0],
                y=features_2d[mask, 1],
                mode='markers',
                name=profile['type'],
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)]
                ),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Feature stationarity check - plot all 4 categories separately
    # Use the selected feature
    feature_to_show = stationarity_feature
    
    # Plot each category separately
    category_colors = {
        'Commercial Long': '#1f77b4',
        'Commercial Short': '#ff7f0e', 
        'Non-Commercial Long': '#2ca02c',
        'Non-Commercial Short': '#d62728'
    }
    
    for cat_name in ['Commercial Long', 'Commercial Short', 'Non-Commercial Long', 'Non-Commercial Short']:
        cat_data = cluster_df[cluster_df['category'] == cat_name].sort_values('date')
        
        if len(cat_data) > 0:
            # Plot raw values
            fig.add_trace(
                go.Scatter(
                    x=cat_data['date'],
                    y=cat_data[feature_to_show],
                    mode='lines',
                    name=cat_name,
                    line=dict(color=category_colors[cat_name], width=1),
                    showlegend=True,
                    opacity=0.7
                ),
                row=2, col=1
            )
    
    # Calculate and plot one overall rolling mean
    weekly_avg = cluster_df.groupby('date')[feature_to_show].mean().sort_index()
    rolling_window = min(13, max(4, len(weekly_avg) // 4))
    
    if len(weekly_avg) >= rolling_window:
        rolling_mean = weekly_avg.rolling(window=rolling_window, center=True).mean()
        
        fig.add_trace(
            go.Scatter(
                x=rolling_mean.index,
                y=rolling_mean,
                mode='lines',
                name=f'Overall {rolling_window}w Mean',
                line=dict(color='black', width=3, dash='solid'),
                showlegend=True,
                opacity=1.0
            ),
            row=2, col=1
        )
    
    # 4. Cluster profiles table
    profile_data = []
    for profile in cluster_profiles:
        if exclude_direction:
            # Without direction column
            profile_data.append([
                f"Cluster {profile['cluster']}",
                profile['type'],
                f"{profile['size']}",
                f"{profile['avg_position']:.1f}",
                f"{profile['activity']:.1f}",
                f"{profile['concentration']:.1f}",
                f"{profile['position_pct_of_oi']:.1f}%"
            ])
        else:
            # With direction column
            profile_data.append([
                f"Cluster {profile['cluster']}",
                profile['type'],
                f"{profile['size']}",
                f"{profile['avg_position']:.1f}",
                f"{profile['direction_bias']:.2f}",
                f"{profile['activity']:.1f}",
                f"{profile['concentration']:.1f}",
                f"{profile['position_pct_of_oi']:.1f}%"
            ])
    
    # Adjust headers based on whether direction is excluded
    if exclude_direction:
        header_values = ['Cluster', 'Type', 'Members', 'Avg Position', 'Activity', 'Concentration', '% of OI']
        num_cols = 7
    else:
        header_values = ['Cluster', 'Type', 'Members', 'Avg Position', 'Direction', 'Activity', 'Concentration', '% of OI']
        num_cols = 8
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=header_values,
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=list(zip(*profile_data)) if profile_data else [[]]*num_cols,
                fill_color='lavender',
                align='left'
            )
        ),
        row=2, col=2
    )
    
    # Update layout with clear time period indication
    window_desc = window_options.get(window_selection, window_selection)
    if window_selection == "compare_all":
        # When in compare all mode, show the most recent period analyzed
        period_desc = f"Most Recent Period: {df_window['report_date_as_yyyy_mm_dd'].min().strftime('%Y-%m-%d')} to {df_window['report_date_as_yyyy_mm_dd'].max().strftime('%Y-%m-%d')}"
        title_text = f"Participant Behavior Clustering Analysis - {period_desc}"
    else:
        title_text = f"Participant Behavior Clustering Analysis ({window_desc})"
    
    fig.update_layout(
        title=title_text,
        height=800,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Category", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Component 1", row=1, col=2)
    fig.update_yaxes(title_text="Component 2", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text=feature_options[stationarity_feature], row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Store stationarity and transition data for later display
    # Analyze stationarity for each category
    stationarity_results = []
    
    for cat_name in ['Commercial Long', 'Commercial Short', 'Non-Commercial Long', 'Non-Commercial Short']:
        cat_data = cluster_df[cluster_df['category'] == cat_name].sort_values('date')
        
        if len(cat_data) > 10:  # Need enough data for meaningful analysis
            feature_values = cat_data[feature_to_show]
            
            # Calculate coefficient of variation
            mean_val = feature_values.mean()
            std_val = feature_values.std()
            cv = std_val / mean_val if mean_val != 0 else 0
            
            # Try ADF test
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(feature_values.dropna())
                adf_pvalue = adf_result[1]
                is_stationary = adf_pvalue < 0.05
            except:
                adf_pvalue = None
                is_stationary = cv < 0.3  # Fallback to CV-based check
            
            stationarity_results.append({
                'Category': cat_name,
                'ADF p-value': f"{adf_pvalue:.3f}" if adf_pvalue is not None else "N/A",
                'Stationary?': "✅ Yes" if is_stationary else "⚠️ No"
            })
    
    # Calculate transitions if requested
    transitions = None
    if show_transitions and len(df_window) > 1:
        # Get previous week's data
        prev_week = latest_week - pd.Timedelta(weeks=1)
        prev_data = cluster_df[cluster_df['date'] == prev_week]
        
        if len(prev_data) > 0:
            # Calculate transitions
            latest_clusters = latest_data[['category', 'cluster']].set_index('category')
            prev_clusters = prev_data[['category', 'cluster']].set_index('category')
            
            transitions = []
            for category in ['Commercial Long', 'Commercial Short', 'Non-Commercial Long', 'Non-Commercial Short']:
                if category in latest_clusters.index and category in prev_clusters.index:
                    prev_cluster = prev_clusters.loc[category, 'cluster']
                    curr_cluster = latest_clusters.loc[category, 'cluster']
                    
                    changed = prev_cluster != curr_cluster
                    transitions.append({
                        'Category': category,
                        'Previous Week': f"Cluster {prev_cluster}",
                        'Current Week': f"Cluster {curr_cluster}",
                        'Changed': '🔄 Yes' if changed else '➡️ No'
                    })
        else:
            transitions = []  # Empty list to trigger the "no data" message
    
    # Show comparison visualization at the bottom if in compare_all mode
    if window_selection == "compare_all" and 'comparison_data' in locals() and comparison_data['all_window_clusters']:
        st.markdown("---")
        st.header("🔍 Cross-Window Clustering Comparison")
        
        # Create comparison plot - now with 5 windows (2x3 grid)
        fig_compare = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"{w['window']}" for w in comparison_data['all_window_clusters'][:6]],
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Determine which features to plot based on comparison_view
        # Adjust indices based on whether direction is excluded
        if comparison_data.get('exclude_direction', False):
            # Without direction: [avg_position_norm, activity, concentration, position_pct_of_oi]
            feature_mapping = {
                'activity_vs_position': {
                    'x_feature': 'avg_position_norm',
                    'y_feature': 'activity',
                    'x_index': 0,
                    'y_index': 1,
                    'x_label': 'Avg Position (norm)',
                    'y_label': 'Activity Level',
                    'x_range': [-0.5, 2],
                    'y_range': [0, 20]
                },
                'concentration_vs_activity': {
                    'x_feature': 'concentration',
                    'y_feature': 'activity',
                    'x_index': 2,
                    'y_index': 1,
                    'x_label': 'Concentration Score',
                    'y_label': 'Activity Level',
                    'x_range': [0, 100],
                    'y_range': [0, 20]
                },
                'concentration_vs_market_share': {
                    'x_feature': 'concentration',
                    'y_feature': 'position_pct_of_oi',
                    'x_index': 2,
                    'y_index': 3,
                    'x_label': 'Concentration Score',
                    'y_label': '% of Open Interest',
                    'x_range': [0, 100],
                    'y_range': [0, 30]
                },
                'position_vs_market_share': {
                    'x_feature': 'avg_position_norm',
                    'y_feature': 'position_pct_of_oi',
                    'x_index': 0,
                    'y_index': 3,
                    'x_label': 'Avg Position (norm)',
                    'y_label': '% of Open Interest',
                    'x_range': [-0.5, 2],
                    'y_range': [0, 30]
                },
                'activity_vs_market_share': {
                    'x_feature': 'activity',
                    'y_feature': 'position_pct_of_oi',
                    'x_index': 1,
                    'y_index': 3,
                    'x_label': 'Activity Level',
                    'y_label': '% of Open Interest',
                    'x_range': [0, 20],
                    'y_range': [0, 30]
                }
            }
        else:
            # With direction: [avg_position_norm, direction, activity, concentration, position_pct_of_oi]
            feature_mapping = {
                'activity_vs_position': {
                    'x_feature': 'avg_position_norm',
                    'y_feature': 'activity',
                    'x_index': 0,
                    'y_index': 2,
                    'x_label': 'Avg Position (norm)',
                    'y_label': 'Activity Level',
                    'x_range': [-0.5, 2],
                    'y_range': [0, 20]
                },
                'concentration_vs_activity': {
                    'x_feature': 'concentration',
                    'y_feature': 'activity',
                    'x_index': 3,
                    'y_index': 2,
                    'x_label': 'Concentration Score',
                    'y_label': 'Activity Level',
                    'x_range': [0, 100],
                    'y_range': [0, 20]
                },
                'concentration_vs_market_share': {
                    'x_feature': 'concentration',
                    'y_feature': 'position_pct_of_oi',
                    'x_index': 3,
                    'y_index': 4,
                    'x_label': 'Concentration Score',
                    'y_label': '% of Open Interest',
                    'x_range': [0, 100],
                    'y_range': [0, 30]
                },
                'position_vs_market_share': {
                    'x_feature': 'avg_position_norm',
                    'y_feature': 'position_pct_of_oi',
                    'x_index': 0,
                    'y_index': 4,
                    'x_label': 'Avg Position (norm)',
                    'y_label': '% of Open Interest',
                    'x_range': [-0.5, 2],
                    'y_range': [0, 30]
                },
                'activity_vs_market_share': {
                    'x_feature': 'activity',
                    'y_feature': 'position_pct_of_oi',
                    'x_index': 2,
                    'y_index': 4,
                    'x_label': 'Activity Level',
                    'y_label': '% of Open Interest',
                    'x_range': [0, 20],
                    'y_range': [0, 30]
                }
            }
        
        # Use comparison_view if set, otherwise default to activity_vs_position
        plot_config = feature_mapping.get(comparison_data['comparison_view'], feature_mapping['activity_vs_position'])
        
        for idx, window_cluster in enumerate(comparison_data['all_window_clusters'][:6]):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            # Plot cluster centers in 2D (using selected features)
            centers = window_cluster['centers']
            
            # Get the scaler to inverse transform centers
            temp_df = window_cluster['cluster_data']
            scaler = StandardScaler()
            scaler.fit(temp_df[comparison_data['feature_cols']])
            centers_raw = scaler.inverse_transform(centers)
            
            for i in range(comparison_data['cluster_count']):
                # Plot cluster center (using inverse transformed values)
                x_val = centers_raw[i][plot_config['x_index']]
                y_val = centers_raw[i][plot_config['y_index']]
                
                fig_compare.add_trace(
                    go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode='markers',
                        marker=dict(
                            size=20,
                            color=colors[i % len(colors)],
                            symbol='star',
                            line=dict(width=2, color='black')
                        ),
                        name=f'Cluster {i}' if idx == 0 else None,
                        showlegend=(idx == 0),
                        hovertext=f"Cluster {i}<br>{plot_config['x_label']}: {x_val:.2f}<br>{plot_config['y_label']}: {y_val:.2f}",
                        hoverinfo='text'
                    ),
                    row=row, col=col
                )
                
                # Plot points in cluster
                cluster_points = window_cluster['cluster_data'][window_cluster['cluster_data']['cluster'] == i]
                if len(cluster_points) > 0:
                    fig_compare.add_trace(
                        go.Scatter(
                            x=cluster_points[plot_config['x_feature']],
                            y=cluster_points[plot_config['y_feature']],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=colors[i % len(colors)],
                                opacity=0.3
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=row, col=col
                    )
            
            # Update axes with appropriate labels and ranges
            fig_compare.update_xaxes(title_text=plot_config['x_label'], row=row, col=col, range=plot_config['x_range'])
            fig_compare.update_yaxes(title_text=plot_config['y_label'], row=row, col=col, range=plot_config['y_range'])
        
        fig_compare.update_layout(
            title=f"Cluster Centers Comparison Across Time Windows (K={comparison_data['cluster_count']})",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Show cluster stability metrics
        st.subheader("📊 Cluster Stability Metrics")
        
        # Calculate stability metrics
        stability_results = []
        
        for i in range(len(comparison_data['all_window_clusters']) - 1):
            curr_window = comparison_data['all_window_clusters'][i]
            next_window = comparison_data['all_window_clusters'][i + 1]
            
            # Compare cluster centers between consecutive windows
            curr_centers = curr_window['centers']
            next_centers = next_window['centers']
            
            # Calculate average distance between corresponding clusters (in scaled space)
            total_distance = 0
            for j in range(comparison_data['cluster_count']):
                distance = np.linalg.norm(curr_centers[j] - next_centers[j])
                total_distance += distance
            
            avg_distance = total_distance / comparison_data['cluster_count']
            
            stability_results.append({
                'Comparison': f"{curr_window['window']} → {next_window['window']}",
                'Avg Center Distance': f"{avg_distance:.3f}",
                'Stability': "✅ Stable" if avg_distance < 0.5 else "⚠️ Changing"
            })
        
        if stability_results:
            stability_df = pd.DataFrame(stability_results)
            st.dataframe(stability_df, use_container_width=True, hide_index=True)
            
            # Overall assessment
            stable_count = sum(1 for r in stability_results if "✅" in r['Stability'])
            if stable_count == len(stability_results):
                st.success("✅ Clustering patterns are stable across all time windows - features appear stationary")
            elif stable_count >= len(stability_results) // 2:
                st.info(f"ℹ️ Clustering patterns are mostly stable ({stable_count}/{len(stability_results)} transitions)")
            else:
                st.warning(f"⚠️ Clustering patterns show significant changes across time windows ({stable_count}/{len(stability_results)} stable)")
        
        # Show feature evolution
        st.subheader("📈 Feature Evolution Across Windows")
        
        feature_evolution = []
        for window_cluster in comparison_data['all_window_clusters']:
            window_stats = {
                'Window': window_cluster['window'],
                'Avg Activity': f"{window_cluster['cluster_data']['activity'].mean():.2f}",
                'Avg Concentration': f"{window_cluster['cluster_data']['concentration'].mean():.1f}",
                'Avg Position Size': f"{window_cluster['cluster_data']['avg_position_norm'].mean():.3f}"
            }
            feature_evolution.append(window_stats)
        
        evolution_df = pd.DataFrame(feature_evolution)
        st.dataframe(evolution_df, use_container_width=True, hide_index=True)
        
        # Show the comparison summary table at the very end
        st.subheader("🔄 Comparing Clustering Results Across Time Windows")
        if 'comparison_df' in locals():
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Display stationarity analysis at the bottom
    if stationarity_results:
        st.markdown("---")
        st.subheader("📊 Stationarity Analysis")
        results_df = pd.DataFrame(stationarity_results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Display transitions at the bottom
    if transitions is not None:
        st.subheader("🔄 Week-over-Week Cluster Assignments")
        if len(transitions) > 0:
            transition_df = pd.DataFrame(transitions)
            st.dataframe(transition_df, use_container_width=True, hide_index=True)
        else:
            st.info("No data available for previous week to compare transitions")
    
    # Add download section at the very bottom
    st.markdown("---")
    st.markdown("### 📥 Download Pre-Clustering Data")
    
    # Convert to CSV
    csv = download_df.to_csv(index=False)
    
    # Create download button
    col1_download, col2_download = st.columns([1, 3])
    with col1_download:
        st.download_button(
            label="Download Raw Data (CSV)",
            data=csv,
            file_name=f"pre_clustering_data_{instrument_name.replace(' ', '_')}_{window_selection}.csv",
            mime="text/csv",
            help="Raw feature data before clustering - no cluster assignments"
        )
    
    with col2_download:
        st.info(f"📊 Dataset: {len(download_df)} rows × {len(feature_cols)} raw features + {len(feature_cols)} scaled features")
    
    return fig