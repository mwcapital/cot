"""
Market Microstructure Analysis for CFTC COT Dashboard
Analyzes market concentration, position distribution, and trader dynamics
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats


def create_market_microstructure_analysis(df, instrument_name):
    """Create comprehensive market microstructure analysis"""
    
    st.markdown("#### 🔬 Market Microstructure Analysis")
    
    # Explanation expander
    with st.expander("📖 Understanding Market Microstructure", expanded=False):
        st.markdown("""
        **What is Market Microstructure Analysis?**
        
        A granular view of how different trader sizes and types interact in the market.
        This analysis reveals the "ecosystem" of market participants.
        
        **Key Metrics:**
        
        1. **Size Distribution**
           - Small traders: Bottom 50% by position size
           - Medium traders: 50-90th percentile
           - Large traders: 90-95th percentile
           - Whales: Top 5%
           
        2. **Market Impact Potential**
           - Measures how much each group could move the market
           - Based on position size × concentration
           
        3. **Liquidity Provision**
           - Identifies who provides vs consumes liquidity
           - Based on position stability and two-sided exposure
           
        4. **Position Persistence**
           - How long different groups maintain positions
           - Indicates trading style (scalping vs investing)
           
        **Microstructure Patterns:**
        - **Healthy**: Diverse participation across all sizes
        - **Top-Heavy**: Dominated by large traders
        - **Bifurcated**: Missing middle (only large and small)
        - **Fragmented**: Many small traders, few large
        
        **Why It Matters:**
        - Reveals market vulnerability to large trader actions
        - Shows liquidity depth and resilience
        - Identifies potential manipulation risks
        - Helps predict volatility patterns
        """)
    
    # Configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        weeks_back = st.selectbox(
            "Compare with:",
            [0, 1, 2, 3, 4],
            format_func=lambda x: "Current Week" if x == 0 else f"{x} Week{'s' if x > 1 else ''} Ago",
            index=0,
            help="Show data from selected week for comparison"
        )
    
    with col2:
        # Only show evolution checkbox for current week
        if weeks_back == 0:
            show_evolution = st.checkbox(
                "Show Evolution Over Time",
                value=True,
                help="Track how microstructure changes"
            )
        else:
            show_evolution = False
            st.empty()  # Keep layout consistent
    
    # Prepare data
    df_micro = df.copy()
    
    # Get the analysis window - single week based on selection
    sorted_dates = sorted(df_micro['report_date_as_yyyy_mm_dd'].unique(), reverse=True)
    
    if weeks_back < len(sorted_dates):
        selected_date = sorted_dates[weeks_back]
        analysis_data = df_micro[df_micro['report_date_as_yyyy_mm_dd'] == selected_date]
        periods = 1
    else:
        # Fallback to latest available
        selected_date = sorted_dates[0]
        analysis_data = df_micro[df_micro['report_date_as_yyyy_mm_dd'] == selected_date]
        periods = 1
    
    if len(analysis_data) > 0:
        # CORRECTED MICROSTRUCTURE CALCULATION
        # Using concentration as % of total OI, not category positions
        
        # Get values for the selected week
        avg_oi = float(analysis_data['open_interest_all'].iloc[0])
        
        # Calculate top trader metrics from OI-based concentration
        top_traders_metrics = {}
        
        for side in ['long', 'short']:
            # Use gross concentration (includes all traders)
            if side == 'long':
                conc_4_pct = float(analysis_data['conc_gross_le_4_tdr_long'].iloc[0])
                conc_8_pct = float(analysis_data['conc_gross_le_8_tdr_long'].iloc[0])
            else:
                conc_4_pct = float(analysis_data['conc_gross_le_4_tdr_short'].iloc[0])
                conc_8_pct = float(analysis_data['conc_gross_le_8_tdr_short'].iloc[0])
            
            # Calculate contracts held by top traders
            # These percentages are of TOTAL OPEN INTEREST
            top_4_contracts = avg_oi * (conc_4_pct / 100)
            top_4_avg = top_4_contracts / 4
            
            # Top 8 total contracts
            top_8_contracts = avg_oi * (conc_8_pct / 100)
            
            # Next 4 (5-8) by subtraction
            next_4_contracts = top_8_contracts - top_4_contracts
            next_4_avg = next_4_contracts / 4
            
            top_traders_metrics[side] = {
                'top_4_avg': top_4_avg,
                'next_4_avg': next_4_avg,
                'top_4_pct_of_oi': conc_4_pct,
                'top_8_pct_of_oi': conc_8_pct
            }
        
        # Calculate category averages for comparison
        category_metrics = []
        
        categories = [
            ('Commercial Long', 'comm_positions_long_all', 'traders_comm_long_all', 'red'),
            ('Commercial Short', 'comm_positions_short_all', 'traders_comm_short_all', 'darkred'),
            ('Non-Commercial Long', 'noncomm_positions_long_all', 'traders_noncomm_long_all', 'blue'),
            ('Non-Commercial Short', 'noncomm_positions_short_all', 'traders_noncomm_short_all', 'darkblue')
        ]
        
        for cat_name, pos_col, trader_col, color in categories:
            if trader_col in analysis_data.columns and pos_col in analysis_data.columns:
                avg_positions = float(analysis_data[pos_col].iloc[0])
                avg_traders = float(analysis_data[trader_col].iloc[0])
                
                if avg_traders > 0:
                    avg_position = avg_positions / avg_traders
                    position_pct_of_oi = (avg_positions / avg_oi) * 100
                    
                    # Determine side for comparison
                    side = 'long' if 'Long' in cat_name else 'short'
                    
                    # Calculate ratios to top traders
                    top_4_avg = top_traders_metrics[side]['top_4_avg']
                    ratio_to_top_4 = top_4_avg / avg_position if avg_position > 0 else 0
                    
                    category_metrics.append({
                        'category': cat_name,
                        'total_traders': avg_traders,
                        'total_positions': avg_positions,
                        'avg_position': avg_position,
                        'position_pct_of_oi': position_pct_of_oi,
                        'ratio_to_top_4': ratio_to_top_4,
                        'color': color,
                        'side': side
                    })
        
        # Combine top trader and category data for visualization
        micro_metrics = []
        
        # Create visualizations with corrected approach
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Position Comparison', 'Market Control Breakdown',
                          'Top Trader Dominance', 'Category vs Top Trader Analysis'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Plot 1: Average Position Comparison
        # Shows top traders vs category averages
        
        # Prepare data for comparison
        groups = []
        avg_positions = []
        colors = []
        labels = []
        
        # Add top traders (both sides)
        for side in ['long', 'short']:
            metrics = top_traders_metrics[side]
            groups.extend([f'Top 4 {side.title()}', f'Next 4 {side.title()}'])
            avg_positions.extend([metrics['top_4_avg'], metrics['next_4_avg']])
            colors.extend(['gold' if side == 'long' else 'darkgoldenrod', 
                         'orange' if side == 'long' else 'darkorange'])
            labels.extend([f"{metrics['top_4_avg']:,.0f}", f"{metrics['next_4_avg']:,.0f}"])
        
        # Add category averages
        for cat_metric in category_metrics:
            groups.append(cat_metric['category'])
            avg_positions.append(cat_metric['avg_position'])
            colors.append(cat_metric['color'])
            labels.append(f"{cat_metric['avg_position']:,.0f}")
        
        fig.add_trace(go.Bar(
            x=groups,
            y=avg_positions,
            marker_color=colors,
            text=labels,
            textposition='auto',
            showlegend=False,
            hovertemplate='%{x}<br>Avg Position: %{y:,.0f} contracts<extra></extra>'
        ), row=1, col=1)
        
        # Add ratio annotations
        for i, cat_metric in enumerate(category_metrics):
            if cat_metric['ratio_to_top_4'] > 0:
                fig.add_annotation(
                    x=cat_metric['category'],
                    y=cat_metric['avg_position'],
                    text=f"{cat_metric['ratio_to_top_4']:.1f}x",
                    showarrow=True,
                    arrowhead=2,
                    yshift=20,
                    row=1, col=1
                )
        
        # Plot 2: Contract Distribution Breakdown
        # Shows actual contract numbers from concentration data
        
        # Get total reportable positions for each side
        total_long = float(analysis_data['tot_rept_positions_long_all'].iloc[0]) if 'tot_rept_positions_long_all' in analysis_data.columns else 0
        total_short = float(analysis_data['tot_rept_positions_short'].iloc[0]) if 'tot_rept_positions_short' in analysis_data.columns else 0
        
        # Calculate actual contracts from OI percentages
        long_metrics = top_traders_metrics['long']
        short_metrics = top_traders_metrics['short']
        
        # Top 4 contracts (from % of total OI)
        long_top4_contracts = avg_oi * (long_metrics['top_4_pct_of_oi'] / 100)
        short_top4_contracts = avg_oi * (short_metrics['top_4_pct_of_oi'] / 100)
        
        # Top 8 total contracts
        long_top8_contracts = avg_oi * (long_metrics['top_8_pct_of_oi'] / 100)
        short_top8_contracts = avg_oi * (short_metrics['top_8_pct_of_oi'] / 100)
        
        # Next 4 (5-8) contracts
        long_next4_contracts = long_top8_contracts - long_top4_contracts
        short_next4_contracts = short_top8_contracts - short_top4_contracts
        
        # Remaining contracts (total reportable minus top 8)
        long_remaining = total_long - long_top8_contracts if total_long > long_top8_contracts else 0
        short_remaining = total_short - short_top8_contracts if total_short > short_top8_contracts else 0
        
        # Create stacked bar showing contract distribution
        fig.add_trace(go.Bar(
            name='Top 4 Traders',
            x=['Long Contracts', 'Short Contracts'],
            y=[long_top4_contracts, short_top4_contracts],
            text=[f'{long_top4_contracts:,.0f}<br>({long_metrics["top_4_pct_of_oi"]:.1f}% of OI)', 
                  f'{short_top4_contracts:,.0f}<br>({short_metrics["top_4_pct_of_oi"]:.1f}% of OI)'],
            textposition='inside',
            marker_color='darkred',
            hovertemplate='%{x}<br>Top 4: %{y:,.0f} contracts<extra></extra>'
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            name='Next 4 Traders (5-8)',
            x=['Long Contracts', 'Short Contracts'],
            y=[long_next4_contracts, short_next4_contracts],
            text=[f'{long_next4_contracts:,.0f}<br>({(long_metrics["top_8_pct_of_oi"]-long_metrics["top_4_pct_of_oi"]):.1f}% of OI)', 
                  f'{short_next4_contracts:,.0f}<br>({(short_metrics["top_8_pct_of_oi"]-short_metrics["top_4_pct_of_oi"]):.1f}% of OI)'],
            textposition='inside',
            marker_color='orange',
            hovertemplate='%{x}<br>Next 4: %{y:,.0f} contracts<extra></extra>'
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            name='Remaining Traders',
            x=['Long Contracts', 'Short Contracts'],
            y=[long_remaining, short_remaining],
            text=[f'{long_remaining:,.0f}', f'{short_remaining:,.0f}'],
            textposition='inside',
            marker_color='lightblue',
            hovertemplate='%{x}<br>Remaining: %{y:,.0f} contracts<extra></extra>'
        ), row=1, col=2)
        
        # Add annotations with calculations
        fig.add_annotation(
            text=f"Total OI: {avg_oi:,.0f}",
            xref="paper", yref="paper",
            x=0.75, y=0.95,
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="lightyellow",
            row=1, col=2
        )
        
        # Update layout for stacked bars
        fig.update_yaxes(title_text="Number of Contracts", row=1, col=2)
        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_layout(barmode='stack')
        
        # Plot 3: Top Trader Dominance
        # Shows how concentration changes from top 4 to top 8
        
        # Create data for both sides
        dominance_data = []
        
        for side in ['long', 'short']:
            metrics = top_traders_metrics[side]
            # Top 4
            dominance_data.append({
                'side': side.title(),
                'traders': 4,
                'concentration': metrics['top_4_pct_of_oi'],
                'label': f"Top 4 {side.title()}"
            })
            # Top 8
            dominance_data.append({
                'side': side.title(),
                'traders': 8,
                'concentration': metrics['top_8_pct_of_oi'],
                'label': f"Top 8 {side.title()}"
            })
        
        dominance_df = pd.DataFrame(dominance_data)
        
        for side in ['Long', 'Short']:
            side_data = dominance_df[dominance_df['side'] == side]
            fig.add_trace(go.Scatter(
                x=side_data['traders'],
                y=side_data['concentration'],
                name=side,
                mode='lines+markers',
                marker=dict(size=12),
                line=dict(width=3, color='green' if side == 'Long' else 'red'),
                hovertemplate='%{text}<br>%{y:.1f}% of Total OI<extra></extra>',
                text=side_data['label']
            ), row=2, col=1)
        
        # Add annotations for concentration increase
        for side, color in [('long', 'green'), ('short', 'red')]:
            metrics = top_traders_metrics[side]
            increase = metrics['top_8_pct_of_oi'] - metrics['top_4_pct_of_oi']
            fig.add_annotation(
                x=6,
                y=(metrics['top_4_pct_of_oi'] + metrics['top_8_pct_of_oi']) / 2,
                text=f"+{increase:.1f}%",
                showarrow=False,
                font=dict(color=color, size=10),
                row=2, col=1
            )
        
        # Plot 4: Category vs Top Trader Analysis
        # Shows which categories likely contain the top traders
        
        if category_metrics:
            # Calculate max value first for positioning
            max_val = max([m['avg_position'] for m in category_metrics] + 
                         [top_traders_metrics['long']['top_4_avg'], 
                          top_traders_metrics['short']['top_4_avg']])
            
            # Prepare scatter data
            for i, cat_metric in enumerate(category_metrics):
                # Get corresponding top trader average
                side = cat_metric['side']
                top_4_avg = top_traders_metrics[side]['top_4_avg']
                
                # Scale down bubble sizes and add minimum size
                bubble_size = max(15, min(50, cat_metric['total_traders'] / 4))
                
                fig.add_trace(go.Scatter(
                    x=[cat_metric['avg_position']],
                    y=[top_4_avg],
                    mode='markers',
                    marker=dict(
                        size=bubble_size,
                        color=cat_metric['color'],
                        line=dict(width=2, color='black'),
                        opacity=0.8
                    ),
                    name=cat_metric['category'],
                    showlegend=False,
                    hovertemplate='%{text}<br>Category Avg: %{x:,.0f}<br>Top 4 Avg: %{y:,.0f}<br>Ratio: %{customdata[0]:.1f}x<br>Traders: %{customdata[1]:.0f}<extra></extra>',
                    text=[cat_metric['category']],
                    customdata=[[cat_metric['ratio_to_top_4'], cat_metric['total_traders']]]
                ), row=2, col=2)
                
                # Add text annotations with arrows pointing to bubbles
                # Position text based on quadrant to avoid overlaps
                x_pos = cat_metric['avg_position']
                y_pos = top_4_avg
                
                # Determine text positioning based on location
                if i == 0:  # Top-left
                    text_x = x_pos - max_val * 0.15
                    text_y = y_pos + max_val * 0.15
                    ax = 20
                    ay = -20
                elif i == 1:  # Top-right
                    text_x = x_pos + max_val * 0.15
                    text_y = y_pos + max_val * 0.15
                    ax = -20
                    ay = -20
                elif i == 2:  # Bottom-left
                    text_x = x_pos - max_val * 0.15
                    text_y = y_pos - max_val * 0.15
                    ax = 20
                    ay = 20
                else:  # Bottom-right
                    text_x = x_pos + max_val * 0.15
                    text_y = y_pos - max_val * 0.15
                    ax = -20
                    ay = 20
                
                # Add category label with arrow
                fig.add_annotation(
                    x=x_pos,
                    y=y_pos,
                    text=f"<b>{cat_metric['category']}</b><br>{cat_metric['total_traders']:.0f} traders",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=cat_metric['color'],
                    ax=ax,
                    ay=ay,
                    bgcolor="white",
                    bordercolor=cat_metric['color'],
                    borderwidth=2,
                    opacity=0.9,
                    font=dict(size=10),
                    row=2, col=2
                )
            
            # Add diagonal line (y=x)
            fig.add_trace(go.Scatter(
                x=[0, max_val * 1.1],
                y=[0, max_val * 1.1],
                mode='lines',
                line=dict(dash='dash', color='gray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ), row=2, col=2)
            
            # Add guide text
            fig.add_annotation(
                x=max_val * 0.55,
                y=max_val * 0.45,
                text="← Categories below line have<br>larger positions than top 4 avg",
                showarrow=False,
                font=dict(size=9, color='gray'),
                opacity=0.7,
                row=2, col=2
            )
        
        # Update layout
        week_label = "Current Week" if weeks_back == 0 else f"{weeks_back} Week{'s' if weeks_back > 1 else ''} Ago"
        fig.update_layout(
            title=f"Market Microstructure Analysis - {week_label} ({selected_date.strftime('%Y-%m-%d')})",
            height=900,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Trader Group", row=1, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Average Position (Contracts)", row=1, col=1)
        fig.update_xaxes(
            title_text="Number of Top Traders", 
            row=2, col=1,
            tickmode='array',
            tickvals=[4, 8],
            ticktext=['4', '8']
        )
        fig.update_yaxes(title_text="% of Total Open Interest", row=2, col=1)
        fig.update_xaxes(title_text="Category Average Position", row=2, col=2)
        fig.update_yaxes(title_text="Top 4 Average Position", row=2, col=2)
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics with corrected calculations
        st.markdown("### Market Structure Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate total traders from categories
        total_traders = sum(m['total_traders'] for m in category_metrics)
        
        # Calculate what % of each side the average top 4 trader controls
        # Get total positions for each side
        total_long = float(analysis_data['tot_rept_positions_long_all'].iloc[0]) if 'tot_rept_positions_long_all' in analysis_data.columns else 0
        total_short = float(analysis_data['tot_rept_positions_short'].iloc[0]) if 'tot_rept_positions_short' in analysis_data.columns else 0
        
        # Calculate percentage of side controlled by each top 4 trader
        long_top4_avg = top_traders_metrics['long']['top_4_avg']
        short_top4_avg = top_traders_metrics['short']['top_4_avg']
        
        long_pct_of_side = (long_top4_avg / total_long * 100) if total_long > 0 else 0
        short_pct_of_side = (short_top4_avg / total_short * 100) if total_short > 0 else 0
        
        # Average of both sides
        avg_top4_control_of_side = (long_pct_of_side + short_pct_of_side) / 2
        
        # Determine likely composition
        if category_metrics:
            min_ratio = min(m['ratio_to_top_4'] for m in category_metrics)
            likely_category = next(m['category'] for m in category_metrics if m['ratio_to_top_4'] == min_ratio)
        else:
            likely_category = "Unknown"
        
        # Calculate structure for each week (moved up to use in current status)
        structure_data = []
        weeks_data = df_micro.tail(52).copy()
        
        # Get latest date for calculations
        latest_date = df_micro['report_date_as_yyyy_mm_dd'].max()
        
        # Calculate empirical percentiles for market structure determination
        # Get historical data for the past 52 weeks
        historical_window = df_micro[df_micro['report_date_as_yyyy_mm_dd'] >= (latest_date - pd.Timedelta(weeks=52))]
        
        if len(historical_window) > 10:  # Need enough data for percentiles
            # Calculate historical control percentages
            historical_controls = []
            for _, week_data in historical_window.iterrows():
                week_oi = week_data['open_interest_all']
                week_total_long = week_data['tot_rept_positions_long_all'] if 'tot_rept_positions_long_all' in week_data else 0
                week_total_short = week_data['tot_rept_positions_short'] if 'tot_rept_positions_short' in week_data else 0
                
                # Long side control
                if week_total_long > 0 and week_oi > 0:
                    week_long_top4_contracts = week_oi * (week_data['conc_gross_le_4_tdr_long'] / 100)
                    week_long_top4_avg = week_long_top4_contracts / 4
                    week_long_pct = (week_long_top4_avg / week_total_long * 100)
                    historical_controls.append(week_long_pct)
                
                # Short side control
                if week_total_short > 0 and week_oi > 0:
                    week_short_top4_contracts = week_oi * (week_data['conc_gross_le_4_tdr_short'] / 100)
                    week_short_top4_avg = week_short_top4_contracts / 4
                    week_short_pct = (week_short_top4_avg / week_total_short * 100)
                    historical_controls.append(week_short_pct)
            
            # Calculate percentiles
            p75 = np.percentile(historical_controls, 75)  # Top 25% most concentrated
            p50 = np.percentile(historical_controls, 50)  # Median
            
            # Now calculate structure for each week for the timeline
            for _, week in weeks_data.iterrows():
                week_oi = week['open_interest_all']
                week_total_long = week['tot_rept_positions_long_all'] if 'tot_rept_positions_long_all' in week else 0
                week_total_short = week['tot_rept_positions_short'] if 'tot_rept_positions_short' in week else 0
                
                # Calculate control percentages for this week
                if week_total_long > 0 and week_oi > 0:
                    week_long_top4_contracts = week_oi * (week['conc_gross_le_4_tdr_long'] / 100)
                    week_long_top4_avg = week_long_top4_contracts / 4
                    week_long_pct = (week_long_top4_avg / week_total_long * 100)
                else:
                    week_long_pct = 0
                    
                if week_total_short > 0 and week_oi > 0:
                    week_short_top4_contracts = week_oi * (week['conc_gross_le_4_tdr_short'] / 100)
                    week_short_top4_avg = week_short_top4_contracts / 4
                    week_short_pct = (week_short_top4_avg / week_total_short * 100)
                else:
                    week_short_pct = 0
                
                # Use max for structure determination
                week_max_control = max(week_long_pct, week_short_pct)
                
                # Assign structure based on percentiles
                if week_max_control > p75:
                    structure_value = 2  # Highly Concentrated
                    structure_label = "High"
                elif week_max_control > p50:
                    structure_value = 1  # Moderately Concentrated
                    structure_label = "Moderate"
                else:
                    structure_value = 0  # Well Distributed
                    structure_label = "Low"
                
                structure_data.append({
                    'date': week['report_date_as_yyyy_mm_dd'],
                    'structure_value': structure_value,
                    'structure_label': structure_label,
                    'control_pct': week_max_control
                })
            
            # Determine market structure
            # For current week, always use the timeline's last entry for consistency
            if weeks_back == 0 and structure_data:
                # Use the most recent week from timeline
                latest_week = structure_data[-1]
                
                if latest_week['structure_value'] == 2:
                    structure = f"🔴 Highly Concentrated (>{p75:.1f}%)"
                elif latest_week['structure_value'] == 1:
                    structure = f"🟠 Moderately Concentrated ({p50:.1f}-{p75:.1f}%)"
                else:
                    structure = f"🟢 Well Distributed (<{p50:.1f}%)"
            else:
                # For historical weeks or if no structure data, calculate based on values
                max_control = max(long_pct_of_side, short_pct_of_side)
                if max_control > p75:
                    structure = f"🔴 Highly Concentrated (>{p75:.1f}%)"
                elif max_control > p50:
                    structure = f"🟠 Moderately Concentrated ({p50:.1f}-{p75:.1f}%)"
                else:
                    structure = f"🟢 Well Distributed (<{p50:.1f}%)"
        else:
            # Fallback to fixed thresholds if not enough historical data
            max_control = max(long_pct_of_side, short_pct_of_side)
            if max_control > 6:
                structure = "🔴 Highly Concentrated"
            elif max_control > 4:
                structure = "🟠 Moderately Concentrated"
            else:
                structure = "🟢 Well Distributed"
        
        with col1:
            st.metric("Total Traders", f"{int(total_traders):,}")
        with col2:
            st.metric("Top-4 Long Control", f"{long_pct_of_side:.1f}% of Longs")
        with col3:
            st.metric("Top-4 Short Control", f"{short_pct_of_side:.1f}% of Shorts")
        with col4:
            st.metric("Top 4 Likely Type", likely_category)
        with col5:
            st.metric("Market Structure", structure)
        
        # Detailed insights
        st.markdown("### Key Insights")
        insights_cols = st.columns(2)
        
        with insights_cols[0]:
            st.markdown("**Top Trader Analysis:**")
            long_metrics = top_traders_metrics['long']
            short_metrics = top_traders_metrics['short']
            
            st.write(f"• Top 4 Long traders: {long_metrics['top_4_avg']:,.0f} contracts each")
            st.write(f"• Top 4 Short traders: {short_metrics['top_4_avg']:,.0f} contracts each")
            st.write(f"• Each top Long controls: {long_pct_of_side:.1f}% of all longs")
            st.write(f"• Each top Short controls: {short_pct_of_side:.1f}% of all shorts")
        
        with insights_cols[1]:
            st.markdown("**Category Comparison:**")
            if category_metrics:
                # Sort by ratio to find most likely categories
                sorted_categories = sorted(category_metrics, key=lambda x: x['ratio_to_top_4'])
                
                st.write(f"• Most likely top 4: {sorted_categories[0]['category']} ({sorted_categories[0]['ratio_to_top_4']:.1f}x)")
                st.write(f"• Least likely top 4: {sorted_categories[-1]['category']} ({sorted_categories[-1]['ratio_to_top_4']:.1f}x)")
                
                # Average positions
                comm_avg = np.mean([m['avg_position'] for m in category_metrics if 'Commercial' in m['category']])
                noncomm_avg = np.mean([m['avg_position'] for m in category_metrics if 'Non-Commercial' in m['category']])
                
                st.write(f"• Commercial avg: {comm_avg:,.0f} contracts")
                st.write(f"• Non-Commercial avg: {noncomm_avg:,.0f} contracts")
        
        # Market Structure History Heatmap
        st.markdown("### Market Structure History (52 Weeks)")
        
        # Create heatmap
        if structure_data:
            # Prepare data for heatmap
            dates = [d['date'] for d in structure_data]
            values = [d['structure_value'] for d in structure_data]
            labels = [d['structure_label'] for d in structure_data]
            control_pcts = [d['control_pct'] for d in structure_data]
            
            # Create custom hover text
            hover_text = []
            for i in range(len(dates)):
                hover_text.append(
                    f"Date: {dates[i].strftime('%Y-%m-%d')}<br>" +
                    f"Status: {labels[i]}<br>" +
                    f"Max Control: {control_pcts[i]:.1f}%"
                )
            
            # Create the timeline
            fig_timeline = go.Figure()
            
            # Create a single row heatmap
            fig_timeline.add_trace(go.Heatmap(
                z=[values],  # Single row
                text=[hover_text],
                hovertemplate='%{text}<extra></extra>',
                colorscale=[
                    [0, '#4CAF50'],      # 0 = Green - Well Distributed
                    [0.33, '#4CAF50'],   # Still green
                    [0.34, '#FF9800'],   # 1 = Orange - Moderately Concentrated
                    [0.66, '#FF9800'],   # Still orange
                    [0.67, '#F44336'],   # 2 = Red - Highly Concentrated
                    [1, '#F44336']       # Still red
                ],
                showscale=False,
                xgap=2,
                ygap=0,
                zmin=0,
                zmax=2
            ))
            
            # Add x-axis labels for months
            # Get unique months from dates
            month_positions = []
            month_labels = []
            current_month = None
            
            for i, date in enumerate(dates):
                month_year = date.strftime('%b %Y')
                if month_year != current_month:
                    month_positions.append(i)
                    month_labels.append(month_year)
                    current_month = month_year
            
            # Update layout
            fig_timeline.update_layout(
                title="52-Week Market Concentration Timeline",
                height=120,
                xaxis=dict(
                    tickmode='array',
                    tickvals=month_positions,
                    ticktext=month_labels,
                    tickangle=0,
                    showgrid=False,
                    side='bottom'
                ),
                yaxis=dict(
                    showticklabels=False, 
                    showgrid=False,
                    fixedrange=True
                ),
                plot_bgcolor='white',
                margin=dict(t=60, b=40, l=20, r=20)
            )
            
            # Add legend
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🟢 **Well Distributed** (<" + f"{p50:.1f}%)")
            with col2:
                st.markdown("🟠 **Moderately Concentrated** (" + f"{p50:.1f}-{p75:.1f}%)")
            with col3:
                st.markdown("🔴 **Highly Concentrated** (>" + f"{p75:.1f}%)")
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Show evolution if requested
        if show_evolution:
            st.markdown("### Microstructure Evolution")
            
            # Calculate metrics over time
            evolution_data = []
            
            for date in df_micro['report_date_as_yyyy_mm_dd'].tail(52).unique():
                date_data = df_micro[df_micro['report_date_as_yyyy_mm_dd'] == date]
                
                # Calculate key metrics
                total_traders = 0
                total_positions = 0
                
                # Sum up traders and positions across categories
                for cat_name, pos_col, trader_col, color in categories:
                    if trader_col in date_data.columns and pos_col in date_data.columns:
                        traders = float(date_data.iloc[0][trader_col]) if pd.notna(date_data.iloc[0][trader_col]) else 0
                        positions = float(date_data.iloc[0][pos_col]) if pd.notna(date_data.iloc[0][pos_col]) else 0
                        total_traders += traders
                        total_positions += positions
                
                # Get market concentration - use maximum as indicator of concentration risk
                long_conc = float(date_data.iloc[0]['conc_gross_le_4_tdr_long']) if 'conc_gross_le_4_tdr_long' in date_data.columns else 0
                short_conc = float(date_data.iloc[0]['conc_gross_le_4_tdr_short']) if 'conc_gross_le_4_tdr_short' in date_data.columns else 0
                max_concentration = max(long_conc, short_conc)
                
                # Calculate average position size
                avg_position_size = total_positions / total_traders if total_traders > 0 else 0
                
                evolution_data.append({
                    'date': date,
                    'total_traders': total_traders,
                    'max_concentration': max_concentration,
                    'long_concentration': long_conc,
                    'short_concentration': short_conc,
                    'avg_position_size': avg_position_size
                })
            
            evolution_df = pd.DataFrame(evolution_data)
            
            # Create evolution chart
            fig_evo = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]]
            )
            
            fig_evo.add_trace(go.Scatter(
                x=evolution_df['date'],
                y=evolution_df['total_traders'],
                name='Total Traders',
                line=dict(color='blue', width=2)
            ), secondary_y=False)
            
            fig_evo.add_trace(go.Scatter(
                x=evolution_df['date'],
                y=evolution_df['long_concentration'],
                name='Long Concentration',
                line=dict(color='green', width=2)
            ), secondary_y=True)
            
            fig_evo.add_trace(go.Scatter(
                x=evolution_df['date'],
                y=evolution_df['short_concentration'],
                name='Short Concentration',
                line=dict(color='red', width=2)
            ), secondary_y=True)
            
            fig_evo.update_layout(
                title="Market Microstructure Evolution (52 weeks)",
                height=400
            )
            fig_evo.update_xaxes(title_text="Date")
            fig_evo.update_yaxes(title_text="Total Traders", secondary_y=False)
            fig_evo.update_yaxes(title_text="Top 4 Concentration (% of OI)", secondary_y=True)
            
            st.plotly_chart(fig_evo, use_container_width=True)
    
    else:
        st.warning("Insufficient data for microstructure analysis")
    
    return fig