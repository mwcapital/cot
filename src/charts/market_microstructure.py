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
    
    # Title is rendered by the caller in display_functions_futures_first.py
    
    # Fixed to current week
    weeks_back = 0
    show_evolution = True

    # Prepare data
    df_micro = df.copy()

    # Get the latest week
    sorted_dates = sorted(df_micro['report_date_as_yyyy_mm_dd'].unique(), reverse=True)
    selected_date = sorted_dates[0]
    analysis_data = df_micro[df_micro['report_date_as_yyyy_mm_dd'] == selected_date]
    
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
                          'Top Trader Dominance', ''),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter", "colspan": 2}, None]],
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
        
        # Update layout
        week_label = "Current Week" if weeks_back == 0 else f"{weeks_back} Week{'s' if weeks_back > 1 else ''} Ago"
        fig.update_layout(
            title=f"Market Microstructure Analysis - {week_label} ({selected_date.strftime('%Y-%m-%d')})",
            height=700,
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
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Insights
        # Calculate what % of each side the average top 4 trader controls
        total_long = float(analysis_data['tot_rept_positions_long_all'].iloc[0]) if 'tot_rept_positions_long_all' in analysis_data.columns else 0
        total_short = float(analysis_data['tot_rept_positions_short'].iloc[0]) if 'tot_rept_positions_short' in analysis_data.columns else 0
        long_top4_avg = top_traders_metrics['long']['top_4_avg']
        short_top4_avg = top_traders_metrics['short']['top_4_avg']
        long_pct_of_side = (long_top4_avg / total_long * 100) if total_long > 0 else 0
        short_pct_of_side = (short_top4_avg / total_short * 100) if total_short > 0 else 0

        # Determine likely composition per side
        long_cats = sorted([m for m in category_metrics if m['side'] == 'long'], key=lambda x: x['ratio_to_top_4'])
        short_cats = sorted([m for m in category_metrics if m['side'] == 'short'], key=lambda x: x['ratio_to_top_4'])

        st.markdown("### Key Insights")
        insights_cols = st.columns(2)

        with insights_cols[0]:
            st.markdown("**Top Trader Analysis:**")
            long_metrics = top_traders_metrics['long']
            short_metrics = top_traders_metrics['short']
            st.write(f"Top 4 Long traders: {long_metrics['top_4_avg']:,.0f} contracts avg each")
            st.write(f"Top 4 Short traders: {short_metrics['top_4_avg']:,.0f} contracts avg each")
            st.write(f"Each top Long controls: {long_pct_of_side:.1f}% of all longs")
            st.write(f"Each top Short controls: {short_pct_of_side:.1f}% of all shorts")

        with insights_cols[1]:
            st.markdown("**Category Comparison:**")
            if long_cats:
                st.write(f"Likely top-4 long: {long_cats[0]['category']} ({long_cats[0]['ratio_to_top_4']:.1f}x ratio)")
            if short_cats:
                st.write(f"Likely top-4 short: {short_cats[0]['category']} ({short_cats[0]['ratio_to_top_4']:.1f}x ratio)")

            comm_avg = np.mean([m['avg_position'] for m in category_metrics if 'Commercial' in m['category']])
            noncomm_avg = np.mean([m['avg_position'] for m in category_metrics if 'Non-Commercial' in m['category']])
            st.write(f"Commercial avg: {comm_avg:,.0f} contracts")
            st.write(f"Non-Commercial avg: {noncomm_avg:,.0f} contracts")

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