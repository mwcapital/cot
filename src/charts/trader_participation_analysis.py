"""
Complete trader participation analysis charts for CFTC COT Dashboard
Includes all analysis types from legacyF.py
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from config import PARTICIPATION_CHART_HEIGHT, CONCENTRATION_COLORS


def create_market_concentration_flow(df, instrument_name):
    """Create market concentration flow analysis"""
    try:
        st.markdown("#### 🌊 Market Concentration Flow Analysis")
        
        # Explanation
        with st.expander("📖 Understanding Market Concentration Flow", expanded=False):
            st.markdown("""
            **What is Market Concentration Flow?**
            
            This analysis tracks how market concentration changes over time by monitoring:
            - The flow of traders entering/exiting the market
            - Changes in position concentration among top traders
            - The relationship between trader count changes and concentration shifts
            
            **Key Insights:**
            - **Rising concentration + Falling traders** = Market consolidation (fewer, larger players)
            - **Falling concentration + Rising traders** = Market democratization (more, smaller players)
            - **Flow intensity** = Rate of change in market structure
            """)
        
        # Calculate metrics
        df_flow = df.copy()
        df_flow = df_flow.sort_values('report_date_as_yyyy_mm_dd')
        
        # Calculate flows
        df_flow['trader_flow'] = df_flow['traders_tot_all'].diff()
        df_flow['concentration_change'] = df_flow['conc_net_le_4_tdr_long_all'].diff()
        
        # Create visualization
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[
                'Market Concentration vs Trader Count',
                'Trader Flow (Week-over-Week Change)',
                'Concentration Change vs Trader Flow'
            ]
        )
        
        # Plot 1: Dual axis - concentration and trader count
        fig.add_trace(
            go.Scatter(
                x=df_flow['report_date_as_yyyy_mm_dd'],
                y=df_flow['conc_net_le_4_tdr_long_all'],
                name='Top 4 Concentration',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Add trader count on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=df_flow['report_date_as_yyyy_mm_dd'],
                y=df_flow['traders_tot_all'],
                name='Total Traders',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Plot 2: Trader flow bars
        colors = ['green' if x > 0 else 'red' for x in df_flow['trader_flow'].fillna(0)]
        fig.add_trace(
            go.Bar(
                x=df_flow['report_date_as_yyyy_mm_dd'],
                y=df_flow['trader_flow'],
                name='Trader Flow',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Plot 3: Scatter - concentration change vs trader flow
        fig.add_trace(
            go.Scatter(
                x=df_flow['trader_flow'],
                y=df_flow['concentration_change'],
                mode='markers',
                name='Flow Relationship',
                marker=dict(
                    size=8,
                    color=df_flow.index,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time")
                )
            ),
            row=3, col=1
        )
        
        # Add quadrant lines for plot 3
        fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, row=3, col=1, line_dash="dash", line_color="gray")
        
        # Update layout
        fig.update_yaxes(title_text="Concentration %", row=1, col=1)
        fig.update_yaxes(title_text="Traders", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Trader Change", row=2, col=1)
        fig.update_yaxes(title_text="Concentration Change", row=3, col=1)
        fig.update_xaxes(title_text="Trader Flow", row=3, col=1)
        
        fig.update_layout(
            height=900,
            title=f"Market Concentration Flow - {instrument_name}",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating market concentration flow: {str(e)}")
        return None


def create_concentration_risk_heatmap(df, instrument_name, conc_metric, time_agg='Weekly', lookback_years=3):
    """Create concentration risk heatmap"""
    try:
        # Calculate lookback date
        lookback_date = df['report_date_as_yyyy_mm_dd'].max() - pd.DateOffset(years=lookback_years)
        df_heatmap = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()
        
        # Define trader categories based on selected concentration metric
        conc_metric_name, conc_metric_column = conc_metric
        
        # Determine which trader categories to include based on concentration metric direction
        if "long" in conc_metric_column.lower():
            # For long concentration metrics, only show long trader categories
            categories = [
                ("Non-Commercial Long", "traders_noncomm_long_all"),
                ("Commercial Long", "traders_comm_long_all")
                # Skip Non-Reportable as they're unlikely to be in top 4/8
            ]
        elif "short" in conc_metric_column.lower():
            # For short concentration metrics, only show short trader categories
            categories = [
                ("Non-Commercial Short", "traders_noncomm_short_all"),
                ("Commercial Short", "traders_comm_short_all")
                # Skip Non-Reportable as they're unlikely to be in top 4/8
            ]
        else:
            # Fallback to all categories if direction can't be determined
            categories = [
                ("Non-Commercial Long", "traders_noncomm_long_all"),
                ("Non-Commercial Short", "traders_noncomm_short_all"),
                ("Commercial Long", "traders_comm_long_all"),
                ("Commercial Short", "traders_comm_short_all"),
                ("Non-Reportable Long", None),
                ("Non-Reportable Short", None)
            ]
        
        # Time aggregation
        if time_agg == "Weekly":
            df_heatmap['period'] = df_heatmap['report_date_as_yyyy_mm_dd']
        elif time_agg == "Monthly":
            df_heatmap['period'] = df_heatmap['report_date_as_yyyy_mm_dd'].dt.to_period('M')
        else:  # Quarterly
            df_heatmap['period'] = df_heatmap['report_date_as_yyyy_mm_dd'].dt.to_period('Q')
        
        # Calculate historical baseline for percentiles (since 2010)
        hist_baseline = df[df['report_date_as_yyyy_mm_dd'] >= pd.Timestamp('2010-01-01')]
        
        # Prepare heatmap data
        heatmap_data = []
        y_labels = []
        
        for cat_name, trader_col in categories:
            if conc_metric_column in df_heatmap.columns:
                y_labels.append(cat_name)
                row_data = []
                
                for period in df_heatmap['period'].unique():
                    period_data = df_heatmap[df_heatmap['period'] == period]
                    
                    if len(period_data) > 0:
                        # Get latest value in period
                        latest_idx = period_data['report_date_as_yyyy_mm_dd'].idxmax()
                        latest_row = period_data.loc[latest_idx]
                        
                        # Calculate concentration score (0-100)
                        if conc_metric_column in latest_row and pd.notna(latest_row[conc_metric_column]):
                            conc_value = float(latest_row[conc_metric_column])
                            conc_score = min(conc_value / 50 * 100, 100)  # Normalize to 0-100
                        else:
                            conc_score = 0
                        
                        # Calculate trader participation score if available
                        trader_score = 0
                        if trader_col and trader_col in latest_row and pd.notna(latest_row[trader_col]):
                            trader_count = float(latest_row[trader_col])
                            if trader_col in hist_baseline.columns:
                                trader_percentile = stats.percentileofscore(
                                    hist_baseline[trader_col].dropna(), 
                                    trader_count
                                )
                                trader_score = 100 - trader_percentile  # Inverse: fewer traders = higher risk
                        
                        # Combined risk score
                        if trader_col:
                            risk_score = (conc_score * 0.7) + (trader_score * 0.3)
                        else:
                            risk_score = conc_score
                        
                        row_data.append(risk_score)
                    else:
                        row_data.append(0)
                
                heatmap_data.append(row_data)
        
        # Create heatmap
        if heatmap_data:
            # Get x-axis labels
            x_labels = []
            for period in df_heatmap['period'].unique():
                if time_agg == "Weekly":
                    x_labels.append(period.strftime('%Y-%m-%d'))
                elif time_agg == "Monthly":
                    x_labels.append(str(period))
                else:
                    x_labels.append(str(period))
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=x_labels,
                y=y_labels,
                colorscale=[
                    [0, 'green'],
                    [0.25, 'yellow'],
                    [0.5, 'orange'],
                    [0.75, 'red'],
                    [1, 'darkred']
                ],
                colorbar=dict(
                    title="Risk Score",
                    tickmode='array',
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=['0<br>Low', '25', '50<br>Medium', '75', '100<br>High']
                ),
                hovertemplate='%{y}<br>%{x}<br>Risk Score: %{z:.1f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Concentration Risk Heatmap - {instrument_name}",
                xaxis=dict(title=f"{time_agg} Periods", tickangle=-45),
                yaxis=dict(title="Trader Categories"),
                height=600
            )
            
            return fig
        else:
            st.warning("No data available for heatmap")
            return None
            
    except Exception as e:
        st.error(f"Error creating concentration risk heatmap: {str(e)}")
        return None


def create_market_structure_quadrant(df, instrument_name, conc_metric, show_evolution=True):
    """Create market structure quadrant analysis"""
    try:
        # Get the latest date
        latest_date = df['report_date_as_yyyy_mm_dd'].max()
        
        # Define specific time points to show
        time_points = []
        time_labels = []
        
        # Current
        time_points.append(latest_date)
        time_labels.append("Current")
        
        if show_evolution:
            # Past 4 weeks (detailed recent evolution)
            for weeks_ago in [1, 2, 3, 4]:
                target_date = latest_date - pd.DateOffset(weeks=weeks_ago)
                time_points.append(target_date)
                time_labels.append(f"{weeks_ago}w ago")
            
            # 2 months ago
            target_date = latest_date - pd.DateOffset(months=2)
            time_points.append(target_date)
            time_labels.append("2m ago")
            
            # 3 months ago (quarter)
            target_date = latest_date - pd.DateOffset(months=3)
            time_points.append(target_date)
            time_labels.append("3m ago")
        
        # Get data for each time point (find closest available date)
        df_points = []
        actual_dates = []
        for target_date in time_points:
            # Find closest date in data
            date_diffs = abs(df['report_date_as_yyyy_mm_dd'] - target_date)
            closest_idx = date_diffs.idxmin()
            df_points.append(df.loc[closest_idx])
            actual_dates.append(df.loc[closest_idx]['report_date_as_yyyy_mm_dd'])
        
        # Calculate historical baselines
        hist_baseline = df[df['report_date_as_yyyy_mm_dd'] >= pd.Timestamp('2010-01-01')]
        
        # Categories to plot
        categories = [
            ('Non-Commercial', 'noncomm_positions_long_all', 'traders_noncomm_long_all', 
             'noncomm_positions_short_all', 'traders_noncomm_short_all'),
            ('Commercial', 'comm_positions_long_all', 'traders_comm_long_all', 
             'comm_positions_short_all', 'traders_comm_short_all')
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add quadrant backgrounds
        fig.add_shape(type="rect", x0=0, y0=50, x1=50, y1=100,
                      fillcolor="rgba(255,0,0,0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=50, y0=50, x1=100, y1=100,
                      fillcolor="rgba(255,165,0,0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50,
                      fillcolor="rgba(255,255,0,0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=50, y0=0, x1=100, y1=50,
                      fillcolor="rgba(0,255,0,0.1)", line=dict(width=0))
        
        # Add quadrant labels
        fig.add_annotation(x=25, y=75, text="<b>Oligopolistic</b><br>Few traders<br>High concentration",
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=75, y=75, text="<b>Crowded<br>Concentration</b><br>Many traders<br>High concentration",
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=25, y=25, text="<b>Specialized</b><br>Few traders<br>Low concentration",
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=75, y=25, text="<b>Democratic</b><br>Many traders<br>Low concentration",
                          showarrow=False, font=dict(size=10))
        
        # Plot data for each category
        colors = {'Non-Commercial': 'blue', 'Commercial': 'red'}
        
        for cat_name, long_pos_col, long_trader_col, short_pos_col, short_trader_col in categories:
            # Determine which columns to use
            if 'long' in conc_metric[1]:
                trader_col = long_trader_col
                label_suffix = " (Long)"
            else:
                trader_col = short_trader_col
                label_suffix = " (Short)"
            
            # Calculate metrics for each specific time point
            x_vals = []
            y_vals = []
            sizes = []
            hover_texts = []
            opacities = []
            
            for i, (point_data, actual_date, label) in enumerate(zip(df_points, actual_dates, time_labels)):
                if pd.notna(point_data[trader_col]) and pd.notna(point_data[conc_metric[1]]):
                    # X-axis: Trader participation percentile
                    trader_count = float(point_data[trader_col])
                    trader_percentile = stats.percentileofscore(
                        hist_baseline[trader_col].dropna(), trader_count
                    )
                    
                    # Y-axis: Concentration percentile (not raw value)
                    concentration = float(point_data[conc_metric[1]])
                    conc_percentile = stats.percentileofscore(
                        hist_baseline[conc_metric[1]].dropna(), concentration
                    )
                    
                    # Size: Total open interest (normalized)
                    oi = float(point_data['open_interest_all'])
                    max_oi = hist_baseline['open_interest_all'].max()
                    size = 20 + (oi / max_oi) * 50  # Size between 20 and 70
                    
                    # Hover text
                    hover = (f"<b>{cat_name + label_suffix}</b><br>"
                           f"Date: {actual_date.strftime('%Y-%m-%d')} ({label})<br>"
                           f"Traders: {trader_count:.0f} ({trader_percentile:.1f}%ile)<br>"
                           f"Concentration: {concentration:.1f}% ({conc_percentile:.1f}%ile)<br>"
                           f"Open Interest: {oi:,.0f}")
                    
                    x_vals.append(trader_percentile)
                    y_vals.append(conc_percentile)
                    sizes.append(size)
                    hover_texts.append(hover)
                    
                    # Set opacity based on time point
                    if i == 0:  # Current
                        opacities.append(1.0)
                    elif i <= 4:  # Past 4 weeks
                        opacities.append(0.7)
                    else:  # 2m and 3m ago
                        opacities.append(0.4)
            
            # Plot evolution path if enabled (connect points with lines)
            if show_evolution and len(x_vals) > 1:
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    line=dict(color=colors[cat_name], width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Plot bubbles for each time point
            if x_vals:
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers+text',
                    name=cat_name + label_suffix,
                    marker=dict(
                        size=sizes,
                        color=colors[cat_name],
                        opacity=opacities,
                        line=dict(width=2, color='white')
                    ),
                    text=[label if i == 0 else "" for i, label in enumerate(time_labels)],
                    textposition="top center",
                    textfont=dict(size=10, color='black'),
                    hovertext=hover_texts,
                    hoverinfo='text'
                ))
                
                # Add arrow from 1 week ago to current for direction
                if show_evolution and len(x_vals) >= 2:
                    fig.add_annotation(
                        x=x_vals[1],  # 1 week ago (starting point)
                        y=y_vals[1],
                        ax=x_vals[0],  # Current (arrow points here)
                        ay=y_vals[0],
                        xref='x',
                        yref='y',
                        axref='x',
                        ayref='y',
                        showarrow=True,
                        arrowhead=2,  # Arrow at the end (current position)
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=colors[cat_name],
                        standoff=10  # Small offset from the bubble
                    )
        
        # Add center lines
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title=f"Market Structure Quadrant - {conc_metric[0]}",
            xaxis=dict(
                title="Trader Participation Percentile →<br>(Few Traders ... Many Traders)",
                range=[0, 100],
                ticksuffix="%"
            ),
            yaxis=dict(
                title="Position Concentration →<br>(Low ... High)",
                range=[0, 100],
                ticksuffix="%"
            ),
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating market structure quadrant: {str(e)}")
        return None


def create_concentration_divergence_analysis(df, instrument_name, divergence_type, conc_side='Long Positions'):
    """Create concentration divergence analysis"""
    try:
        # Filter data to periods with enough history
        df_div = df[df['report_date_as_yyyy_mm_dd'] >= pd.Timestamp('2015-01-01')].copy()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=[
                'Concentration Levels',
                'Divergence Score',
                'Probability Distribution (%)'
            ]
        )
        
        if divergence_type == "Category Divergence (Commercial vs Non-Commercial)":
            # Calculate concentration scores based on selected side
            if conc_side == "Long Positions":
                # Calculate position shares relative to total reportable long positions
                df_div['comm_position_share'] = (df_div['comm_positions_long_all'] / df_div['tot_rept_positions_long_all']) * 100
                df_div['noncomm_position_share'] = ((df_div['noncomm_positions_long_all'] + df_div['noncomm_postions_spread_all']) / df_div['tot_rept_positions_long_all']) * 100
                
                # Trader shares relative to total reportable long traders
                df_div['comm_trader_share'] = (df_div['traders_comm_long_all'] / df_div['traders_tot_rept_long_all']) * 100
                df_div['noncomm_trader_share'] = ((df_div['traders_noncomm_long_all'] + df_div['traders_noncomm_spread_all']) / df_div['traders_tot_rept_long_all']) * 100
                
            elif conc_side == "Short Positions":
                # Calculate position shares relative to total reportable short positions
                df_div['comm_position_share'] = (df_div['comm_positions_short_all'] / df_div['tot_rept_positions_short']) * 100
                df_div['noncomm_position_share'] = ((df_div['noncomm_positions_short_all'] + df_div['noncomm_postions_spread_all']) / df_div['tot_rept_positions_short']) * 100
                
                # Trader shares relative to total reportable short traders
                df_div['comm_trader_share'] = (df_div['traders_comm_short_all'] / df_div['traders_tot_rept_short_all']) * 100
                df_div['noncomm_trader_share'] = ((df_div['traders_noncomm_short_all'] + df_div['traders_noncomm_spread_all']) / df_div['traders_tot_rept_short_all']) * 100
            
            # Calculate concentration: Position Share / Trader Share
            df_div['comm_concentration'] = np.where(
                df_div['comm_trader_share'] > 0,
                df_div['comm_position_share'] / df_div['comm_trader_share'],
                0
            )
            df_div['noncomm_concentration'] = np.where(
                df_div['noncomm_trader_share'] > 0,
                df_div['noncomm_position_share'] / df_div['noncomm_trader_share'],
                0
            )
            
            # Clean infinities
            df_div['comm_concentration'] = df_div['comm_concentration'].replace([np.inf, -np.inf], 0).fillna(0)
            df_div['noncomm_concentration'] = df_div['noncomm_concentration'].replace([np.inf, -np.inf], 0).fillna(0)
            
            # Calculate divergence
            df_div['divergence'] = (df_div['noncomm_concentration'] - df_div['comm_concentration']) * 10
            
            # Plot concentrations
            fig.add_trace(go.Scatter(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['comm_concentration'],
                name='Commercial',
                line=dict(color='red', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['noncomm_concentration'],
                name='Non-Commercial',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
        elif divergence_type == "Directional Divergence (Long vs Short)":
            # Calculate concentration based on trader category
            trader_category = conc_side  # This is actually trader_cat from the UI
            
            if trader_category == "All Traders":
                # Use top 4 net concentration metrics only
                df_div['long_concentration'] = df_div['conc_net_le_4_tdr_long_all']
                df_div['short_concentration'] = df_div['conc_net_le_4_tdr_short_all']
                long_label = 'Long Side (Top 4 Net)'
                short_label = 'Short Side (Top 4 Net)'
                
            elif trader_category == "Commercial":
                # Use API-provided percentage of open interest values
                df_div['long_concentration'] = df_div['pct_of_oi_comm_long_all']
                df_div['short_concentration'] = df_div['pct_of_oi_comm_short_all']
                long_label = 'Commercial Long (% of OI)'
                short_label = 'Commercial Short (% of OI)'
                
            else:  # Non-Commercial
                # Use API-provided percentage of open interest values
                df_div['long_concentration'] = df_div['pct_of_oi_noncomm_long_all']
                df_div['short_concentration'] = df_div['pct_of_oi_noncomm_short_all']
                long_label = 'Non-Commercial Long (% of OI)'
                short_label = 'Non-Commercial Short (% of OI)'
            
            # Calculate divergence
            df_div['divergence'] = df_div['long_concentration'] - df_div['short_concentration']
            
            # Plot concentrations
            fig.add_trace(go.Scatter(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['long_concentration'],
                name=long_label,
                line=dict(color='green', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_div['report_date_as_yyyy_mm_dd'],
                y=df_div['short_concentration'],
                name=short_label,
                line=dict(color='red', width=2)
            ), row=1, col=1)
        
        # Plot divergence bars
        colors = ['green' if x > 0 else 'red' for x in df_div['divergence']]
        fig.add_trace(go.Bar(
            x=df_div['report_date_as_yyyy_mm_dd'],
            y=df_div['divergence'],
            name='Divergence',
            marker_color=colors,
            hovertemplate='Date: %{x}<br>Divergence: %{y:.2f}%<extra></extra>'
        ), row=2, col=1)
        
        # Add zero line
        fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Plot percentage histogram (probability distribution)
        divergence_clean = df_div['divergence'].dropna()
        
        # Get current (most recent) divergence value
        current_divergence = df_div['divergence'].iloc[-1]
        
        # Create histogram data
        hist_values, bin_edges = np.histogram(divergence_clean, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Convert to percentages (probability)
        total_observations = len(divergence_clean)
        probabilities = (hist_values / total_observations) * 100
        
        # Create color array - highlight the bar containing current value
        bar_colors = []
        for i, (bin_center, bin_start, bin_end) in enumerate(zip(bin_centers, bin_edges[:-1], bin_edges[1:])):
            if bin_start <= current_divergence < bin_end:
                bar_colors.append('red')  # Highlight current value bar
            else:
                bar_colors.append('lightblue')
        
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=probabilities,
            name='Probability %',
            marker_color=bar_colors,
            width=(bin_edges[1] - bin_edges[0]) * 0.8,  # Bar width
            showlegend=False,
            hovertemplate='Divergence: %{x:.2f}<br>Probability: %{y:.1f}%<extra></extra>'
        ), row=3, col=1)
        
        # Add vertical line for current value
        fig.add_vline(
            x=current_divergence,
            row=3, col=1,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Current: {current_divergence:.1f}",
            annotation_position="top",
            annotation_font_size=10
        )
        
        # Update layout
        fig.update_layout(
            title=f"Concentration Divergence Analysis - {divergence_type.split('(')[0]}",
            height=900,
            showlegend=True,
            hovermode='x unified',
            xaxis=dict(
                matches='x2',  # Link x-axis of subplot 1 with subplot 2
                showticklabels=True
            ),
            xaxis2=dict(
                matches='x',  # Link x-axis of subplot 2 with subplot 1
                showticklabels=True
            ),
            # Don't link xaxis3 as it shows divergence values, not dates
            xaxis3=dict(
                showticklabels=True
            )
        )
        
        # Update axes
        if divergence_type.startswith("Category"):
            fig.update_yaxes(title_text="Concentration Score", row=1, col=1)
        else:
            fig.update_yaxes(title_text="Concentration %", row=1, col=1)
        fig.update_yaxes(title_text="Divergence (%)", row=2, col=1)
        fig.update_yaxes(title_text="Probability (%)", row=3, col=1)
        
        # Configure x-axes
        fig.update_xaxes(type='date', row=1, col=1)
        fig.update_xaxes(type='date', row=2, col=1)
        fig.update_xaxes(title_text="Divergence Value", row=3, col=1)  # This one is not a date axis
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating concentration divergence analysis: {str(e)}")
        return None


def create_single_component_view(df_hetero, column_name, title, instrument_name, description):
    """Create a single component view for heterogeneity index"""
    fig = go.Figure()
    
    # Add the component line
    fig.add_trace(go.Scatter(
        x=df_hetero['report_date_as_yyyy_mm_dd'],
        y=df_hetero[column_name],
        name=title,
        line=dict(color='darkblue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 139, 0.1)'
    ))
    
    # Add horizontal reference lines
    fig.add_hline(y=25, line_dash="dot", line_color="green", opacity=0.5)
    fig.add_hline(y=50, line_dash="dot", line_color="orange", opacity=0.5)
    fig.add_hline(y=75, line_dash="dot", line_color="red", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title} - {instrument_name}<br><sub>{description}</sub>",
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        height=500,
        showlegend=False,
        hovermode='x unified',
        yaxis=dict(
            title=title,
            range=[0, 100],
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0', '25 (Low)', '50 (Moderate)', '75 (High)', '100']
        ),
        xaxis=dict(title='Date'),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig


def create_heterogeneity_index(df, instrument_name, component_view="Full Index"):
    """Create heterogeneity index analysis - EXACT from legacyF.py"""
    try:
        # Calculate heterogeneity components
        df_hetero = df.copy()
        
        # Component 1: Directional Opposition (25%)
        # Measures divergence in positioning relative to each group's own history
        window = 52  # 52-week lookback
        min_periods = 26  # Need at least 6 months of data
        
        # Calculate net positioning for each group
        df_hetero['comm_net'] = df_hetero['comm_positions_long_all'] - df_hetero['comm_positions_short_all']
        df_hetero['noncomm_net'] = df_hetero['noncomm_positions_long_all'] - df_hetero['noncomm_positions_short_all']
        
        # Calculate z-scores of net positions
        df_hetero['comm_net_mean'] = df_hetero['comm_net'].rolling(window, min_periods=min_periods).mean()
        df_hetero['comm_net_std'] = df_hetero['comm_net'].rolling(window, min_periods=min_periods).std()
        df_hetero['comm_net_zscore'] = np.where(
            df_hetero['comm_net_std'] > 0,
            (df_hetero['comm_net'] - df_hetero['comm_net_mean']) / df_hetero['comm_net_std'],
            0
        )
        
        df_hetero['noncomm_net_mean'] = df_hetero['noncomm_net'].rolling(window, min_periods=min_periods).mean()
        df_hetero['noncomm_net_std'] = df_hetero['noncomm_net'].rolling(window, min_periods=min_periods).std()
        df_hetero['noncomm_net_zscore'] = np.where(
            df_hetero['noncomm_net_std'] > 0,
            (df_hetero['noncomm_net'] - df_hetero['noncomm_net_mean']) / df_hetero['noncomm_net_std'],
            0
        )
        
        # Divergence = how differently positioned they are
        df_hetero['directional_divergence_raw'] = abs(df_hetero['comm_net_zscore'] - df_hetero['noncomm_net_zscore'])
        
        # Direct scaling: z-score diff of 0→0, 4→100 (capped at 100)
        df_hetero['directional_opposition_scaled'] = np.minimum(df_hetero['directional_divergence_raw'] * 25, 100)
        
        # Scale to component (0-25)
        df_hetero['directional_opposition'] = df_hetero['directional_opposition_scaled'] * 0.25
        
        # Component 2: Flow Intensity/Urgency (25%)
        # Measures unusual urgency in positioning changes
        flow_window = 52  # 52-week for flow statistics
        flow_min_periods = 26
        
        # Calculate week-over-week changes in net positions
        df_hetero['comm_flow'] = df_hetero['comm_net'].diff()
        df_hetero['noncomm_flow'] = df_hetero['noncomm_net'].diff()
        
        # Calculate z-scores of flows (measure of urgency)
        df_hetero['comm_flow_mean'] = df_hetero['comm_flow'].rolling(flow_window, min_periods=flow_min_periods).mean()
        df_hetero['comm_flow_std'] = df_hetero['comm_flow'].rolling(flow_window, min_periods=flow_min_periods).std()
        df_hetero['comm_flow_zscore'] = np.where(
            df_hetero['comm_flow_std'] > 0,
            (df_hetero['comm_flow'] - df_hetero['comm_flow_mean']) / df_hetero['comm_flow_std'],
            0
        )
        
        df_hetero['noncomm_flow_mean'] = df_hetero['noncomm_flow'].rolling(flow_window, min_periods=flow_min_periods).mean()
        df_hetero['noncomm_flow_std'] = df_hetero['noncomm_flow'].rolling(flow_window, min_periods=flow_min_periods).std()
        df_hetero['noncomm_flow_zscore'] = np.where(
            df_hetero['noncomm_flow_std'] > 0,
            (df_hetero['noncomm_flow'] - df_hetero['noncomm_flow_mean']) / df_hetero['noncomm_flow_std'],
            0
        )
        
        # Flow intensity divergence = difference in flow urgency
        df_hetero['flow_intensity_raw'] = abs(df_hetero['comm_flow_zscore'] - df_hetero['noncomm_flow_zscore'])
        
        # Direct scaling: z-score diff of 0→0, 4→100 (capped at 100)
        df_hetero['flow_intensity_scaled'] = np.minimum(df_hetero['flow_intensity_raw'] * 25, 100)
        
        # Scale to component weight (0-25)
        df_hetero['flow_divergence'] = df_hetero['flow_intensity_scaled'] * 0.25
        
        # Component 3: Percentile Distance Approach (25%)
        # Measures divergence between commercial and non-commercial average position sizes using percentile rankings
        
        # Calculate average positions for each group
        df_hetero['nc_long_avg'] = df_hetero['noncomm_positions_long_all'] / df_hetero['traders_noncomm_long_all']
        df_hetero['c_long_avg'] = df_hetero['comm_positions_long_all'] / df_hetero['traders_comm_long_all']
        
        # Replace inf/nan with 0
        df_hetero['nc_long_avg'] = df_hetero['nc_long_avg'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df_hetero['c_long_avg'] = df_hetero['c_long_avg'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate percentile rank for each group's average position (52-week history)
        df_hetero['nc_long_avg_percentile'] = df_hetero['nc_long_avg'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        df_hetero['c_long_avg_percentile'] = df_hetero['c_long_avg'].rolling(window, min_periods=min_periods).rank(pct=True) * 100
        
        # Absolute distance between percentiles
        df_hetero['percentile_distance'] = abs(df_hetero['nc_long_avg_percentile'] - df_hetero['c_long_avg_percentile'])
        
        # Direct scaling: already 0-100, just scale to component weight (0-25)
        df_hetero['commitment_divergence'] = df_hetero['percentile_distance'] * 0.25
        
        # Component 4: Cross-Category Positioning Divergence (25%)
        # Measures when commercials and non-commercials are on opposite sides
        
        # Calculate % of OI for each category (using API values if available)
        if 'pct_of_oi_noncomm_long_all' in df_hetero.columns:
            df_hetero['pct_oi_nc_long'] = df_hetero['pct_of_oi_noncomm_long_all']
            df_hetero['pct_oi_nc_short'] = df_hetero['pct_of_oi_noncomm_short_all']
            df_hetero['pct_oi_c_long'] = df_hetero['pct_of_oi_comm_long_all']
            df_hetero['pct_oi_c_short'] = df_hetero['pct_of_oi_comm_short_all']
        else:
            # Calculate manually if API values not available
            df_hetero['pct_oi_nc_long'] = (df_hetero['noncomm_positions_long_all'] / df_hetero['open_interest_all']) * 100
            df_hetero['pct_oi_nc_short'] = (df_hetero['noncomm_positions_short_all'] / df_hetero['open_interest_all']) * 100
            df_hetero['pct_oi_c_long'] = (df_hetero['comm_positions_long_all'] / df_hetero['open_interest_all']) * 100
            df_hetero['pct_oi_c_short'] = (df_hetero['comm_positions_short_all'] / df_hetero['open_interest_all']) * 100
        
        # Calculate cross-category averages
        # When NC long and C short align (same direction)
        df_hetero['aligned_nc_long_c_short'] = (df_hetero['pct_oi_nc_long'] + df_hetero['pct_oi_c_short']) / 2
        
        # When NC short and C long align (opposite direction)
        df_hetero['aligned_nc_short_c_long'] = (df_hetero['pct_oi_nc_short'] + df_hetero['pct_oi_c_long']) / 2
        
        # Cross-category divergence: difference between the two alignments
        df_hetero['cross_category_divergence_raw'] = abs(
            df_hetero['aligned_nc_long_c_short'] - df_hetero['aligned_nc_short_c_long']
        )
        
        # Direct scaling: 0→0, 50%→100 (typical max around 40-50%)
        df_hetero['cross_category_scaled'] = np.minimum(df_hetero['cross_category_divergence_raw'] * 2, 100)
        
        # Scale to component weight (0-25)
        df_hetero['directional_bias_divergence'] = df_hetero['cross_category_scaled'] * 0.25
        
        # Combine all components into final index
        df_hetero['heterogeneity_index'] = (
            df_hetero['directional_opposition'] +
            df_hetero['flow_divergence'] +
            df_hetero['commitment_divergence'] +
            df_hetero['directional_bias_divergence']
        ).clip(0, 100)
        
        # Handle component view selection
        if component_view == "Directional Opposition":
            return create_single_component_view(df_hetero, 'directional_opposition_scaled', 
                                              'Directional Opposition (0-100)', instrument_name,
                                              "Z-score divergence in net positions")
        elif component_view == "Flow Intensity":
            return create_single_component_view(df_hetero, 'flow_intensity_scaled', 
                                              'Flow Intensity (0-100)', instrument_name,
                                              "Z-score divergence in position flows")
        elif component_view == "Percentile Distance":
            return create_single_component_view(df_hetero, 'percentile_distance', 
                                              'Percentile Distance (0-100)', instrument_name,
                                              "Distance between average position percentiles")
        elif component_view == "Cross-Category Positioning":
            return create_single_component_view(df_hetero, 'cross_category_scaled', 
                                              'Cross-Category Positioning (0-100)', instrument_name,
                                              "Opposing market positions between groups")
        
        # Default: Create full visualization
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.3, 0.2],
            subplot_titles=[
                'Heterogeneity Index (0-100)',
                'Component Breakdown',
                'Regime Classification'
            ]
        )
        
        # 1. Main Heterogeneity Index
        fig.add_trace(
            go.Scatter(
                x=df_hetero['report_date_as_yyyy_mm_dd'],
                y=df_hetero['heterogeneity_index'],
                name='Heterogeneity Index',
                line=dict(color='black', width=2),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add regime zones with subtle colors
        fig.add_hrect(y0=0, y1=25, fillcolor="lightgreen", opacity=0.3, row=1, col=1, line_width=0)
        fig.add_hrect(y0=25, y1=50, fillcolor="lightyellow", opacity=0.3, row=1, col=1, line_width=0)
        fig.add_hrect(y0=50, y1=75, fillcolor="lightcoral", opacity=0.2, row=1, col=1, line_width=0)
        fig.add_hrect(y0=75, y1=100, fillcolor="lightpink", opacity=0.3, row=1, col=1, line_width=0)
        
        # 2. Component Breakdown
        components = ['directional_opposition', 'flow_divergence', 'commitment_divergence', 'directional_bias_divergence']
        colors = ['red', 'blue', 'green', 'orange']
        labels = ['Directional Opposition (25%)', 'Flow Intensity (25%)', 'Percentile Distance (25%)', 'Cross-Category Positioning (25%)']
        
        for comp, color, label in zip(components, colors, labels):
            fig.add_trace(
                go.Scatter(
                    x=df_hetero['report_date_as_yyyy_mm_dd'],
                    y=df_hetero[comp],
                    name=label,
                    stackgroup='components',
                    fillcolor=color,
                    line=dict(color=color, width=0.5)
                ),
                row=2, col=1
            )
        
        # 3. Regime Classification
        df_hetero['regime'] = pd.cut(
            df_hetero['heterogeneity_index'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low Divergence', 'Moderate', 'High Divergence', 'Extreme']
        )
        
        # Create regime indicator
        regime_colors = {
            'Low Divergence': 'green',
            'Moderate': 'yellow',
            'High Divergence': 'orange',
            'Extreme': 'red'
        }
        
        for regime, color in regime_colors.items():
            mask = df_hetero['regime'] == regime
            fig.add_trace(
                go.Scatter(
                    x=df_hetero.loc[mask, 'report_date_as_yyyy_mm_dd'],
                    y=[1] * mask.sum(),
                    mode='markers',
                    marker=dict(color=color, size=10, symbol='square'),
                    name=regime,
                    showlegend=True
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_yaxes(title_text="Index Value", range=[0, 100], row=1, col=1, 
                        showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Contribution", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Regime", range=[0, 2], row=3, col=1, showticklabels=False)
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        fig.update_layout(
            title=dict(
                text=f"Market Heterogeneity Analysis - {instrument_name}",
                font=dict(size=16),
                x=0.5,
                xanchor='center'
            ),
            height=900,
            showlegend=True,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=60, r=40, t=60, b=60),  # Tighter margins for full width
            autosize=True
        )
        
        # Remove all annotations to keep chart clean
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating heterogeneity index: {str(e)}")
        return None


def create_spreading_activity_analysis(df, instrument_name):
    """Create spreading activity analysis"""
    try:
        # Copy and prepare data
        df_spread = df.copy()
        df_spread = df_spread.sort_values('report_date_as_yyyy_mm_dd')
        
        # Calculate spread-to-directional ratio
        df_spread['spread_directional_ratio'] = df_spread['traders_noncomm_spread_all'] / (
            df_spread['traders_noncomm_long_all'] + df_spread['traders_noncomm_short_all']
        )
        
        # Clean any infinities or NaN
        df_spread['spread_directional_ratio'] = df_spread['spread_directional_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # Create subplots with secondary y-axis for the first subplot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=[
                'Non-Commercial Spread/Directional Trader Ratio',
                'Probability Distribution (Since 2010)'
            ],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Plot 1a: Spread trader count as bars at the bottom
        fig.add_trace(
            go.Bar(
                x=df_spread['report_date_as_yyyy_mm_dd'],
                y=df_spread['traders_noncomm_spread_all'],
                name='Spread Traders',
                marker_color='darkgray',
                opacity=0.7,
                yaxis='y2',
                hovertemplate='Date: %{x}<br>Spread Traders: %{y}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Plot 1b: Main ratio over time
        fig.add_trace(
            go.Scatter(
                x=df_spread['report_date_as_yyyy_mm_dd'],
                y=df_spread['spread_directional_ratio'],
                name='Spread/Directional Ratio',
                line=dict(color='purple', width=2),
                hovertemplate='Date: %{x}<br>Ratio: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False
        )
        
        
        # Plot 2: Distribution since 2010
        df_dist = df_spread[df_spread['report_date_as_yyyy_mm_dd'] >= pd.Timestamp('2010-01-01')]
        ratio_clean = df_dist['spread_directional_ratio'].dropna()
        
        # Create histogram data
        hist_values, bin_edges = np.histogram(ratio_clean, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Convert to percentages (probability)
        total_observations = len(ratio_clean)
        probabilities = (hist_values / total_observations) * 100
        
        # Get current value for highlighting
        current_ratio = df_spread['spread_directional_ratio'].iloc[-1]
        
        # Create color array - highlight the bar containing current value
        bar_colors = []
        for i, (bin_center, bin_start, bin_end) in enumerate(zip(bin_centers, bin_edges[:-1], bin_edges[1:])):
            if bin_start <= current_ratio < bin_end:
                bar_colors.append('red')  # Highlight current value bar
            else:
                bar_colors.append('lightblue')
        
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=probabilities,
            name='Probability %',
            marker_color=bar_colors,
            width=(bin_edges[1] - bin_edges[0]) * 0.8,
            showlegend=False,
            hovertemplate='Ratio: %{x:.3f}<br>Probability: %{y:.1f}%<extra></extra>'
        ), row=2, col=1)
        
        # Add vertical line for current value
        fig.add_vline(
            x=current_ratio,
            row=2, col=1,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Current: {current_ratio:.3f}",
            annotation_position="top",
            annotation_font_size=10
        )
        
        # Update layout
        fig.update_layout(
            title=f"Non-Commercial Spreading Activity - {instrument_name}",
            height=700,
            showlegend=False,
            hovermode='x unified',
            xaxis=dict(
                matches='x2',
                showticklabels=True
            ),
            xaxis2=dict(
                matches='x',
                showticklabels=True
            )
        )
        
        # Update axes
        fig.update_yaxes(title_text="Spread/Directional Ratio", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Spread Traders", row=1, col=1, secondary_y=True, showgrid=False)
        fig.update_yaxes(title_text="Probability (%)", row=2, col=1)
        fig.update_xaxes(title_text="Ratio Value", row=2, col=1)
        
        # Configure x-axes
        fig.update_xaxes(type='date', row=1, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating spreading activity analysis: {str(e)}")
        return None