"""
Cross-Asset Analysis for CFTC COT Dashboard
Exact implementation from legacyF.py
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_fetcher import fetch_cftc_data


def create_cross_asset_analysis(selected_instruments, trader_category, api_token, lookback_start, show_week_ago, instruments_db):
    """Create cross-asset positioning analysis chart"""
    try:
        # Store data for all instruments
        instrument_data = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data for each instrument
        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Fetching data for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))
            
            # Fetch data
            df = fetch_cftc_data(instrument, api_token)
            
            if df is not None and not df.empty:
                # Filter data from lookback start date
                df = df[df['report_date_as_yyyy_mm_dd'] >= pd.Timestamp(lookback_start)]
                
                if not df.empty:
                    # Define column mappings
                    category_columns = {
                        "Non-Commercial": {
                            "long": "noncomm_positions_long_all",
                            "short": "noncomm_positions_short_all",
                            "net": "net_noncomm_positions"
                        },
                        "Commercial": {
                            "long": "comm_positions_long_all",
                            "short": "comm_positions_short_all",
                            "net": "net_comm_positions"
                        },
                        "Non-Reportable": {
                            "long": "nonrept_positions_long_all",
                            "short": "nonrept_positions_short_all"
                        }
                    }
                    
                    cols = category_columns[trader_category]
                    
                    # Calculate net if not present
                    if trader_category == "Non-Reportable" and "net" not in cols:
                        df['net_nonrept_positions'] = df[cols['long']] - df[cols['short']]
                        cols['net'] = 'net_nonrept_positions'
                    
                    # Calculate net as % of open interest
                    if 'open_interest_all' in df.columns and cols.get('net') in df.columns:
                        df['net_pct_oi'] = (df[cols['net']] / df['open_interest_all'] * 100).fillna(0)
                        
                        # Calculate z-score
                        mean_pct = df['net_pct_oi'].mean()
                        std_pct = df['net_pct_oi'].std()
                        
                        if std_pct > 0:
                            # Get latest and week-ago values
                            latest_pct = df['net_pct_oi'].iloc[-1] if len(df) > 0 else 0
                            latest_z = (latest_pct - mean_pct) / std_pct
                            
                            week_ago_z = None
                            if show_week_ago and len(df) > 1:
                                week_ago_pct = df['net_pct_oi'].iloc[-2]
                                week_ago_z = (week_ago_pct - mean_pct) / std_pct
                            
                            instrument_data[instrument] = {
                                'z_score': latest_z,
                                'week_ago_z': week_ago_z,
                                'net_pct': latest_pct,
                                'mean': mean_pct,
                                'std': std_pct
                            }
        
        progress_bar.empty()
        status_text.empty()
        
        if not instrument_data:
            st.error("No valid data found for selected instruments")
            return None
        
        # Sort by z-score
        sorted_instruments = sorted(instrument_data.items(), key=lambda x: x[1]['z_score'], reverse=True)
        
        # Create the chart
        fig = go.Figure()
        
        # Prepare data for plotting
        instruments_full = [item[0] for item in sorted_instruments]
        # Shorten instrument names - take only the part before first "-"
        instruments_short = [name.split('-')[0].strip() for name in instruments_full]
        z_scores = [item[1]['z_score'] for item in sorted_instruments]
        
        # Nice mint/salad green color
        bar_color = '#7DCEA0'  # Soft mint green
        
        # Add bars
        fig.add_trace(go.Bar(
            x=instruments_short,
            y=z_scores,
            name='Current Z-Score',
            marker=dict(color=bar_color),
            text=[f"{z:.2f}" for z in z_scores],
            textposition='outside',
            hovertemplate='<b>%{customdata[3]}</b><br>' +  # Show full name in hover
                         'Z-Score: %{y:.2f}<br>' +
                         'Net %: %{customdata[0]:.1f}%<br>' +
                         'Mean %: %{customdata[1]:.1f}%<br>' +
                         'Std %: %{customdata[2]:.1f}%<extra></extra>',
            customdata=[[item[1]['net_pct'], item[1]['mean'], item[1]['std'], full_name] 
                       for item, full_name in zip(sorted_instruments, instruments_full)]
        ))
        
        # Add week-ago markers if requested
        if show_week_ago:
            week_ago_zs = [item[1]['week_ago_z'] for item in sorted_instruments]
            valid_week_ago = [(i, z) for i, z in enumerate(week_ago_zs) if z is not None]
            
            if valid_week_ago:
                indices, week_z_values = zip(*valid_week_ago)
                fig.add_trace(go.Scatter(
                    x=[instruments_short[i] for i in indices],
                    y=week_z_values,
                    mode='markers',
                    name='Week Ago',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color='purple',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='<b>%{customdata}</b><br>Week Ago Z-Score: %{y:.2f}<extra></extra>',
                    customdata=[instruments_full[i] for i in indices]
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{trader_category} Net Positioning Z-Scores (since {lookback_start.year})",
            xaxis_title="",
            yaxis_title="Z-Score",
            height=600,
            showlegend=True,
            hovermode='x unified',
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                gridcolor='lightgray'
            ),
            xaxis=dict(
                tickangle=-45
            ),
            plot_bgcolor='white',
            bargap=0.2
        )
        
        # Add reference lines
        fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1, 
                     annotation_text="", annotation_position="right")
        fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1,
                     annotation_text="", annotation_position="right")
        fig.add_hline(y=1, line_dash="dot", line_color="gray", line_width=1)
        fig.add_hline(y=-1, line_dash="dot", line_color="gray", line_width=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating cross-asset analysis: {str(e)}")
        return None


def create_cross_asset_wow_changes(selected_instruments, trader_category, api_token, instruments_db):
    """Create week-over-week changes analysis chart"""
    try:
        # Store data for all instruments
        instrument_data = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data for each instrument
        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Fetching data for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))
            
            # Fetch data
            df = fetch_cftc_data(instrument, api_token)
            
            if df is not None and not df.empty:
                # Sort by date
                df = df.sort_values('report_date_as_yyyy_mm_dd')
                
                if len(df) >= 2:  # Need at least 2 observations for change
                    # Define column mappings
                    category_columns = {
                        "Non-Commercial": {
                            "long": "noncomm_positions_long_all",
                            "short": "noncomm_positions_short_all",
                            "net": "net_noncomm_positions"
                        },
                        "Commercial": {
                            "long": "comm_positions_long_all",
                            "short": "comm_positions_short_all",
                            "net": "net_comm_positions"
                        },
                        "Non-Reportable": {
                            "long": "nonrept_positions_long_all",
                            "short": "nonrept_positions_short_all"
                        }
                    }
                    
                    cols = category_columns[trader_category]
                    
                    # Calculate net if not present
                    if trader_category == "Non-Reportable" and "net" not in cols:
                        df['net_nonrept_positions'] = df[cols['long']] - df[cols['short']]
                        cols['net'] = 'net_nonrept_positions'
                    
                    # Calculate week-over-week change as % of OI
                    if len(df) >= 2 and 'open_interest_all' in df.columns and cols.get('net') in df.columns:
                        latest_net = df[cols['net']].iloc[-1]
                        prev_net = df[cols['net']].iloc[-2]
                        latest_oi = df['open_interest_all'].iloc[-1]
                        
                        if latest_oi > 0:
                            change_pct_oi = ((latest_net - prev_net) / latest_oi) * 100
                            
                            # Get current date
                            current_date = df['report_date_as_yyyy_mm_dd'].iloc[-1]
                            
                            instrument_data[instrument] = {
                                'change_pct_oi': change_pct_oi,
                                'latest_net': latest_net,
                                'prev_net': prev_net,
                                'latest_oi': latest_oi,
                                'current_date': current_date
                            }
        
        progress_bar.empty()
        status_text.empty()
        
        if not instrument_data:
            st.error("No valid data found for selected instruments")
            return None
        
        # Sort by change percentage (largest positive to largest negative)
        sorted_instruments = sorted(instrument_data.items(), key=lambda x: x[1]['change_pct_oi'], reverse=True)
        
        # Create the chart
        fig = go.Figure()
        
        # Prepare data for plotting
        instruments_full = [item[0] for item in sorted_instruments]
        # Shorten instrument names - take only the part before first "-"
        instruments_short = [name.split('-')[0].strip() for name in instruments_full]
        changes = [item[1]['change_pct_oi'] for item in sorted_instruments]
        
        # Nice mint/salad green color
        bar_color = '#7DCEA0'  # Soft mint green
        
        # Add bars
        fig.add_trace(go.Bar(
            x=instruments_short,
            y=changes,
            name='WoW Change',
            marker=dict(color=bar_color),
            text=[f"{c:.2f}%" for c in changes],
            textposition='outside',
            hovertemplate='<b>%{customdata[4]}</b><br>' +  # Show full name in hover
                         'WoW Change: %{y:.2f}% of OI<br>' +
                         'Latest Net: %{customdata[0]:,.0f}<br>' +
                         'Previous Net: %{customdata[1]:,.0f}<br>' +
                         'Open Interest: %{customdata[2]:,.0f}<br>' +
                         'Date: %{customdata[3]}<extra></extra>',
            customdata=[[item[1]['latest_net'], item[1]['prev_net'], item[1]['latest_oi'], 
                        item[1]['current_date'].strftime('%Y-%m-%d'), full_name] 
                       for item, full_name in zip(sorted_instruments, instruments_full)]
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{trader_category} Week-over-Week Position Changes (% of OI)",
            xaxis_title="",
            yaxis_title="Change (% of Open Interest)",
            height=600,
            showlegend=False,
            hovermode='x unified',
            yaxis=dict(
                zeroline=True,
                zerolinewidth=3,
                zerolinecolor='black',
                gridcolor='lightgray'
            ),
            xaxis=dict(
                tickangle=-45
            ),
            plot_bgcolor='white',
            bargap=0.2
        )
        
        # Add current date annotation (top-right corner)
        if instrument_data:
            sample_date = list(instrument_data.values())[0]['current_date']
            fig.add_annotation(
                x=1,
                y=1,
                xref='paper',
                yref='paper',
                text=f"Data as of: {sample_date.strftime('%Y-%m-%d')}",
                showarrow=False,
                font=dict(size=10),
                xanchor='right',
                yanchor='top'
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating week-over-week changes analysis: {str(e)}")
        return None


# Placeholder functions for other chart types - can be implemented later if needed
def create_positioning_concentration_charts(selected_instruments, trader_category, api_token, instruments_db):
    """Create time series and bar charts for positioning concentration analysis"""
    try:
        # Store data for all instruments
        all_data = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data for each instrument
        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Fetching data for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))
            
            # Fetch data
            df = fetch_cftc_data(instrument, api_token)
            
            if df is not None and not df.empty:
                # Sort by date
                df = df.sort_values('report_date_as_yyyy_mm_dd')
                
                # Define column mappings
                category_columns = {
                    "Non-Commercial": {
                        "long": "noncomm_positions_long_all",
                        "short": "noncomm_positions_short_all",
                        "net": "net_noncomm_positions"
                    },
                    "Commercial": {
                        "long": "comm_positions_long_all",
                        "short": "comm_positions_short_all",
                        "net": "net_comm_positions"
                    },
                    "Non-Reportable": {
                        "long": "nonrept_positions_long_all",
                        "short": "nonrept_positions_short_all"
                    }
                }
                
                cols = category_columns[trader_category]
                
                # Calculate net if not present
                if trader_category == "Non-Reportable" and "net" not in cols:
                    df['net_nonrept_positions'] = df[cols['long']] - df[cols['short']]
                    cols['net'] = 'net_nonrept_positions'
                
                # Calculate net as % of open interest
                if 'open_interest_all' in df.columns and cols.get('net') in df.columns:
                    df['net_pct_oi'] = (df[cols['net']] / df['open_interest_all'] * 100).fillna(0)
                    
                    # Get trader count columns based on category
                    trader_count_cols = {
                        "Non-Commercial": {
                            "long": "traders_noncomm_long_all",
                            "short": "traders_noncomm_short_all"
                        },
                        "Commercial": {
                            "long": "traders_comm_long_all", 
                            "short": "traders_comm_short_all"
                        },
                        "Non-Reportable": {
                            # Non-reportable doesn't have separate trader counts in the API
                            "long": None,
                            "short": None
                        }
                    }
                    
                    trader_cols = trader_count_cols[trader_category]
                    
                    # Calculate average position per trader if trader count data is available
                    avg_pos_long = None
                    avg_pos_short = None
                    
                    if trader_cols['long'] and trader_cols['short']:
                        if trader_cols['long'] in df.columns and trader_cols['short'] in df.columns:
                            # Calculate average position per trader
                            df['avg_pos_long'] = df[cols['long']] / df[trader_cols['long']].replace(0, 1)
                            df['avg_pos_short'] = df[cols['short']] / df[trader_cols['short']].replace(0, 1)
                            avg_pos_long = df['avg_pos_long']
                            avg_pos_short = df['avg_pos_short']
                    
                    # Store the data
                    all_data[instrument] = {
                        'dates': df['report_date_as_yyyy_mm_dd'],
                        'net_pct_oi': df['net_pct_oi'],
                        'latest_pct': df['net_pct_oi'].iloc[-1] if len(df) > 0 else 0,
                        'avg_pos_long': avg_pos_long,
                        'avg_pos_short': avg_pos_short
                    }
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            st.error("No valid data found for selected instruments")
            return None, None
        
        # Create time series chart
        time_series_fig = go.Figure()
        
        # Define a color palette for different instruments
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Add traces for each instrument
        for idx, (instrument, data) in enumerate(all_data.items()):
            # Shorten name for display
            short_name = instrument.split('-')[0].strip()
            
            time_series_fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data['net_pct_oi'],
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate=f'<b>{instrument}</b><br>' +
                             'Date: %{x}<br>' +
                             'Net % of OI: %{y:.1f}%<extra></extra>'
            ))
        
        # Update time series layout
        time_series_fig.update_layout(
            title=f"{trader_category} Positioning as % of Open Interest",
            xaxis_title="Date",
            yaxis_title="Net Positioning (% of OI)",
            height=500,
            hovermode='x unified',
            yaxis=dict(
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                gridcolor='lightgray',
                ticksuffix='%'
            ),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Create bar chart for latest values
        bar_chart_fig = go.Figure()
        
        # Sort by latest percentage
        sorted_instruments = sorted(all_data.items(), 
                                  key=lambda x: x[1]['latest_pct'], 
                                  reverse=True)
        
        # Prepare data for bar chart
        instruments_full = [item[0] for item in sorted_instruments]
        instruments_short = [name.split('-')[0].strip() for name in instruments_full]
        latest_values = [item[1]['latest_pct'] for item in sorted_instruments]
        
        # Use mint green for positive, red for negative
        colors_bar = ['#7DCEA0' if v > 0 else '#E74C3C' for v in latest_values]
        
        # Add bars
        bar_chart_fig.add_trace(go.Bar(
            x=instruments_short,
            y=latest_values,
            marker=dict(color=colors_bar),
            text=[f"{v:.1f}%" for v in latest_values],
            textposition='outside',
            hovertemplate='<b>%{customdata}</b><br>' +
                         'Net % of OI: %{y:.1f}%<extra></extra>',
            customdata=instruments_full
        ))
        
        # Update bar chart layout
        bar_chart_fig.update_layout(
            title=f"Latest {trader_category} Positioning (% of OI)",
            xaxis_title="",
            yaxis_title="Net Positioning (% of OI)",
            height=400,
            showlegend=False,
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                gridcolor='lightgray',
                ticksuffix='%'
            ),
            xaxis=dict(
                tickangle=-45
            ),
            bargap=0.2
        )
        
        # Add annotation with data date
        if sorted_instruments:
            latest_date = all_data[sorted_instruments[0][0]]['dates'].iloc[-1]
            bar_chart_fig.add_annotation(
                text=f"Data as of: {latest_date.strftime('%Y-%m-%d')}",
                xref="paper", yref="paper",
                x=0.99, y=0.99,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right", yanchor="top"
            )
        
        # Create average position per trader time series chart
        avg_positions_fig = go.Figure()
        
        # Check if we have valid data for average positions
        has_avg_data = any(all_data[inst]['avg_pos_long'] is not None for inst in all_data)
        
        if has_avg_data:
            # Add traces for each instrument - both long and short
            for idx, (instrument, data) in enumerate(all_data.items()):
                if data['avg_pos_long'] is not None and data['avg_pos_short'] is not None:
                    # Shorten name for display
                    short_name = instrument.split('-')[0].strip()
                    
                    # Add long positions line
                    avg_positions_fig.add_trace(go.Scatter(
                        x=data['dates'],
                        y=data['avg_pos_long'],
                        mode='lines',
                        name=f'{short_name} Long',
                        line=dict(color=colors[idx % len(colors)], width=2),
                        hovertemplate=f'<b>{instrument}</b><br>' +
                                     'Date: %{x}<br>' +
                                     f'{trader_category} Long Avg: %{{y:,.0f}} contracts/trader<extra></extra>'
                    ))
                    
                    # Add short positions line
                    avg_positions_fig.add_trace(go.Scatter(
                        x=data['dates'],
                        y=data['avg_pos_short'],
                        mode='lines',
                        name=f'{short_name} Short',
                        line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
                        hovertemplate=f'<b>{instrument}</b><br>' +
                                     'Date: %{x}<br>' +
                                     f'{trader_category} Short Avg: %{{y:,.0f}} contracts/trader<extra></extra>'
                    ))
            
            # Update average positions layout
            avg_positions_fig.update_layout(
                title=f"Average {trader_category} Position per Trader",
                xaxis_title="Date",
                yaxis_title="Average Position per Trader (Contracts)",
                height=500,
                hovermode='x unified',
                yaxis=dict(
                    gridcolor='lightgray',
                    tickformat=','
                ),
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(count=2, label="2Y", step="year", stepmode="backward"),
                            dict(count=5, label="5Y", step="year", stepmode="backward"),
                            dict(step="all", label="All")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        else:
            # Show info message if no trader count data available
            avg_positions_fig.add_annotation(
                text=f"Average position per trader data not available for {trader_category} category",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray"),
                xanchor="center", yanchor="middle"
            )
            avg_positions_fig.update_layout(
                title=f"Average {trader_category} Position per Trader",
                xaxis_title="Date",
                yaxis_title="Average Position per Trader (Contracts)",
                height=400
            )
        
        return time_series_fig, bar_chart_fig, avg_positions_fig
        
    except Exception as e:
        st.error(f"Error creating positioning concentration charts: {str(e)}")
        return None, None, None


def create_cross_asset_participation_comparison(selected_instruments, api_token, instruments_db):
    """Create cross-asset participation comparison showing trader count trends"""
    try:
        # Store data for all instruments
        all_data = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data for each instrument
        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Fetching data for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))
            
            # Fetch data
            df = fetch_cftc_data(instrument, api_token)
            
            if df is not None and not df.empty:
                # Sort by date
                df = df.sort_values('report_date_as_yyyy_mm_dd')
                
                # Store the data
                all_data[instrument] = {
                    'dates': df['report_date_as_yyyy_mm_dd'],
                    'total_traders': df['traders_tot_all'],
                    'noncomm_long': df['traders_noncomm_long_all'],
                    'noncomm_short': df['traders_noncomm_short_all'],
                    'comm_long': df['traders_comm_long_all'],
                    'comm_short': df['traders_comm_short_all'],
                    'open_interest': df['open_interest_all']
                }
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            st.error("No valid data found for selected instruments")
            return None
        
        # Create figure with subplots
        from plotly.subplots import make_subplots
        import plotly.express as px
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Trader Count', 'Trader Count YoY % Change', 
                           'Avg Position per Trader', 'Participation Score'),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Define colors for instruments
        colors = px.colors.qualitative.Plotly[:len(selected_instruments)]
        
        # 1. Total trader count time series
        for idx, (instrument, data) in enumerate(all_data.items()):
            short_name = instrument.split('-')[0].strip()
            
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data['total_traders'],
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx], width=2),
                showlegend=True,
                legendgroup=short_name,
                hovertemplate=f'<b>{instrument}</b><br>Date: %{{x}}<br>Traders: %{{y:,.0f}}<extra></extra>'
            ), row=1, col=1)
        
        # 2. YoY % change in trader count
        for idx, (instrument, data) in enumerate(all_data.items()):
            short_name = instrument.split('-')[0].strip()
            
            # Calculate YoY change
            import pandas as pd
            trader_series = pd.Series(data['total_traders'].values, index=data['dates'])
            yoy_change = trader_series.pct_change(52) * 100  # 52 weeks = 1 year
            
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=yoy_change,
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx], width=2),
                showlegend=False,
                legendgroup=short_name,
                hovertemplate=f'<b>{instrument}</b><br>Date: %{{x}}<br>YoY Change: %{{y:.1f}}%<extra></extra>'
            ), row=1, col=2)
        
        # 3. Average position per trader
        for idx, (instrument, data) in enumerate(all_data.items()):
            short_name = instrument.split('-')[0].strip()
            
            # Calculate avg position per trader
            avg_position = data['open_interest'] / data['total_traders'].replace(0, 1)
            
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=avg_position,
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx], width=2),
                showlegend=False,
                legendgroup=short_name,
                hovertemplate=f'<b>{instrument}</b><br>Date: %{{x}}<br>Avg Position: %{{y:,.0f}}<extra></extra>'
            ), row=2, col=1)
        
        # 4. Participation score (traders as % of max historical)
        for idx, (instrument, data) in enumerate(all_data.items()):
            short_name = instrument.split('-')[0].strip()
            
            # Calculate participation score
            max_traders = data['total_traders'].max()
            participation_score = (data['total_traders'] / max_traders) * 100
            
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=participation_score,
                mode='lines',
                name=short_name,
                line=dict(color=colors[idx], width=2),
                showlegend=False,
                legendgroup=short_name,
                hovertemplate=f'<b>{instrument}</b><br>Date: %{{x}}<br>Participation: %{{y:.1f}}%<extra></extra>'
            ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Cross-Asset Participation Comparison",
                y=0.98,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=20)
            ),
            height=900,
            hovermode='x unified',
            margin=dict(t=120, b=50, l=60, r=40)
        )
        
        # Update axes
        fig.update_yaxes(title_text="Number of Traders", row=1, col=1)
        fig.update_yaxes(title_text="YoY Change %", row=1, col=2)
        fig.update_yaxes(title_text="Avg Position Size", row=2, col=1)
        fig.update_yaxes(title_text="Participation %", row=2, col=2)
        
        # Add zero line for YoY change
        fig.add_hline(y=0, row=1, col=2, line_dash="dash", line_color="gray")
        
        # Add range selector to first subplot
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                yanchor="top",
                y=1.06
            ),
            row=1, col=1
        )
        
        # Update the y-axis domain of the first row to leave space for range selector
        fig.update_yaxes(domain=[0.55, 0.92], row=1, col=1)
        fig.update_yaxes(domain=[0.55, 0.92], row=1, col=2)
        fig.update_yaxes(domain=[0, 0.37], row=2, col=1)
        fig.update_yaxes(domain=[0, 0.37], row=2, col=2)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating participation comparison: {str(e)}")
        return None


def create_relative_strength_matrix(selected_instruments, api_token, time_period, instruments_db):
    """Create relative strength heatmap matrix"""
    st.info("Relative strength matrix - to be implemented")
    return None


def create_market_structure_matrix(all_instruments_data, selected_instruments, concentration_metric='average_net_4'):
    """Create market structure matrix showing instruments on 2x2 grid based on trader count and concentration percentiles"""
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare data for all selected instruments
        scatter_data = []
        
        # Calculate 5-year lookback date
        lookback_date = pd.Timestamp.now() - pd.DateOffset(years=5)
        
        for idx, instrument in enumerate(selected_instruments):
            status_text.text(f"Calculating percentiles for {instrument}...")
            progress_bar.progress((idx + 1) / len(selected_instruments))
            
            if instrument in all_instruments_data:
                df = all_instruments_data[instrument]
                
                # Filter data for 5-year lookback
                df_5yr = df[df['report_date_as_yyyy_mm_dd'] >= lookback_date].copy()
                
                if len(df_5yr) < 10:  # Need sufficient data for percentiles
                    continue
                
                # Get latest data
                latest_idx = df['report_date_as_yyyy_mm_dd'].idxmax()
                latest_data = df.loc[latest_idx]
                
                # Get raw values
                trader_count = latest_data.get('traders_tot_all', 0)
                open_interest = latest_data.get('open_interest_all', 0)
                
                # Get concentration based on selected metric
                if concentration_metric == 'conc_gross_le_4_tdr_long':
                    concentration = latest_data.get('conc_gross_le_4_tdr_long', 0)
                elif concentration_metric == 'conc_gross_le_4_tdr_short':
                    concentration = latest_data.get('conc_gross_le_4_tdr_short', 0)
                elif concentration_metric == 'conc_gross_le_8_tdr_long':
                    concentration = latest_data.get('conc_gross_le_8_tdr_long', 0)
                elif concentration_metric == 'conc_gross_le_8_tdr_short':
                    concentration = latest_data.get('conc_gross_le_8_tdr_short', 0)
                elif concentration_metric == 'conc_net_le_4_tdr_long_all':
                    concentration = latest_data.get('conc_net_le_4_tdr_long_all', 0)
                elif concentration_metric == 'conc_net_le_4_tdr_short_all':
                    concentration = latest_data.get('conc_net_le_4_tdr_short_all', 0)
                elif concentration_metric == 'conc_net_le_8_tdr_long_all':
                    concentration = latest_data.get('conc_net_le_8_tdr_long_all', 0)
                elif concentration_metric == 'conc_net_le_8_tdr_short_all':
                    concentration = latest_data.get('conc_net_le_8_tdr_short_all', 0)
                else:
                    # Default to gross top 4 long if metric not found
                    concentration = latest_data.get('conc_gross_le_4_tdr_long', 0)
                
                # Calculate percentiles for trader count
                trader_percentile = (df_5yr['traders_tot_all'] <= trader_count).sum() / len(df_5yr) * 100
                
                # Calculate percentiles for concentration metric
                # Use the selected metric directly
                metric_col = concentration_metric
                if metric_col in df_5yr.columns:
                    conc_percentile = (df_5yr[metric_col] <= concentration).sum() / len(df_5yr) * 100
                else:
                    conc_percentile = 50  # Default if column not found
                
                # Add to scatter data
                scatter_data.append({
                    'instrument': instrument,
                    'trader_count': trader_count,
                    'concentration': concentration,
                    'trader_percentile': trader_percentile,
                    'conc_percentile': conc_percentile,
                    'open_interest': open_interest,
                    'short_name': instrument.split('-')[0].strip()
                })
        
        progress_bar.empty()
        status_text.empty()
        
        if not scatter_data:
            st.warning("Insufficient historical data for percentile calculations")
            return None
        
        # Create scatter plot
        fig = go.Figure()
        
        # Define quadrant colors based on percentiles (50th percentile as threshold)
        colors = []
        for item in scatter_data:
            if item['trader_percentile'] > 50:
                if item['conc_percentile'] < 50:
                    colors.append('#2ECC71')  # High traders, Low concentration (best)
                else:
                    colors.append('#F39C12')  # High traders, High concentration
            else:
                if item['conc_percentile'] < 50:
                    colors.append('#3498DB')  # Low traders, Low concentration
                else:
                    colors.append('#E74C3C')  # Low traders, High concentration (worst)
        
        # Add scatter points using percentiles
        fig.add_trace(go.Scatter(
            x=[d['trader_percentile'] for d in scatter_data],
            y=[d['conc_percentile'] for d in scatter_data],
            mode='markers+text',
            marker=dict(
                size=[np.sqrt(d['open_interest']/1000) for d in scatter_data],  # Scale bubble size
                color=colors,
                line=dict(width=2, color='white'),
                sizemode='diameter',
                sizemin=10,
                sizeref=2
            ),
            text=[d['short_name'] for d in scatter_data],
            textposition='top center',
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Trader Count Percentile: %{x:.1f}%<br>' +
                         'Concentration Percentile: %{y:.1f}%<br>' +
                         'Actual Traders: %{customdata[1]:,.0f}<br>' +
                         'Actual Concentration: %{customdata[2]:.1f}%<br>' +
                         'Open Interest: %{customdata[3]:,.0f}<extra></extra>',
            customdata=[[d['instrument'], d['trader_count'], d['concentration'], d['open_interest']] 
                       for d in scatter_data]
        ))
        
        # Add quadrant lines at 50th percentile
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(
            text="Below Median Concentration<br>Above Median Traders",
            xref="paper", yref="paper",
            x=0.75, y=0.25,
            showarrow=False,
            font=dict(size=12, color="green"),
            opacity=0.8
        )
        
        fig.add_annotation(
            text="Above Median Concentration<br>Above Median Traders",
            xref="paper", yref="paper",
            x=0.75, y=0.75,
            showarrow=False,
            font=dict(size=12, color="orange"),
            opacity=0.8
        )
        
        fig.add_annotation(
            text="Below Median Concentration<br>Below Median Traders",
            xref="paper", yref="paper",
            x=0.25, y=0.25,
            showarrow=False,
            font=dict(size=12, color="blue"),
            opacity=0.8
        )
        
        fig.add_annotation(
            text="Above Median Concentration<br>Below Median Traders",
            xref="paper", yref="paper",
            x=0.25, y=0.75,
            showarrow=False,
            font=dict(size=12, color="red"),
            opacity=0.8
        )
        
        # Get concentration metric label
        metric_labels = {
            'conc_gross_le_4_tdr_long': 'Gross Top 4 Long',
            'conc_gross_le_4_tdr_short': 'Gross Top 4 Short',
            'conc_gross_le_8_tdr_long': 'Gross Top 8 Long',
            'conc_gross_le_8_tdr_short': 'Gross Top 8 Short',
            'conc_net_le_4_tdr_long_all': 'Net Top 4 Long',
            'conc_net_le_4_tdr_short_all': 'Net Top 4 Short',
            'conc_net_le_8_tdr_long_all': 'Net Top 8 Long',
            'conc_net_le_8_tdr_short_all': 'Net Top 8 Short'
        }
        metric_label = metric_labels.get(concentration_metric, concentration_metric)
        
        # Update layout
        fig.update_layout(
            title=f"Market Structure Matrix - Percentile Based (5-Year Lookback)",
            xaxis_title="Trader Count Percentile",
            yaxis_title=f"Concentration Percentile ({metric_label})",
            height=600,
            showlegend=False,
            xaxis=dict(
                range=[0, 100],
                gridcolor='lightgray',
                zeroline=False,
                ticksuffix='%'
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor='lightgray',
                zeroline=False,
                ticksuffix='%'
            )
        )
        
        # Add size legend
        fig.add_annotation(
            text="Bubble size = Open Interest",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10),
            xanchor="left", yanchor="top"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating market structure matrix: {str(e)}")
        return None