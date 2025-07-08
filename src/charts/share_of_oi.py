"""
Share of Open Interest chart functionality
"""

import streamlit as st
import plotly.graph_objects as go


def create_share_of_oi_chart(df, calculation_side, chart_title):
    """Create share of open interest time series chart"""
    try:
        if df.empty:
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Use API percentages directly
        df_calc = df.copy()
        
        # Check if API percentage columns exist
        api_pct_cols = ['pct_of_oi_comm_long_all', 'pct_of_oi_noncomm_long_all', 
                       'pct_of_oi_noncomm_spread', 'pct_of_oi_nonrept_long_all',
                       'pct_of_oi_comm_short_all', 'pct_of_oi_noncomm_short_all',
                       'pct_of_oi_nonrept_short_all']
        
        has_api_pct = all(col in df.columns for col in api_pct_cols)
        
        if has_api_pct:
            # Use API percentages directly
            if calculation_side == "Long Side":
                df_calc['comm_pct'] = df_calc['pct_of_oi_comm_long_all']
                df_calc['noncomm_pct'] = df_calc['pct_of_oi_noncomm_long_all']
                df_calc['nonrept_pct'] = df_calc['pct_of_oi_nonrept_long_all']
                df_calc['spread_pct'] = df_calc['pct_of_oi_noncomm_spread']
            else:  # Short Side
                df_calc['comm_pct'] = df_calc['pct_of_oi_comm_short_all']
                df_calc['noncomm_pct'] = df_calc['pct_of_oi_noncomm_short_all']
                df_calc['nonrept_pct'] = df_calc['pct_of_oi_nonrept_short_all']
                df_calc['spread_pct'] = df_calc['pct_of_oi_noncomm_spread']
        else:
            # Fallback to manual calculation if API columns not available
            st.warning("API percentage columns not found. Using manual calculation.")
            
            if calculation_side == "Long Side":
                # Long side total: NonComm Long + Spread + Comm Long + NonRep Long
                df_calc['total_side'] = (
                    df_calc['noncomm_positions_long_all'] + 
                    df_calc['noncomm_postions_spread_all'] +
                    df_calc['comm_positions_long_all'] + 
                    df_calc['nonrept_positions_long_all']
                )
                
                # Calculate percentages
                df_calc['comm_pct'] = (df_calc['comm_positions_long_all'] / df_calc['total_side']) * 100
                df_calc['noncomm_pct'] = (df_calc['noncomm_positions_long_all'] / df_calc['total_side']) * 100
                df_calc['nonrept_pct'] = (df_calc['nonrept_positions_long_all'] / df_calc['total_side']) * 100
            else:  # Short Side
                # Short side total: NonComm Short + Spread + Comm Short + NonRep Short
                df_calc['total_side'] = (
                    df_calc['noncomm_positions_short_all'] + 
                    df_calc['noncomm_postions_spread_all'] +
                    df_calc['comm_positions_short_all'] + 
                    df_calc['nonrept_positions_short_all']
                )
                
                # Calculate percentages
                df_calc['comm_pct'] = (df_calc['comm_positions_short_all'] / df_calc['total_side']) * 100
                df_calc['noncomm_pct'] = (df_calc['noncomm_positions_short_all'] / df_calc['total_side']) * 100
                df_calc['nonrept_pct'] = (df_calc['nonrept_positions_short_all'] / df_calc['total_side']) * 100
            
            # Spread percentage is same for both sides
            df_calc['spread_pct'] = (df_calc['noncomm_postions_spread_all'] / df_calc['total_side']) * 100
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=df_calc['report_date_as_yyyy_mm_dd'],
            y=df_calc['comm_pct'],
            name='Commercial',
            line=dict(width=2, color='blue'),
            mode='lines',
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_calc['report_date_as_yyyy_mm_dd'],
            y=df_calc['noncomm_pct'],
            name='Non-Commercial',
            line=dict(width=2, color='green'),
            mode='lines',
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_calc['report_date_as_yyyy_mm_dd'],
            y=df_calc['spread_pct'],
            name='Non-Commercial Spread',
            line=dict(width=2, color='orange'),
            mode='lines',
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_calc['report_date_as_yyyy_mm_dd'],
            y=df_calc['nonrept_pct'],
            name='Non-Reportable',
            line=dict(width=2, color='gray'),
            mode='lines',
            stackgroup='one'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{chart_title} - Share of Open Interest ({calculation_side})",
            xaxis_title="Date",
            yaxis=dict(
                title="Share of Open Interest (%)",
                tickformat=".1f",
                ticksuffix="%",
                range=[0, 100]
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600,
            showlegend=True
        )
        
        # Add range selector
        fig.update_layout(
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
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating share of OI chart: {str(e)}")
        return None