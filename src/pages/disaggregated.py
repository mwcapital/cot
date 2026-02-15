"""
Disaggregated COT Report page
Uses the CFTC Disaggregated Futures report (dataset 72hh-3qpy)
4 reportable categories: Producer/Merchant, Swap Dealers, Managed Money, Other Reportables
"""
import streamlit as st
from disaggregated_dashboard import display_disagg_dashboard

st.title("Disaggregated COT Report")
st.markdown("Disaggregated Futures positioning with 4 trader categories: "
            "Producer/Merchant, Swap Dealers, Managed Money, Other Reportables")

# API Token - shared via session state (initialized on Legacy page)
if 'api_token' not in st.session_state:
    st.session_state.api_token = "3CKjkFN6jIIHgSkIJH19i7VhK"

with st.expander("API Configuration", expanded=False):
    api_token = st.text_input(
        "API Token (optional):",
        value=st.session_state.api_token,
        type="password",
        help="Enter your CFTC API token for higher rate limits.",
        key="disagg_api_token_input"
    )
    st.session_state.api_token = api_token

st.markdown("---")

# Display the disaggregated overview table
display_disagg_dashboard(st.session_state.api_token)
