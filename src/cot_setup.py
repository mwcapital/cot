import pandas as pd
import streamlit as st
import nasdaqdatalink
import json
import os
from streamlit_chat import message
from plotting import plot_cftc_data
from plotting2 import display_historical_analysis

# Page configuration - correctly using the JPEG file as page icon
st.set_page_config(
    page_title="CFTC Monitor",
    layout="wide",
    initial_sidebar_state='collapsed',
    page_icon="app_icon.jpeg"  # This is the correct way to reference your icon
)

# Title for the setup page
st.title("CFTC - Set Up")

# API Key configuration section
st.subheader("API Key Configuration")

# Check if API key already exists in session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# API key input field
api_key_input = st.text_input(
    "Enter Nasdaq Data Link API Key",
    value=st.session_state.api_key if st.session_state.api_key else "",
    type="password"
)

# Button to save API key
if st.button("Submit API Key"):
    if api_key_input:
        # Store API Key in session state & configure Nasdaq Data Link
        st.session_state.api_key = api_key_input
        nasdaqdatalink.ApiConfig.api_key = api_key_input
        st.success(f"API Key saved successfully! Using: {api_key_input[:5]}******")
    else:
        st.warning("Please enter an API key.")

# Load instrument mapping section
st.subheader("Instrument Configuration")

## Strictly load instrument mapping from instruments.json
import os

# Find the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to instruments.json
instrument_mapping_file = os.path.join(script_dir, "instruments.json")
st.session_state.instrument_mapping = json.load(open(instrument_mapping_file, "r"))

# Dataset and instrument configuration
st.subheader("Data Selection")

# Dataset selection dropdown with descriptions
dataset_options = {
    "QDL/FON": "Futures and Options",
    "QDL/LFON": "Legacy Futures and Options",
    "QDL/FCR": "Futures and Options - Concentration Ratios",
    "QDL/CITS": "Commodity Index Trader Supplemental Data"
}

# Create the selectbox with formatted options
dataset_code = st.selectbox(
    "Select Dataset Code",
    options=list(dataset_options.keys()),
    format_func=lambda x: f"{x} - {dataset_options[x]}",
    help="Choose the type of CFTC report data you want to retrieve"
)

# Display description of selected dataset
st.info(f"**Selected Dataset:** {dataset_options[dataset_code]}")

# Allow user to select from predefined instruments (using session state mapping)
selected_instrument = st.selectbox(
    "Select an Instrument",
    list(st.session_state.instrument_mapping.keys())
)
instrument_code = st.session_state.instrument_mapping[selected_instrument]

# Data format options
col1, col2 = st.columns(2)

with col1:
    # Only show Legacy checkbox when QDL/LFON is selected
    if dataset_code == "QDL/LFON":
        # Default to True for QDL/LFON to make L appear by default
        use_legacy = st.checkbox("Use Legacy Format", value=True,
                                 help="Available for QDL/LFON dataset")
    else:
        use_legacy = False
        st.write("**Legacy Format:** Not applicable for this dataset")

    # Dropdown for selecting F or FO with descriptions
    base_type_options = {
        "F": "Futures Only",
        "FO": "Futures and Options Combined",
        "CITS": "Commodity Index Trader Supplemental"
    }

    base_type = st.selectbox(
        "Select Base Type",
        options=["F", "FO", "CITS"],
        format_func=lambda x: f"{x} - {base_type_options[x]}",
        help="Choose whether to include only futures or both futures and options"
    )

with col2:
    # Special handling for QDL/FCR - only ALL is available
    if dataset_code == "QDL/FCR":
        all_or_chg = "ALL"
        st.selectbox(
            "Select Data Type",
            ["ALL"],
            format_func=lambda x: "ALL - All positions",
            help="Only ALL positions available for FCR dataset"
        )
    else:
        # Dropdown for selecting data type with descriptions
        data_type_options = {
            "ALL": "All positions (absolute numbers)",
            "CHG": "Change in positions (week-over-week)",

        }

        # Filter available options based on dataset
        available_data_types = ["ALL", "CHG"]


        all_or_chg = st.selectbox(
            "Select Data Type",
            options=available_data_types,
            format_func=lambda x: f"{x} - {data_type_options[x]}",
            help="Choose the type of position data"
        )

    # Multi-select for optional suffix with descriptions
    suffix_descriptions = {
        "_CR": "Concentration Ratios - Top 4/8 traders",
        "_NT": "Number of Traders - Count by category",
        "_OI": "Open Interest - Total outstanding contracts"
    }

    # Conditional logic based on dataset
    if dataset_code == "QDL/FCR":
        # For FCR, only _CR is available
        suffix_options = ["_CR"]
        available_suffixes = st.multiselect(
            "Select Additional Categories",
            suffix_options,
            default=["_CR"],  # Pre-select _CR for FCR
            format_func=lambda x: suffix_descriptions[x],
            help="Only CR (Concentration Ratios) available for FCR dataset"
        )
    elif dataset_code in ["QDL/FON", "QDL/LFON"]:
        # For FON and LFON, _CR is not available
        # Additional check for LFON with legacy and CHG
        if dataset_code == "QDL/LFON" and use_legacy and all_or_chg == "CHG":
            # No additional categories available for LFON with legacy and CHG
            suffix_options = []
            st.write("**Additional Categories:** Not available for Legacy CHG data")
            available_suffixes = []
        else:
            suffix_options = ["_NT", "_OI"]
            available_suffixes = st.multiselect(
                "Select Additional Categories",
                suffix_options,
                format_func=lambda x: suffix_descriptions[x],
                help="Select additional data categories to include"
            )
    else:  # QDL/CITS
        suffix_options = ["_CR", "_NT", "_OI"]
        available_suffixes = st.multiselect(
            "Select Additional Categories",
            suffix_options,
            format_func=lambda x: suffix_descriptions[x],
            help="Select additional data categories to include"
        )

# Use available_suffixes instead of selected_suffixes
selected_suffixes = available_suffixes

# Construct the type category dynamically
prefix = f"{base_type}_L" if use_legacy else base_type
type_category_options = [f"{prefix}_{all_or_chg}{suffix}" for suffix in [""] + selected_suffixes]

# Dropdown for selecting the final Type & Category
type_category = st.selectbox(
    "Select Type & Category",
    type_category_options,
    help="This is the final constructed query parameter combining all your selections"
)

# Section for managing instruments (add and remove)
st.subheader("Manage Instruments")

# Using expanders to keep the UI clean
with st.expander("Add a New Instrument"):
    new_instrument_name = st.text_input("Enter Product Name", placeholder="e.g., Copper Futures")
    new_instrument_code = st.text_input("Enter Instrument Code", placeholder="e.g., HG")

    # In the "Add Instrument" section
    if st.button("Add Instrument"):
        if new_instrument_name and new_instrument_code:
            st.session_state.instrument_mapping[new_instrument_name] = new_instrument_code
            # Save updated mapping to file
            try:
                with open(instrument_mapping_file, "w") as f:
                    json.dump(st.session_state.instrument_mapping, f, indent=4)
                st.success(f"Added: {new_instrument_name} ({new_instrument_code})")
                st.rerun()
            except Exception as e:
                st.error(f"Error saving instruments.json: {e}")
        else:
            st.warning("Please enter both a product name and a code.")

with st.expander("Remove an Instrument"):
    instrument_to_remove = st.selectbox(
        "Select Instrument to Remove",
        list(st.session_state.instrument_mapping.keys()),
        index=None
    )

    if st.button("Remove Instrument"):
        if instrument_to_remove:
            if instrument_to_remove == selected_instrument:
                st.warning(
                    "Cannot remove the currently selected instrument. Please select a different instrument first.")
            else:
                del st.session_state.instrument_mapping[instrument_to_remove]
                # Save updated mapping to file
                try:
                    with open(instrument_mapping_file, "w") as f:
                        json.dump(st.session_state.instrument_mapping, f, indent=4)
                    st.success(f"Removed: {instrument_to_remove}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving instruments.json: {e}")
        else:
            st.warning("Please select an instrument to remove.")

# Display the selected configuration with explanations
st.subheader("Selected Configuration")
config_col1, config_col2 = st.columns(2)

with config_col1:
    st.write(f"**Dataset Code:** {dataset_code}")
    st.write(f"**Instrument Code:** {instrument_code}")
    st.write(f"**Type & Category:** {type_category}")

with config_col2:
    # Add explanation of what will be retrieved
    st.write("**This will retrieve:**")

    # Explain the data based on selections
    if dataset_code == "QDL/FON":
        st.write("• Disaggregated COT data")
    elif dataset_code == "QDL/LFON":
        st.write("• Legacy COT format data")
    elif dataset_code == "QDL/FCR":
        st.write("• Concentration ratio data")
    else:
        st.write("• Commodity index trader data")

    if all_or_chg == "ALL":
        st.write("• Absolute position numbers")
    elif all_or_chg == "CHG":
        st.write("• Week-over-week changes")
    elif all_or_chg == "OLD":
        st.write("• Old crop year positions")
    elif all_or_chg == "OTR":
        st.write("• Other crop year positions")

    if "_CR" in selected_suffixes:
        st.write("• Concentration ratios")
    if "_NT" in selected_suffixes:
        st.write("• Number of traders")
    if "_OI" in selected_suffixes:
        st.write("• Open interest data")

# Check for API key first
if 'api_key' not in st.session_state or not st.session_state.api_key:
    st.info("Please enter your API key to begin.")
else:
    # API key exists
    # Add a fetch data button
    fetch_button = st.button("Fetch Data")

    # Only fetch if button is pressed OR if we need to refresh after changes
    if fetch_button or ('needs_refresh' in st.session_state and st.session_state.needs_refresh):
        with st.spinner(f"Fetching {selected_instrument} data..."):
            try:
                # Fetch full data using your simpler approach
                data = nasdaqdatalink.get_table(
                    dataset_code,  # Example: 'QDL/FON'
                    contract_code=instrument_code,  # Example: '067651'
                    type=type_category  # Example: 'F_ALL', 'FO_CHG'
                )

                # Display success and show the data
                st.success(f"Successfully retrieved {len(data)} rows of data")

                # Save to session state
                st.session_state.cftc_data = data
                st.session_state.current_instrument = selected_instrument
                st.session_state.current_type = type_category
                st.session_state.dataset_code = dataset_code  # Add this line

                # Clear refresh flag if it exists
                if 'needs_refresh' in st.session_state:
                    del st.session_state.needs_refresh

            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.session_state.cftc_data = None

    # Check if we have data in session state
    if 'cftc_data' in st.session_state and st.session_state.cftc_data is not None:
        data = st.session_state.cftc_data

        # Check if the current selection matches what's in session state
        if ('current_instrument' in st.session_state and
                'current_type' in st.session_state and
                (st.session_state.current_instrument != selected_instrument or
                 st.session_state.current_type != type_category)):
            st.warning("Data is for a different instrument/type combination. Please fetch new data.")
            st.session_state.needs_refresh = True
        else:
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())

            # Add download option
            csv = data.to_csv()
            file_name = f"{instrument_code}_{type_category}.csv"
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=file_name,
                mime="text/csv",
            )

            # Display plots
            plot_cftc_data(data)

            # Filter out columns that aren't suitable for plotting
            plotable_cols = [col for col in data.columns if col not in ['None', 'contract_code', 'type', 'date']]
            # Call the historical analysis function
            display_historical_analysis(data, plotable_cols)

    else:
        if not fetch_button:  # Only show this if the button wasn't just pressed
            st.info("No data loaded. Please click 'Fetch Data' to retrieve data.")



st.markdown("----------------------------------")

st.markdown("""
## 📌 Understanding the Report Structure

The dataset is structured into three main parts: **Type, Category, and Sub-category**.



### **1 Choose Core Dataset **
This defines the **scope** of the report.
- **FO** – Futures and Options Combined 
- **F** – Futures Only 
 

### **2 if you want legacy tick the legacy too**

### **3 Choose  Category (Position Type)**
This specifies how positions are categorized.
- **ALL** – All positions included  
- **CHG** – Changes in positions   
 

### **4 Choose Sub category **
These sub-categories provide additional analysis:
- **_CR** – Concentration Ratios: Largest traders’ positions   
(this is needed if you want to see top 4 longs and shorts)
- **_NT** – Number of Traders in the market 
- **_OI** – Open Interest: Total outstanding contracts   

You can select different combinations of these to customize your analysis.

""")





st.markdown("""
## Understanding Legacy vs. Non-Legacy CFTC Reports  

### Legacy Reports  
The **Legacy COT Report** has been published since 1968 and provides a simplified breakdown of market positions:  
- **Commercial Traders**: Entities hedging against price risk (e.g., producers, manufacturers).  
- **Non-Commercial Traders**: Speculative traders such as hedge funds and large investors.  
- **Non-Reportable Traders**: Small traders whose positions are too small to be categorized.  

These reports aggregate **futures-only** and **futures + options combined** data with minimal granularity.  

### Non-Legacy (Disaggregated) Reports  
Introduced in **2009**, the **Disaggregated COT Report** offers more detailed trader classifications:  
- **Producer/Merchant/Processor/User**: Entities using futures to hedge physical market risk.  
- **Swap Dealers**: Financial institutions using swaps and futures for risk management.  
- **Money Managers**: Hedge funds and institutional investors.  
- **Other Reportables**: Large traders that do not fit other categories.  

This version provides better transparency on speculative vs. hedging activity.  

### Key Differences  
| Feature               | Legacy Report          | Non-Legacy (Disaggregated) |
|----------------------|----------------------|---------------------------|
| **First Available**  | 1968                 | 2009                      |
| **Trader Breakdown** | 3 Categories         | 4 Categories               |
| **Hedge Funds**      | Not Separate         | Categorized as Money Managers |
| **Swap Dealers**     | Not Identified       | Categorized Separately |
| **Transparency**     | Lower                | Higher                     |

**When to Use Each:**  
- Use **Legacy Reports** for long-term historical analysis .  
- Use **Non-Legacy Reports** for more precise trader classification.  
""")
