import pandas as pd
import streamlit as st
import nasdaqdatalink
import json
import os

from plotting import plot_cftc_data


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

# Strictly load instrument mapping from instruments.json
instrument_mapping_file = "instruments.json"
st.session_state.instrument_mapping = json.load(open(instrument_mapping_file, "r"))

# Dataset and instrument configuration
st.subheader("Data Selection")

# Dataset selection dropdow
dataset_code = st.selectbox(
    "Select Dataset Code",
    ["QDL/FON", "QDL/LFON", "QDL/FCR", "QDL/CITS"]
)

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
        use_legacy = st.checkbox("Use Legacy Format", value=False,
                                 help="Available for QDL/LFON dataset")
    else:
        use_legacy = False
        st.write("**Legacy Format:** Not applicable for this dataset")

    # Dropdown for selecting F or FO
    base_type = st.selectbox(
        "Select Base Type",
        ["F", "FO", "CITS"],
        help="F = Futures, FO = Futures & Options, CITS = Commodity Index Traders"
    )

with col2:
    # Dropdown for selecting _ALL or _CHG
    all_or_chg = st.selectbox(
        "Select Data Type",
        ["ALL", "CHG"],
        help="ALL = All positions, CHG = Change in positions"
    )

    # Multi-select for optional suffix (_CR, _NT, _OI)
    suffix_options = ["_CR", "_NT", "_OI"]
    selected_suffixes = st.multiselect(
        "Select Additional Categories",
        suffix_options,
        help="CR = Commercial, NT = Non-Commercial, OI = Open Interest"
    )

# Construct the type category dynamically
prefix = f"{base_type}_L" if use_legacy else base_type
type_category_options = [f"{prefix}_{all_or_chg}{suffix}" for suffix in [""] + selected_suffixes]

# Dropdown for selecting the final Type & Category
type_category = st.selectbox("Select Type & Category", type_category_options)

# Section for managing instruments (add and remove)
st.subheader("Manage Instruments")

# Using expanders to keep the UI clean
with st.expander("Add a New Instrument"):
    new_instrument_name = st.text_input("Enter Product Name", placeholder="e.g., Copper Futures")
    new_instrument_code = st.text_input("Enter Instrument Code", placeholder="e.g., HG")

    if st.button("Add Instrument"):
        if new_instrument_name and new_instrument_code:
            st.session_state.instrument_mapping[new_instrument_name] = new_instrument_code
            # Save updated mapping to file
            try:
                with open(instrument_mapping_file, "w") as f:
                    json.dump(st.session_state.instrument_mapping, f, indent=4)
                st.success(f"Added: {new_instrument_name} ({new_instrument_code})")
                st.experimental_rerun()  # Refresh the app to update the selectbox and removal options
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
                    st.experimental_rerun()  # Refresh the app to update the selectbox and removal options
                except Exception as e:
                    st.error(f"Error saving instruments.json: {e}")
        else:
            st.warning("Please select an instrument to remove.")




# Display the selected configuration
st.subheader("Selected Configuration")
st.write(f"**Dataset Code:** {dataset_code}")
st.write(f"**Instrument Code:** {instrument_code}")
st.write(f"**Type & Category:** {type_category}")


# Add a fetch data button with a simple spinner for loading state
if st.button("Fetch Data"):
    with st.spinner(f"Fetching {selected_instrument} data..."):
            # Fetch full data using your simpler approach
            data = nasdaqdatalink.get_table(
                dataset_code,  # Example: 'QDL/FON'
                contract_code=instrument_code,  # Example: '067651'
                type=type_category  # Example: 'F_ALL', 'FO_CHG'
            )

            # Display success and show the data
            st.success(f"Successfully retrieved {len(data)} rows of data")
            st.subheader("Data Preview")
            st.dataframe(data)

            # Save to session state for potential use elsewhere
            st.session_state.cftc_data = data

            # Add download option
            csv = data.to_csv()
            file_name = f"{instrument_code}_{type_category}.csv"
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=file_name,
                mime="text/csv",
            )


data = st.session_state.cftc_data
# Call the plotting function
plot_cftc_data(data)


#
# # Page configuration
# st.set_page_config(page_title="CFTC Monitor", layout="wide", initial_sidebar_state='collapsed', page_icon='app_icon.jpeg')
#
# # Store API Key in session state & configure Nasdaq Data Link
# api_key = 'Lfzdy9CuRUL-xnywmDPy'
# st.session_state.api_key = api_key
# nasdaqdatalink.ApiConfig.api_key = api_key
# st.success(f"Using API Key: {api_key[:5]}******")
# # Button to confirm API key
# if st.button("Submit API Key"):
#     st.success("API Key saved successfully!")
#
#
# # Strictly load instrument mapping from instruments.json
# instrument_mapping_file = "instruments.json"
# if not os.path.exists(instrument_mapping_file):
#     st.error("instruments.json file not found. Please ensure the file exists in the app directory with the instrument mappings.")
#     st.stop()  # Stop execution if the file doesn’t exist
#
# try:
#     with open(instrument_mapping_file, "r") as f:
#         st.session_state.instrument_mapping = json.load(f)
# except json.JSONDecodeError as e:
#     st.error(f"Error decoding instruments.json: {e}")
#     st.stop()  # Stop execution if the JSON is invalid
# except Exception as e:
#     st.error(f"Error loading instruments.json: {e}")
#     st.stop()  # Stop execution for other file-related errors
#
# # Streamlit UI
# st.title("CFTC - Set Up")
#
#
# # Button to confirm API key
# if st.button("Submit API Key"):
#     st.success("API Key saved successfully!")
#
# # Dataset selection dropdown
# dataset_code = st.selectbox(
#     "Select Dataset Code",
#     ["QDL/FON", "QDL/LFON", "QDL/FCR", "QDL/CITS"]
# )
#
#
#
#
# # Instrument selection, addition, and removal
# st.subheader("Instrument Selection")
#
# # Allow user to select from predefined instruments (using session state mapping)
# selected_instrument = st.selectbox("Select an Instrument", list(st.session_state.instrument_mapping.keys()))
# instrument_code = st.session_state.instrument_mapping[selected_instrument]
#
#
# # Checkbox for Legacy selection
# use_legacy = st.checkbox("Use Legacy Format if QDL/LFON is selected ", value=False)
#
# # Dropdown for selecting F or FO
# base_type = st.selectbox("Select Base Type", ["F", "FO", "CITS"])
#
# # Dropdown for selecting _ALL or _CHG
# all_or_chg = st.selectbox("Select Data Type", ["ALL", "CHG"])
#
# # Multi-select for optional suffix (_CR, _NT, _OI)
# suffix_options = ["_CR", "_NT", "_OI"]
# selected_suffixes = st.multiselect("Select Additional Categories", suffix_options)
#
# # Construct the type category dynamically
# prefix = f"{base_type}_L" if use_legacy else base_type
# type_category_options = [f"{prefix}_{all_or_chg}{suffix}" for suffix in [""] + selected_suffixes]
#
# # Dropdown for selecting the final Type & Category
# type_category = st.selectbox("Select Type & Category", type_category_options)
#
#
# st.write(f"**Dataset Code:** {dataset_code}")
# st.write(f"**Instrument Code:** {instrument_code}")
# st.write(f"**Type & Category:** {type_category}")
#
#
#
#
#
#
#
# # Section for managing instruments (add and remove)
# st.write("### Manage Instruments")
#
# # Add a new instrument
# st.write("#### Add a New Instrument")
# new_instrument_name = st.text_input("Enter Product Name", placeholder="e.g., Copper Futures")
# new_instrument_code = st.text_input("Enter Instrument Code", placeholder="e.g., 123456")
#
# if st.button("Add Instrument"):
#     if new_instrument_name and new_instrument_code:
#         st.session_state.instrument_mapping[new_instrument_name] = new_instrument_code
#         # Save updated mapping to file
#         try:
#             with open(instrument_mapping_file, "w") as f:
#                 json.dump(st.session_state.instrument_mapping, f, indent=4)
#             st.success(f"Added: {new_instrument_name} ({new_instrument_code})")
#             st.experimental_rerun()  # Refresh the app to update the selectbox and removal options
#         except Exception as e:
#             st.error(f"Error saving instruments.json: {e}")
#     else:
#         st.warning("Please enter both a product name and a code.")
#
# # Remove an instrument
# st.write("#### Remove an Instrument")
# instrument_to_remove = st.selectbox("Select Instrument to Remove", list(st.session_state.instrument_mapping.keys()), index=None)
# if st.button("Remove Instrument"):
#     if instrument_to_remove:
#         if instrument_to_remove == selected_instrument:
#             st.warning("Cannot remove the currently selected instrument. Please select a different instrument first.")
#         else:
#             del st.session_state.instrument_mapping[instrument_to_remove]
#             # Save updated mapping to file
#             try:
#                 with open(instrument_mapping_file, "w") as f:
#                     json.dump(st.session_state.instrument_mapping, f, indent=4)
#                 st.success(f"Removed: {instrument_to_remove}")
#                 st.experimental_rerun()  # Refresh the app to update the selectbox and removal options
#             except Exception as e:
#                 st.error(f"Error saving instruments.json: {e}")
#     else:
#         st.warning("Please select an instrument to remove.")
#
# # Update session state with the selected instrument
# st.session_state.instrument_code = instrument_code
# st.session_state.selected_instrument = selected_instrument
#
# st.write(f"**Selected Instrument Code:** {instrument_code}")
#















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
