# CFTC COT Dashboard - Project Structure

## Overview
This project has been refactored into a modular architecture for better maintainability and scalability.

## Project Structure

```
src/
├── app.py                    # Main application entry point
├── config.py                 # Configuration settings and constants
├── data_fetcher.py          # Data fetching and processing functions
├── ui_components.py         # UI components and layouts
├── charts/                  # Chart creation modules
│   ├── __init__.py
│   ├── base_charts.py       # Basic time series and plotly charts
│   ├── seasonality_charts.py # Seasonality analysis charts
│   └── participation_charts.py # Trader participation analysis charts
├── functions.py             # Legacy functions (to be integrated)
└── legacyF.py              # Original monolithic file (kept for reference)
```

## Module Descriptions

### `app.py`
- Main Streamlit application
- Orchestrates all components
- Handles page routing and state management

### `config.py`
- Centralized configuration settings
- API endpoints and dataset codes
- Chart settings and color schemes
- Column definitions

### `data_fetcher.py`
- `load_instruments_database()`: Loads instrument metadata
- `fetch_cftc_data()`: Fetches data from CFTC API
- `get_asset_class()`: Maps commodities to asset classes

### `ui_components.py`
- `render_sidebar()`: Instrument selection sidebar
- `render_key_metrics()`: Key metrics display
- `render_data_table()`: Raw data table with download options
- `render_chart_selector()`: Chart type selection

### `charts/base_charts.py`
- `create_plotly_chart()`: Generic time series charts with dual y-axis support

### `charts/seasonality_charts.py`
- `create_seasonality_chart()`: Seasonality analysis with percentile zones

### `charts/participation_charts.py`
- `create_participation_density_dashboard()`: Avg position per trader analysis
- `create_trader_breakdown_charts()`: Detailed trader analysis

## Running the Application

```bash
# From the project root
./venv_new/bin/streamlit run src/app.py

# Or if you're in the src directory
../venv_new/bin/streamlit run app.py
```

## Next Steps

1. **Complete Migration**: Move remaining functions from `legacyF.py` to appropriate modules
2. **Add More Charts**: 
   - Percentile analysis charts
   - Cross-asset comparison
   - Momentum indicators
3. **Enhance UI**: 
   - Add more interactive features
   - Improve responsive design
4. **Add Tests**: Create unit tests for each module
5. **Documentation**: Add docstrings and type hints

## Benefits of New Architecture

1. **Modularity**: Each module has a single responsibility
2. **Maintainability**: Easier to find and fix issues
3. **Scalability**: Easy to add new features without affecting existing code
4. **Testability**: Individual modules can be tested in isolation
5. **Reusability**: Charts and components can be reused across different views