# Instrument Management

This directory organizes all instrument-related data and tools for the CFTC COT Dashboard.

## Directory Structure

### 📁 LegacyF/
Contains all data and tools specific to the Legacy Futures COT report:
- `instruments_LegacyF.json` - Main instruments database for Legacy Futures
- `CFTC_API_COLUMN_MAPPING.md` - Complete CFTC API column name reference
- `FUT86_16.txt` - Historical COT data file (1986-2016)
- `instrument_database.py` - Script to fetch and update instruments from CFTC API
- `compare_json_files.py` - Tool to compare JSON database versions

### 📁 futures/
Contains futures price-related data and mapping files:
- `futures_symbols_current.json` - Current futures symbols mapping
- `futures_symbols_enhanced.json` - Enhanced futures symbols with additional metadata
- `futures_cot_mapping.py` - COT to futures symbol mapping utility

### 📁 Supplemental/
Contains data and tools for the Supplemental COT report:
- `instrument_database_supplemental.py` - Script to fetch and update Supplemental report instruments
- `instruments_Supplemental.json` - Instruments database for Supplemental COT report

## Usage

### Update Legacy Futures Instruments Database
```bash
cd LegacyF
python instrument_database.py
```
This fetches the latest instruments from CFTC API and updates the database.

### Compare Database Versions
```bash
cd LegacyF
python compare_json_files.py
```
Compare original and new JSON files to identify changes.

### Working with Futures Data
The `futures/` directory contains mapping files that link COT instruments to their corresponding futures symbols for price data integration.

## Data Flow

1. **COT Data**: Fetched via CFTC API using instrument codes from `LegacyF/instruments_LegacyF.json`
2. **Futures Prices**: Mapped using files in `futures/` directory
3. **Display**: Combined in the dashboard application

## Important Notes

- The application loads `instruments_LegacyF.json` from the LegacyF folder
- Always backup databases before updating
- CFTC adds new instruments regularly - periodic updates recommended
- Use the column mapping reference in LegacyF for correct API field names

## File Paths Used by Application

- Primary database: `instrument_management/LegacyF/instruments_LegacyF.json`
- Column mapping: `instrument_management/LegacyF/CFTC_API_COLUMN_MAPPING.md`
- Futures mappings: `instrument_management/futures/futures_symbols_*.json`