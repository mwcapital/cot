# Instrument Management

This folder contains all tools and files related to managing the CFTC instruments database.

## Files

### Data Files
- `instruments_LegacyF_original.json` - The original/current instruments database used by the app
- `instruments_LegacyF_new.json` - Newly generated database from CFTC API

### Scripts
- `instrument_database.py` - Fetches all instruments from CFTC API and generates the JSON database
- `compare_json_files.py` - Compares two JSON files to identify differences

## Usage

### Generate New Instruments Database
```bash
cd instrument_management
source ../venv_new/bin/activate
python instrument_database.py
```
This creates `instruments_LegacyF_new.json` with the latest instruments from CFTC.

### Compare Original vs New
```bash
python compare_json_files.py
```
This compares the original and new JSON files to show what's changed.

### Deploy New Database
If satisfied with the new database:
```bash
# Backup current
cp ../instruments_LegacyF.json ./instruments_LegacyF_backup_$(date +%Y%m%d).json

# Deploy new
cp instruments_LegacyF_new.json ../instruments_LegacyF.json
```

## Notes
- The app expects `instruments_LegacyF.json` in the root directory
- Always backup before replacing the production database
- New instruments are added regularly to CFTC, so periodic updates are recommended