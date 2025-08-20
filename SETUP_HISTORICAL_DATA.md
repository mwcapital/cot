# Setting Up Historical COT Data

The historical COT data file (FUT86_16.txt, 160MB) is too large for regular Git but is essential for accessing data from 1986-2016.

## Option 1: Using Git LFS (Recommended)

### Step 1: Install Git LFS
```bash
# On macOS with Homebrew
brew install git-lfs

# Or download from: https://git-lfs.github.com/
```

### Step 2: Initialize Git LFS in your repository
```bash
git lfs install
```

### Step 3: Track the historical data file
```bash
git lfs track "instrument_management/FUT86_16.txt"
git add .gitattributes
```

### Step 4: Add and commit the file
```bash
git add instrument_management/FUT86_16.txt
git commit -m "Add historical COT data file via Git LFS"
git push origin main
```

## Option 2: Manual Download

If you're cloning this repository on a new machine and don't have the historical data file:

1. Download FUT86_16.txt from the CFTC website or your backup location
2. Place it in: `instrument_management/FUT86_16.txt`
3. The app will automatically detect and use it

## What This File Provides

- **Pre-2000 Data**: Extends many instruments back to 1986
- **Complete WTI History**: Full crude oil data from 1986-present
- **Automatic Stitching**: The app automatically combines this with API data
- **Gap Detection**: Warns if there are gaps between historical and current data

## File Structure

The FUT86_16.txt file contains:
- Date range: 1986-2016
- All CFTC COT report fields
- Multiple instrument name variations (handles name changes over time)

## Verifying It Works

After setup, when fetching WTI-PHYSICAL in the app, you should see:
- Data starting from 1986 (not just 2000 or 2022)
- Approximately 1,876 total records
- 40 years of continuous data

## Troubleshooting

If historical data isn't loading:
1. Check the file exists at `instrument_management/FUT86_16.txt`
2. Ensure the file is the complete 160MB version
3. Check for any warnings in the app about data gaps