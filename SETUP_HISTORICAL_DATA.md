# Historical COT Data Setup

This repository uses Git LFS to manage the historical COT data file (FUT86_16.txt, 160MB) which provides data from 1986-2016.

## Prerequisites

Ensure Git LFS is installed on your system:

```bash
# On macOS with Homebrew
brew install git-lfs

# Or download from: https://git-lfs.github.com/
```

## Cloning the Repository

When cloning this repository, Git LFS will automatically download the historical data file:

```bash
git clone https://github.com/mwcapital/COT-Monitor.git
cd COT-Monitor
```

If the LFS file didn't download automatically:

```bash
git lfs pull
```

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