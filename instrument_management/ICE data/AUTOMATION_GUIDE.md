# ICE COT Data Automation Guide

This guide explains how to automatically update your ICE COT database weekly without manual downloads.

## Quick Start

The system can automatically:
1. Download the latest data from ICE website every week
2. Compare with your existing database
3. Append only new records
4. Create backups before updating
5. Log all updates

## Files Overview

| File | Purpose |
|------|---------|
| `update_ice_data.py` | Main updater script (downloads & appends data) |
| `weekly_update.sh` | Shell script wrapper for automation |
| `update_log.json` | Tracks all updates and their status |
| `backups/` | Automatic backups before each update |
| `update_logs/` | Detailed logs of each update run |

---

## Data Source

ICE provides direct CSV download URLs with a predictable pattern:

```
Current Year: https://www.ice.com/publicdocs/futures/COTHist2025.csv
Previous Year: https://www.ice.com/publicdocs/futures/COTHist2024.csv
```

**Update Schedule:**
- ICE releases COT data: **Fridays** (for positions as of Tuesday close)
- Recommended update time: **Friday evenings** or **Saturday mornings**

---

## Manual Update

To manually update the database:

```bash
cd "instrument_management/ICE data"
source ../../venv/bin/activate
python update_ice_data.py
```

**Output:**
- New records appended to `ICE_COT_Historical_Merged.csv`
- Backup created in `backups/` folder
- Update logged in `update_log.json`

---

## Automation Options

### Option 1: Local Cron Job (Recommended for Mac/Linux)

Set up a weekly cron job to run the update automatically.

#### Setup Steps:

1. **Make the shell script executable:**
```bash
chmod +x "instrument_management/ICE data/weekly_update.sh"
```

2. **Edit your crontab:**
```bash
crontab -e
```

3. **Add this line for Friday 6 PM updates:**
```bash
0 18 * * 5 /Users/makson/Desktop/COT-Analysis/instrument_management/ICE\ data/weekly_update.sh
```

**Cron Schedule Examples:**
```bash
# Every Friday at 6:00 PM
0 18 * * 5 /path/to/weekly_update.sh

# Every Saturday at 9:00 AM
0 9 * * 6 /path/to/weekly_update.sh

# Every Friday at 11:59 PM
59 23 * * 5 /path/to/weekly_update.sh
```

**Cron Format:**
```
* * * * *
│ │ │ │ │
│ │ │ │ └─── Day of week (0-7, 0 and 7 = Sunday)
│ │ │ └───── Month (1-12)
│ │ └─────── Day of month (1-31)
│ └───────── Hour (0-23)
└─────────── Minute (0-59)
```

4. **Verify cron job is set:**
```bash
crontab -l
```

5. **Check logs:**
```bash
ls "instrument_management/ICE data/update_logs/"
cat "instrument_management/ICE data/update_logs/update_YYYYMMDD_HHMMSS.log"
```

---

### Option 2: Mac Launchd (Alternative to Cron on macOS)

Create a Launch Agent for more reliable scheduling on Mac.

#### Setup Steps:

1. **Create launch agent file:**
```bash
nano ~/Library/LaunchAgents/com.cotanalysis.ice-update.plist
```

2. **Add this content:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.cotanalysis.ice-update</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/makson/Desktop/COT-Analysis/instrument_management/ICE data/weekly_update.sh</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>5</integer>  <!-- Friday = 5 -->
        <key>Hour</key>
        <integer>18</integer>  <!-- 6 PM -->
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/makson/Desktop/COT-Analysis/instrument_management/ICE data/update_logs/launchd_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/makson/Desktop/COT-Analysis/instrument_management/ICE data/update_logs/launchd_stderr.log</string>
</dict>
</plist>
```

3. **Load the launch agent:**
```bash
launchctl load ~/Library/LaunchAgents/com.cotanalysis.ice-update.plist
```

4. **Verify it's loaded:**
```bash
launchctl list | grep cotanalysis
```

5. **Test it manually:**
```bash
launchctl start com.cotanalysis.ice-update
```

6. **Unload if needed:**
```bash
launchctl unload ~/Library/LaunchAgents/com.cotanalysis.ice-update.plist
```

---

### Option 3: GitHub Actions (Cloud Automation)

Automate via GitHub Actions if your repository is on GitHub.

#### Setup Steps:

1. **Create workflow file:**
```bash
mkdir -p .github/workflows
```

2. **Create `.github/workflows/update-ice-data.yml`:**
```yaml
name: Update ICE COT Data

on:
  schedule:
    # Run every Friday at 18:00 UTC (adjust for your timezone)
    - cron: '0 18 * * 5'
  workflow_dispatch:  # Allow manual trigger

jobs:
  update:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pandas requests

    - name: Run update script
      run: |
        cd "instrument_management/ICE data"
        python update_ice_data.py

    - name: Commit and push if changes
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add "instrument_management/ICE data/ICE_COT_Historical_Merged.csv"
        git add "instrument_management/ICE data/update_log.json"
        git diff --quiet && git diff --staged --quiet || (git commit -m "Auto-update ICE COT data [skip ci]" && git push)
```

**Pros:**
- Runs in the cloud (no local machine needed)
- Free for public repositories
- Automatic git commits

**Cons:**
- Requires GitHub repository
- Need to commit large CSV files to repo

---

### Option 4: Python Script Scheduler (Cross-Platform)

Use Python's `schedule` library for a platform-independent solution.

#### Setup:

1. **Install schedule library:**
```bash
pip install schedule
```

2. **Create `scheduler.py`:**
```python
import schedule
import time
import subprocess
from pathlib import Path

def run_update():
    """Run the update script"""
    print(f"Running update at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    script_dir = Path(__file__).parent
    update_script = script_dir / 'update_ice_data.py'

    result = subprocess.run(
        ['python', str(update_script)],
        cwd=script_dir,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Update completed successfully")
    else:
        print("✗ Update failed")
        print(result.stderr)

# Schedule update every Friday at 6 PM
schedule.every().friday.at("18:00").do(run_update)

print("ICE COT Data Scheduler Started")
print("Updates scheduled for: Every Friday at 6:00 PM")
print("Press Ctrl+C to stop\n")

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
```

3. **Run the scheduler:**
```bash
python scheduler.py &
```

---

## Monitoring & Logs

### Check Update Status

**View update log:**
```bash
cat "instrument_management/ICE data/update_log.json"
```

**View latest update:**
```bash
tail -100 "instrument_management/ICE data/update_logs/update_*.log" | tail -50
```

**Check database size:**
```bash
ls -lh "instrument_management/ICE data/ICE_COT_Historical_Merged.csv"
```

### Update Log Structure

```json
{
  "updates": [
    {
      "timestamp": "2025-01-11 18:00:00",
      "year_checked": 2025,
      "new_records": 52,
      "total_records_after": 7941,
      "backup_file": "ICE_COT_Historical_Merged_backup_20250111_180000.csv",
      "status": "success",
      "validation": {
        "duplicates": 0,
        "missing_values": 0
      }
    }
  ],
  "last_update": "2025-01-11 18:00:00",
  "total_updates": 15
}
```

---

## Troubleshooting

### Issue: No new records found (when you expect them)

**Possible causes:**
1. ICE hasn't released this week's data yet
2. Database is already up to date
3. Check the ICE website manually

**Solution:**
- Wait until Friday evening or Saturday morning
- Verify data availability at: https://www.ice.com/report/122

### Issue: Download fails

**Possible causes:**
1. No internet connection
2. ICE website is down
3. URL structure changed

**Solution:**
```bash
# Test the download URL manually
curl https://www.ice.com/publicdocs/futures/COTHist2025.csv
```

### Issue: Permission denied (cron job)

**Solution:**
```bash
# Ensure script is executable
chmod +x "instrument_management/ICE data/weekly_update.sh"

# Check cron has necessary permissions
# On Mac: System Preferences > Security & Privacy > Full Disk Access > Add Terminal
```

### Issue: Virtual environment not found

**Solution:**
Update the `VENV_PATH` in `weekly_update.sh` to match your venv location:
```bash
VENV_PATH="../../venv"  # Adjust this path
```

---

## Best Practices

1. **Backups**: The script automatically creates backups before each update in `backups/` folder

2. **Monitor Logs**: Check update logs periodically to ensure updates are working

3. **Test First**: Run the update manually a few times before setting up automation

4. **Timing**: Schedule updates for Friday evenings or Saturday mornings (after ICE releases data)

5. **Notifications** (Optional): Add email notifications to the shell script:
```bash
# Add to weekly_update.sh
if [ $? -eq 0 ]; then
    echo "Update successful" | mail -s "ICE COT Update Success" your@email.com
else
    echo "Update failed" | mail -s "ICE COT Update FAILED" your@email.com
fi
```

---

## Manual Download Alternative

If automation fails, you can always manually download from:

**ICE Report Center:**
https://www.ice.com/report/122

**Direct Download Links:**
- 2025: https://www.ice.com/publicdocs/futures/COTHist2025.csv
- 2024: https://www.ice.com/publicdocs/futures/COTHist2024.csv

Then run the update script manually:
```bash
python update_ice_data.py
```

---

## Summary

**Recommended Setup for Mac:**
1. Use **Cron Job** (Option 1) for simplicity
2. Schedule for **Friday 6 PM** or **Saturday 9 AM**
3. Check logs weekly for the first month
4. Set it and forget it!

**Quick Setup:**
```bash
# 1. Make script executable
chmod +x "instrument_management/ICE data/weekly_update.sh"

# 2. Add to crontab
crontab -e

# 3. Add this line
0 18 * * 5 /Users/makson/Desktop/COT-Analysis/instrument_management/ICE\ data/weekly_update.sh

# 4. Save and exit
# Done! Updates will run automatically every Friday at 6 PM
```

---

## Support

For issues or questions:
- Check the update logs in `update_logs/` folder
- Review the `update_log.json` file
- Manually test with: `python update_ice_data.py`
- Verify ICE data availability: https://www.ice.com/report/122
