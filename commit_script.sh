#!/bin/bash
cd /Users/makson/Desktop/COT-Analysis

# First show git status
echo "=== Current Git Status ==="
git status

# Show git diff
echo -e "\n=== Git Diff ==="
git diff

# Show recent commits
echo -e "\n=== Recent Commits ==="
git log -3 --oneline

# Add all modified files (not untracked files)
echo -e "\n=== Adding modified files ==="
git add -u

# Also add specific files related to the implementation
git add src/charts/cross_asset_analysis.py
git add src/charts/market_microstructure.py
git add src/app.py
git add src/legacyF.py

# Create the commit
echo -e "\n=== Creating commit ==="
git commit -m "$(cat <<'EOF'
feat: implement Cross-Asset Analysis and Market Structure Matrix

- Add Cross-Asset Analysis implementation from legacyF.py
  - Multi-instrument positioning comparison charts
  - Net positioning trends across assets
  - Relative positioning metrics
  
- Implement Market Structure Matrix with percentile-based comparisons
  - Historical percentile rankings for positioning metrics
  - Cross-asset correlation analysis
  - Market regime identification
  
- Fix session state management for improved UI stability
  - Resolve state persistence issues
  - Improve data caching mechanisms
  
- Add 10-year lookback period option for long-term analysis
  - Extend historical data range
  - Support decade-long trend analysis

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

# Show final status
echo -e "\n=== Final Git Status ==="
git status