#!/bin/bash

# Refined cleanup - remove additional unused files identified

echo "Refined cleanup - removing additional unused files..."
echo ""

# Remove unused chart files
echo "Removing unused chart files..."
rm -f src/charts/time_series.py
rm -f src/charts/trader_analysis.py
rm -f src/charts/market_state_clusters.py

# Remove empty directories (ui, utils, data) since they only contain __init__.py
echo "Removing empty placeholder directories..."
rm -rf src/ui
rm -rf src/utils
rm -rf src/data

# Remove __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove .DS_Store files
echo "Removing .DS_Store files..."
find . -name ".DS_Store" -delete

# Remove old cleanup scripts (keeping this one for reference)
echo "Removing old cleanup scripts..."
rm -f cleanup_old_files.sh
rm -f cleanup_new_architecture.sh
rm -f cleanup_final.sh
rm -f safe_cleanup.sh

# Remove other utility scripts
rm -f commit_script.sh
rm -f git_commands.sh

# Remove old README in src
rm -f src/README.md

echo ""
echo "Cleanup complete!"
echo ""
echo "Final structure:"
echo "==============="
echo ""
echo "Root directory:"
ls -la | grep -v "^d" | grep -v "__pycache__" | grep -v ".DS_Store"
echo ""
echo "src/ directory:"
ls -la src/ | grep -v "__pycache__" | grep -v ".DS_Store"
echo ""
echo "src/charts/ directory:"
ls -la src/charts/ | grep -v "__pycache__" | grep -v ".DS_Store"
echo ""
echo "Total Python files in src/charts/:"
ls src/charts/*.py 2>/dev/null | wc -l
echo ""
echo "Essential files preserved:"
echo "- streamlit_app.py (entry point)"
echo "- src/main.py (main app logic)"
echo "- src/config.py (configuration)"
echo "- src/data_fetcher.py (data fetching)"
echo "- src/display_functions_exact.py (display functions)"
echo "- src/multi_instrument_handler.py (multi-instrument logic)"
echo "- instruments_LegacyF.json (instrument database)"
echo "- All imported chart modules in src/charts/"
echo ""
echo "To verify the app still works:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Run: streamlit run streamlit_app.py"
echo "3. Test all chart types"