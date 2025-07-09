#!/bin/bash

# Script to remove old architecture files
# Run with: bash cleanup_old_files.sh

echo "Cleaning up old architecture files..."

# Old architecture files in src/
rm -f src/legacyF.py
rm -f src/legacyF.py.backup
rm -f src/app.py
rm -f src/app_modular.py
rm -f src/main_restructured.py
rm -f src/handler_functions.py
rm -f src/refactored_handlers.py
rm -f src/display_functions.py
rm -f src/functions.py
rm -f src/ui_components.py

# Unused chart files in src/charts/
rm -f src/charts/base_charts.py
rm -f src/charts/concentration_momentum.py
rm -f src/charts/market_state_clusters.py
rm -f src/charts/momentum_charts.py
rm -f src/charts/participant_behavior_clusters.py
rm -f src/charts/participation_charts.py
rm -f src/charts/percentile_charts.py
rm -f src/charts/seasonality_charts.py
rm -f src/charts/share_of_oi.py
rm -f src/charts/time_series.py
rm -f src/charts/trader_analysis.py
rm -f src/charts/trader_participation_analysis.py

# Test and temporary files in root
rm -f test_app.py
rm -f check_json.py
rm -f clustering_demo.py
rm -f clustering_no_sklearn.py
rm -f find_function.py
rm -f find_main.py
rm -f find_main_and_percentile.py
rm -f find_main_function.py
rm -f find_momentum.py
rm -f find_percentile_ui.py
rm -f find_search_ui.py
rm -f migrate_to_modular.py
rm -f move_files.py
rm -f move_files.sh
rm -f organize_files.py
rm -f participation_divergence_demo.py
rm -f participation_divergence_implementation.py
rm -f participation_divergence_output.txt
rm -f participation_example.py
rm -f search_multi.py
rm -f temp_find_function.py
rm -f test_clustering.py
rm -f test_heterogeneity.py
rm -f test_microstructure.py
rm -f test_regime_detection.py
rm -f trader_estimate.py

# Log files
rm -f streamlit.log
rm -f streamlit_app.log
rm -f streamlit_new.log
rm -f streamlit_output.log
rm -f streamlit_test.log

# Old database file
rm -f instruments_LegacyF.json

# Remove empty directories (only if empty)
rmdir src/data 2>/dev/null
rmdir src/ui 2>/dev/null
rmdir src/utils 2>/dev/null

echo "Cleanup complete!"
echo ""
echo "Remaining structure:"
echo "==================="
find src -name "*.py" -type f | grep -v __pycache__ | sort
echo ""
echo "Other important files:"
ls *.json *.txt *.toml *.md 2>/dev/null | grep -v instruments_LegacyF.json

echo ""
echo "To commit these deletions:"
echo "git add -A"
echo "git commit -m \"refactor: remove old architecture files and clean up project structure\""