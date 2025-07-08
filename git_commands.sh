#!/bin/bash
cd /Users/makson/Desktop/COT-Analysis
echo "=== Git Status ==="
git status
echo -e "\n=== Git Diff ==="
git diff
echo -e "\n=== Git Log (last 3 commits) ==="
git log -3 --oneline