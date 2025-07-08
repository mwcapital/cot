#!/usr/bin/env python3
import re

# Read the file and find the function
with open('src/legacyF.py', 'r') as f:
    lines = f.readlines()

# Find the line number where create_percentile_chart is defined
for i, line in enumerate(lines, 1):
    if 'def create_percentile_chart' in line:
        print(f"Found 'def create_percentile_chart' at line {i}")
        # Print the function and some context
        start = max(0, i-1)
        end = min(len(lines), i+100)
        for j in range(start, end):
            if j >= i-1 and j < len(lines):
                print(f"{j+1}: {lines[j]}", end='')
                # Look for the cumulative percentile curve
                if 'cumulative' in lines[j].lower() or 'go.Scatter' in lines[j]:
                    print(f"\n*** Found potential cumulative curve at line {j+1} ***\n")
                # Look for layout configuration
                if 'update_layout' in lines[j] or 'legend' in lines[j]:
                    print(f"\n*** Found layout/legend configuration at line {j+1} ***\n")
        break
else:
    print("Function 'create_percentile_chart' not found")