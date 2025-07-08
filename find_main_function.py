#!/usr/bin/env python3
import re

# Read the file
with open('/Users/makson/Desktop/COT-Analysis/src/legacyF.py', 'r') as f:
    content = f.read()
    lines = content.split('\n')

# Find 'def main'
for i, line in enumerate(lines):
    if line.strip().startswith('def main'):
        print(f"Found 'def main' at line {i+1}")
        # Print the next 100 lines to see the structure
        for j in range(i, min(i+100, len(lines))):
            print(f"{j+1:4d}: {lines[j]}")
        break

# Also search for where percentile-related UI might be
print("\n\n" + "="*80)
print("Looking for percentile-related UI code:")
print("="*80)

for i, line in enumerate(lines):
    if ('percentile' in line.lower() and 'chart_type' in line) or \
       ('"Percentile Analysis"' in line) or \
       ('create_percentile_chart' in line and 'column' in line):
        # Print context
        start = max(0, i-20)
        end = min(len(lines), i+20)
        print(f"\nFound at line {i+1}:")
        for j in range(start, end):
            marker = ">>>" if j == i else "   "
            print(f"{j+1:4d}: {marker} {lines[j]}")