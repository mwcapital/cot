#!/usr/bin/env python3

with open('/Users/makson/Desktop/COT-Analysis/src/legacyF.py', 'r') as f:
    lines = f.readlines()

# Find def main
main_line = None
for i, line in enumerate(lines):
    if line.strip().startswith('def main'):
        main_line = i + 1
        print(f"Found 'def main' at line {main_line}")
        break

# Find percentile-related UI elements
print("\nSearching for percentile-related UI elements:")
for i, line in enumerate(lines):
    line_lower = line.lower()
    if 'percentile' in line_lower and any(x in line_lower for x in ['selectbox', 'column', 'multiselect', 'select']):
        print(f"Line {i+1}: {line.strip()}")

# Find where create_percentile_chart is called
print("\nSearching for create_percentile_chart calls:")
for i, line in enumerate(lines):
    if 'create_percentile_chart' in line:
        # Show context
        start = max(0, i - 30)
        end = min(len(lines), i + 5)
        print(f"\nFound at line {i+1}:")
        print("Context:")
        for j in range(start, end):
            marker = ">>> " if j == i else "    "
            print(f"{j+1:4d}: {marker}{lines[j].rstrip()}")