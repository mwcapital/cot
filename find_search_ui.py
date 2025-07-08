#!/usr/bin/env python3
import re

# Read the backup file
with open('src/legacyF.py.backup', 'r') as f:
    content = f.read()

# Find lines containing "Choose search method"
pattern = r'.*Choose search method.*'
matches = re.finditer(pattern, content, re.IGNORECASE)

for match in matches:
    # Find the line number
    line_num = content[:match.start()].count('\n') + 1
    print(f"Found at line {line_num}")
    
    # Extract context around the match
    lines = content.split('\n')
    start = max(0, line_num - 5)
    end = min(len(lines), line_num + 100)
    
    print("\nContext:")
    for i in range(start, end):
        if i < len(lines):
            print(f"{i+1}: {lines[i]}")