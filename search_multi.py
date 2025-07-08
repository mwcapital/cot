#!/usr/bin/env python3
import re

# Read the backup file
with open('src/legacyF.py.backup', 'r') as f:
    lines = f.readlines()

# Find lines containing "def handle_multi_instrument_flow"
for i, line in enumerate(lines):
    if 'def handle_multi_instrument_flow' in line:
        print(f"Found at line {i+1}")
        # Print the function and the next 100 lines
        for j in range(i, min(i+100, len(lines))):
            print(f"{j+1}: {lines[j]}", end="")