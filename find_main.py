#!/usr/bin/env python3
import re

# Read the backup file
with open('src/legacyF.py.backup', 'r') as f:
    lines = f.readlines()

# Find lines containing "def main"
for i, line in enumerate(lines):
    if 'def main' in line:
        print(f"Found at line {i+1}: {line.strip()}")
        print(f"Starting from line {i+1}")
        break