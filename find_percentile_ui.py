#!/usr/bin/env python3
import re

# Read the file
with open('/Users/makson/Desktop/COT-Analysis/src/legacyF.py', 'r') as f:
    content = f.read()
    
# Find all occurrences of create_percentile_chart calls
pattern = r'create_percentile_chart\([^)]+\)'
matches = re.finditer(pattern, content)

for match in matches:
    # Get 100 lines before and 20 lines after the match
    start = max(0, content.rfind('\n', 0, match.start() - 2000))
    end = min(len(content), content.find('\n', match.end() + 500))
    
    context = content[start:end]
    lines = context.split('\n')
    
    # Find line numbers
    line_num = content[:match.start()].count('\n') + 1
    
    print(f"Found create_percentile_chart at line ~{line_num}")
    print("Context:")
    print("-" * 80)
    
    # Print with line numbers
    start_line = content[:start].count('\n') + 1
    for i, line in enumerate(lines[:50]):  # Show first 50 lines of context
        print(f"{start_line + i:4d}: {line}")
    print("...")
    print("-" * 80)
    print()