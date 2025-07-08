#!/usr/bin/env python3

with open('/Users/makson/Desktop/COT-Analysis/src/display_functions_exact.py', 'r') as f:
    lines = f.readlines()
    
    # Find the function
    func_line = None
    for i, line in enumerate(lines):
        if 'def display_trader_participation_chart' in line:
            func_line = i + 1  # Line numbers start at 1
            print(f"1. Function starts at line: {func_line}")
            break
    
    if func_line:
        # Look for analysis_type selection
        for i in range(func_line - 1, min(func_line + 200, len(lines))):
            if 'analysis_type' in lines[i] and 'selectbox' in lines[i]:
                print(f"\n2. analysis_type selection happens at line: {i + 1}")
                print(f"   Content: {lines[i].strip()}")
                
                # Find the options
                j = i
                while j < min(i + 20, len(lines)):
                    if '[' in lines[j] and ']' in lines[j]:
                        print(f"\n3. Analysis types available (line {j + 1}):")
                        print(f"   {lines[j].strip()}")
                        break
                    j += 1
                break
        
        # Look for how analyses are handled
        print("\n4. Structure of how different analyses are handled:")
        for i in range(func_line - 1, min(func_line + 300, len(lines))):
            if 'if analysis_type ==' in lines[i] or 'elif analysis_type ==' in lines[i]:
                print(f"   Line {i + 1}: {lines[i].strip()}")