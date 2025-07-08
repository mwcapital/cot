#!/usr/bin/env python3
import sys

def find_function_in_file(filepath, function_name):
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            
        for i, line in enumerate(lines):
            if f'def {function_name}' in line:
                print(f"Found at line {i + 1}")
                # Print the function definition and some context
                start = max(0, i - 2)
                end = min(len(lines), i + 50)
                
                for j in range(start, end):
                    print(f"{j + 1:4d}: {lines[j]}", end='')
                return i + 1
                
        print(f"Function {function_name} not found")
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    find_function_in_file('src/legacyF.py', 'create_single_variable_momentum_dashboard')