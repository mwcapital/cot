#!/usr/bin/env python3
import os
import shutil
import glob

# Define the base directory
base_dir = "/Users/makson/Desktop/COT-Analysis"
os.chdir(base_dir)

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("tests", exist_ok=True)
os.makedirs("examples", exist_ok=True)

# Move instruments_LegacyF.json to data/
if os.path.exists("instruments_LegacyF.json"):
    shutil.move("instruments_LegacyF.json", "data/")
    print("Moved instruments_LegacyF.json to data/")

# Move all test files to tests/
for test_file in glob.glob("test_*.py"):
    shutil.move(test_file, "tests/")
    print(f"Moved {test_file} to tests/")

# Move demo files to examples/
if os.path.exists("clustering_demo.py"):
    shutil.move("clustering_demo.py", "examples/")
    print("Moved clustering_demo.py to examples/")

for participation_file in glob.glob("participation_*.py"):
    shutil.move(participation_file, "examples/")
    print(f"Moved {participation_file} to examples/")

# Move all find_*.py files to examples/
for find_file in glob.glob("find_*.py"):
    shutil.move(find_file, "examples/")
    print(f"Moved {find_file} to examples/")

print("\nAll files moved successfully!")