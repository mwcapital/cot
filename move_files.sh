#!/bin/bash

# Move instruments_LegacyF.json to data/
mv instruments_LegacyF.json data/

# Move all test files to tests/
mv test_*.py tests/

# Move demo files to examples/
mv clustering_demo.py examples/
mv participation_*.py examples/

# Move all find_*.py files to examples/
mv find_*.py examples/