#!/usr/bin/env python3
"""
Script to organize COT-Analysis project files
"""

import os
import shutil
from pathlib import Path

def organize_project():
    """Organize project files into proper structure"""
    
    # Define file movements
    moves = {
        # Data files
        'instruments_LegacyF.json': 'data/instruments_LegacyF.json',
        
        # Test files
        'test_app.py': 'tests/test_app.py',
        'test_clustering.py': 'tests/test_clustering.py',
        
        # Example/demo files  
        'clustering_demo.py': 'examples/clustering_demo.py',
        'clustering_no_sklearn.py': 'examples/clustering_no_sklearn.py',
        'participation_divergence_demo.py': 'examples/participation_divergence_demo.py',
        'participation_divergence_implementation.py': 'examples/participation_divergence_implementation.py',
        'participation_divergence_output.txt': 'examples/participation_divergence_output.txt',
        'participation_example.py': 'examples/participation_example.py',
        'find_function.py': 'examples/find_function.py',
        'find_main_and_percentile.py': 'examples/find_main_and_percentile.py',
        'find_main_function.py': 'examples/find_main_function.py',
        'find_momentum.py': 'examples/find_momentum.py',
        'find_percentile_ui.py': 'examples/find_percentile_ui.py',
    }
    
    # Perform moves
    for src, dst in moves.items():
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.exists():
            # Create destination directory if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.move(str(src_path), str(dst_path))
                print(f"✓ Moved {src} -> {dst}")
            except Exception as e:
                print(f"✗ Failed to move {src}: {e}")
        else:
            print(f"- Skipped {src} (not found)")
    
    # Clean up old virtual environment if requested
    if input("\nRemove venv_new directory? (y/n): ").lower() == 'y':
        try:
            shutil.rmtree('venv_new')
            print("✓ Removed venv_new directory")
        except Exception as e:
            print(f"✗ Failed to remove venv_new: {e}")
    
    print("\n✅ Project organization complete!")
    print("\nNext steps:")
    print("1. Create a new virtual environment: python -m venv venv")
    print("2. Activate it: source venv/bin/activate")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run the app: streamlit run streamlit_app.py")

if __name__ == "__main__":
    organize_project()