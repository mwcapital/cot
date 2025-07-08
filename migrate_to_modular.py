#!/usr/bin/env python3
"""
Migration script to help transition from legacyF.py to modular structure
"""

import os
import shutil
from pathlib import Path


def create_backup():
    """Create backup of current files"""
    backup_dir = Path("backup_before_migration")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "src/legacyF.py",
        "src/app.py",
        "streamlit_app.py"
    ]
    
    for file in files_to_backup:
        if Path(file).exists():
            dest = backup_dir / Path(file).name
            shutil.copy2(file, dest)
            print(f"✓ Backed up {file} to {dest}")


def update_streamlit_app():
    """Update main entry point to use modular app"""
    content = '''"""
CFTC COT Data Dashboard
Main entry point for the Streamlit application
"""

import streamlit as st
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import configuration
from config import PAGE_CONFIG

# Configure the page
st.set_page_config(**PAGE_CONFIG)

# Import main app function - using modular version
from app_modular import main

if __name__ == "__main__":
    main()
'''
    
    with open("streamlit_app.py", "w") as f:
        f.write(content)
    print("✓ Updated streamlit_app.py to use modular structure")


def create_run_scripts():
    """Create convenient run scripts"""
    
    # Create run_legacy.py
    legacy_content = '''#!/usr/bin/env python3
"""Run the legacy monolithic version"""
import subprocess
subprocess.run(["streamlit", "run", "src/legacyF.py"])
'''
    
    with open("run_legacy.py", "w") as f:
        f.write(legacy_content)
    os.chmod("run_legacy.py", 0o755)
    print("✓ Created run_legacy.py for running legacy version")
    
    # Create run_modular.py
    modular_content = '''#!/usr/bin/env python3
"""Run the new modular version"""
import subprocess
subprocess.run(["streamlit", "run", "streamlit_app.py"])
'''
    
    with open("run_modular.py", "w") as f:
        f.write(modular_content)
    os.chmod("run_modular.py", 0o755)
    print("✓ Created run_modular.py for running modular version")


def check_imports():
    """Check if all required modules exist"""
    required_modules = [
        "src/config.py",
        "src/data_fetcher.py",
        "src/ui_components.py",
        "src/display_functions.py",
        "src/charts/__init__.py",
        "src/charts/base_charts.py",
        "src/charts/seasonality_charts.py",
        "src/charts/participation_charts.py",
        "src/charts/share_of_oi.py",
        "src/charts/percentile_charts.py",
        "src/charts/momentum_charts.py",
        "src/charts/trader_analysis.py",
        "src/charts/time_series.py"
    ]
    
    print("\n📋 Checking required modules:")
    all_exist = True
    
    for module in required_modules:
        if Path(module).exists():
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module} - MISSING!")
            all_exist = False
    
    return all_exist


def main():
    """Run the migration process"""
    print("🚀 CFTC COT Dashboard Migration Tool")
    print("=" * 50)
    
    # Step 1: Create backups
    print("\n1️⃣ Creating backups...")
    create_backup()
    
    # Step 2: Check modules
    print("\n2️⃣ Checking module structure...")
    modules_ok = check_imports()
    
    if not modules_ok:
        print("\n⚠️  Some modules are missing!")
        print("Please ensure all chart modules are created before proceeding.")
        return
    
    # Step 3: Update entry points
    print("\n3️⃣ Updating entry points...")
    update_streamlit_app()
    create_run_scripts()
    
    # Step 4: Final instructions
    print("\n✅ Migration preparation complete!")
    print("\n📝 Next steps:")
    print("1. Test the modular version: python run_modular.py")
    print("2. Compare with legacy version: python run_legacy.py")
    print("3. Once satisfied, you can remove src/legacyF.py")
    print("\n💡 Tips:")
    print("- The modular version is in src/app_modular.py")
    print("- All chart functions are now in src/charts/")
    print("- Display functions are in src/display_functions.py")
    print("- Configuration is in src/config.py")


if __name__ == "__main__":
    main()