#!/usr/bin/env python3
"""
Compare two JSON files to see if they are structurally and content-wise similar
"""

import json
from collections import defaultdict

def compare_json_files(file1, file2):
    """Compare two JSON files ignoring order"""
    
    with open(file1, 'r') as f:
        data1 = json.load(f)
    
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    print("=" * 60)
    print(f"COMPARING: {file1} vs {file2}")
    print("=" * 60)
    
    # Compare top-level keys
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    
    print("\n📋 TOP-LEVEL KEYS:")
    print(f"File 1 keys: {sorted(keys1)}")
    print(f"File 2 keys: {sorted(keys2)}")
    
    if keys1 != keys2:
        print("⚠️  Different keys!")
        print(f"  Only in file 1: {keys1 - keys2}")
        print(f"  Only in file 2: {keys2 - keys1}")
    else:
        print("✅ Same top-level keys")
    
    # Compare statistics
    print("\n📊 STATISTICS:")
    print(f"Total instruments - Original: {data1.get('total_count', len(data1.get('all_instruments', [])))}")
    print(f"Total instruments - New:      {data2.get('total_count', len(data2.get('all_instruments', [])))}")
    
    # Compare all_instruments
    if 'all_instruments' in data1 and 'all_instruments' in data2:
        set1 = set(data1['all_instruments'])
        set2 = set(data2['all_instruments'])
        
        print(f"\n📈 INSTRUMENTS:")
        print(f"Original count: {len(set1)}")
        print(f"New count:      {len(set2)}")
        
        if set1 != set2:
            only_in_original = set1 - set2
            only_in_new = set2 - set1
            
            if only_in_original:
                print(f"\n⚠️  Only in original ({len(only_in_original)} instruments):")
                for inst in sorted(list(only_in_original)[:5]):
                    print(f"  - {inst}")
                if len(only_in_original) > 5:
                    print(f"  ... and {len(only_in_original) - 5} more")
            
            if only_in_new:
                print(f"\n⚠️  Only in new ({len(only_in_new)} instruments):")
                for inst in sorted(list(only_in_new)[:5]):
                    print(f"  - {inst}")
                if len(only_in_new) > 5:
                    print(f"  ... and {len(only_in_new) - 5} more")
        else:
            print("✅ Exact same instruments in both files")
    
    # Compare exchanges structure
    if 'exchanges' in data1 and 'exchanges' in data2:
        print("\n🏢 EXCHANGES:")
        exchanges1 = set(data1['exchanges'].keys())
        exchanges2 = set(data2['exchanges'].keys())
        
        print(f"Original exchanges: {len(exchanges1)}")
        print(f"New exchanges:      {len(exchanges2)}")
        
        # Show sample exchanges
        print("\nSample exchanges from original:")
        for ex in sorted(list(exchanges1))[:5]:
            print(f"  - {ex}")
        
        print("\nSample exchanges from new:")
        for ex in sorted(list(exchanges2))[:5]:
            print(f"  - {ex}")
        
        # Compare a specific exchange structure (CBT)
        if 'CBT' in data1['exchanges'] and 'CBT' in data2['exchanges']:
            print("\n🔍 DETAILED COMPARISON FOR CBT EXCHANGE:")
            
            cbt1 = data1['exchanges']['CBT']
            cbt2 = data2['exchanges']['CBT']
            
            groups1 = set(cbt1.keys())
            groups2 = set(cbt2.keys())
            
            print(f"Original CBT groups: {sorted(groups1)}")
            print(f"New CBT groups:      {sorted(groups2)}")
            
            # Check NATURAL RESOURCES in CBT
            if 'NATURAL RESOURCES' in cbt1 and 'NATURAL RESOURCES' in cbt2:
                nr1 = cbt1['NATURAL RESOURCES']
                nr2 = cbt2['NATURAL RESOURCES']
                
                subgroups1 = set(nr1.keys())
                subgroups2 = set(nr2.keys())
                
                print(f"\nCBT > NATURAL RESOURCES subgroups:")
                print(f"  Original: {sorted(subgroups1)}")
                print(f"  New:      {sorted(subgroups2)}")
                
                # Check PRECIOUS METALS
                if 'PRECIOUS METALS' in nr1 and 'PRECIOUS METALS' in nr2:
                    pm1 = nr1['PRECIOUS METALS']
                    pm2 = nr2['PRECIOUS METALS']
                    
                    commodities1 = set(pm1.keys())
                    commodities2 = set(pm2.keys())
                    
                    print(f"\nCBT > NATURAL RESOURCES > PRECIOUS METALS commodities:")
                    print(f"  Original: {sorted(commodities1)}")
                    print(f"  New:      {sorted(commodities2)}")
                    
                    # Check GOLD instruments
                    if 'GOLD' in pm1 and 'GOLD' in pm2:
                        gold1 = set(pm1['GOLD'])
                        gold2 = set(pm2['GOLD'])
                        
                        print(f"\nGOLD instruments:")
                        print(f"  Original: {gold1}")
                        print(f"  New:      {gold2}")
                        if gold1 == gold2:
                            print("  ✅ Exact match!")
    
    # Compare commodity groups
    if 'commodity_groups' in data1 and 'commodity_groups' in data2:
        print("\n📁 COMMODITY GROUPS:")
        groups1 = set(data1['commodity_groups'].keys())
        groups2 = set(data2['commodity_groups'].keys())
        
        print(f"Original groups: {sorted(groups1)}")
        print(f"New groups:      {sorted(groups2)}")
        
        if groups1 == groups2:
            print("✅ Same commodity groups")
        else:
            print("⚠️  Different commodity groups")
    
    # Compare commodity subgroups
    if 'commodity_subgroups' in data1 and 'commodity_subgroups' in data2:
        print("\n📂 COMMODITY SUBGROUPS:")
        subgroups1 = set(data1['commodity_subgroups'].keys())
        subgroups2 = set(data2['commodity_subgroups'].keys())
        
        print(f"Original count: {len(subgroups1)}")
        print(f"New count:      {len(subgroups2)}")
        
        # Show differences if any
        if subgroups1 != subgroups2:
            only_orig = subgroups1 - subgroups2
            only_new = subgroups2 - subgroups1
            
            if only_orig:
                print(f"\nOnly in original ({len(only_orig)}):")
                for sg in sorted(list(only_orig))[:10]:
                    print(f"  - {sg}")
            
            if only_new:
                print(f"\nOnly in new ({len(only_new)}):")
                for sg in sorted(list(only_new))[:10]:
                    print(f"  - {sg}")

if __name__ == "__main__":
    compare_json_files("instruments_LegacyF.json", "instruments_LegacyF_new.json")