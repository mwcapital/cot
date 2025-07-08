import json

with open('instruments_LegacyF.json', 'r') as f:
    data = json.load(f)
    
print("Top-level keys:", list(data.keys()))
print("\nHas 'all_instruments':", 'all_instruments' in data)

# Count total instruments
total = 0
if 'exchanges' in data:
    for exchange, groups in data['exchanges'].items():
        for group, subgroups in groups.items():
            for subgroup, commodities in subgroups.items():
                for commodity, instruments in commodities.items():
                    total += len(instruments)
                    
print(f"\nTotal instruments in exchanges: {total}")