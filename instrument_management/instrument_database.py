#!/usr/bin/env python3
"""
Script to generate instruments_LegacyF.json from CFTC API
This fetches all available instruments and organizes them using the API's commodity hierarchy
"""

import json
import time
from datetime import datetime
from collections import defaultdict
from sodapy import Socrata

# API Configur1ation
CFTC_API_BASE = "publicreporting.cftc.gov"
DATASET_CODE = "6dca-aqww"
API_TOKEN = "3CKjkFN6jIIHgSkIJH19i7VhK"

def fetch_all_instruments():
    """Fetch all unique instruments from CFTC API with all their metadata"""
    print("Connecting to CFTC API...")
    client = Socrata(CFTC_API_BASE, API_TOKEN, timeout=30)  # Increase timeout
    
    # Get current year dynamically
    current_year = datetime.now().year
    year_filter = f"report_date_as_yyyy_mm_dd >= '{current_year}-01-01T00:00:00.000'"
    
    try:
        print(f"Fetching unique instruments with reports from {current_year}...")
        print("This may take a moment...")
        # Get the key fields including commodity_group_name, commodity_subgroup_name, and contract code
        # Filter for only instruments with current year reports
        results = client.get(
            DATASET_CODE,
            where=year_filter,
            select="DISTINCT market_and_exchange_names, commodity_name, commodity_group_name, commodity_subgroup_name, cftc_commodity_code, cftc_contract_market_code",
            limit=10000  # Reduced limit to avoid timeout
        )
        
        print(f"Found {len(results)} unique instrument records")
        return results
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Retrying with smaller limit...")
        try:
            results = client.get(
                DATASET_CODE,
                where=year_filter,
                select="DISTINCT market_and_exchange_names, commodity_name, commodity_group_name, commodity_subgroup_name, cftc_contract_market_code",
                limit=5000
            )
            print(f"Found {len(results)} unique instrument records")
            return results
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            return []
    finally:
        client.close()

def organize_instruments(instruments):
    """Organize instruments using the commodity hierarchy from the API"""
    print("Organizing instruments using API's commodity hierarchy...")
    
    # First, collect all unique instruments with their best available code
    unique_instruments = {}
    for record in instruments:
        instrument_name = record.get('market_and_exchange_names', '').strip()
        contract_code = record.get('cftc_contract_market_code', '').strip()
        
        if not instrument_name:
            continue
            
        # If we haven't seen this instrument, or if we have but this one has a code and the previous didn't
        if instrument_name not in unique_instruments or (contract_code and not unique_instruments[instrument_name].get('cftc_contract_market_code')):
            unique_instruments[instrument_name] = record
    
    print(f"Found {len(unique_instruments)} unique instruments after deduplication")
    
    # Main hierarchical structure
    exchanges = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    # Additional lookup structures
    commodity_groups = defaultdict(set)
    commodity_subgroups = defaultdict(set)
    commodities = defaultdict(set)
    all_instruments = []
    
    # Process each unique instrument
    codes_found = 0
    no_codes = 0
    for instrument_name, record in unique_instruments.items():
        instrument_name = record.get('market_and_exchange_names', '').strip()
        commodity_name = record.get('commodity_name', '').strip()
        commodity_group = record.get('commodity_group_name', '').strip()
        commodity_subgroup = record.get('commodity_subgroup_name', '').strip()
        contract_code = record.get('cftc_contract_market_code', '').strip()
        
        if not instrument_name:
            continue
        
        # Append contract code to instrument name if available
        if contract_code:
            instrument_name_with_code = f"{instrument_name} ({contract_code})"
            codes_found += 1
        else:
            instrument_name_with_code = instrument_name
            no_codes += 1
        
        # Extract exchange from the instrument name (it's always after the last " - ")
        exchange = "UNKNOWN"
        if " - " in instrument_name:
            exchange = instrument_name.rsplit(" - ", 1)[1].strip()
            # Clean up exchange name (remove trailing spaces, standardize)
            exchange = exchange.strip()
            
            # Map to standard abbreviations for known exchanges
            exchange_map = {
                "CHICAGO BOARD OF TRADE": "CBT",
                "CHICAGO MERCANTILE EXCHANGE": "CME",
                "COMMODITY EXCHANGE INC.": "COMEX",
                "NEW YORK MERCANTILE EXCHANGE": "NYMEX",
                "ICE FUTURES U.S.": "ICE",
                "CBOE FUTURES EXCHANGE": "CFE",
                "MINNEAPOLIS GRAIN EXCHANGE": "MGE",
                "KANSAS CITY BOARD OF TRADE": "KCBT",
                "NEW YORK COTTON EXCHANGE": "NYCE",
                "NEW YORK BOARD OF TRADE": "NYBOT",
                "ICE FUTURES EUROPE": "ICE EU",
                "ICE FUTURES ENERGY DIV": "ICE ENERGY",
                "CME COMMODITY EXCHANGE INC.": "COMEX"
            }
            exchange = exchange_map.get(exchange, exchange)
        
        # Use the API's commodity hierarchy
        group = commodity_group if commodity_group else "UNSPECIFIED"
        subgroup = commodity_subgroup if commodity_subgroup else "UNSPECIFIED"
        commodity = commodity_name.upper() if commodity_name else "UNSPECIFIED"
        
        # Add to hierarchical structure using the name with code
        if instrument_name_with_code not in exchanges[exchange][group][subgroup][commodity]:
            exchanges[exchange][group][subgroup][commodity].append(instrument_name_with_code)
        
        # Add to lookup structures using the name with code
        commodity_groups[group].add(instrument_name_with_code)
        commodity_subgroups[subgroup].add(instrument_name_with_code)
        commodities[commodity].add(instrument_name_with_code)
        
        if instrument_name_with_code not in all_instruments:
            all_instruments.append(instrument_name_with_code)
    
    print(f"Organized {len(all_instruments)} unique instruments")
    print(f"  - With contract codes: {codes_found}")
    print(f"  - Without contract codes: {no_codes}")
    
    # Convert defaultdicts to regular dicts and sets to sorted lists
    exchanges_dict = {}
    for exchange, groups in exchanges.items():
        exchanges_dict[exchange] = {}
        for group, subgroups in groups.items():
            exchanges_dict[exchange][group] = {}
            for subgroup, commodities_data in subgroups.items():
                exchanges_dict[exchange][group][subgroup] = {}
                for commodity, instruments_list in commodities_data.items():
                    exchanges_dict[exchange][group][subgroup][commodity] = sorted(instruments_list)
    
    # Convert sets to sorted lists for lookup structures
    commodity_groups_dict = {k: sorted(list(v)) for k, v in commodity_groups.items()}
    commodity_subgroups_dict = {k: sorted(list(v)) for k, v in commodity_subgroups.items()}
    commodities_dict = {k: sorted(list(v)) for k, v in commodities.items()}
    
    return {
        "exchanges": exchanges_dict,
        "commodity_groups": commodity_groups_dict,
        "commodity_subgroups": commodity_subgroups_dict,
        "commodities": commodities_dict,
        "all_instruments": sorted(all_instruments)
    }

def generate_statistics(data):
    """Generate statistics about the instruments database"""
    stats = {
        "total_count": len(data["all_instruments"]),
        "exchanges_count": len(data["exchanges"]),
        "groups_count": len(data["commodity_groups"]),
        "subgroups_count": len(data["commodity_subgroups"]),
        "commodities_count": len(data["commodities"]),
        "last_updated": datetime.now().isoformat(),
        "api_info": {
            "source": "CFTC Public Reporting",
            "dataset": DATASET_CODE,
            "api_base": CFTC_API_BASE
        }
    }
    return stats

def main():
    """Main function to generate the instruments database"""
    print("=" * 60)
    print("CFTC Instruments Database Generator")
    print("=" * 60)
    
    start_time = time.time()
    
    # Fetch all instruments
    instruments = fetch_all_instruments()
    
    # Organize into hierarchical structure using API's commodity hierarchy
    organized_data = organize_instruments(instruments)
    
    # Generate statistics
    stats = generate_statistics(organized_data)
    
    # Combine all data
    final_data = {**organized_data, **stats}
    
    # Save to JSON file - using _new suffix to not overwrite original
    output_file = "instrument_management/instruments_LegacyF_new.json"
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2, sort_keys=False)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ Database generation complete!")
    print(f"📊 Total instruments: {stats['total_count']}")
    print(f"🏢 Exchanges: {stats['exchanges_count']}")
    print(f"📁 Commodity Groups: {stats['groups_count']}")
    print(f"📂 Commodity Subgroups: {stats['subgroups_count']}")
    print(f"📦 Commodities: {stats['commodities_count']}")
    print(f"⏱️  Time taken: {elapsed_time:.2f} seconds")
    print(f"💾 Saved to: {output_file}")
    print("=" * 60)
    
    # Show sample of structure for verification
    print("\nSample of generated structure:")
    print("-" * 40)
    for exchange in list(final_data['exchanges'].keys())[:2]:
        print(f"Exchange: {exchange}")
        for group in list(final_data['exchanges'][exchange].keys())[:2]:
            print(f"  Group: {group}")
            for subgroup in list(final_data['exchanges'][exchange][group].keys())[:2]:
                print(f"    Subgroup: {subgroup}")
                for commodity in list(final_data['exchanges'][exchange][group][subgroup].keys())[:2]:
                    instruments = final_data['exchanges'][exchange][group][subgroup][commodity]
                    print(f"      Commodity: {commodity} ({len(instruments)} instruments)")

if __name__ == "__main__":
    main()