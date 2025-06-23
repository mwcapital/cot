'''
need those packages

import json
import requests
from sodapy import Socrata
import pandas as pd
from collections import defaultdict
'''


def fetch_all_cftc_instruments(dataset_code: str, api_token: str = None):
    """
    Fetch all unique instruments from the CFTC COT dataset
    and organize them by exchange code and category hierarchy
    """
    try:
        # Initialize Socrata client
        client = Socrata("publicreporting.cftc.gov", api_token or None)

        # Query unique combinations including market code
        select_fields = [
            "cftc_market_code",
            "market_and_exchange_names",
            "commodity_group_name",
            "commodity_subgroup_name",
            "commodity_name"
        ]
        results = client.get(
            dataset_code,
            select=",".join(select_fields),
            group=",".join(select_fields),
            limit=5000
        )

        # Prepare nested structures
        exchanges = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(list)
                )
            )
        )
        commodity_groups = defaultdict(list)
        commodity_subgroups = defaultdict(list)
        commodities = defaultdict(list)
        all_instruments = []

        for record in results:
            # Ensure all fields present
            if all(k in record for k in select_fields):
                code = record['cftc_market_code']
                instrument_full = record['market_and_exchange_names']
                group = record['commodity_group_name']
                subgroup = record['commodity_subgroup_name']
                commodity = record['commodity_name']

                # Track all unique instruments
                all_instruments.append(instrument_full)

                # Use code as key for exchange
                exchanges[code][group][subgroup][commodity].append(instrument_full)

                # Build flat lists
                commodity_groups[group].append(instrument_full)
                commodity_subgroups[subgroup].append(instrument_full)
                commodities[commodity].append(instrument_full)

        # Convert to regular dicts
        exchanges_dict = {
            code: {
                grp: {
                    subgrp: dict(comms)
                    for subgrp, comms in subgroups.items()
                }
                for grp, subgroups in groups.items()
            }
            for code, groups in exchanges.items()
        }

        result = {
            "exchanges": exchanges_dict,
            "commodity_groups": dict(commodity_groups),
            "commodity_subgroups": dict(commodity_subgroups),
            "commodities": dict(commodities),
            "all_instruments": sorted(set(all_instruments)),
            "total_count": len(set(all_instruments)),
            "exchanges_count": len(exchanges_dict),
            "groups_count": len(commodity_groups),
            "subgroups_count": len(commodity_subgroups),
            "commodities_count": len(commodities),
            "last_updated": pd.Timestamp.now().isoformat(),
            "api_info": {
                "dataset_id": dataset_code,
                "base_url": f"https://publicreporting.cftc.gov/resource/{dataset_code}.json",
                "hierarchy_fields": select_fields,
                "note": "Hierarchy: Exchange Code -> Commodity Group -> Subgroup -> Commodity -> Instruments"
            }
        }

        return result

    except Exception as e:
        print(f"Error fetching instruments: {e}")
        return None

    finally:
        if 'client' in locals():
            client.close()

def save_instruments_to_file(instruments_data, filename:str):
    """Save the instruments data to a JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(instruments_data, f, indent=2, ensure_ascii=False)
        print(f"Instruments saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving to file: {e}")
        return False