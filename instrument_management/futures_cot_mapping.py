#!/usr/bin/env python3
"""
Create mapping between futures price symbols and COT instruments
"""

import json
import re

# Manual mapping of futures symbols to COT instrument names
# Based on analysis of both datasets
FUTURES_TO_COT_MAPPING = {
    # Currency futures
    "AN": ["AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE"],
    "BN": ["BRITISH POUND - CHICAGO MERCANTILE EXCHANGE", "BRITISH POUND - ICE FUTURES U.S."],
    "CN": ["CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE"],
    "FN": ["EURO FX - CHICAGO MERCANTILE EXCHANGE"],
    "FX": ["EURO FX - CHICAGO MERCANTILE EXCHANGE"],  # Day session
    "JN": ["JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE"],
    "MP": ["MEXICAN PESO - CHICAGO MERCANTILE EXCHANGE"],
    "SN": ["SWISS FRANC - CHICAGO MERCANTILE EXCHANGE"],
    "DX": ["USD INDEX - ICE FUTURES U.S."],

    # Index futures
    "ES": ["E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
           "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",
           "MICRO E-MINI S&P 500 INDEX - CHICAGO MERCANTILE EXCHANGE"],
    "EN": ["E-MINI NASDAQ - CHICAGO MERCANTILE EXCHANGE",
           "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE"],
    "ER": ["RUSSELL 2000 - ICE FUTURES U.S.",
           "RUSSELL 2000 MINI - ICE FUTURES U.S."],
    "YM": ["DJIA x $5 - CHICAGO BOARD OF TRADE",
           "E-MINI DOW - CHICAGO BOARD OF TRADE"],
    "MD": ["E-MIDCAP 400 - CHICAGO MERCANTILE EXCHANGE"],
    "NK": ["NIKKEI - CHICAGO MERCANTILE EXCHANGE"],
    "AX": ["DAX INDEX - EUREX"],  # German DAX
    "CA": ["CAC40 INDEX - EURONEXT"],  # CAC40
    "XU": ["DJ EURO STOXX 50 - EUREX"],
    "LX": ["FTSE 100 - ICE FUTURES EUROPE"],
    "HS": ["HANG SENG - HONG KONG"],
    "AP": ["SPI 200 - ASX"],  # Australian index

    # Agricultural futures
    "ZC": ["CORN - CHICAGO BOARD OF TRADE"],
    "ZS": ["SOYBEANS - CHICAGO BOARD OF TRADE"],
    "ZW": ["WHEAT-SRW - CHICAGO BOARD OF TRADE"],
    "W": ["WHEAT-SRW - CHICAGO BOARD OF TRADE"],
    "KW": ["WHEAT-HRW - CHICAGO BOARD OF TRADE"],
    "MW": ["WHEAT-HRSpring - MIAX FUTURES EXCHANGE"],
    "ZM": ["SOYBEAN MEAL - CHICAGO BOARD OF TRADE"],
    "ZL": ["SOYBEAN OIL - CHICAGO BOARD OF TRADE"],
    "ZO": ["OATS - CHICAGO BOARD OF TRADE"],
    "ZR": ["ROUGH RICE - CHICAGO BOARD OF TRADE"],
    "CC": ["COCOA - ICE FUTURES U.S."],
    "KC": ["COFFEE C - ICE FUTURES U.S."],
    "SB": ["SUGAR NO. 11 - ICE FUTURES U.S."],
    "CT": ["COTTON NO. 2 - ICE FUTURES U.S."],
    "JO": ["FRZN CONCENTRATED ORANGE JUICE - ICE FUTURES U.S."],
    "LB": ["LUMBER - CHICAGO MERCANTILE EXCHANGE"],

    # Livestock futures
    "ZT": ["LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE"],
    "ZF": ["FEEDER CATTLE - CHICAGO MERCANTILE EXCHANGE"],
    "ZZ": ["LEAN HOGS - CHICAGO MERCANTILE EXCHANGE"],
    "DA": ["MILK CLASS III - CHICAGO MERCANTILE EXCHANGE"],

    # Energy futures
    "ZU": ["WTI CRUDE OIL - NEW YORK MERCANTILE EXCHANGE",
           "CRUDE OIL- NEW YORK MERCANTILE EXCHANGE"],
    "ZB": ["RBOB GASOLINE- NEW YORK MERCANTILE EXCHANGE"],
    "ZH": ["HEATING OIL- NEW YORK MERCANTILE EXCHANGE"],
    "ZN": ["NATURAL GAS - NEW YORK MERCANTILE EXCHANGE"],
    "BG": ["GASOIL - ICE FUTURES EUROPE"],

    # Metals futures
    "ZG": ["GOLD - COMMODITY EXCHANGE INC.",
           "MICRO GOLD - COMMODITY EXCHANGE INC."],
    "ZI": ["SILVER - COMMODITY EXCHANGE INC."],
    "ZK": ["COPPER - COMMODITY EXCHANGE INC.",
           "COPPER-GRADE #1 - COMMODITY EXCHANGE INC."],
    "ZP": ["PLATINUM - NEW YORK MERCANTILE EXCHANGE"],
    "ZA": ["PALLADIUM - NEW YORK MERCANTILE EXCHANGE"],

    # Financial/Interest Rate futures
    "US": ["U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE"],
    "TY": ["10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE"],
    "FB": ["5-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE"],
    "TU": ["2-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE"],
    "FF": ["30-DAY FED FUNDS - CHICAGO BOARD OF TRADE"],
    "DT": ["EURO BUND - EUREX"],
    "UB": ["EURO BOBL - EUREX"],
    "UZ": ["EURO SCHATZ - EUREX"],
    "GS": ["LONG GILT - ICE FUTURES EUROPE"],
    "CB": ["CANADIAN 10 YR BOND - MONTREAL EXCHANGE"],
    "GI": ["GSCI - CHICAGO MERCANTILE EXCHANGE"]  # Goldman Sachs Commodity Index
}

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def find_cot_code(cot_name, instruments_data):
    """Find COT code for a given instrument name"""
    # Search through all_instruments for the code
    if 'all_instruments' in instruments_data:
        for instrument in instruments_data['all_instruments']:
            # Extract the name and code from format like "GOLD - COMMODITY EXCHANGE INC. (088691)"
            if cot_name.upper() in instrument.upper():
                match = re.search(r'\((\w+)\)', instrument)
                if match:
                    return match.group(1)
    return None

def create_mapping():
    """Create comprehensive mapping between futures symbols and COT data"""

    # Load existing data
    futures_symbols = load_json('/Users/makson/Desktop/COT-Analysis/instrument_management/futures_symbols_current.json')
    cot_instruments = load_json('/Users/makson/Desktop/COT-Analysis/instrument_management/LegacyF/instruments_LegacyF.json')

    # Create enhanced futures symbols with COT mapping
    enhanced_futures = futures_symbols.copy()

    # Add COT mapping to each symbol
    for symbol, info in enhanced_futures['futures_symbols'].items():
        # Initialize COT mapping for this symbol
        info['cot_mapping'] = {
            'instruments': [],
            'codes': [],
            'matched': False
        }

        # Check if we have a mapping for this symbol
        if symbol in FUTURES_TO_COT_MAPPING:
            cot_names = FUTURES_TO_COT_MAPPING[symbol]

            for cot_name in cot_names:
                # Find the COT code
                cot_code = find_cot_code(cot_name, cot_instruments)

                if cot_code:
                    info['cot_mapping']['instruments'].append(cot_name)
                    info['cot_mapping']['codes'].append(cot_code)
                    info['cot_mapping']['matched'] = True

    # Save enhanced file
    output_file = '/Users/makson/Desktop/COT-Analysis/instrument_management/futures_symbols_enhanced.json'
    with open(output_file, 'w') as f:
        json.dump(enhanced_futures, f, indent=2)

    print(f"Enhanced mapping saved to {output_file}")

    # Print summary
    matched = sum(1 for s, i in enhanced_futures['futures_symbols'].items()
                  if i['cot_mapping']['matched'])
    total = len(enhanced_futures['futures_symbols'])

    print(f"\nMapping Summary:")
    print(f"Total symbols: {total}")
    print(f"Matched with COT: {matched}")
    print(f"Unmatched: {total - matched}")

    # List unmatched symbols
    unmatched = [s for s, i in enhanced_futures['futures_symbols'].items()
                 if not i['cot_mapping']['matched']]
    if unmatched:
        print(f"\nUnmatched symbols: {', '.join(unmatched)}")

    return enhanced_futures

if __name__ == "__main__":
    create_mapping()