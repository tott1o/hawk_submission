"""
normalize.py
------------
Text normalization for business names and addresses.
Lowercases, expands abbreviations, removes punctuation.
"""

import re
import pandas as pd


# Common business name abbreviations to expand
NAME_ABBREVS = {
    r'\bcorp\b':   'corporation',
    r'\bco\b':     'company',
    r'\bcos\b':    'companies',
    r'\bltd\b':    'limited',
    r'\bpvt\b':    'private',
    r'\bpte\b':    'private',
    r'\binc\b':    'incorporated',
    r'\bllc\b':    'limited liability company',
    r'\bllp\b':    'limited liability partnership',
    r'\blp\b':     'limited partnership',
    r'\bplc\b':    'public limited company',
    r'\bintl\b':   'international',
    r'\bsvc\b':    'service',
    r'\bsvcs\b':   'services',
    r'\bmfg\b':    'manufacturing',
    r'\btech\b':   'technology',
    r'\btechnol\b': 'technology',
    r'\bgrp\b':    'group',
    r'\bassoc\b':  'associates',
    r'\bmgmt\b':   'management',
    r'&':          'and',
}

# Common address abbreviations to expand
ADDR_ABBREVS = {
    r'\brd\b':    'road',
    r'\bst\b':    'street',
    r'\bave\b':   'avenue',
    r'\bav\b':    'avenue',
    r'\bblvd\b':  'boulevard',
    r'\bdr\b':    'drive',
    r'\bln\b':    'lane',
    r'\bct\b':    'court',
    r'\bpl\b':    'place',
    r'\bsq\b':    'square',
    r'\bfwy\b':   'freeway',
    r'\bhwy\b':   'highway',
    r'\bpkwy\b':  'parkway',
    r'\bfte\b':   'suite',
    r'\bste\b':   'suite',
    r'\bapt\b':   'apartment',
    r'\bflr\b':   'floor',
    r'\bfl\b':    'floor',
    r'\bn\b':     'north',
    r'\bs\b':     'south',
    r'\be\b':     'east',
    r'\bw\b':     'west',
    r'\bnw\b':    'northwest',
    r'\bne\b':    'northeast',
    r'\bsw\b':    'southwest',
    r'\bse\b':    'southeast',
}


def normalize_name(text):
    """
    Normalize a business name:
    - Lowercase
    - Expand common abbreviations (Corp -> corporation, etc.)
    - Remove punctuation (but KEEP Unicode letters like Devanagari/Hindi)
    - Collapse extra whitespace
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text).lower().strip()
    for pattern, replacement in NAME_ABBREVS.items():
        text = re.sub(pattern, replacement, text)
    # Keep: Unicode letters (\w covers Devanagari, Latin, etc.), digits, spaces
    # Remove: punctuation and symbols only
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'_', ' ', text)          # \w includes underscore, remove it
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_address(text):
    """
    Normalize an address:
    - Lowercase
    - Expand common abbreviations (Rd -> road, St -> street, etc.)
    - Remove punctuation (but KEEP Unicode letters)
    - Collapse extra whitespace
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text).lower().strip()
    for pattern, replacement in ADDR_ABBREVS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_combined_text(df):
    """
    Create a single searchable string per row by combining
    normalized business_name + business_address.
    Used as input to TF-IDF blocking.
    """
    names   = df['business_name'].apply(normalize_name)
    addrs   = df['business_address'].apply(normalize_address)
    return names + ' ' + addrs


if __name__ == '__main__':
    # Quick test
    tests = [
        ("McDonald's Corp.", "123 Main St, NYC"),
        ("Tata Pvt Ltd",     "MG Road, Near SBI ATM, Bengaluru"),
        ("AT&T Inc",         "1 Telecom Blvd, Dallas"),
    ]
    print("Normalization tests:\n")
    for name, addr in tests:
        print(f"  Name: '{name}' -> '{normalize_name(name)}'")
        print(f"  Addr: '{addr}' -> '{normalize_address(addr)}'")
        print()
