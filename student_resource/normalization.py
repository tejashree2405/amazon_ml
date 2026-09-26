import re
import pandas as pd

LEGAL_SUFFIXES = {
    "llc", "inc", "ltd", "limited", "pvt", "private", "corp", "corporation", 
    "co", "company", "llp", "lp", "pc", "pllc", "incorporated"
}

def normalize_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_core_name(norm_name):
    tokens = norm_name.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)

def get_sorted_tokens(text):
    return " ".join(sorted(text.split()))

def normalize_address(address):
    # symmetric abbreviation handling
    if pd.isna(address) or not isinstance(address, str):
        return ""
    addr = normalize_text(address)
    replacements = {
        r'\bblvd\b': 'boulevard',
        r'\bave\b': 'avenue',
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bste\b': 'suite',
        r'\bapt\b': 'apartment',
        r'\bhwy\b': 'highway',
        r'\bpkwy\b': 'parkway',
    }
    for pat, rep in replacements.items():
        addr = re.sub(pat, rep, addr)
    return addr

def extract_numeric(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    nums = re.findall(r'\d+', text)
    return " ".join(nums)

def preprocess_dataframe(df):
    """
    Applies normalization routines to a DataFrame inplace and returns it.
    Expects 'business_name' and 'business_address' columns.
    """
    df['norm_name'] = df['business_name'].apply(normalize_text)
    df['core_name'] = df['norm_name'].apply(get_core_name)
    df['sorted_name'] = df['core_name'].apply(get_sorted_tokens)
    df['norm_address'] = df['business_address'].apply(normalize_address)
    df['addr_numbers'] = df['norm_address'].apply(extract_numeric)
    return df
