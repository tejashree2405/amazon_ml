import pandas as pd
import numpy as np

try:
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz.fuzz import token_sort_ratio, ratio
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    import difflib

def get_ratio(s1, s2):
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2: 
        return 0.0
    if HAS_RAPIDFUZZ:
        return ratio(s1, s2) / 100.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def get_token_sort_ratio(s1, s2):
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2: 
        return 0.0
    if HAS_RAPIDFUZZ:
        return token_sort_ratio(s1, s2) / 100.0
    return difflib.SequenceMatcher(None, " ".join(sorted(s1.split())), " ".join(sorted(s2.split()))).ratio()

def get_jaccard_word(s1, s2):
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2:
        return 0.0
    set1, set2 = set(s1.split()), set(s2.split())
    u = set1.union(set2)
    if not u: return 0.0
    return len(set1.intersection(set2)) / len(u)

def get_jaccard_char_ngram(s1, s2, n=3):
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2:
        return 0.0
    set1 = set([s1[i:i+n] for i in range(len(s1)-n+1)])
    set2 = set([s2[i:i+n] for i in range(len(s2)-n+1)])
    u = set1.union(set2)
    if not u: return 0.0
    return len(set1.intersection(set2)) / len(u)

def exact_first_token(s1, s2):
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2:
        return 0.0
    t1, t2 = s1.split(), s2.split()
    if not t1 or not t2: return 0.0
    return 1.0 if t1[0] == t2[0] else 0.0

def compute_pair_features(pairs_df):
    """
    Computes pair-level features. 
    pairs_df must contain the following columns for both _s1 and _cand:
    norm_name, core_name, sorted_name, norm_address, addr_numbers, country, source_type (S2/S3)
    """
    
    # 1. Name Match Features
    pairs_df['feat_name_ratio'] = pairs_df.apply(lambda r: get_ratio(r['norm_name_s1'], r['norm_name_cand']), axis=1)
    pairs_df['feat_core_ratio'] = pairs_df.apply(lambda r: get_ratio(r['core_name_s1'], r['core_name_cand']), axis=1)
    pairs_df['feat_name_sort_ratio'] = pairs_df.apply(lambda r: get_token_sort_ratio(r['core_name_s1'], r['core_name_cand']), axis=1)
    pairs_df['feat_name_jaccard'] = pairs_df.apply(lambda r: get_jaccard_word(r['core_name_s1'], r['core_name_cand']), axis=1)
    pairs_df['feat_name_char3_jaccard'] = pairs_df.apply(lambda r: get_jaccard_char_ngram(r['core_name_s1'], r['core_name_cand'], n=3), axis=1)
    pairs_df['feat_name_brand_anchor'] = pairs_df.apply(lambda r: exact_first_token(r['core_name_s1'], r['core_name_cand']), axis=1)
    
    # 2. Address Match Features
    pairs_df['feat_addr_ratio'] = pairs_df.apply(lambda r: get_ratio(r['norm_address_s1'], r['norm_address_cand']), axis=1)
    pairs_df['feat_addr_jaccard'] = pairs_df.apply(lambda r: get_jaccard_word(r['norm_address_s1'], r['norm_address_cand']), axis=1)
    pairs_df['feat_addr_char4_jaccard'] = pairs_df.apply(lambda r: get_jaccard_char_ngram(r['norm_address_s1'], r['norm_address_cand'], n=4), axis=1)
    
    # numeric house-number / postal code matching
    pairs_df['feat_addr_numbers_jaccard'] = pairs_df.apply(lambda r: get_jaccard_word(r['addr_numbers_s1'], r['addr_numbers_cand']), axis=1)
    
    # 3. Country and Source Indicators
    pairs_df['feat_country_agree'] = (pairs_df['country_s1'] == pairs_df['country_cand']).astype(int)
    
    # Source indicators (S1-S2 vs S1-S3)
    pairs_df['feat_is_s2'] = pairs_df['entity_id_cand'].astype(str).str.startswith('S2-').astype(int)
    pairs_df['feat_is_s3'] = pairs_df['entity_id_cand'].astype(str).str.startswith('S3-').astype(int)
    
    # Drop blocker rank or any position-based features. None are computed here.
    return pairs_df
