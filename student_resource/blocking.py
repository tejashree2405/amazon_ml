import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Blocker:
    def __init__(self, k_fuzzy=10):
        self.k_fuzzy = k_fuzzy
        
        # Exact indexes
        self.exact_name_idx = defaultdict(list)
        self.exact_core_idx = defaultdict(list)
        self.exact_sorted_idx = defaultdict(list)
        
        # Fuzzy models
        self.char_tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
        self.word_tf = TfidfVectorizer(analyzer='word', stop_words='english', min_df=2)
        self.addr_tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
        
        # using brute force for sparse cosine
        self.char_nn = NearestNeighbors(n_neighbors=self.k_fuzzy, metric='cosine', algorithm='brute', n_jobs=-1)
        self.word_nn = NearestNeighbors(n_neighbors=self.k_fuzzy, metric='cosine', algorithm='brute', n_jobs=-1)
        self.addr_nn = NearestNeighbors(n_neighbors=self.k_fuzzy, metric='cosine', algorithm='brute', n_jobs=-1)
        
    def fit(self, df):
        logging.info(f"Building exact indexes for {len(df)} records...")
        for _, row in df.iterrows():
            eid = row['entity_id']
            if row.get('norm_name'):
                self.exact_name_idx[row['norm_name']].append(eid)
            if row.get('core_name'):
                self.exact_core_idx[row['core_name']].append(eid)
            if row.get('sorted_name'):
                self.exact_sorted_idx[row['sorted_name']].append(eid)
                
        logging.info("Building fuzzy indexes...")
        if len(df) > 0:
            char_X = self.char_tf.fit_transform(df['core_name'].fillna(''))
            self.char_nn.fit(char_X)
            
            word_X = self.word_tf.fit_transform(df['core_name'].fillna(''))
            self.word_nn.fit(word_X)
            
            addr_X = self.addr_tf.fit_transform(df['norm_address'].fillna(''))
            self.addr_nn.fit(addr_X)
            
        self.ref_ids = df['entity_id'].values

    def query(self, df_q):
        logging.info(f"Querying for {len(df_q)} records...")
        has_records = len(self.ref_ids) > 0
        
        if has_records:
            # Only query up to available records
            k = min(self.k_fuzzy, len(self.ref_ids))
            
            char_q = self.char_tf.transform(df_q['core_name'].fillna(''))
            _, char_inds = self.char_nn.kneighbors(char_q, n_neighbors=k)
            
            word_q = self.word_tf.transform(df_q['core_name'].fillna(''))
            _, word_inds = self.word_nn.kneighbors(word_q, n_neighbors=k)
            
            addr_q = self.addr_tf.transform(df_q['norm_address'].fillna(''))
            _, addr_inds = self.addr_nn.kneighbors(addr_q, n_neighbors=k)

        results = []
        for i, row in df_q.iterrows():
            candidates = set()
            
            # Exact
            if row.get('norm_name'):
                candidates.update(self.exact_name_idx.get(row['norm_name'], []))
            if row.get('core_name'):
                candidates.update(self.exact_core_idx.get(row['core_name'], []))
            if row.get('sorted_name'):
                candidates.update(self.exact_sorted_idx.get(row['sorted_name'], []))
                
            # Fuzzy
            if has_records:
                candidates.update([self.ref_ids[idx] for idx in char_inds[i]])
                candidates.update([self.ref_ids[idx] for idx in word_inds[i]])
                candidates.update([self.ref_ids[idx] for idx in addr_inds[i]])
            
            results.append({
                'source1_entity_id': row['entity_id'],
                'candidate_entity_ids': ",".join(list(candidates))
            })
            
        return results

def run_blocker(s1_df, s2_df, s3_df, k_fuzzy=20):
    countries = s1_df['country'].dropna().unique()
    all_results = []
    
    for country in countries:
        logging.info(f"--- Blocking for country: {country} ---")
        s1_c = s1_df[s1_df['country'] == country].copy().reset_index(drop=True)
        s2_c = s2_df[s2_df['country'] == country].copy().reset_index(drop=True)
        s3_c = s3_df[s3_df['country'] == country].copy().reset_index(drop=True)
        
        ref_df = pd.concat([s2_c, s3_c], ignore_index=True)
        if len(ref_df) == 0:
            for s1_id in s1_c['entity_id']:
                all_results.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ""})
            continue
            
        blocker = Blocker(k_fuzzy=k_fuzzy)
        blocker.fit(ref_df)
        res = blocker.query(s1_c)
        all_results.extend(res)
        
    return pd.DataFrame(all_results)
