import sys
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from normalization import preprocess_dataframe
from blocking import run_blocker

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    splits_dir = os.path.join(root_dir, "dataset", "splits")
    dataset_dir = os.path.join(root_dir, "dataset", "train")
    
    val_s1_path = os.path.join(splits_dir, "val_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")
    out_path = os.path.join(splits_dir, "val_candidate_pairs.tsv")
    
    logging.info("Loading validation queries (S1)...")
    val_s1 = pd.read_csv(val_s1_path, sep="\t", dtype=str)
    
    # We load S2 and S3 in chunks if memory becomes an issue, but for now we try to load them in full.
    logging.info("Loading reference datasets (S2, S3)...")
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str)
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str)
    
    logging.info("Preprocessing validation queries...")
    val_s1 = preprocess_dataframe(val_s1)
    
    logging.info("Preprocessing reference datasets...")
    s2_df = preprocess_dataframe(s2_df)
    s3_df = preprocess_dataframe(s3_df)
    
    logging.info("Running blocker...")
    results_df = run_blocker(val_s1, s2_df, s3_df, k_fuzzy=20)
    
    logging.info(f"Saving candidates to {out_path}...")
    results_df.to_csv(out_path, sep="\t", index=False)
    logging.info("Done!")

if __name__ == "__main__":
    main()
