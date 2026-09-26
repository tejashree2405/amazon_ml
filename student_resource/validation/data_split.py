import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(root_dir, "dataset", "train")
    splits_dir = os.path.join(root_dir, "dataset", "splits")
    os.makedirs(splits_dir, exist_ok=True)

    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")

    print(f"Loading {s1_path}...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    
    print(f"Loading {gt_path}...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")

    print("Merging to identify singletons and countries...")
    merged = pd.merge(s1_df, gt_df, left_on="entity_id", right_on="source1_entity_id", how="inner")
    
    merged["is_singleton"] = merged["matched_entity_ids"] == ""
    merged["country"] = merged["country"].fillna("UNKNOWN")
    merged["stratify_col"] = merged["country"] + "_" + merged["is_singleton"].astype(str)

    print(f"Total S1 entities: {len(merged)}")
    singleton_count = merged["is_singleton"].sum()
    print(f"Singleton rate: {singleton_count / len(merged):.2%}")

    # Normal 80/20 split
    print("Creating normal 80/20 train/val split...")
    train_df, val_df = train_test_split(
        merged, 
        test_size=0.20, 
        random_state=42, 
        stratify=merged["stratify_col"]
    )

    # For stress test: train on US only, validate on India
    # We will save an "unseen_country" val set which is all of India from the validation set (or the whole dataset).
    # Let's use all of India from the val_df to keep it disjoint from any potential US training set.
    val_unseen_country_df = val_df[val_df["country"].str.upper() == "INDIA"].copy()

    # Save validation splits (we only need the validation harness first)
    val_s1 = val_df[["entity_id", "business_name", "business_address", "country"]]
    val_gt = val_df[["source1_entity_id", "matched_entity_ids"]]

    val_unseen_s1 = val_unseen_country_df[["entity_id", "business_name", "business_address", "country"]]
    val_unseen_gt = val_unseen_country_df[["source1_entity_id", "matched_entity_ids"]]

    val_s1_path = os.path.join(splits_dir, "val_source1.tsv")
    val_gt_path = os.path.join(splits_dir, "val_ground_truth.tsv")
    val_unseen_s1_path = os.path.join(splits_dir, "val_unseen_source1.tsv")
    val_unseen_gt_path = os.path.join(splits_dir, "val_unseen_ground_truth.tsv")

    print("Saving validation sets...")
    val_s1.to_csv(val_s1_path, sep="\t", index=False)
    val_gt.to_csv(val_gt_path, sep="\t", index=False)
    val_unseen_s1.to_csv(val_unseen_s1_path, sep="\t", index=False)
    val_unseen_gt.to_csv(val_unseen_gt_path, sep="\t", index=False)

    print(f"Validation set size: {len(val_s1)} (Singletons: {val_df['is_singleton'].sum()})")
    print(f"Unseen country (India) validation size: {len(val_unseen_s1)}")
    print("Done!")

if __name__ == "__main__":
    main()
