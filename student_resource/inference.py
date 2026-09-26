import os
import gc
import logging
import joblib
import pandas as pd
from catboost import CatBoostClassifier

from normalization import preprocess_dataframe
from blocking import run_blocker
from train_pipeline import expand_candidates, build_features
from train_catboost import predict_pairs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(root_dir, "dataset", "test")
    models_dir = os.path.join(root_dir, "models")
    out_dir = os.path.join(root_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load data
    logging.info("Loading test datasets...")
    test_s1 = preprocess_dataframe(pd.read_csv(os.path.join(dataset_dir, "test_source1.tsv"), sep="\t", dtype=str))
    test_s2 = preprocess_dataframe(pd.read_csv(os.path.join(dataset_dir, "test_source2.tsv"), sep="\t", dtype=str))
    test_s3 = preprocess_dataframe(pd.read_csv(os.path.join(dataset_dir, "test_source3.tsv"), sep="\t", dtype=str))
    ref_df = pd.concat([test_s2, test_s3], ignore_index=True)
    
    # 2. Block
    logging.info("Running blocker for test set...")
    test_cands_raw = pd.DataFrame(run_blocker(test_s1, test_s2, test_s3, k_fuzzy=20))
    test_cands_raw.to_csv(os.path.join(out_dir, "candidate_pairs.tsv"), sep="\t", index=False)
    
    test_pairs = expand_candidates(test_cands_raw)
    
    if len(test_pairs) == 0:
        logging.info("No candidates generated!")
        with open(os.path.join(out_dir, "matching_results.tsv"), "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for _, row in test_s1.iterrows():
                f.write(f"{row['entity_id']}\t\n")
        return
        
    # 3. Features
    test_df = build_features(test_pairs, test_s1, ref_df)
    
    # 4. Load models
    logging.info("Loading models...")
    model = CatBoostClassifier()
    model.load_model(os.path.join(models_dir, "catboost_model.cbm"))
    calibrator = joblib.load(os.path.join(models_dir, "calibrator.pkl"))
    with open(os.path.join(models_dir, "threshold.txt"), "r") as f:
        threshold = float(f.read().strip())
        
    features = [
        'feat_name_ratio', 'feat_core_ratio', 'feat_name_sort_ratio',
        'feat_name_jaccard', 'feat_name_char3_jaccard', 'feat_name_brand_anchor',
        'feat_addr_ratio', 'feat_addr_jaccard', 'feat_addr_char4_jaccard',
        'feat_addr_numbers_jaccard', 'feat_country_agree', 'feat_is_s2', 'feat_is_s3'
    ]
    
    # 5. Predict
    logging.info("Predicting...")
    raw_probs = predict_pairs(model, test_df, features)
    cal_probs = calibrator.predict_proba(raw_probs)
    test_df['probability'] = cal_probs
    
    # 6. Apply Threshold
    logging.info(f"Applying threshold {threshold:.2f}...")
    filtered = test_df[test_df['probability'] >= threshold]
    
    pred_dict = {}
    for s1, group in filtered.groupby('source1_entity_id'):
        pred_dict[s1] = set(group['candidate_entity_id'].tolist())
        
    # Write output
    logging.info("Writing output...")
    with open(os.path.join(out_dir, "matching_results.tsv"), "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for _, row in test_s1.iterrows():
            s1 = row['entity_id']
            matches = pred_dict.get(s1, set())
            f.write(f"{s1}\t{','.join(list(matches))}\n")
            
    logging.info("Inference complete.")

if __name__ == "__main__":
    main()
