import os
import gc
import pandas as pd
import logging
import numpy as np

# Configurable paths (defaults for Colab)
DATASET_ROOT = os.getenv('DATASET_ROOT', r'C:/Users/tejas/OneDrive/Desktop/mlamazon/student_resource/dataset')
OUTPUT_ROOT = os.getenv('OUTPUT_ROOT', r'C:/Users/tejas/OneDrive/Desktop/mlamazon/student_resource/output')

from normalization import preprocess_dataframe
from blocking import run_blocker
from pair_features import compute_pair_features
from hard_negative_mining import label_candidates, mine_hard_negatives
from train_catboost import train_pairwise_classifier, predict_pairs, get_feature_importance
from calibration import ProbabilityCalibrator
from validation.threshold_search import load_ground_truth, generate_predictions
from validation.metrics import macro_f05
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def expand_candidates(cand_df):
    """Expands comma-separated candidates into a long format dataframe"""
    records = []
    for _, row in cand_df.iterrows():
        s1 = row['source1_entity_id']
        cands = row['candidate_entity_ids']
        if pd.isna(cands) or not cands:
            continue
        for c in cands.split(','):
            c = c.strip()
            if c:
                records.append({'source1_entity_id': s1, 'candidate_entity_id': c})
    return pd.DataFrame(records)

def build_features(pairs_long, s1_df, ref_df):
    logging.info("Joining pairs with metadata...")
    # Add S1 data
    s1_cols = ['entity_id', 'norm_name', 'core_name', 'sorted_name', 'norm_address', 'addr_numbers', 'country']
    df = pd.merge(pairs_long, s1_df[s1_cols], left_on='source1_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={c: f"{c}_s1" for c in s1_cols if c != 'entity_id'})
    df.drop(columns=['entity_id'], inplace=True)
    
    # Add S2/S3 data
    df = pd.merge(df, ref_df[s1_cols], left_on='candidate_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={c: f"{c}_cand" for c in s1_cols if c != 'entity_id'})
    df.drop(columns=['entity_id'], inplace=True)
    
    logging.info("Computing pair features...")
    df = compute_pair_features(df)
    return df

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_root = DATASET_ROOT

    splits_dir = os.path.join(dataset_root, "splits")
    dataset_dir = os.path.join(dataset_root, "train")
    
    features = [
        'feat_name_ratio', 'feat_core_ratio', 'feat_name_sort_ratio',
        'feat_name_jaccard', 'feat_name_char3_jaccard', 'feat_name_brand_anchor',
        'feat_addr_ratio', 'feat_addr_jaccard', 'feat_addr_char4_jaccard',
        'feat_addr_numbers_jaccard', 'feat_country_agree', 'feat_is_s2', 'feat_is_s3'
    ]
    
    # 1. Load data
    logging.info("Loading baseline datasets...")
    s2_df = preprocess_dataframe(pd.read_csv(os.path.join(dataset_dir, "train_source2.tsv"), sep="\t", dtype=str))
    s3_df = preprocess_dataframe(pd.read_csv(os.path.join(dataset_dir, "train_source3.tsv"), sep="\t", dtype=str))
    ref_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # Determine train split vs validation split
    # Wait, for a true train pipeline, we need train candidates. 
    # For now, we will use a small sample of train_source1 to simulate training pipeline
    train_s1 = preprocess_dataframe(pd.read_csv(os.path.join(dataset_dir, "train_source1.tsv"), sep="\t", dtype=str, nrows=50000))
    val_s1 = preprocess_dataframe(pd.read_csv(os.path.join(splits_dir, "val_source1.tsv"), sep="\t", dtype=str, nrows=10000))
    
    gt_dict = load_ground_truth(os.path.join(dataset_dir, "train_ground_truth.tsv"))
    
    # 2. Block
    logging.info("Running blocker for training set...")
    train_cands_raw = pd.DataFrame(run_blocker(train_s1, s2_df, s3_df, k_fuzzy=10))
    train_pairs = expand_candidates(train_cands_raw)
    
    logging.info("Running blocker for validation set...")
    val_cands_raw = pd.DataFrame(run_blocker(val_s1, s2_df, s3_df, k_fuzzy=10))
    val_pairs = expand_candidates(val_cands_raw)
    
    # 3. Features
    train_df = build_features(train_pairs, train_s1, ref_df)
    val_df = build_features(val_pairs, val_s1, ref_df)
    
    # 4. Label & Mine Hard Negatives
    logging.info("Labeling candidates and mining hard negatives...")
    train_df = label_candidates(train_df, gt_dict)
    # We pass n_negatives_per_source scaled down for this sample
    train_sampled = mine_hard_negatives(train_df, n_negatives_per_source=20000)
    
    # Validate set just needs labeling
    val_df = label_candidates(val_df, gt_dict)
    
    # 5. Train Model
    logging.info("Training CatBoost model...")
    model = train_pairwise_classifier(train_sampled, val_df, features)
    
    # Log Importance
    imp = get_feature_importance(model, features)
    logging.info(f"Feature Importances:\n{imp}")
    
    # 6. Predict and Calibrate
    logging.info("Predicting and calibrating probabilities...")
    # Get raw probabilities on the unbalanced validation set
    raw_val_probs = predict_pairs(model, val_df, features)
    
    calibrator = ProbabilityCalibrator(method='isotonic')
    calibrator.fit(raw_val_probs, val_df['label'])
    
    cal_val_probs = calibrator.predict_proba(raw_val_probs)
    val_df['probability'] = cal_val_probs
    
    # 7. Threshold Search Integration
    logging.info("Running Threshold Search...")
    # Prepare preds_df for threshold_search
    preds_df = val_df[['source1_entity_id', 'candidate_entity_id', 'probability']]
    
    thresholds = np.arange(0.30, 0.99, 0.02)
    best_t, best_f05 = 0, 0
    
    for t in thresholds:
        pred_dict = generate_predictions(preds_df, t)
        for s1 in val_s1['entity_id']:
            if s1 not in pred_dict: pred_dict[s1] = set()
            
        metrics = macro_f05(gt_dict, pred_dict)
        if metrics['overall'] > best_f05:
            best_f05 = metrics['overall']
            best_t = t
            
    logging.info(f"Pipeline Complete! Best Calibrated Threshold: {best_t:.2f} -> Validation F0.5: {best_f05:.4f}")
    # ---------------------------------------------------------
    # Validation output generation
    # ---------------------------------------------------------
    val_out_dir = os.path.join(OUTPUT_ROOT, "validation")
    os.makedirs(val_out_dir, exist_ok=True)
    # Save candidate pairs (raw blocker output)
    val_cands_path = os.path.join(val_out_dir, "candidate_pairs.tsv")
    val_cands_raw.to_csv(val_cands_path, sep="\t", index=False)
    # Generate matching results
    pred_dict_val = generate_predictions(val_df[['source1_entity_id','candidate_entity_id','probability']], best_t)
    for s1 in val_s1['entity_id']:
        pred_dict_val.setdefault(s1, set())
    match_path = os.path.join(val_out_dir, "matching_results.tsv")
    rows = []
    for s1 in val_s1['entity_id']:
        matches = sorted(pred_dict_val.get(s1, []))
        rows.append({"source1_entity_id": s1, "matched_entity_ids": ",".join(matches) if matches else ""})
    pd.DataFrame(rows).to_csv(match_path, sep="\t", index=False)
    # Validation metrics file
    metrics_path = os.path.join(val_out_dir, "validation_score.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Best threshold: {best_t:.2f}\n")
        f.write(f"Validation F0.5: {best_f05:.4f}\n")
        f.write(f"Number of validation Source 1 entities: {len(val_s1)}\n")
        f.write(f"Number of candidate pairs: {len(val_cands_raw)}\n")
        f.write(f"Number of predicted matches: {sum(len(v) for v in pred_dict_val.values())}\n")
    
    # 8. Save Artifacts
    artifacts_dir = os.path.join(root_dir, "models")
    os.makedirs(artifacts_dir, exist_ok=True)
    logging.info("Saving model and calibrator...")
    model.save_model(os.path.join(artifacts_dir, "catboost_model.cbm"))
    joblib.dump(calibrator, os.path.join(artifacts_dir, "calibrator.pkl"))
    with open(os.path.join(artifacts_dir, "threshold.txt"), "w") as f:
        f.write(str(best_t))
    logging.info("Training pipeline completely finished and artifacts saved.")

if __name__ == "__main__":
    main()
