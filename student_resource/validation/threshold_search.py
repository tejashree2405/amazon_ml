import pandas as pd
import numpy as np
import argparse
import json
from .metrics import macro_f05

def load_ground_truth(path):
    gt_df = pd.read_csv(path, sep="\t", dtype=str)
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna("")
    y_true_dict = {}
    for _, row in gt_df.iterrows():
        s1 = row['source1_entity_id']
        matches = row['matched_entity_ids'].split(",") if row['matched_entity_ids'] else []
        y_true_dict[s1] = set([m.strip() for m in matches if m.strip()])
    return y_true_dict

def load_candidates(path):
    cand_df = pd.read_csv(path, sep="\t", dtype=str)
    cand_df['candidate_entity_ids'] = cand_df['candidate_entity_ids'].fillna("")
    cand_dict = {}
    for _, row in cand_df.iterrows():
        s1 = row['source1_entity_id']
        cands = row['candidate_entity_ids'].split(",") if row['candidate_entity_ids'] else []
        cand_dict[s1] = set([c.strip() for c in cands if c.strip()])
    return cand_dict

def generate_predictions(preds_df, threshold):
    # preds_df: source1_entity_id, candidate_entity_id, probability
    filtered = preds_df[preds_df['probability'] >= threshold]
    pred_dict = {}
    for s1, group in filtered.groupby('source1_entity_id'):
        pred_dict[s1] = set(group['candidate_entity_id'].tolist())
    return pred_dict

def calculate_candidate_recall(y_true_dict, cand_dict):
    total_true_matches = 0
    found_matches = 0
    for s1, trues in y_true_dict.items():
        if not trues:
            continue
        cands = cand_dict.get(s1, set())
        total_true_matches += len(trues)
        found_matches += len(trues & cands)
    return found_matches / total_true_matches if total_true_matches > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", help="Path to predictions TSV (columns: source1_entity_id, candidate_entity_id, probability)", required=True)
    parser.add_argument("--gt", help="Path to validation ground truth TSV", required=True)
    parser.add_argument("--cands", help="Path to validation candidates TSV (optional, for recall calc)")
    args = parser.parse_args()

    print("Loading Ground Truth...")
    y_true = load_ground_truth(args.gt)

    if args.cands:
        print("Loading Candidates to check recall...")
        cands = load_candidates(args.cands)
        recall = calculate_candidate_recall(y_true, cands)
        print(f"Candidate Recall: {recall:.4%}")

    print("Loading Predictions...")
    preds_df = pd.read_csv(args.preds, sep="\t")
    # ensure missing s1 entities are in pred_dict as empty sets
    all_s1 = set(y_true.keys())

    print("Running threshold search...")
    thresholds = np.arange(0.30, 0.99, 0.01)
    results = []

    for t in thresholds:
        pred_dict = generate_predictions(preds_df, t)
        # Add empty sets for S1s with no predictions above threshold
        for s1 in all_s1:
            if s1 not in pred_dict:
                pred_dict[s1] = set()
                
        metrics = macro_f05(y_true, pred_dict)
        results.append((t, metrics['overall'], metrics['singleton_only'], metrics['non_singleton_only']))

    # Find optimal threshold
    results.sort(key=lambda x: x[1], reverse=True)
    opt_t, opt_overall, opt_sing, opt_non_sing = results[0]
    
    print(f"\n--- Best Threshold: {opt_t:.2f} ---")
    print(f"Overall F0.5:        {opt_overall:.4f}")
    print(f"Singleton F0.5:      {opt_sing:.4f}")
    print(f"Non-Singleton F0.5:  {opt_non_sing:.4f}")

    # Plateau check
    print("\n--- Plateau Check ---")
    plateau_valid = True
    for dt in [-0.05, -0.02, 0.02, 0.05]:
        check_t = round(opt_t + dt, 2)
        if 0.30 <= check_t <= 0.98:
            # find the result for check_t
            res = next((r for r in results if abs(r[0] - check_t) < 0.001), None)
            if res:
                diff = opt_overall - res[1]
                print(f"T={check_t:.2f}: F0.5={res[1]:.4f} (Delta: -{diff:.4f})")
                if diff > 0.005:
                    plateau_valid = False

    if not plateau_valid:
        print("\nWARNING: Threshold plateau is narrow! The F0.5 drop is > 0.005 within +/- 0.05 of the optimal threshold.")
        print("This indicates the validation harness or model might not be robust.")
    else:
        print("\nSUCCESS: Threshold plateau is wide and stable.")

if __name__ == "__main__":
    main()
