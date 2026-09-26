def f05_score(true_matches: set, pred_matches: set) -> float:
    """
    Computes the F0.5 score for a single S1 entity.
    Precision is weighted twice as much as recall.
    """
    is_true_singleton = len(true_matches) == 0
    is_pred_singleton = len(pred_matches) == 0
    
    if is_true_singleton:
        # For singletons, empty prediction is perfect, any prediction is 0
        return 1.0 if is_pred_singleton else 0.0
    
    if is_pred_singleton:
        # True matches exist but none were predicted -> Recall is 0
        return 0.0

    tp = len(true_matches & pred_matches)
    fp = len(pred_matches - true_matches)
    fn = len(true_matches - pred_matches)
    
    if tp == 0:
        return 0.0
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def macro_f05(y_true_dict: dict, y_pred_dict: dict) -> dict:
    """
    Computes the macro-averaged F0.5 score over all S1 entities in y_true_dict.
    Returns overall score, singleton-only score, and non-singleton-only score.
    
    Args:
        y_true_dict: dict mapping s1_id -> set of true matched IDs
        y_pred_dict: dict mapping s1_id -> set of predicted matched IDs
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []
    
    for s1_id, true_matches in y_true_dict.items():
        pred_matches = y_pred_dict.get(s1_id, set())
        score = f05_score(true_matches, pred_matches)
        
        scores.append(score)
        if len(true_matches) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)
            
    overall = sum(scores) / len(scores) if scores else 0.0
    singleton_overall = sum(singleton_scores) / len(singleton_scores) if singleton_scores else 0.0
    non_singleton_overall = sum(non_singleton_scores) / len(non_singleton_scores) if non_singleton_scores else 0.0
    
    return {
        "overall": overall,
        "singleton_only": singleton_overall,
        "non_singleton_only": non_singleton_overall,
        "total_entities": len(scores),
        "total_singletons": len(singleton_scores),
        "total_non_singletons": len(non_singleton_scores)
    }

# Quick test
if __name__ == "__main__":
    true_dict = {
        "S1-1": {"S2-1", "S3-1"},
        "S1-2": set(), # singleton
        "S1-3": {"S2-2"}
    }
    pred_dict = {
        "S1-1": {"S2-1"}, # partial match
        "S1-2": set(), # correct singleton
        "S1-3": {"S2-2", "S2-3"} # false positive
    }
    
    res = macro_f05(true_dict, pred_dict)
    print(res)
