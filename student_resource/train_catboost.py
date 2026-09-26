import pandas as pd
import numpy as np
import json
import logging

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_pairwise_classifier(train_df, val_df, features):
    if not HAS_CATBOOST:
        raise ImportError("catboost is not installed. Run 'pip install catboost'.")
        
    X_train, y_train = train_df[features], train_df['label']
    X_val, y_val = val_df[features], val_df['label']
    
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function='Logloss',
        eval_metric='Logloss',
        early_stopping_rounds=150,
        random_seed=42,
        task_type='CPU', # Or GPU if available
        verbose=100
    )
    
    logging.info("Training CatBoost pairwise classifier...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    logging.info(f"Best iteration: {model.get_best_iteration()}")
    logging.info(f"Best validation logloss: {model.get_best_score()['validation']['Logloss']:.4f}")
    
    return model

def predict_pairs(model, df, features):
    """
    Returns probability of class 1.
    """
    pool = Pool(df[features])
    return model.predict_proba(pool)[:, 1]

def get_feature_importance(model, features):
    importance = model.get_feature_importance()
    return pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
