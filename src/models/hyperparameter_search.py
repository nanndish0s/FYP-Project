"""
Post-hoc hyperparameter validation for the Vocalytics RF models.

Runs GridSearchCV on the training set only (no test set leakage) across
n_estimators, max_depth, and min_samples_leaf to confirm that the fixed
configuration (n_estimators=100, max_depth=10) is near-optimal.

Results saved to: results/hyperparameter_search.csv
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import QuantileTransformer

project_root = Path(__file__).parent.parent.parent
data_dir     = project_root / 'data' / 'processed'
results_dir  = project_root / 'results'
results_dir.mkdir(exist_ok=True)

# ── 1. Load data (identical setup to train_big5_proxy_models.py) ───────────
print("Loading data...")
features_df = pd.read_csv(data_dir / 'recruitview_features_all.csv')
metadata_df = pd.read_csv(data_dir / 'recruitview_metadata.csv')

exclude_cols = [
    'video_id', 'transcript', 'file_name', 'question',
    'curiosity_score', 'critical_thinking_score', 'creativity_score',
    'curiosity_reasoning', 'critical_thinking_reasoning', 'creativity_reasoning',
    'openness', 'conscientiousness', 'extraversion',
    'agreeableness', 'neuroticism', 'overall_personality',
    'interview_score', 'overall_performance',
    'word_count', 'sentence_count',
]
feature_cols = [c for c in features_df.columns if c not in exclude_cols]

merged = features_df[['video_id'] + feature_cols].merge(
    metadata_df[['video_id', 'openness', 'conscientiousness', 'extraversion']],
    on='video_id', how='inner',
)
print(f"  Samples  : {len(merged)}")
print(f"  Features : {len(feature_cols)}")

X = merged[feature_cols].fillna(merged[feature_cols].mean()).values

# ── 2. Build targets (identical to train_big5_proxy_models.py) ─────────────
qt = QuantileTransformer(output_distribution='uniform', random_state=42)
openness_q          = qt.fit_transform(merged[['openness']]).flatten() * 4 + 1
conscientiousness_q = qt.fit_transform(merged[['conscientiousness']]).flatten() * 4 + 1
extraversion_q      = qt.fit_transform(merged[['extraversion']]).flatten() * 4 + 1
creativity_q        = (openness_q + extraversion_q) / 2

targets = {
    'curiosity':         openness_q,
    'critical_thinking': conscientiousness_q,
    'creativity':        creativity_q,
}

# ── 3. Parameter grid ──────────────────────────────────────────────────────
# Centred around the deployed config (n_estimators=100, max_depth=10)
# to confirm it is near-optimal within a realistic search space.
param_grid = {
    'n_estimators':     [50, 100, 200, 300],
    'max_depth':        [5, 10, 15, None],
    'min_samples_leaf': [1, 2, 4],
}
DEPLOYED = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 1}

# ── 4. Run grid search per trait (train set only — no test leakage) ────────
all_results = []

for trait, y in targets.items():
    print(f"\n{'='*60}")
    print(f"  Trait: {trait.upper()}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    gs = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        refit=False,
    )
    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    best_score  = gs.best_score_

    # Score of the deployed configuration
    deployed_idx = None
    for i, params in enumerate(gs.cv_results_['params']):
        if params == DEPLOYED:
            deployed_idx = i
            break

    deployed_score = (
        gs.cv_results_['mean_test_score'][deployed_idx]
        if deployed_idx is not None else None
    )

    print(f"\n  Best params  : {best_params}")
    print(f"  Best CV R²   : {best_score:.4f}")
    deployed_score_str = f"{deployed_score:.4f}" if deployed_score is not None else "N/A"
    print(f"  Deployed CV R²  (n=100, depth=10): {deployed_score_str}")
    if deployed_score is not None:
        gap = best_score - deployed_score
        print(f"  Gap (best - deployed)           : {gap:.4f}")
        near_optimal = gap < 0.01
        print(f"  Near-optimal?                   : {'YES' if near_optimal else 'NO'}")

    all_results.append({
        'trait':                   trait,
        'best_n_estimators':       best_params['n_estimators'],
        'best_max_depth':          str(best_params['max_depth']),
        'best_min_samples_leaf':   best_params['min_samples_leaf'],
        'best_cv_r2':              round(best_score, 4),
        'deployed_cv_r2':          round(deployed_score, 4) if deployed_score is not None else None,
        'gap_best_minus_deployed': round(best_score - deployed_score, 4) if deployed_score is not None else None,
        'near_optimal':            bool(best_score - deployed_score < 0.01) if deployed_score is not None else None,
        'deployed_config':         'n_estimators=100, max_depth=10, min_samples_leaf=1',
    })

# ── 5. Save results ────────────────────────────────────────────────────────
out_path = results_dir / 'hyperparameter_search.csv'
pd.DataFrame(all_results).to_csv(out_path, index=False)
print(f"\n\nResults saved to: {out_path}")
print("\nSummary:")
print(pd.DataFrame(all_results).to_string(index=False))
