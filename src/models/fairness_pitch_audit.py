"""
Fairness Audit — Pitch-Based Proxy Grouping
============================================
RecruitView provides no demographic metadata (e.g. gender), so a direct
Demographic Parity audit is not possible. As a proxy, candidates are split
into pitch quartiles (pitch_mean is a well-documented acoustic correlate of
speaker gender — e.g. Titze, 1989), and model error (MAE) and mean predicted
score are compared across quartiles for each C3 trait.

A Kruskal-Wallis test checks whether differences across pitch quartiles are
statistically significant (p < 0.05).

Out-of-fold (5-fold CV) predictions are used so every one of the 2,011
candidates contributes to the audit, then calibrated with the production
isotonic regressors.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer

project_root = Path(__file__).parent.parent.parent
models_dir   = project_root / 'models'
data_dir     = project_root / 'data' / 'processed'
results_dir  = project_root / 'results'
results_dir.mkdir(exist_ok=True)

RF_PARAMS = dict(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
TRAITS = ['curiosity', 'critical_thinking', 'creativity']

# ── 1. Load data ──────────────────────────────────────────────────────────────
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

# ── 2. Build targets (same as production training) ────────────────────────────
qt = QuantileTransformer(output_distribution='uniform', random_state=42)
openness_q          = qt.fit_transform(merged[['openness']]).flatten() * 4 + 1
conscientiousness_q = qt.fit_transform(merged[['conscientiousness']]).flatten() * 4 + 1
extraversion_q      = qt.fit_transform(merged[['extraversion']]).flatten() * 4 + 1

targets = {
    'curiosity':         openness_q,
    'critical_thinking': conscientiousness_q,
    'creativity':        (openness_q + extraversion_q) / 2,
}

# ── 3. Pitch quartile groups ───────────────────────────────────────────────────
pitch = merged['pitch_mean'].values
quartile_labels = pd.qcut(pitch, q=4, labels=['Q1 (lowest pitch)', 'Q2', 'Q3', 'Q4 (highest pitch)'])
merged['pitch_quartile'] = quartile_labels

# ── 4. OOF predictions + calibration, per trait ────────────────────────────────
print("\nGenerating out-of-fold predictions (5-fold CV)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

summary_rows = []
kruskal_rows = []

for trait in TRAITS:
    y = targets[trait]
    oof_raw = np.zeros(len(merged))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X[train_idx], y[train_idx])
        oof_raw[val_idx] = rf.predict(X[val_idx])

    with open(models_dir / f'{trait}_calibration.pkl', 'rb') as f:
        iso = pickle.load(f)
    oof_cal = np.clip(iso.predict(oof_raw), 1.0, 5.0)

    abs_err = np.abs(oof_cal - y)

    print(f"\n{trait.upper()}")
    mae_by_group = []
    pred_by_group = []
    for q in ['Q1 (lowest pitch)', 'Q2', 'Q3', 'Q4 (highest pitch)']:
        mask = merged['pitch_quartile'] == q
        mae_q = abs_err[mask.values].mean()
        pred_q = oof_cal[mask.values].mean()
        n_q = mask.sum()
        mae_by_group.append(abs_err[mask.values])
        pred_by_group.append(oof_cal[mask.values])
        print(f"  {q:20s}  n={n_q:4d}  MAE={mae_q:.4f}  mean_pred={pred_q:.3f}")
        summary_rows.append({
            'trait': trait, 'pitch_quartile': q, 'n': n_q,
            'mae': round(mae_q, 4), 'mean_predicted_score': round(pred_q, 3),
        })

    h_mae, p_mae = kruskal(*mae_by_group)
    h_pred, p_pred = kruskal(*pred_by_group)
    print(f"  Kruskal-Wallis (MAE across quartiles)        : H={h_mae:.3f}  p={p_mae:.4f}")
    print(f"  Kruskal-Wallis (mean score across quartiles) : H={h_pred:.3f}  p={p_pred:.4f}")
    kruskal_rows.append({
        'trait': trait, 'metric': 'MAE', 'H_statistic': round(h_mae, 4), 'p_value': round(p_mae, 4),
        'significant_at_0.05': bool(p_mae < 0.05),
    })
    kruskal_rows.append({
        'trait': trait, 'metric': 'mean_predicted_score', 'H_statistic': round(h_pred, 4), 'p_value': round(p_pred, 4),
        'significant_at_0.05': bool(p_pred < 0.05),
    })

# ── 5. Save results ───────────────────────────────────────────────────────────
df_summary = pd.DataFrame(summary_rows)
df_kruskal = pd.DataFrame(kruskal_rows)

df_summary.to_csv(results_dir / 'fairness_pitch_audit_summary.csv', index=False)
df_kruskal.to_csv(results_dir / 'fairness_pitch_audit_kruskal.csv', index=False)

print(f"\nSaved: {results_dir / 'fairness_pitch_audit_summary.csv'}")
print(f"Saved: {results_dir / 'fairness_pitch_audit_kruskal.csv'}")
