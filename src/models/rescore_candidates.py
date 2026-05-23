"""
Rescore all 2,011 RecruitView candidates using the trained Big 5-proxy
Random Forest models + Isotonic Regression calibration.

Replaces the LLM-generated curiosity_score, critical_thinking_score,
creativity_score columns in recruitview_metadata.csv with model predictions
grounded in human-assessed Big 5 personality scores.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
models_dir   = project_root / 'models'
data_dir     = project_root / 'data' / 'processed'

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
    metadata_df[['video_id']],
    on='video_id',
    how='inner',
)
X = merged[feature_cols].fillna(
    features_df[feature_cols].mean()
).values

print(f"  Candidates : {len(merged)}")
print(f"  Features   : {len(feature_cols)}")

# ── 2. Load models and calibrators ───────────────────────────────────────────
print("\nLoading models and calibrators...")
models      = {}
calibrators = {}

for trait in ['curiosity', 'critical_thinking', 'creativity']:
    with open(models_dir / f'{trait}_model_recruitview.pkl', 'rb') as f:
        models[trait] = pickle.load(f)
    with open(models_dir / f'{trait}_calibration.pkl', 'rb') as f:
        calibrators[trait] = pickle.load(f)
    print(f"  Loaded: {trait}")

# ── 3. Predict scores for all 2,011 candidates ───────────────────────────────
print("\nScoring all candidates...")
scores = {}

for trait in ['curiosity', 'critical_thinking', 'creativity']:
    raw        = models[trait].predict(X)
    calibrated = calibrators[trait].predict(raw)
    calibrated = np.clip(calibrated, 1.0, 5.0)
    scores[trait] = calibrated
    print(f"  {trait:25s}  mean: {calibrated.mean():.3f}  "
          f"min: {calibrated.min():.3f}  max: {calibrated.max():.3f}")

# ── 4. Update metadata CSV ────────────────────────────────────────────────────
print("\nUpdating recruitview_metadata.csv...")

score_df = pd.DataFrame({
    'video_id':                merged['video_id'].values,
    'curiosity_score':         np.round(scores['curiosity'], 3),
    'critical_thinking_score': np.round(scores['critical_thinking'], 3),
    'creativity_score':        np.round(scores['creativity'], 3),
})

# Drop old LLM scores and merge in new RF scores
metadata_updated = metadata_df.drop(
    columns=['curiosity_score', 'critical_thinking_score', 'creativity_score'],
    errors='ignore'
).merge(score_df, on='video_id', how='left')

metadata_updated.to_csv(data_dir / 'recruitview_metadata.csv', index=False)
print(f"  Saved {len(metadata_updated)} rows to recruitview_metadata.csv")
print("\nDone. Restart the Flask backend to serve updated candidate scores.")
