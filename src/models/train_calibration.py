"""
Data-Driven Calibration Training
Replaces the heuristic calibrate_score() with Isotonic Regression
learned from out-of-fold predictions on the RecruitView dataset.

Ground truth targets (Big 5 proxies, scaled 1-5):
  curiosity         <- openness
  critical_thinking <- conscientiousness
  creativity        <- (openness + extraversion) / 2
                       Grajzel et al. (2023): openness -> originality, extraversion -> fluency
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer

project_root = Path(__file__).parent.parent.parent
models_dir   = project_root / 'models'
data_dir     = project_root / 'data' / 'processed'

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
    on='video_id',
    how='inner',
)
print(f"  Aligned samples : {len(merged)}")
print(f"  Feature columns : {len(feature_cols)}")

X = merged[feature_cols].fillna(merged[feature_cols].mean()).values

# QuantileTransformer: maps each Big 5 score to uniform distribution in [1, 5]
qt = QuantileTransformer(output_distribution='uniform', random_state=42)

openness_q          = qt.fit_transform(merged[['openness']]).flatten() * 4 + 1
conscientiousness_q = qt.fit_transform(merged[['conscientiousness']]).flatten() * 4 + 1
extraversion_q      = qt.fit_transform(merged[['extraversion']]).flatten() * 4 + 1

targets = {
    'curiosity':         openness_q,
    'critical_thinking': conscientiousness_q,
    'creativity':        (openness_q + extraversion_q) / 2,
}

# ── 2. Load trained RF models ────────────────────────────────────────────────
print("\nLoading trained models...")
trained_models = {}
for trait in TRAITS:
    model_file = models_dir / f'{trait}_model_recruitview.pkl'
    with open(model_file, 'rb') as f:
        trained_models[trait] = pickle.load(f)
    print(f"  Loaded: {model_file.name}")

# ── 3. Generate out-of-fold predictions ──────────────────────────────────────
print("\nGenerating out-of-fold predictions (5-fold CV)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = {trait: np.zeros(len(merged)) for trait in TRAITS}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"  Fold {fold + 1}/5 ...")
    X_train, X_val = X[train_idx], X[val_idx]

    for trait in TRAITS:
        y_train = targets[trait][train_idx]
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        oof_preds[trait][val_idx] = rf.predict(X_val)

# ── 4. Fit Isotonic Regression calibrators ───────────────────────────────────
print("\nFitting Isotonic Regression calibrators...")
calibrators = {}

for trait in TRAITS:
    y_true = targets[trait]
    y_raw  = oof_preds[trait]

    iso = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds='clip')
    iso.fit(y_raw, y_true)
    calibrators[trait] = iso

    y_cal = iso.predict(y_raw)
    mae   = np.abs(y_cal - y_true).mean()
    print(f"  {trait:25s}  raw mean: {y_raw.mean():.3f}  "
          f"calibrated mean: {y_cal.mean():.3f}  MAE: {mae:.4f}")

# ── 5. Save calibration models ───────────────────────────────────────────────
print("\nSaving calibration models...")
for trait, iso in calibrators.items():
    out_path = models_dir / f'{trait}_calibration.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(iso, f)
    print(f"  Saved: {out_path.name}")

print("\nCalibration training complete.")
print("Restart the Flask backend to load the new calibrators.")
