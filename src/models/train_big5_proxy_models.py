"""
Train C3 Models Using Big 5 Personality Scores as Validated Proxies

Ground truth mapping (justified by psychological literature):
  Curiosity         <- openness
                       (Silvia & Christensen, 2020; r = .50)
  Critical Thinking <- conscientiousness
                       (Hu et al., 2023; r = .191)
  Creativity        <- (openness + extraversion) / 2
                       Openness predicts originality (Grajzel et al., 2023; r = .20)
                       Extraversion predicts fluency (Nusbaum & Silvia, 2020; r = .18)

Normalisation: QuantileTransformer (uniform output distribution 1-5)
  Maps each Big 5 score to its percentile rank in the training population.
  This spreads targets uniformly across 1-5, preventing the clustering
  around the mean that occurs with linear (MinMax) scaling.
  A score of 4.0 means the candidate ranked in the top 25% of the
  RecruitView reference population on that trait.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import QuantileTransformer

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
    # Excluded: quantity measures — response length should not predict personality
    # word_count is retained only as a validity pre-check in the API layer
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

# ── 2. Build targets using QuantileTransformer ────────────────────────────────
# Maps each Big 5 dimension to a uniform distribution in [1, 5]
# so scores spread across the full range instead of clustering at the mean
print("\nBuilding targets with QuantileTransformer...")

qt = QuantileTransformer(output_distribution='uniform', random_state=42)

# Fit on the combined pool of openness + extraversion values
# so both dimensions share the same percentile scale
openness_q         = qt.fit_transform(merged[['openness']]).flatten() * 4 + 1
conscientiousness_q = qt.fit_transform(merged[['conscientiousness']]).flatten() * 4 + 1
extraversion_q     = qt.fit_transform(merged[['extraversion']]).flatten() * 4 + 1

# Creativity = average of openness (originality) and extraversion (fluency)
creativity_q = (openness_q + extraversion_q) / 2

targets = {
    'curiosity':         openness_q,
    'critical_thinking': conscientiousness_q,
    'creativity':        creativity_q,
}

for trait, y in targets.items():
    print(f"  {trait:25s}  min={y.min():.2f}  max={y.max():.2f}  mean={y.mean():.2f}  std={y.std():.2f}")

# ── 3. Train one RF model per C3 trait ───────────────────────────────────────
print("\nTraining models...")
results = []

for trait, y in targets.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    cv  = cross_val_score(rf, X, y, cv=5, scoring='r2').mean()

    print(f"\n  {trait.upper()}")
    print(f"    R²   : {r2:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    CV R²: {cv:.4f}")

    out_path = models_dir / f'{trait}_model_recruitview.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"    Saved: {out_path.name}")

    results.append({
        'trait':       trait,
        'big5_proxy':  'openness' if trait == 'curiosity' else
                       'conscientiousness' if trait == 'critical_thinking' else
                       'openness + extraversion (avg)',
        'normaliser':  'QuantileTransformer (uniform 1-5)',
        'r2':   r2,
        'mae':  mae,
        'cv_r2': cv,
        'samples': len(merged),
    })

# ── 4. Save performance report ───────────────────────────────────────────────
df_results = pd.DataFrame(results)
df_results.to_csv(project_root / 'results' / 'big5_proxy_model_performance.csv', index=False)
print("\nPerformance report saved.")
print("Now run train_calibration.py to recalibrate.")
