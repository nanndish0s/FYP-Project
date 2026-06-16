"""
Ablation and Baseline Comparison Study for Chapter 6
=====================================================
Runs four ablation variants and two baselines, reporting MAE, RMSE, R², and
Pearson r for all three C3 traits across every condition.

Ablation variants
  1. Full model      : RF (35 features) + Isotonic Regression calibration
  2. No calibration  : RF (35 features), raw predictions only
  3. Acoustic only   : RF (32 acoustic features) + calibration
  4. Lexical only    : RF (3 lexical features) + calibration

Baselines
  5. Mean predictor  : always predicts the training-set mean (naive baseline)
  6. Linear Regression: OLS on same 35 features + calibration (simpler model)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer

project_root = Path(__file__).parent.parent.parent
data_dir     = project_root / 'data' / 'processed'

# ── Feature column lists ──────────────────────────────────────────────────────
ACOUSTIC_COLS = [
    'pitch_mean', 'pitch_std', 'energy_mean', 'energy_std',
    *[f'mfcc_{i}_mean' for i in range(13)],
    *[f'mfcc_{i}_std'  for i in range(13)],
    'spectral_centroid_mean', 'zcr_mean',
]
LEXICAL_COLS = ['avg_word_length', 'vocab_diversity', 'filler_word_ratio']
ALL_FEATURE_COLS = ACOUSTIC_COLS + LEXICAL_COLS

RF_PARAMS = dict(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
TRAITS = ['curiosity', 'critical_thinking', 'creativity']

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
features_df = pd.read_csv(data_dir / 'recruitview_features_all.csv')
metadata_df = pd.read_csv(data_dir / 'recruitview_metadata.csv')

merged = features_df[['video_id'] + ALL_FEATURE_COLS].merge(
    metadata_df[['video_id', 'openness', 'conscientiousness', 'extraversion']],
    on='video_id', how='inner',
)
print(f"  Samples  : {len(merged)}")
print(f"  Features : {len(ALL_FEATURE_COLS)} ({len(ACOUSTIC_COLS)} acoustic + {len(LEXICAL_COLS)} lexical)")

X_all = merged[ALL_FEATURE_COLS].fillna(merged[ALL_FEATURE_COLS].mean()).values
X_ac  = merged[ACOUSTIC_COLS].fillna(merged[ACOUSTIC_COLS].mean()).values
X_lex = merged[LEXICAL_COLS].fillna(merged[LEXICAL_COLS].mean()).values

# ── 2. Build targets ──────────────────────────────────────────────────────────
qt = QuantileTransformer(output_distribution='uniform', random_state=42)
openness_q          = qt.fit_transform(merged[['openness']]).flatten() * 4 + 1
conscientiousness_q = qt.fit_transform(merged[['conscientiousness']]).flatten() * 4 + 1
extraversion_q      = qt.fit_transform(merged[['extraversion']]).flatten() * 4 + 1

targets = {
    'curiosity':         openness_q,
    'critical_thinking': conscientiousness_q,
    'creativity':        (openness_q + extraversion_q) / 2,
}

# Fixed 80/20 split (same seed as production training)
idx_train, idx_test = train_test_split(
    np.arange(len(merged)), test_size=0.2, random_state=42
)

def metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    return {'variant': label, 'mae': mae, 'rmse': rmse, 'r2': r2, 'pearson_r': r}

def fit_calibrator(X_feat, y_true_full):
    """5-fold OOF calibration on training split."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_tr = X_feat[idx_train]
    y_tr = y_true_full[idx_train]
    oof  = np.zeros(len(idx_train))
    for tr_idx, val_idx in kf.split(X_tr):
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof[val_idx] = rf.predict(X_tr[val_idx])
    iso = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds='clip')
    iso.fit(oof, y_tr)
    return iso

# ── 3. Run all variants ───────────────────────────────────────────────────────
all_results = []

for trait in TRAITS:
    y = targets[trait]
    y_train, y_test = y[idx_train], y[idx_test]

    print(f"\n{'='*60}")
    print(f"  TRAIT: {trait.upper()}")
    print(f"{'='*60}")

    # ── Variant 1: Full model (RF all features + calibration) ─────────────
    rf_full = RandomForestRegressor(**RF_PARAMS)
    rf_full.fit(X_all[idx_train], y_train)
    raw_full = rf_full.predict(X_all[idx_test])
    iso_full = fit_calibrator(X_all, y)
    cal_full = iso_full.predict(raw_full)
    r1 = metrics(y_test, cal_full, 'Full model (RF + calibration)')
    all_results.append({**r1, 'trait': trait})
    print(f"  [Full]          MAE={r1['mae']:.4f}  RMSE={r1['rmse']:.4f}  R²={r1['r2']:.4f}  r={r1['pearson_r']:.4f}")

    # ── Variant 2: No calibration (RF all features, raw output) ──────────
    raw_nocal = rf_full.predict(X_all[idx_test])  # same RF, no calibration
    r2v = metrics(y_test, raw_nocal, 'No calibration (RF raw)')
    all_results.append({**r2v, 'trait': trait})
    print(f"  [No calibration] MAE={r2v['mae']:.4f}  RMSE={r2v['rmse']:.4f}  R²={r2v['r2']:.4f}  r={r2v['pearson_r']:.4f}")

    # ── Variant 3: Acoustic features only ────────────────────────────────
    rf_ac = RandomForestRegressor(**RF_PARAMS)
    rf_ac.fit(X_ac[idx_train], y_train)
    raw_ac = rf_ac.predict(X_ac[idx_test])
    iso_ac = fit_calibrator(X_ac, y)
    cal_ac = iso_ac.predict(raw_ac)
    r3 = metrics(y_test, cal_ac, 'Acoustic only')
    all_results.append({**r3, 'trait': trait})
    print(f"  [Acoustic only]  MAE={r3['mae']:.4f}  RMSE={r3['rmse']:.4f}  R²={r3['r2']:.4f}  r={r3['pearson_r']:.4f}")

    # ── Variant 4: Lexical features only ─────────────────────────────────
    rf_lex = RandomForestRegressor(**RF_PARAMS)
    rf_lex.fit(X_lex[idx_train], y_train)
    raw_lex = rf_lex.predict(X_lex[idx_test])
    iso_lex = fit_calibrator(X_lex, y)
    cal_lex = iso_lex.predict(raw_lex)
    r4 = metrics(y_test, cal_lex, 'Lexical only')
    all_results.append({**r4, 'trait': trait})
    print(f"  [Lexical only]   MAE={r4['mae']:.4f}  RMSE={r4['rmse']:.4f}  R²={r4['r2']:.4f}  r={r4['pearson_r']:.4f}")

    # ── Baseline 5: Mean predictor ────────────────────────────────────────
    mean_pred = np.full(len(y_test), y_train.mean())
    r5 = metrics(y_test, mean_pred, 'Baseline: Mean predictor')
    all_results.append({**r5, 'trait': trait})
    print(f"  [Mean predictor] MAE={r5['mae']:.4f}  RMSE={r5['rmse']:.4f}  R²={r5['r2']:.4f}  r={r5['pearson_r']:.4f}")

    # ── Baseline 6: Linear Regression ────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_all[idx_train], y_train)
    raw_lr = lr.predict(X_all[idx_test])
    # fit calibrator on LR OOF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lr = np.zeros(len(idx_train))
    for tr_idx, val_idx in kf.split(X_all[idx_train]):
        lr2 = LinearRegression()
        lr2.fit(X_all[idx_train][tr_idx], y_train[tr_idx])
        oof_lr[val_idx] = lr2.predict(X_all[idx_train][val_idx])
    iso_lr = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds='clip')
    iso_lr.fit(oof_lr, y_train)
    cal_lr = iso_lr.predict(raw_lr)
    r6 = metrics(y_test, cal_lr, 'Baseline: Linear Regression')
    all_results.append({**r6, 'trait': trait})
    print(f"  [Linear Regr.]   MAE={r6['mae']:.4f}  RMSE={r6['rmse']:.4f}  R²={r6['r2']:.4f}  r={r6['pearson_r']:.4f}")

# ── 4. Save results ───────────────────────────────────────────────────────────
results_dir = project_root / 'results'
results_dir.mkdir(exist_ok=True)

df_out = pd.DataFrame(all_results)[['trait', 'variant', 'mae', 'rmse', 'r2', 'pearson_r']]
df_out = df_out.round(4)
out_path = results_dir / 'ablation_results.csv'
df_out.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print("FULL RESULTS TABLE")
print('='*60)
print(df_out.to_string(index=False))
print(f"\nSaved to: {out_path}")
