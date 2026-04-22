"""
Train Random Forest models for C3 soft skill prediction (full dataset).
Uses the same 2011-sample RecruitView dataset and identical pipeline
as train_big5_models.py for a fair side-by-side comparison.
Ground truth: LLM-generated labels (Groq/LLaMA 3.3-70b) from interview transcripts.
"""
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("C3 Soft Skill Prediction — Random Forest (Full Dataset)")
print("Ground truth: LLM-generated labels (Groq/LLaMA 3.3-70b)")
print("=" * 80)

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("\n1. Loading dataset...")
df = pd.read_csv('data/processed/recruitview_features_all.csv')
print(f"   Samples: {df.shape[0]}, Columns: {df.shape[1]}")

# ── 2. Define features and targets ───────────────────────────────────────────
print("\n2. Preparing features and labels...")

exclude_cols = [
    'video_id',
    'curiosity_score', 'critical_thinking_score', 'creativity_score',
]
feature_cols = [c for c in df.columns if c not in exclude_cols]
c3_targets = ['curiosity_score', 'critical_thinking_score', 'creativity_score']

X = df[feature_cols].values
print(f"   Features: {len(feature_cols)}")

mask = ~np.isnan(X).any(axis=1)
X = X[mask]
df_clean = df[mask].reset_index(drop=True)
print(f"   Samples after dropping NaN rows: {len(df_clean)}")

# ── 3. Train/test split (same seed as Big 5 for comparable splits) ────────────
print("\n3. Splitting data (80% train, 20% test)...")
X_train, X_test, idx_train, idx_test = train_test_split(
    X, np.arange(len(X)), test_size=0.2, random_state=42
)
print(f"   Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ── 4. Train models ───────────────────────────────────────────────────────────
print("\n4. Training Random Forest models for each C3 trait...")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

models = {}
results = []
feature_importance_data = []

for trait in c3_targets:
    y = df_clean[trait].values
    y_train, y_test = y[idx_train], y[idx_test]

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred  = rf.predict(X_test)

    train_mae  = mean_absolute_error(y_train, y_train_pred)
    test_mae   = mean_absolute_error(y_test,  y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
    train_r2   = r2_score(y_train, y_train_pred)
    test_r2    = r2_score(y_test,  y_test_pred)

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    print(f"\n   [C3] {trait}")
    print(f"      Test MAE:  {test_mae:.4f}")
    print(f"      Test RMSE: {test_rmse:.4f}")
    print(f"      Test R²:   {test_r2:.4f}")
    print(f"      CV MAE:    {cv_mae:.4f}")

    models[trait] = rf
    results.append({
        'trait': trait,
        'type': 'C3',
        'train_mae':  round(train_mae,  4),
        'test_mae':   round(test_mae,   4),
        'train_rmse': round(train_rmse, 4),
        'test_rmse':  round(test_rmse,  4),
        'train_r2':   round(train_r2,   4),
        'test_r2':    round(test_r2,    4),
        'cv_mae':     round(cv_mae,     4),
        'n_features': len(feature_cols),
        'n_train':    X_train.shape[0],
        'n_test':     X_test.shape[0],
    })

    with open(f'models/c3_{trait}_model.pkl', 'wb') as f:
        pickle.dump(rf, f)

    top_idx = np.argsort(rf.feature_importances_)[-10:][::-1]
    for rank, fi in enumerate(top_idx, 1):
        feature_importance_data.append({
            'trait': trait,
            'rank': rank,
            'feature': feature_cols[fi],
            'importance': round(rf.feature_importances_[fi], 6),
        })

# ── 5. Save results ───────────────────────────────────────────────────────────
print("\n5. Saving results...")
df_results = pd.DataFrame(results)
df_results.to_csv('results/c3_model_performance.csv', index=False)
print("   Saved: results/c3_model_performance.csv")

pd.DataFrame(feature_importance_data).to_csv('results/c3_feature_importance.csv', index=False)
print("   Saved: results/c3_feature_importance.csv")

# ── 6. Save test predictions ──────────────────────────────────────────────────
print("\n6. Saving test predictions...")
pred_rows = []
for i, orig_idx in enumerate(idx_test):
    row = {'video_id': df_clean.loc[orig_idx, 'video_id']}
    for trait in c3_targets:
        actual    = df_clean.loc[orig_idx, trait]
        predicted = models[trait].predict(X_test[i].reshape(1, -1))[0]
        row[f'{trait}_actual']    = round(actual,    4)
        row[f'{trait}_predicted'] = round(predicted, 4)
        row[f'{trait}_error']     = round(abs(actual - predicted), 4)
    pred_rows.append(row)

pd.DataFrame(pred_rows).to_csv('results/c3_test_predictions.csv', index=False)
print("   Saved: results/c3_test_predictions.csv")

# ── 7. Combined comparison table (Big 5 + C3) ────────────────────────────────
print("\n7. Building combined comparison table...")
big5_path = 'results/big5_model_performance.csv'
if os.path.exists(big5_path):
    df_big5 = pd.read_csv(big5_path)
    df_combined = pd.concat([df_big5, df_results], ignore_index=True)
    df_combined.to_csv('results/all_model_performance.csv', index=False)
    print("   Saved: results/all_model_performance.csv")

    print("\n" + "=" * 80)
    print("FULL COMPARISON: Big 5 (Human-Rated) vs C3 (LLM-Rated)")
    print("=" * 80)
    print(df_combined[['trait', 'type', 'test_mae', 'test_rmse', 'test_r2', 'cv_mae']].to_string(index=False))
else:
    print("   (Big 5 results not found — run train_big5_models.py first)")
    print("\n" + "=" * 80)
    print("C3 RESULTS SUMMARY")
    print("=" * 80)
    print(df_results[['trait', 'type', 'test_mae', 'test_rmse', 'test_r2', 'cv_mae']].to_string(index=False))

# ── 8. Comparison bar chart ───────────────────────────────────────────────────
print("\n8. Generating comparison R² chart...")
os.makedirs('results/plots', exist_ok=True)

if os.path.exists(big5_path):
    df_plot = df_combined.copy()
else:
    df_plot = df_results.copy()

color_map = {'Big5': '#4C72B0', 'Performance': '#DD8452', 'C3': '#55A868'}
colors = [color_map[t] for t in df_plot['type']]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(df_plot['trait'], df_plot['test_r2'], color=colors)
ax.set_xlabel('Trait')
ax.set_ylabel('R² Score (Test Set)')
ax.set_title('Big 5 (Human-Rated) vs C3 (LLM-Rated) — Random Forest R² Comparison')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(rotation=15, ha='right')

for bar, val in zip(bars, df_plot['test_r2']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.legend(handles=[
    Patch(color='#4C72B0', label='Big 5 (Human-Rated)'),
    Patch(color='#DD8452', label='Performance (Human-Rated)'),
    Patch(color='#55A868', label='C3 (LLM-Rated)'),
], loc='upper right')

plt.tight_layout()
plt.savefig('results/plots/big5_vs_c3_r2_comparison.png', dpi=150)
plt.close()
print("   Saved: results/plots/big5_vs_c3_r2_comparison.png")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print("  Models:  models/c3_*_model.pkl")
print("  Results: results/c3_*.csv  |  results/all_model_performance.csv")
print("  Chart:   results/plots/big5_vs_c3_r2_comparison.png")
