"""
Train Random Forest models for Big 5 personality trait prediction.
Uses acoustic + lexical features from RecruitView to predict human-rated
Big 5 scores (observer-rated via pairwise judgments — human ground truth).
"""
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Big 5 Personality Trait Prediction — Random Forest")
print("Ground truth: human observer-rated (RecruitView pairwise judgments)")
print("=" * 80)

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("\n1. Loading dataset...")
df = pd.read_csv('data/processed/ml_ready_big5.csv')
print(f"   Samples: {df.shape[0]}, Columns: {df.shape[1]}")

# ── 2. Define features and targets ───────────────────────────────────────────
print("\n2. Preparing features and labels...")

exclude_cols = [
    'video_id',
    'curiosity_score', 'critical_thinking_score', 'creativity_score',
    'openness', 'conscientiousness', 'extraversion', 'agreeableness',
    'neuroticism', 'interview_score', 'overall_performance'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

big5_targets = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
extended_targets = big5_targets + ['interview_score', 'overall_performance']

X = df[feature_cols].values
print(f"   Features: {len(feature_cols)}")
print(f"   Feature names: {feature_cols}")

# Drop rows with any NaN in features
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
df_clean = df[mask].reset_index(drop=True)
print(f"   Samples after dropping NaN rows: {len(df_clean)}")

# ── 3. Train/test split ───────────────────────────────────────────────────────
print("\n3. Splitting data (80% train, 20% test)...")
X_train, X_test, idx_train, idx_test = train_test_split(
    X, np.arange(len(X)), test_size=0.2, random_state=42
)
print(f"   Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ── 4. Train models ───────────────────────────────────────────────────────────
print("\n4. Training Random Forest models for each Big 5 trait...")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

models = {}
results = []
feature_importance_data = []

for trait in extended_targets:
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
    y_test_pred = rf.predict(X_test)

    train_mae  = mean_absolute_error(y_train, y_train_pred)
    test_mae   = mean_absolute_error(y_test,  y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
    train_r2   = r2_score(y_train, y_train_pred)
    test_r2    = r2_score(y_test,  y_test_pred)

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    label = "Big5" if trait in big5_targets else "Perf"
    print(f"\n   [{label}] {trait}")
    print(f"      Test MAE:  {test_mae:.4f}")
    print(f"      Test RMSE: {test_rmse:.4f}")
    print(f"      Test R²:   {test_r2:.4f}")
    print(f"      CV MAE:    {cv_mae:.4f}")

    models[trait] = rf
    results.append({
        'trait': trait,
        'type': 'Big5' if trait in big5_targets else 'Performance',
        'train_mae': round(train_mae, 4),
        'test_mae':  round(test_mae, 4),
        'train_rmse': round(train_rmse, 4),
        'test_rmse':  round(test_rmse, 4),
        'train_r2':  round(train_r2, 4),
        'test_r2':   round(test_r2, 4),
        'cv_mae':    round(cv_mae, 4),
        'n_features': len(feature_cols),
        'n_train': X_train.shape[0],
        'n_test':  X_test.shape[0],
    })

    # Save model
    with open(f'models/big5_{trait}_model.pkl', 'wb') as f:
        pickle.dump(rf, f)

    # Top 10 feature importances
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
df_results.to_csv('results/big5_model_performance.csv', index=False)
print("   Saved: results/big5_model_performance.csv")

df_importance = pd.DataFrame(feature_importance_data)
df_importance.to_csv('results/big5_feature_importance.csv', index=False)
print("   Saved: results/big5_feature_importance.csv")

# ── 6. Save test predictions ──────────────────────────────────────────────────
print("\n6. Saving test predictions...")
pred_rows = []
for i, orig_idx in enumerate(idx_test):
    row = {'video_id': df_clean.loc[orig_idx, 'video_id']}
    for trait in extended_targets:
        actual = df_clean.loc[orig_idx, trait]
        predicted = models[trait].predict(X_test[i].reshape(1, -1))[0]
        row[f'{trait}_actual']    = round(actual, 4)
        row[f'{trait}_predicted'] = round(predicted, 4)
        row[f'{trait}_error']     = round(abs(actual - predicted), 4)
    pred_rows.append(row)

pd.DataFrame(pred_rows).to_csv('results/big5_test_predictions.csv', index=False)
print("   Saved: results/big5_test_predictions.csv")

# ── 7. Summary table ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(df_results[['trait', 'type', 'test_mae', 'test_rmse', 'test_r2', 'cv_mae']].to_string(index=False))

# ── 8. Bar chart of R² per trait ─────────────────────────────────────────────
print("\n7. Generating R² comparison chart...")
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#4C72B0' if t in big5_targets else '#DD8452' for t in df_results['trait']]
bars = ax.bar(df_results['trait'], df_results['test_r2'], color=colors)
ax.set_xlabel('Trait')
ax.set_ylabel('R² Score (Test Set)')
ax.set_title('Big 5 & Performance Prediction — Random Forest R² Scores')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
for bar, val in zip(bars, df_results['test_r2']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#4C72B0', label='Big 5'),
                   Patch(color='#DD8452', label='Performance')], loc='upper right')
plt.tight_layout()
os.makedirs('results/plots', exist_ok=True)
plt.savefig('results/plots/big5_r2_scores.png', dpi=150)
plt.close()
print("   Saved: results/plots/big5_r2_scores.png")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"  Models saved to:  models/big5_*_model.pkl")
print(f"  Results saved to: results/big5_*.csv")
print(f"  Chart saved to:   results/plots/big5_r2_scores.png")
