"""
Generates Figure 6.1 — Predicted vs Actual Score Scatter Plots
One panel per C3 trait, saved as figure6_1.png
Run from the project root: python generate_figure6_1.py
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# ── Load data ─────────────────────────────────────────────────────────────────
data_dir = Path('data/processed')
features_df = pd.read_csv(data_dir / 'recruitview_features_all.csv')
metadata_df = pd.read_csv(data_dir / 'recruitview_metadata.csv')

exclude_cols = [
    'video_id', 'transcript', 'file_name', 'question',
    'curiosity_score', 'critical_thinking_score', 'creativity_score',
    'curiosity_reasoning', 'critical_thinking_reasoning', 'creativity_reasoning',
    'openness', 'conscientiousness', 'extraversion',
    'agreeableness', 'neuroticism', 'overall_personality',
    'interview_score', 'overall_performance', 'word_count', 'sentence_count',
]
feature_cols = [c for c in features_df.columns if c not in exclude_cols]

merged = features_df[['video_id'] + feature_cols].merge(
    metadata_df[['video_id', 'openness', 'conscientiousness', 'extraversion']],
    on='video_id', how='inner'
)

X = merged[feature_cols].fillna(merged[feature_cols].mean()).values

qt = QuantileTransformer(output_distribution='uniform', random_state=42)
oq = qt.fit_transform(merged[['openness']]).flatten() * 4 + 1
cq = qt.fit_transform(merged[['conscientiousness']]).flatten() * 4 + 1
eq = qt.fit_transform(merged[['extraversion']]).flatten() * 4 + 1

targets = {
    'Curiosity':         oq,
    'Critical Thinking': cq,
    'Creativity':        (oq + eq) / 2,
}
model_keys = {
    'Curiosity':         'curiosity',
    'Critical Thinking': 'critical_thinking',
    'Creativity':        'creativity',
}

X_train, X_test, idx_tr, idx_te = train_test_split(
    X, np.arange(len(X)), test_size=0.2, random_state=42
)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))
fig.suptitle(
    'Figure 6.1 – Predicted vs Actual C3 Scores (Test Set, n = 403)',
    fontsize=14, fontweight='bold', y=1.02
)

colours = ['#2563EB', '#7C3AED', '#059669']

for idx, (trait, colour) in enumerate(zip(targets.keys(), colours)):
    ax = fig.add_subplot(1, 3, idx + 1)

    y      = targets[trait]
    y_test = y[idx_te]

    key = model_keys[trait]
    with open(f'models/{key}_model_recruitview.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open(f'models/{key}_calibration.pkl', 'rb') as f:
        cal = pickle.load(f)

    y_pred = cal.predict(rf.predict(X_test))
    r, _   = pearsonr(y_test, y_pred)

    # Scatter
    ax.scatter(y_test, y_pred, alpha=0.35, s=18, color=colour, edgecolors='none')

    # Perfect prediction line
    lims = [1, 5]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.6, label='Perfect prediction')

    # Trend line
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, 5, 100)
    ax.plot(x_line, p(x_line), color=colour, linewidth=2, label=f'Trend (r = {r:.3f})')

    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_xlabel('Actual Score (Ground Truth)', fontsize=11)
    ax.set_ylabel('Predicted Score', fontsize=11)
    ax.set_title(trait, fontsize=12, fontweight='bold', color=colour)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    # Annotate r value prominently
    ax.text(0.97, 0.05, f'Pearson r = {r:.3f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='right', va='bottom', color=colour,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figure6_1.png', dpi=200, bbox_inches='tight')
print("Saved: figure6_1.png")
plt.show()
