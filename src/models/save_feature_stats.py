"""
Save training feature statistics (mean, std, min, max) from the RecruitView dataset.
Used at inference time to clip live recording features to the training distribution,
preventing out-of-distribution predictions caused by the domain gap between
studio-quality training audio and live browser microphone recordings.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
data_dir     = project_root / 'data' / 'processed'
models_dir   = project_root / 'models'

features_df = pd.read_csv(data_dir / 'recruitview_features_all.csv')

exclude = [
    'video_id', 'transcript', 'file_name', 'question',
    'curiosity_score', 'critical_thinking_score', 'creativity_score',
    'curiosity_reasoning', 'critical_thinking_reasoning', 'creativity_reasoning',
    'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism',
    'overall_personality', 'interview_score', 'overall_performance',
    'word_count', 'sentence_count',
]
feature_cols = [c for c in features_df.columns if c not in exclude]
X = features_df[feature_cols]

stats = {
    'feature_cols': feature_cols,
    'mean':         X.mean().to_dict(),
    'std':          X.std().to_dict(),
    'min':          X.min().to_dict(),
    'max':          X.max().to_dict(),
    # 3-sigma bounds — clips extreme outliers while preserving natural variation
    'clip_low':     (X.mean() - 3 * X.std()).to_dict(),
    'clip_high':    (X.mean() + 3 * X.std()).to_dict(),
}

out_path = models_dir / 'feature_stats.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(stats, f)

print(f"Saved feature statistics for {len(feature_cols)} features")
print(f"Saved to: {out_path}")
print("\nSample clip bounds:")
for feat in ['pitch_mean', 'pitch_std', 'energy_mean', 'vocab_diversity', 'filler_word_ratio']:
    print(f"  {feat:25s}  clip [{stats['clip_low'][feat]:.3f}, {stats['clip_high'][feat]:.3f}]")
