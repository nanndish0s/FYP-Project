import pandas as pd
import numpy as np

# Load metadata and features
metadata = pd.read_csv('data/processed/recruitview_metadata.csv')
features = pd.read_csv('data/processed/recruitview_features_all.csv')

# Merge
df = pd.merge(metadata, features, on='video_id')

# Find high scorers (top 20% for each trait)
print("HIGH SCORING SAMPLES (Top 20% for each trait)")
print("="*70)

for trait in ['curiosity_score', 'critical_thinking_score', 'creativity_score']:
    threshold = df[trait].quantile(0.80)
    high_scorers = df[df[trait] >= threshold].sort_values(trait, ascending=False).head(5)
    
    print(f"\n{trait.upper().replace('_', ' ')} (Threshold: {threshold:.2f})")
    print("-"*70)
    
    for idx, row in high_scorers.iterrows():
        print(f"\nVideo ID: {row['video_id']} | Score: {row[trait]:.2f}")
        print(f"  Word Count: {row.get('word_count', 'N/A')}")
        print(f"  Vocab Diversity: {row.get('vocab_diversity', 'N/A'):.3f}" if 'vocab_diversity' in row else "  Vocab Diversity: N/A")
        print(f"  Pitch Mean: {row.get('pitch_mean', 'N/A'):.1f}" if 'pitch_mean' in row else "  Pitch Mean: N/A")
        print(f"  Energy Mean: {row.get('energy_mean', 'N/A'):.4f}" if 'energy_mean' in row else "  Energy Mean: N/A")
        print(f"  Transcript (first 100 chars): {row['transcript'][:100]}...")
