import pandas as pd
import numpy as np

# Load features which includes scores
df_feats = pd.read_csv('data/processed/recruitview_features_all.csv')
print(f"Loaded {len(df_feats)} feature samples")

# Load metadata for transcripts
df_meta = pd.read_csv('data/processed/recruitview_metadata.csv')
print(f"Loaded {len(df_meta)} metadata samples")

# Merge on video_id
df = pd.merge(df_feats, df_meta[['video_id', 'transcript']], on='video_id')
print(f"Merged dataset size: {len(df)}")

print("\n" + "="*80)
print("FEATURE PROFILE OF HIGH SCORERS (Score >= 3.5)")
print("="*80)

traits = ['curiosity_score', 'critical_thinking_score', 'creativity_score']

for trait in traits:
    # Filter high scorers (using 3.5 as realistic high threshold based on previous analysis)
    high_scorers = df[df[trait] >= 3.5]
    
    if len(high_scorers) == 0:
        print(f"\n⚠️ No samples found with {trait} >= 3.5")
        continue
        
    print(f"\n🧠 {trait.upper().replace('_SCORE', '')} (N={len(high_scorers)})")
    
    # Calculate feature stats
    avg_word_count = high_scorers['word_count'].mean()
    avg_vocab = high_scorers['vocab_diversity'].mean()
    avg_pitch = high_scorers['pitch_mean'].mean()
    avg_pitch_std = high_scorers['pitch_std'].mean()
    avg_energy = high_scorers['energy_mean'].mean()
    avg_filler = high_scorers['filler_word_ratio'].mean()
    
    print(f"  Word Count:      {avg_word_count:.1f} (vs {df['word_count'].mean():.1f} avg)")
    print(f"  Vocab Diversity: {avg_vocab:.3f} (vs {df['vocab_diversity'].mean():.3f} avg)")
    print(f"  Filler Ratio:    {avg_filler:.3f} (vs {df['filler_word_ratio'].mean():.3f} avg)")
    print(f"  Pitch Mean:      {avg_pitch:.1f} Hz")
    print(f"  Pitch Std:       {avg_pitch_std:.1f} Hz (Variation)")
    print(f"  Energy Mean:     {avg_energy:.4f}")
    
    # Show best example
    best_candidate = high_scorers.sort_values(trait, ascending=False).iloc[0]
    print(f"\n  🏆 Top Scorer (ID: {best_candidate['video_id']}, Score: {best_candidate[trait]:.2f})")
    print(f"  Word Count: {best_candidate['word_count']}")
    print(f"  Pitch: {best_candidate['pitch_mean']:.1f} Hz")
    print(f"  Transcript: \"{best_candidate['transcript'][:200]}...\"")
    print("-" * 40)
