"""
Merge all features and create ML-ready dataset for 44 samples
"""
import pandas as pd

print("=" * 80)
print("MERGING FEATURES - 44 SAMPLES")
print("=" * 80)

# Load all datasets
print("\n1. Loading datasets...")
df_labels = pd.read_csv('data/processed/combined_44_with_c3_labels.csv')
df_acoustic = pd.read_csv('data/processed/acoustic_features_44csv')
df_lexical = pd.read_csv('data/processed/lexical_features_44.csv')

print(f"   Labels: {len(df_labels)} samples")
print(f"   Acoustic: {len(df_acoustic)} samples × {len(df_acoustic.columns)-1} features")
print(f"   Lexical: {len(df_lexical)} samples × {len(df_lexical.columns)-1} features")

# Merge on video_id
print("\n2. Merging datasets...")
ml_dataset = df_labels[['video_id', 'transcript', 'word_count', 
                         'curiosity_score', 'critical_thinking_score', 'creativity_score']].copy()
ml_dataset = ml_dataset.merge(df_acoustic, on='video_id', how='inner')
ml_dataset = ml_dataset.merge(df_lexical.drop('word_count', axis=1), on='video_id', how='inner')

print(f"\n✅ Merged dataset:")
print(f"   Samples: {len(ml_dataset)}")
print(f"   Total columns: {len(ml_dataset.columns)}")
print(f"   Feature columns: {len(ml_dataset.columns) - 6}")  # Subtract metadata + 3 scores

# Save
output_file = 'data/processed/ml_ready_dataset_44.csv'
ml_dataset.to_csv(output_file, index=False)

print(f"\n💾 Saved to: {output_file}")

print("\n" + "=" * 80)
print("✅ ML-READY DATASET COMPLETE!")
print(f"   {len(ml_dataset)} samples × {len(ml_dataset.columns)} columns")
print("   Ready for model training!")
print("=" * 80)
