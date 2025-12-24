"""
Merge all features and labels into ML-ready dataset
Combines: Acoustic (518) + Lexical (21) + C3 Labels (3)
"""
import pandas as pd

print("=" * 80)
print("Creating ML-Ready Dataset")
print("=" * 80)

# Load all components
print("\n1. Loading feature sets and labels...")
df_acoustic = pd.read_csv('data/processed/acoustic_features.csv')
df_lexical = pd.read_csv('data/processed/lexical_features.csv')
df_labels = pd.read_csv('data/processed/sample_with_c3_labels.csv')

print(f"   Acoustic features: {df_acoustic.shape}")
print(f"   Lexical features: {df_lexical.shape}")
print(f"   C3 labels: {df_labels.shape}")

# Merge on video_id
print("\n2. Merging datasets...")
df_merged = df_acoustic.merge(df_lexical, on='video_id', how='inner')
print(f"   After acoustic + lexical merge: {df_merged.shape}")

# Add C3 labels
df_final = df_merged.merge(
    df_labels[['video_id', 'curiosity_score', 'critical_thinking_score', 'creativity_score']],
    on='video_id',
    how='inner'
)
print(f"   After adding C3 labels: {df_final.shape}")

# Reorder columns: video_id, labels, features
label_cols = ['curiosity_score', 'critical_thinking_score', 'creativity_score']
feature_cols = [col for col in df_final.columns if col not in ['video_id'] + label_cols]

final_col_order = ['video_id'] + label_cols + feature_cols
df_final = df_final[final_col_order]

# Save
output_path = 'data/processed/ml_ready_dataset.csv'
df_final.to_csv(output_path, index=False)

print(f"\n3. Saved ML-ready dataset to: {output_path}")

# Summary statistics
print(f"\n4. Dataset Summary:")
print(f"   Total samples: {df_final.shape[0]}")
print(f"   Total features: {len(feature_cols)}")
print(f"     - Acoustic: 518")
print(f"     - Lexical: 21")
print(f"   Target variables: {len(label_cols)}")
print(f"   Total columns: {df_final.shape[1]}")

# Show C3 label distribution
print(f"\n5. C3 Label Distribution:")
for label in label_cols:
    print(f"   {label}:")
    print(f"     Mean: {df_final[label].mean():.2f}")
    print(f"     Std: {df_final[label].std():.2f}")
    print(f"     Range: [{df_final[label].min()}, {df_final[label].max()}]")

# Show sample
print(f"\n6. Sample (first 3 rows, first 8 columns):")
print(df_final.iloc[:3, :8])

print("\n" + "=" * 80)
print("ML-Ready Dataset Complete!")
print("=" * 80)
print(f"\n✓ Ready for model training!")
print(f"   Input: {len(feature_cols)} features")
print(f"   Output: 3 C3 scores (Curiosity, Critical Thinking, Creativity)")
print(f"   Samples: {df_final.shape[0]}")
