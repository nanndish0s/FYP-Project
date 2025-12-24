"""
Combine original and newly extracted transcripts
"""
import pandas as pd

# Load both datasets
df_original = pd.read_csv('data/processed/sample_transcripts.csv')
df_new = pd.read_csv('data/processed/all_transcripts.csv')

print("=" * 70)
print("COMBINING TRANSCRIPT DATASETS")
print("=" * 70)

print(f"\nOriginal dataset: {len(df_original)} transcripts")
print(f"New extraction: {len(df_new)} transcripts")

# Check overlap
overlap = set(df_original['video_id']) & set(df_new['video_id'])
print(f"\nOverlap: {len(overlap)} videos")
print(f"Unique new videos: {len(df_new) - len(overlap)}")

# Combine
combined = pd.concat([df_original, df_new], ignore_index=True)
combined = combined.drop_duplicates(subset='video_id', keep='first')

print(f"\n📊 FINAL COMBINED DATASET:")
print(f"   Total transcripts: {len(combined)}")
print(f"   Total unique videos: {combined['video_id'].nunique()}")

# Save
combined.to_csv('data/processed/combined_transcripts.csv', index=False)

print(f"\n💾 Saved to: data/processed/combined_transcripts.csv")

# Show statistics
print(f"\n📈 Statistics:")
print(f"   Mean word count: {combined['word_count'].mean():.1f}")
print(f"   Min word count: {combined['word_count'].min()}")
print(f"   Max word count: {combined['word_count'].max()}")

print(f"\n🎉 Dataset expanded from 23 to {len(combined)} samples!")
print(f"   Increase: +{len(combined) - 23} samples ({(len(combined)/23)*100:.0f}% of original)")
print("=" * 70)
