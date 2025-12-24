"""
Combine original 23 labeled samples with new 21 labeled samples
"""
import pandas as pd

# Load both datasets
df_original = pd.read_csv('data/processed/sample_with_c3_labels.csv')
df_new = pd.read_csv('data/processed/new_21_labeled.csv')

print("=" * 80)
print("COMBINING LABELED DATASETS")
print("=" * 80)

print(f"\nOriginal labeled: {len(df_original)} samples")
print(f"New labeled: {len(df_new)} samples")

# Combine
combined = pd.concat([df_original, df_new], ignore_index=True)

print(f"\n✅ Combined total: {len(combined)} samples")
print(f"   Increase: +{len(combined) - 23} samples")

# Save
output_file = 'data/processed/combined_44_with_c3_labels.csv'
combined.to_csv(output_file, index=False)

print(f"\n💾 Saved to: {output_file}")

# Show C3 statistics
print(f"\n📈 C3 Score Statistics (All 44 Samples):")
for trait in ['curiosity', 'critical_thinking', 'creativity']:
    scores = combined[f'{trait}_score']
    print(f"\n  {trait.replace('_', ' ').title()}:")
    print(f"    Mean: {scores.mean():.2f}")
    print(f"    Std:  {scores.std():.2f}")
    print(f"    Range: {scores.min()}-{scores.max()}")

print("\n" + "=" * 80)
print("🎉 DATASET EXPANSION COMPLETE!")
print(f"   23 samples → 44 samples (191% of original)")
print("=" * 80)
