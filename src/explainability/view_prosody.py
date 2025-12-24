"""View SHAP prosody contributions"""
import pandas as pd

df = pd.read_csv('results/prosody_contributions.csv')

print("=" * 80)
print("PROSODIC CATEGORY CONTRIBUTIONS TO C3 PREDICTIONS")
print("=" * 80)

for trait in ['Curiosity', 'Critical Thinking', 'Creativity']:
    print(f"\n{trait}:")
    trait_data = df[df['trait'] == trait].head(5)
    for _, row in trait_data.iterrows():
        print(f"  {row['category']:30s}: {row['avg_impact']:.4f}")
