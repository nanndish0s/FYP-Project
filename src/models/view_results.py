"""Display model training results"""
import pandas as pd

print("=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

df_perf = pd.read_csv('results/model_performance.csv')
print(df_perf.to_string(index=False))

print("\n" + "=" * 80)
print("TOP 5 IMPORTANT FEATURES PER TRAIT")
print("=" * 80)

df_imp = pd.read_csv('results/feature_importance.csv')

for trait in ['Curiosity', 'Critical Thinking', 'Creativity']:
    print(f"\n{trait}:")
    trait_features = df_imp[df_imp['trait'] == trait].head(5)
    for _, row in trait_features.iterrows():
        print(f"  {int(row['rank'])}. {row['feature']}: {row['importance']:.4f}")

print("\n" + "=" * 80)
print("TEST SET PREDICTIONS")
print("=" * 80)

df_pred = pd.read_csv('results/test_predictions.csv')
print(df_pred.to_string(index=False))
