import pandas as pd

df = pd.read_csv('data/processed/recruitview_metadata.csv')

print('RECRUITVIEW SCORE DISTRIBUTION ANALYSIS')
print('='*60)
print(f'Total samples: {len(df)}')
print()

for col in ['curiosity_score', 'critical_thinking_score', 'creativity_score']:
    print(f'{col.replace("_", " ").upper()}:')
    print(f'  Mean: {df[col].mean():.2f}')
    print(f'  Median: {df[col].median():.2f}')
    print(f'  Min: {df[col].min():.2f}')
    print(f'  Max: {df[col].max():.2f}')
    high = len(df[df[col] >= 4.0])
    mid = len(df[df[col] >= 3.5])
    low = len(df[df[col] < 3.0])
    print(f'  Scores >= 4.0: {high} ({high/len(df)*100:.1f}%)')
    print(f'  Scores >= 3.5: {mid} ({mid/len(df)*100:.1f}%)')
    print(f'  Scores < 3.0: {low} ({low/len(df)*100:.1f}%)')
    print()
