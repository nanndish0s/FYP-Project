"""
View C3 labeled transcripts in a readable format
"""
import pandas as pd

# Load labeled data
df = pd.read_csv('data/processed/sample_with_c3_labels.csv')

print("=" * 100)
print("C3 LABELED TRANSCRIPTS - SUMMARY")
print("=" * 100)
print(f"\nTotal transcripts: {len(df)}")
print(f"Successfully labeled: {len(df[df['labeling_status'] == 'success'])}")

# Show overview table
print("\n" + "=" * 100)
print("OVERVIEW - All Transcripts")
print("=" * 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
print(df[['video_id', 'word_count', 'curiosity_score', 'critical_thinking_score', 'creativity_score']].to_string(index=False))

# Show detailed view of first 3
print("\n" + "=" * 100)
print("DETAILED VIEW - First 3 Examples")
print("=" * 100)

for i in range(min(3, len(df))):
    sample = df.iloc[i]
    print(f"\n{'='*100}")
    print(f"TRANSCRIPT {i+1}: {sample['video_id']}")
    print('='*100)
    print(f"Word Count: {sample['word_count']}")
    
    print(f"\nTranscript Preview:")
    print(f"  {sample['transcript'][:250]}...")
    
    print(f"\n--- C3 SCORES & REASONING ---")
    print(f"\n✦ CURIOSITY: {int(sample['curiosity_score'])}/5")
    print(f"  Reasoning: {sample['curiosity_reasoning']}")
    
    print(f"\n✦ CRITICAL THINKING: {int(sample['critical_thinking_score'])}/5")
    print(f"  Reasoning: {sample['critical_thinking_reasoning']}")
    
    print(f"\n✦ CREATIVITY: {int(sample['creativity_score'])}/5")
    print(f"  Reasoning: {sample['creativity_reasoning']}")

print("\n" + "=" * 100)
print("To view all data in Excel/Spreadsheet:")
print("Open: data/processed/sample_with_c3_labels.csv")
print("=" * 100)
