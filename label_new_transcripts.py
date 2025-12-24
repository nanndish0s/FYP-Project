"""
Label 21 new transcripts with C3 scores using existing pipeline
"""
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.labeling.groq_labeler import GroqLabeler
from tqdm import tqdm
import time

def main():
    print("=" * 80)
    print("LABELING NEW TRANSCRIPTS - GROQ API")
    print("=" * 80)
    
    # Load the 21 new transcripts
    print("\n📊 Loading new transcripts...")
    df_new = pd.read_csv('data/processed/all_transcripts.csv')
    print(f"   New transcripts: {len(df_new)}")
    
    # Initialize labeler
    print("\n🤖 Initializing Groq API (llama-3.3-70b-versatile)...")
    labeler = GroqLabeler(model_name="llama-3.3-70b-versatile")
    
    # Estimate time
    estimated_time = len(df_new) * 4.8 / 60
    print(f"\n⏱️  Estimated time: {estimated_time:.1f} minutes (~4.8 sec/transcript)")
    print("💰 Cost: FREE (Groq generous tier)\n")
    
    # Label transcripts
    results = []
    success_count = 0
    error_count = 0
    
    print("🚀 Starting labeling...\n")
    
    for idx, row in tqdm(df_new.iterrows(), total=len(df_new), desc="Labeling"):
        try:
            c3_result = labeler.label_transcript(
                transcript=row['transcript']
            )
            
            if c3_result:
                results.append({
                    'video_id': row['video_id'],
                    'transcript': row['transcript'],
                    'word_count': row['word_count'],
                    'curiosity_score': c3_result['curiosity_score'],
                    'critical_thinking_score': c3_result['critical_thinking_score'],
                    'creativity_score': c3_result['creativity_score'],
                    'curiosity_reasoning': c3_result['curiosity_reasoning'],
                    'critical_thinking_reasoning': c3_result['critical_thinking_reasoning'],
                    'creativity_reasoning': c3_result['creativity_reasoning']
                })
                success_count += 1
            else:
                error_count += 1
                print(f"\n❌ Failed to label {row['video_id']}")
        
        except Exception as e:
            print(f"\n❌ Error labeling {row['video_id']}: {str(e)}")
            error_count += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Save results
    df_labeled = pd.DataFrame(results)
    output_file = 'data/processed/new_transcripts_with_c3_labels.csv'
    df_labeled.to_csv(output_file, index=False)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("LABELING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Successfully labeled: {success_count}/{len(df_new)}")
    print(f"❌ Errors: {error_count}")
    print(f"\n💾 Saved to: {output_file}")
    
    # Show C3 statistics
    if len(df_labeled) > 0:
        print("\n📈 C3 Score Statistics (New Transcripts):")
        for trait in ['curiosity', 'critical_thinking', 'creativity']:
            scores = df_labeled[f'{trait}_score']
            print(f"\n   {trait.replace('_', ' ').title()}:")
            print(f"      Mean: {scores.mean():.2f}")
            print(f"      Std:  {scores.std():.2f}")
            print(f"      Min:  {scores.min()}")
            print(f"      Max:  {scores.max()}")
    
    print("\n" + "=" * 80)
    print(f"🎉 Now combine with original 23 to get {success_count + 23} total labeled samples!")
    print("=" * 80)
    
    return df_labeled

if __name__ == "__main__":
    main()
