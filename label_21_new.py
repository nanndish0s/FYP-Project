"""
Simple manual labeling script for 21 new transcripts
Uses the correct groq_labeler API signature
"""
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from labeling.groq_labeler import GroqLabeler
from labeling.prompt_template import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from tqdm import tqdm

def main():
    print("=" * 80)
    print("MANUAL LABELING - 21 NEW TRANSCRIPTS")
    print("=" * 80)
    
    # Load new transcripts
    df = pd.read_csv('data/processed/all_transcripts.csv')
    print(f"\n📊 Transcripts to label: {len(df)}")
    
    # Initialize labeler
    labeler = GroqLabeler(model_name="llama-3.3-70b-versatile")
    print(f"\n🤖 Groq API initialized")
    print(f"⏱️  Estimated time: ~{len(df) * 5 / 60:.1f} minutes\n")
    
    # Label each transcript
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        # Format prompts
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT_TEMPLATE.format(transcript=row['transcript'])
        
        # Get C3 labels
        result = labeler.label_transcript(
            transcript=row['transcript'],
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        if result:
            results.append({
                'video_id': row['video_id'],
                'transcript': row['transcript'],
                'word_count': row['word_count'],
                'curiosity_score': result['curiosity']['score'],
                'critical_thinking_score': result['critical_thinking']['score'],
                'creativity_score': result['creativity']['score'],
                'curiosity_reasoning': result['curiosity']['reasoning'],
                'critical_thinking_reasoning': result['critical_thinking']['reasoning'],
                'creativity_reasoning': result['creativity']['reasoning']
            })
        else:
            print(f"\n❌ Failed: {row['video_id']}")
    
    # Save results
    df_labeled = pd.DataFrame(results)
    df_labeled.to_csv('data/processed/new_21_labeled.csv', index=False)
    
    print(f"\n\n{'='*80}")
    print(f"✅ SUCCESS! Labeled {len(results)}/21 transcripts")
    print(f"💾 Saved to: data/processed/new_21_labeled.csv")
    
    # Show stats
    print(f"\n📈 C3 Statistics (New 21):")
    for trait in ['curiosity', 'critical_thinking', 'creativity']:
        scores = df_labeled[f'{trait}_score']
        print(f"  {trait.title()}: Mean={scores.mean():.2f}, Std={scores.std():.2f}")
    
    print(f"\n{'='*80}")
    return df_labeled

if __name__ == "__main__":
    main()
