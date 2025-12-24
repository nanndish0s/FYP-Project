"""
Batch LLM Labeling for Mass Extracted Transcripts
Modified version to handle large batch of transcripts efficiently
"""
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from labeling.groq_labeler import GroqLabeler
from labeling.prompt_template import C3_EVALUATION_PROMPT
from tqdm import tqdm
import json
from datetime import datetime

def main(
    input_file="data/processed/all_transcripts.csv",
    output_file="data/processed/all_with_c3_labels.csv",
    checkpoint_file="data/processed/labeling_checkpoint.json",
    checkpoint_interval=10
):
    """
    Label all transcripts with C3 scores
    
    Args:
        input_file: CSV with video_id, transcript, word_count
        output_file: CSV to save labeled results
        checkpoint_file: Save progress periodically
        checkpoint_interval: Save every N transcripts
    """
    
    print("=" * 80)
    print("BATCH C3 LABELING - GROQ API")
    print("=" * 80)
    
    # Load transcripts
    print(f"\n📊 Loading transcripts from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Total transcripts: {len(df):,}")
    
    # Initialize labeler
    print("\n🤖 Initializing Groq API...")
    labeler = GroqLabeler(model_name="llama-3.3-70b-versatile")
    
    # Check for checkpoint
    labeled_ids = set()
    results = []
    
    if os.path.exists(checkpoint_file):
        print(f"\n📂 Loading checkpoint from {checkpoint_file}...")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            labeled_ids = set(checkpoint['labeled_ids'])
            print(f"   Previously labeled: {len(labeled_ids):,} transcripts")
            
        # Load partial results
        if os.path.exists(output_file):
            df_partial = pd.read_csv(output_file)
            results = df_partial.to_dict('records')
            print(f"   Loaded {len(results):,} existing labels")
    
    # Filter unlabeled
    df_unlabeled = df[~df['video_id'].isin(labeled_ids)].reset_index(drop=True)
    print(f"\n🔄 Transcripts to label: {len(df_unlabeled):,}")
    
    if len(df_unlabeled) == 0:
        print("✅ All transcripts already labeled!")
        return pd.read_csv(output_file)
    
    # Estimate time and cost
    estimated_time = len(df_unlabeled) * 4.8 / 60
    print(f"\n⏱️  Estimated time: {estimated_time:.1f} minutes")
    print(f"   (~4.8 seconds per transcript)")
    print(f"\n💰 Cost: FREE (Groq generous tier)")
    
    # Process transcripts
    success_count = 0
    error_count = 0
    
    print(f"\n🚀 Starting labeling...\n")
    
    with tqdm(total=len(df_unlabeled), desc="Labeling") as pbar:
        for idx, row in df_unlabeled.iterrows():
            try:
                # Generate C3 labels
                c3_result = labeler.label_transcript(
                    transcript=row['transcript'],
                    video_id=row['video_id']
                )
                
                if c3_result:
                    # Add to results
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
                    
                    labeled_ids.add(row['video_id'])
                    success_count += 1
                else:
                    error_count += 1
                
                # Save checkpoint periodically
                if (idx + 1) % checkpoint_interval == 0:
                    save_checkpoint(checkpoint_file, labeled_ids, output_file, results)
                    pbar.set_postfix({
                        'success': success_count,
                        'errors': error_count
                    })
                
            except Exception as e:
                print(f"\n❌ Error labeling {row['video_id']}: {str(e)}")
                error_count += 1
            
            pbar.update(1)
    
    # Final save
    save_checkpoint(checkpoint_file, labeled_ids, output_file, results)
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("LABELING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Successfully labeled: {success_count:,}/{len(df_unlabeled):,}")
    print(f"❌ Errors: {error_count:,}")
    print(f"\n📊 Total dataset size: {len(results):,} samples")
    print(f"   Previous: 23 samples")
    print(f"   New: {len(results):,} samples") 
    print(f"   Increase: {len(results) - 23:,} samples ({(len(results)/23)*100:.0f}% of original)")
    
    print(f"\n💾 Saved to: {output_file}")
    print("=" * 80)
    
    # Show C3 score statistics
    df_final = pd.DataFrame(results)
    print(f"\n📈 C3 Score Statistics:")
    for trait in ['curiosity', 'critical_thinking', 'creativity']:
        scores = df_final[f'{trait}_score']
        print(f"\n   {trait.replace('_', ' ').title()}:")
        print(f"      Mean: {scores.mean():.2f}")
        print(f"      Std:  {scores.std():.2f}")
        print(f"      Min:  {scores.min()}")
        print(f"      Max:  {scores.max()}")
    
    return df_final

def save_checkpoint(checkpoint_file, labeled_ids, output_file, results):
    """Save current progress"""
    # Save checkpoint
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'labeled_count': len(labeled_ids),
            'labeled_ids': list(labeled_ids)
        }, f)
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
