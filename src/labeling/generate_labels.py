"""
Main script to generate C3 labels using Groq API
Processes transcripts and saves results with labels
"""
import pandas as pd
from tqdm import tqdm
import json
import os
from prompt_template import get_system_prompt, create_prompt
from groq_labeler import GroqLabeler


def main(
    input_file="../../data/processed/sample_transcripts.csv",
    output_file="../../data/processed/sample_with_c3_labels.csv",
    limit=None
):
    """
    Generate C3 labels for transcripts using Gemini API.
    
    Args:
        input_file: Path to CSV with transcripts
        output_file: Path to save labeled data
        limit: Optional limit on number of transcripts to process (for testing)
    """
    
    print("=" * 70)
    print("C3 Label Generation with Gemini API")
    print("=" * 70)
    
    # Load transcripts
    print(f"\nLoading transcripts from {input_file}...")
    df = pd.read_csv(input_file)
    
    if limit:
        df = df.head(limit)
        print(f"Processing first {limit} transcripts (test mode)")
    
    print(f"Total transcripts to process: {len(df)}")
    
    # Initialize Groq labeler
    print("\nInitializing Groq API...")
    labeler = GroqLabeler(model_name="llama-3.3-70b-versatile")
    
    # Get prompts
    system_prompt = get_system_prompt()
    
    # Process each transcript
    results  = []
    
    print(f"\nGenerating C3 labels...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        video_id = row['video_id']
        transcript = row['transcript']
        word_count = row['word_count']
        
        # Create user prompt
        user_prompt = create_prompt(transcript)
        
        # Get labels
        labels = labeler.label_transcript(transcript, system_prompt, user_prompt)
        
        if labels:
            result = {
                'video_id': video_id,
                'transcript': transcript,
                'word_count': word_count,
                'curiosity_score': labels['curiosity']['score'],
                'curiosity_reasoning': labels['curiosity']['reasoning'],
                'critical_thinking_score': labels['critical_thinking']['score'],
                'critical_thinking_reasoning': labels['critical_thinking']['reasoning'],
                'creativity_score': labels['creativity']['score'],
                'creativity_reasoning': labels['creativity']['reasoning'],
                'labeling_status': 'success'
            }
        else:
            result = {
                'video_id': video_id,
                'transcript': transcript,
                'word_count': word_count,
                'curiosity_score': None,
                'curiosity_reasoning': None,
                'critical_thinking_score': None,
                'critical_thinking_reasoning': None,
                'creativity_score': None,
                'creativity_reasoning': None,
                'labeling_status': 'failed'
            }
        
        results.append(result)
    
    # Create DataFrame
    df_labeled = pd.DataFrame(results)
    
    # Save results
    df_labeled.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print("Results Summary")
    print('='*70)
    
    successful = df_labeled[df_labeled['labeling_status'] == 'success']
    print(f"Successfully labeled: {len(successful)}/{len(df_labeled)}")
    
    if len(successful) > 0:
        print(f"\nC3 Score Statistics:")
        print(f"Curiosity - Mean: {successful['curiosity_score'].mean():.2f}, " 
              f"Min: {successful['curiosity_score'].min()}, "
              f"Max: {successful['curiosity_score'].max()}")
        print(f"Critical Thinking - Mean: {successful['critical_thinking_score'].mean():.2f}, "
              f"Min: {successful['critical_thinking_score'].min()}, "
              f"Max: {successful['critical_thinking_score'].max()}")
        print(f"Creativity - Mean: {successful['creativity_score'].mean():.2f}, "
              f"Min: {successful['creativity_score'].min()}, "
              f"Max: {successful['creativity_score'].max()}")
        
        # Show sample
        print(f"\nSample labeled entry:")
        sample = successful.iloc[0]
        print(f"Video ID: {sample['video_id']}")
        print(f"Word count: {sample['word_count']}")
        print(f"Scores: Curiosity={sample['curiosity_score']}, "
              f"Critical Thinking={sample['critical_thinking_score']}, "
              f"Creativity={sample['creativity_score']}")
        print(f"Curiosity reasoning: {sample['curiosity_reasoning'][:100]}...")
    
    print(f"\nSaved to: {output_file}")
    print("=" * 70)
    
    return df_labeled


if __name__ == "__main__":
    # Run full batch
    print("Running C3 label generation on ALL transcripts...")
    print("This will process all 23 samples.\n")
    
    df_labeled = main()  # No limit - process all
    
    print("\n" + "=" * 70)
    print("LABELING COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: data/processed/sample_with_c3_labels.csv")
