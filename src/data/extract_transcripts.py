"""
Extract sample transcripts from CMU-MOSEI TimestampedWordVectors dataset.
Samples 50-100 random video segments for C3 labeling.
"""
import h5py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def extract_transcripts(
    csd_file="data/raw/debug_test.csd",
    output_file="data/processed/sample_transcripts.csv",
    sample_size=100,
    min_word_count=20,
    random_seed=42
):
    """
    Extract sample transcripts from CMU-MOSEI dataset.
    
    Args:
        csd_file: Path to TimestampedWordVectors CSD file
        output_file: Path to save extracted transcripts
        sample_size: Number of samples to extract
        min_word_count: Minimum words required in transcript
        random_seed: Random seed for reproducibility
    """
    
    print("=" * 70)
    print("Extracting Sample Transcripts from CMU-MOSEI")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load HDF5 file
    print(f"\nLoading dataset from {csd_file}...")
    with h5py.File(csd_file, 'r') as f:
        # Navigate to glove_vectors group
        glove_vectors = f['glove_vectors']
        video_ids = list(glove_vectors.keys())
        
        print(f"Total video segments available: {len(video_ids):,}")
        
        # Extract transcripts
        transcripts_data = []
        
        print("\nExtracting transcripts...")
        for video_id in tqdm(video_ids):
            video_group = glove_vectors[video_id]
            
            # Check if 'words' dataset exists
            if 'words' in video_group:
                words_data = video_group['words'][:]
                
                # Decode words (they're stored as bytes)
                if len(words_data) > 0:
                    # Words are stored as byte strings
                    words = []
                    for word_bytes in words_data:
                        try:
                            # Handle different encodings
                            if isinstance(word_bytes, bytes):
                                word = word_bytes.decode('utf-8')
                            else:
                                word = str(word_bytes)
                            words.append(word)
                        except:
                            continue
                    
                    transcript = ' '.join(words)
                    word_count = len(words)
                    
                    # Filter by minimum word count
                    if word_count >= min_word_count:
                        transcripts_data.append({
                            'video_id': video_id,
                            'transcript': transcript,
                            'word_count': word_count
                        })
    
    print(f"\nExtracted {len(transcripts_data):,} transcripts with >= {min_word_count} words")
    
    # Convert to DataFrame
    df = pd.DataFrame(transcripts_data)
    
    # Sample random subset
    np.random.seed(random_seed)
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=random_seed)
        print(f"Sampled {sample_size} random transcripts")
    else:
        df_sample = df
        print(f"Using all {len(df)} transcripts (less than requested sample size)")
    
    # Sort by video_id for consistency
    df_sample = df_sample.sort_values('video_id').reset_index(drop=True)
    
    # Save to CSV
    df_sample.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    print(f"Sample size: {len(df_sample)}")
    print(f"Word count - Mean: {df_sample['word_count'].mean():.1f}")
    print(f"Word count - Min: {df_sample['word_count'].min()}")
    print(f"Word count - Max: {df_sample['word_count'].max()}")
    print("=" * 70)
    
    return df_sample

if __name__ == "__main__":
    # Extract 100 sample transcripts
    extract_transcripts(
        sample_size=100,
        min_word_count=20
    )
