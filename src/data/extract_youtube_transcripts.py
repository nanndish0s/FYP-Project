"""
Extract transcripts from YouTube using video IDs from CMU-MOSEI metadata.
Samples 100 videos and fetches their captions for C3 labeling.
"""
import pandas as pd
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from tqdm import tqdm
import time
import os

def extract_video_id(filename):
    """Extract YouTube video ID from CMU-MOSEI filename."""
    # Filenames are like: "SqAiJrvHXNA_3" or "245243_0"
    # The video ID is before the underscore
    parts = filename.split('_')
    if len(parts) >= 2:
        # Check if it looks like a YouTube ID (contains letters)
        video_id = parts[0]
        if any(c.isalpha() for c in video_id):
            return video_id
    return None

def get_transcript(video_id):
    """Fetch transcript from YouTube for a given video ID."""
    try:
        # Create API instance and fetch transcript
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        
        # Convert to raw data (list of dicts)
        transcript_list = fetched_transcript.to_raw_data()
        
        # Combine all text entries
        full_text = ' '.join([entry['text'] for entry in transcript_list])
        
        # Clean up common artifacts
        full_text = full_text.replace('\n', ' ')
        full_text = ' '.join(full_text.split())  # Remove extra whitespace
        
        return full_text, len(transcript_list)
        
    except TranscriptsDisabled:
        return None, "Transcripts disabled"
    except NoTranscriptFound:
        return None, "No transcript found"
    except VideoUnavailable:
        return None, "Video unavailable"
    except Exception as e:
        return None, f"Error: {str(e)}"

def main(
    metadata_file="data/raw/metadata.csv",
    output_file="data/processed/sample_transcripts.csv",
    sample_size=100,
    min_word_count=50
):
    """
    Extract transcripts from YouTube for CMU-MOSEI videos.
    
    Args:
        metadata_file: Path to metadata CSV
        output_file: Path to save extracted transcripts
        sample_size: Number of transcripts to extract
        min_word_count: Minimum words required in transcript
    """
    
    print("=" * 70)
    print("Extracting YouTube Transcripts for CMU-MOSEI")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load metadata
    print(f"\nLoading metadata from {metadata_file}...")
    df_meta = pd.read_csv(metadata_file)
    
    print(f"Total entries in metadata: {len(df_meta):,}")
    
    # Extract unique video IDs
    print("\nExtracting video IDs...")
    df_meta['video_id'] = df_meta['filename'].apply(extract_video_id)
    
    # Filter out non-YouTube IDs (numeric-only IDs)
    df_meta = df_meta[df_meta['video_id'].notna()]
    
    unique_video_ids = df_meta['video_id'].unique()
    print(f"Unique YouTube video IDs: {len(unique_video_ids):,}")
    
    # Sample random videos
    np.random.seed(42)
    sampled_ids = np.random.choice(
        unique_video_ids, 
        size=min(sample_size * 2, len(unique_video_ids)),  # Sample 2x to account for failures
        replace=False
    )
    
    print(f"\nSampling {len(sampled_ids)} videos (target: {sample_size} successful)...")
    
    # Fetch transcripts
    results = []
    successful = 0
    
    with tqdm(total=len(sampled_ids), desc="Fetching transcripts") as pbar:
        for video_id in sampled_ids:
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            
            transcript, error = get_transcript(video_id)
            
            if transcript:
                word_count = len(transcript.split())
                
                if word_count >= min_word_count:
                    results.append({
                        'video_id': video_id,
                        'transcript': transcript,
                        'word_count': word_count,
                        'status': 'success'
                    })
                    successful += 1
                    
                    # Stop if we have enough
                    if successful >= sample_size:
                        pbar.update(len(sampled_ids) - pbar.n)
                        break
            else:
                results.append({
                    'video_id': video_id,
                    'transcript': None,
                    'word_count': 0,
                    'status': error
                })
            
            pbar.update(1)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Filter successful ones
    df_successful = df_results[df_results['status'] == 'success'].copy()
    df_successful = df_successful.drop('status', axis=1)
    df_successful = df_successful.reset_index(drop=True)
    
    # Save to CSV
    df_successful.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("Results:")
    print('='*70)
    print(f"Successfully extracted: {len(df_successful)}/{len(sampled_ids)}")
    print(f"Success rate: {len(df_successful)/len(sampled_ids)*100:.1f}%")
    
    # Show failure breakdown
    failure_counts = df_results[df_results['status'] != 'success']['status'].value_counts()
    if len(failure_counts) > 0:
        print(f"\nFailure breakdown:")
        for status, count in failure_counts.items():
            print(f"  {status}: {count}")
    
    print(f"\nSaved to: {output_file}")
    
    # Print statistics
    if len(df_successful) > 0:
        print("\n" + "=" * 70)
        print("Statistics:")
        print("=" * 70)
        print(f"Sample size: {len(df_successful)}")
        print(f"Word count - Mean: {df_successful['word_count'].mean():.1f}")
        print(f"Word count - Min: {df_successful['word_count'].min()}")
        print(f"Word count - Max: {df_successful['word_count'].max()}")
        print("=" * 70)
        
        # Show first few examples
        print("\nSample transcripts (first 100 chars):")
        for i in range(min(3, len(df_successful))):
            print(f"\n{i+1}. Video: {df_successful.iloc[i]['video_id']}")
            print(f"   Words: {df_successful.iloc[i]['word_count']}")
            print(f"   Text: {df_successful.iloc[i]['transcript'][:100]}...")
    
    return df_successful

if __name__ == "__main__":
    main(sample_size=100, min_word_count=50)
