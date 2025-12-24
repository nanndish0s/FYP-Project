"""
MASS Transcript Extraction from CMU-MOSEI (ALL Videos)
Processes ALL ~3,800 video IDs to maximize dataset size
Includes progress saving, batch processing, and error recovery
"""
import pandas as pd
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from tqdm import tqdm
import time
import os
import json
from datetime import datetime

class MassTranscriptExtractor:
    def __init__(self, 
                 metadata_file="data/raw/metadata.csv",
                 output_file="data/processed/all_transcripts.csv",
                 progress_file="data/processed/extraction_progress.json",
                 min_word_count=50,
                 batch_size=100):
        """
        Initialize mass transcript extractor
        
        Args:
            metadata_file: CMU-MOSEI metadata CSV
            output_file: Where to save successful transcripts
            progress_file: Track extraction progress (for resuming)
            min_word_count: Minimum words per transcript
            batch_size: Save progress every N videos
        """
        self.metadata_file = metadata_file
        self.output_file = output_file
        self.progress_file = progress_file
        self.min_word_count = min_word_count
        self.batch_size = batch_size
        
        # Statistics
        self.stats = {
            'total_attempted': 0,
            'successful': 0,
            'transcripts_disabled': 0,
            'no_transcript': 0,
            'video_unavailable': 0,
            'other_errors': 0
        }
        
    def extract_video_id(self, filename):
        """Extract YouTube video ID from CMU-MOSEI filename"""
        parts = filename.split('_')
        if len(parts) >= 2:
            video_id = parts[0]
            # YouTube IDs contain letters (not purely numeric)
            if any(c.isalpha() for c in video_id):
                return video_id
        return None
    
    def get_transcript(self, video_id):
        """Fetch transcript from YouTube"""
        try:
            ytt_api = YouTubeTranscriptApi()
            fetched_transcript = ytt_api.fetch(video_id)
            transcript_list = fetched_transcript.to_raw_data()
            
            # Combine all text
            full_text = ' '.join([entry['text'] for entry in transcript_list])
            full_text = full_text.replace('\n', ' ')
            full_text = ' '.join(full_text.split())
            
            return full_text, "success"
            
        except TranscriptsDisabled:
            return None, "transcripts_disabled"
        except NoTranscriptFound:
            return None, "no_transcript"
        except VideoUnavailable:
            return None, "video_unavailable"
        except Exception as e:
            return None, f"error_{str(e)[:30]}"
    
    def load_progress(self):
        """Load extraction progress if exists"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                print(f"\n📂 Found previous progress: {data['processed_count']} videos already processed")
                return set(data['processed_ids']), data['stats']
        return set(), None
    
    def save_progress(self, processed_ids, partial_results):
        """Save current progress"""
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'processed_count': len(processed_ids),
            'processed_ids': list(processed_ids),
            'stats': self.stats
        }
        
        # Save progress file
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f)
        
        # Save partial results
        if partial_results:
            df_partial = pd.DataFrame(partial_results)
            df_partial.to_csv(self.output_file, index=False)
            print(f"\n💾 Progress saved: {len(partial_results)} transcripts so far")
    
    def extract_all(self):
        """Main extraction loop - process ALL videos"""
        print("=" * 80)
        print("MASS TRANSCRIPT EXTRACTION - CMU-MOSEI")
        print("=" * 80)
        
        # Load metadata
        print(f"\n📊 Loading metadata from {self.metadata_file}...")
        df_meta = pd.read_csv(self.metadata_file)
        print(f"   Total entries: {len(df_meta):,}")
        
        # Extract video IDs
        print("\n🎬 Extracting YouTube video IDs...")
        df_meta['video_id'] = df_meta['filename'].apply(self.extract_video_id)
        df_meta = df_meta[df_meta['video_id'].notna()]
        
        unique_video_ids = df_meta['video_id'].unique()
        print(f"   Unique YouTube IDs found: {len(unique_video_ids):,}")
        
        # Load previous progress
        processed_ids, loaded_stats = self.load_progress()
        if loaded_stats:
            self.stats = loaded_stats
        
        # Filter out already processed
        remaining_ids = [vid for vid in unique_video_ids if vid not in processed_ids]
        print(f"\n🔄 Videos to process: {len(remaining_ids):,}")
        
        if len(remaining_ids) == 0:
            print("✅ All videos already processed!")
            return pd.read_csv(self.output_file)
        
        # Load existing results
        results = []
        if os.path.exists(self.output_file):
            existing_df = pd.read_csv(self.output_file)
            results = existing_df.to_dict('records')
            print(f"   Loaded {len(results)} existing transcripts")
        
        # Process all remaining videos
        print(f"\n🚀 Starting mass extraction...")
        print(f"   Estimated time: {len(remaining_ids) * 0.6 / 60:.1f} minutes")
        print(f"   (0.5s delay + 0.1s processing per video)\n")
        
        with tqdm(total=len(remaining_ids), desc="Extracting") as pbar:
            for i, video_id in enumerate(remaining_ids):
                # Rate limiting
                time.sleep(0.5)
                
                # Fetch transcript
                transcript, status = self.get_transcript(video_id)
                
                self.stats['total_attempted'] += 1
                
                if transcript:
                    word_count = len(transcript.split())
                    
                    if word_count >= self.min_word_count:
                        results.append({
                            'video_id': video_id,
                            'transcript': transcript,
                            'word_count': word_count
                        })
                        self.stats['successful'] += 1
                else:
                    # Track error types
                    if 'disabled' in status:
                        self.stats['transcripts_disabled'] += 1
                    elif 'no_transcript' in status:
                        self.stats['no_transcript'] += 1
                    elif 'unavailable' in status:
                        self.stats['video_unavailable'] += 1
                    else:
                        self.stats['other_errors'] += 1
                
                processed_ids.add(video_id)
                pbar.update(1)
                
                # Save progress periodically
                if (i + 1) % self.batch_size == 0:
                    self.save_progress(processed_ids, results)
                    pbar.set_postfix({
                        'success': self.stats['successful'],
                        'rate': f"{self.stats['successful']/(self.stats['total_attempted'])*100:.1f}%"
                    })
        
        # Final save
        self.save_progress(processed_ids, results)
        
        # Create final DataFrame
        df_final = pd.DataFrame(results)
        df_final.to_csv(self.output_file, index=False)
        
        # Print final statistics
        self.print_statistics(df_final, len(unique_video_ids))
        
        return df_final
    
    def print_statistics(self, df, total_videos):
        """Print extraction statistics"""
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE!")
        print("=" * 80)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total videos in dataset: {total_videos:,}")
        print(f"   Videos attempted: {self.stats['total_attempted']:,}")
        print(f"   ✅ Successful extractions: {self.stats['successful']:,}")
        print(f"   Success rate: {self.stats['successful']/self.stats['total_attempted']*100:.1f}%")
        
        print(f"\n❌ Failure Breakdown:")
        print(f"   Transcripts disabled: {self.stats['transcripts_disabled']:,}")
        print(f"   No transcript found: {self.stats['no_transcript']:,}")
        print(f"   Video unavailable: {self.stats['video_unavailable']:,}")
        print(f"   Other errors: {self.stats['other_errors']:,}")
        
        if len(df) > 0:
            print(f"\n📝 Transcript Statistics:")
            print(f"   Total transcripts: {len(df):,}")
            print(f"   Average word count: {df['word_count'].mean():.1f}")
            print(f"   Median word count: {df['word_count'].median():.1f}")
            print(f"   Min word count: {df['word_count'].min():,}")
            print(f"   Max word count: {df['word_count'].max():,}")
            
            print(f"\n   Word count distribution:")
            print(f"      50-100 words:   {len(df[df['word_count'].between(50, 100)]):,}")
            print(f"      100-200 words:  {len(df[df['word_count'].between(100, 200)]):,}")
            print(f"      200-500 words:  {len(df[df['word_count'].between(200, 500)]):,}")
            print(f"      500+ words:     {len(df[df['word_count'] >= 500]):,}")
        
        print(f"\n💾 Saved to: {self.output_file}")
        print("=" * 80)
        
        # Show sample transcripts
        if len(df) > 0:
            print(f"\n📄 Sample Transcripts (first 3):")
            for i in range(min(3, len(df))):
                print(f"\n   {i+1}. Video: {df.iloc[i]['video_id']}")
                print(f"      Words: {df.iloc[i]['word_count']:,}")
                print(f"      Preview: {df.iloc[i]['transcript'][:150]}...")

def main():
    """Run mass extraction"""
    extractor = MassTranscriptExtractor(
        metadata_file="data/raw/metadata.csv",
        output_file="data/processed/all_transcripts.csv",
        progress_file="data/processed/extraction_progress.json",
        min_word_count=50,
        batch_size=100  # Save every 100 videos
    )
    
    df_results = extractor.extract_all()
    
    print(f"\n🎉 Extraction complete! Got {len(df_results):,} transcripts")
    print(f"   Previous dataset: 23 samples")
    print(f"   New dataset: {len(df_results):,} samples")
    print(f"   Increase: {len(df_results) - 23:,} samples ({(len(df_results)/23)*100:.0f}% of original)")
    
    return df_results

if __name__ == "__main__":
    main()
