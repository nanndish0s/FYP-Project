import os
import pandas as pd
import numpy as np
import librosa
import parselmouth
from pydub import AudioSegment
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import json
import warnings
from static_ffmpeg import add_paths

# Add static-ffmpeg paths to environment
add_paths()

warnings.filterwarnings('ignore')

def extract_features_from_audio(audio_path):
    """
    Extracts acoustic features from a WAV file using librosa and parselmouth.
    Consistent with the 'Live Assessment' pipeline.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        features = {}
        
        # 1. Pitch (Acoustic)
        try:
            # Using librosa yin
            pitches = librosa.yin(y, fmin=50, fmax=500)
            features['pitch_mean'] = float(np.mean(pitches))
            features['pitch_std'] = float(np.std(pitches))
        except:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            
        # 2. Energy/RMS
        rms = librosa.feature.rms(y=y)
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        
        # 3. MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
        # 4. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        
        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        
        return features
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def extract_lexical_features(transcript):
    """
    Extracts basic lexical features from text.
    """
    if not transcript or not isinstance(transcript, str):
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'vocab_diversity': 0,
            'filler_word_ratio': 0
        }
        
    words = transcript.lower().split()
    sentences = [s.strip() for s in transcript.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    word_count = len(words)
    if word_count == 0:
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'vocab_diversity': 0,
            'filler_word_ratio': 0
        }
        
    filler_words = {'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically'}
    filler_count = sum(1 for w in words if w in filler_words)
    
    return {
        'word_count': word_count,
        'sentence_count': len(sentences),
        'avg_word_length': np.mean([len(w) for w in words]),
        'vocab_diversity': len(set(words)) / word_count,
        'filler_word_ratio': filler_count / word_count
    }

def process_batch(df_metadata, start=0, limit=10):
    """
    Processes a batch of samples: download -> extract -> delete.
    """
    results = []
    temp_dir = 'data/raw/recruitview_temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    subset = df_metadata.iloc[start:start+limit]
    
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Batch {start}-{start+limit}"):
        video_id = row['video_id']
        file_name = row['file_name']
        
        try:
            # 1. Download MP4
            local_mp4 = hf_hub_download(
                repo_id='AI4A-lab/RecruitView',
                filename=file_name,
                repo_type='dataset',
                local_dir=temp_dir
            )
            
            # 2. Convert to WAV (16kHz, mono)
            wav_path = local_mp4.replace('.mp4', '.wav')
            audio = AudioSegment.from_file(local_mp4)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")
            
            # 3. Extract Acoustic features
            acoustic_feats = extract_features_from_audio(wav_path)
            
            # 4. Extract Lexical features
            lexical_feats = extract_lexical_features(row['transcript'])
            
            if acoustic_feats:
                # Combine all
                combined = {'video_id': video_id}
                combined.update(acoustic_feats)
                combined.update(lexical_feats)
                # Add scores
                combined['curiosity_score'] = row['curiosity_score']
                combined['critical_thinking_score'] = row['critical_thinking_score']
                combined['creativity_score'] = row['creativity_score']
                
                results.append(combined)
            
            # 5. Cleanup
            if os.path.exists(local_mp4): os.remove(local_mp4)
            if os.path.exists(wav_path): os.remove(wav_path)
            
        except Exception as e:
            print(f"Failed to process {video_id}: {e}")
            
    return results

def run_extraction(start_idx=0, batch_size=2011):
    metadata_path = 'data/processed/recruitview_metadata.csv'
    output_path = 'data/processed/recruitview_features_all.csv'
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found: {metadata_path}")
        return
        
    df_meta = pd.read_csv(metadata_path)
    df_meta = df_meta.iloc[start_idx : start_idx + batch_size]
    
    print(f"🚀 Starting mass extraction for {len(df_meta)} samples...")
    print(f"   Range: {start_idx} to {start_idx + len(df_meta)}")
    
    # Check for existing progress
    if os.path.exists(output_path):
        all_results_df = pd.read_csv(output_path)
        existing_ids = set(all_results_df['video_id'].astype(str).tolist())
        print(f"   Found {len(existing_ids)} existing samples in {output_path}")
    else:
        all_results_df = pd.DataFrame()
        existing_ids = set()

    # Process in chunks of 20 to avoid memory buildup and show progress
    chunk_size = 20
    for i in range(0, len(df_meta), chunk_size):
        chunk = df_meta.iloc[i : i + chunk_size]
        # Skip if all in chunk exist
        if all(str(row['video_id']) in existing_ids for _, row in chunk.iterrows()):
            continue
            
        print(f"\n📦 Processing chunk {i//chunk_size + 1}/{(len(df_meta)-1)//chunk_size + 1}...")
        results = process_batch(df_meta, start=i, limit=chunk_size)
        
        if results:
            chunk_df = pd.DataFrame(results)
            # Filter out already existing ids (redundancy check)
            chunk_df = chunk_df[~chunk_df['video_id'].astype(str).isin(existing_ids)]
            all_results_df = pd.concat([all_results_df, chunk_df], ignore_index=True)
            
            # Save progress after each chunk
            all_results_df.to_csv(output_path, index=False)
            print(f"💾 Saved progress. Total: {len(all_results_df)}")
            
            # Update existing_ids for next chunk
            existing_ids.update(chunk_df['video_id'].astype(str).tolist())

    print(f"\n✅ Extraction complete. Total samples: {len(all_results_df)}")
    print(f"📂 Final dataset saved to {output_path}")

if __name__ == "__main__":
    import sys
    start = 0
    limit = 2011
    
    if len(sys.argv) > 1:
        start = int(sys.argv[1])
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])
        
    run_extraction(start, limit)
