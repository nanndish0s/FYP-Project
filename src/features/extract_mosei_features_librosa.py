import pandas as pd
import numpy as np
import librosa
import os
from tqdm import tqdm
import warnings
import static_ffmpeg

# Suppress warnings
warnings.filterwarnings('ignore')

def extract_acoustic_features(audio_path):
    """Same logic as mass_extract_recruitview.py"""
    try:
        y, sr = librosa.load(audio_path)
        
        # 1. Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # 2. Energy/RMS
        rms = np.mean(librosa.feature.rms(y=y))
        
        # 3. MFCCs (20)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # 4. Spectral Centroid
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 5. Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        features = {
            'pitch_mean': pitch,
            'energy_mean': rms,
            'spec_centroid_mean': spec_cent,
            'zcr_mean': zcr
        }
        for i, val in enumerate(mfccs_mean):
            features[f'mfcc_{i+1}_mean'] = val
            
        return features
    except Exception as e:
        print(f"Error extracting acoustic features: {e}")
        return None

def extract_lexical_features(transcript):
    """Same logic as mass_extract_recruitview.py"""
    if not isinstance(transcript, str):
        return None
        
    words = transcript.split()
    word_count = len(words)
    sentence_count = transcript.count('.') + transcript.count('?') + transcript.count('!')
    if sentence_count == 0: sentence_count = 1
    
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    unique_words = len(set(words))
    vocab_diversity = unique_words / word_count if word_count > 0 else 0
    
    filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically']
    filler_count = sum(1 for w in words if w.lower() in filler_words)
    filler_ratio = filler_count / word_count if word_count > 0 else 0
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'vocab_diversity': vocab_diversity,
        'filler_word_ratio': filler_ratio
    }

def main():
    # Load 44-sample MOSEI metadata
    df = pd.read_csv('data/processed/combined_44_with_c3_labels.csv')
    print(f"Aligning features for {len(df)} MOSEI samples...")
    
    all_features = []
    
    # We need access to the audio files for the 44 samples.
    # From previous context, they should be in data/raw/mosei_audio/
    audio_dir = 'data/raw/mosei_audio'
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row['video_id']
        audio_path = os.path.join(audio_dir, f"{video_id}.wav")
        
        if not os.path.exists(audio_path):
            # Try mp4 if wav doesn't exist
            audio_path = os.path.join(audio_dir, f"{video_id}.mp4")
            
        if os.path.exists(audio_path):
            acoustic = extract_acoustic_features(audio_path)
            lexical = extract_lexical_features(row['transcript'])
            
            if acoustic and lexical:
                combined = {
                    'video_id': video_id,
                    'curiosity_score': row['curiosity_score'],
                    'critical_thinking_score': row['critical_thinking_score'],
                    'creativity_score': row['creativity_score'],
                    **acoustic,
                    **lexical
                }
                all_features.append(combined)
        else:
            print(f"Audio not found for {video_id}")
            
    features_df = pd.DataFrame(all_features)
    features_df.to_csv('data/processed/mosei_features_aligned.csv', index=False)
    print(f"✅ Aligned features saved to data/processed/mosei_features_aligned.csv")

if __name__ == "__main__":
    main()
