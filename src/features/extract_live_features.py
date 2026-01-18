"""
Live Feature Extraction Module
Extracts simplified acoustic + lexical features from live audio
Compatible with trained Random Forest models
"""
import librosa
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.features.extract_lexical_features import extract_lexical_features as extract_lexical

class LiveFeatureExtractor:
    def __init__(self):
        """Initialize live feature extractor"""
        self.sample_rate = 16000
        
    def extract_acoustic_features(self, audio_path):
        """
        Extract simplified acoustic features from audio file using Librosa.
        Aligned with mass_extract_recruitview.py (Prioritizes consistency with training data).
        """
        print("\n🎵 Extracting acoustic features...")
        
        # Load audio at 16kHz
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        features = {}
        
        # 1. Pitch (Acoustic) - Use 50-500Hz range and include unvoiced frames in mean (matches training)
        try:
            pitches = librosa.yin(y, fmin=50, fmax=500)
            features['pitch_mean'] = float(np.mean(pitches))
            features['pitch_std'] = float(np.std(pitches))
        except:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        
        # 3. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        
        # 4. Energy (RMS) features
        rms = librosa.feature.rms(y=y)
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        
        # 5. MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        print(f"✅ Extracted {len(features)} acoustic features")
        return features
    
    def extract_lexical_features(self, transcript):
        """
        Extracts lexical features using same logic as training (split() based).
        """
        if not transcript or not isinstance(transcript, str):
            defaults = ['word_count', 'sentence_count', 'avg_word_length', 'vocab_diversity', 'filler_word_ratio']
            return {k: 0 for k in defaults}
            
        words = transcript.lower().split()
        sentences = [s.strip() for s in transcript.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        word_count = len(words)
        if word_count == 0:
            return {'word_count':0, 'sentence_count':0, 'avg_word_length':0, 'vocab_diversity':0, 'filler_word_ratio':0}
            
        filler_words = {'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically'}
        filler_count = sum(1 for w in words if w in filler_words)
        
        return {
            'word_count': word_count,
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]),
            'vocab_diversity': len(set(words)) / word_count,
            'filler_word_ratio': filler_count / word_count
        }

    def extract_all_features(self, audio_path, transcript):
        """
        Extract all features from audio and transcript
        """
        print("\n" + "=" * 70)
        print("EXTRACTING FEATURES FOR LIVE ASSESSMENT")
        print("=" * 70)
        
        # Acoustic features
        acoustic_features = self.extract_acoustic_features(audio_path)
        
        # Lexical features
        print("\n📝 Extracting lexical features...")
        lexical_features = self.extract_lexical_features(transcript)
        print(f"✅ Extracted {len(lexical_features)} lexical features")
        
        # Combine
        all_features = {**acoustic_features, **lexical_features}
        
        # Verbose Logging for Debugging
        print("\n📊 FEATURE SUMMARY FOR PREDICTION:")
        print(f"   - Pitch Mean: {all_features.get('pitch_mean', 0):.2f} Hz (Training Mean: ~205)")
        print(f"   - Energy Mean: {all_features.get('energy_mean', 0):.4f} (Training Mean: ~0.04)")
        print(f"   - Vocab Diversity: {all_features.get('vocab_diversity', 0):.4f} (Training Mean: ~0.74)")
        print(f"   - Word Count: {all_features.get('word_count', 0)}")
        
        print(f"\n✅ Total features: {len(all_features)}")
        print("=" * 70)
        
        return all_features

# Example usage
if __name__ == "__main__":
    extractor = LiveFeatureExtractor()
    
    # Test on sample audio
    # features = extractor.extract_all_features(
    #     audio_path="data/temp/test_recording.wav",
    #     transcript="This is a test transcript"
    # )
    # print(f"\nExtracted {len(features)} features")
