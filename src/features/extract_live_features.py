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
        Extract simplified acoustic features from audio file using Librosa
        
        Extracts ~50 key features that approximate COVAREP features:
        - Pitch (F0) statistics
        - Spectral features
        - MFCCs
        - Energy features
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict of acoustic features
        """
        print("\n🎵 Extracting acoustic features...")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        features = {}
        
        # 1. Pitch (F0) features - most important for voice
        f0 = librosa.yin(y, fmin=50, fmax=400)  # F0 estimation
        f0_valid = f0[f0 > 0]  # Remove unvoiced frames
        
        if len(f0_valid) > 0:
            features['f0_mean'] = np.mean(f0_valid)
            features['f0_std'] = np.std(f0_valid)
            features['f0_min'] = np.min(f0_valid)
            features['f0_max'] = np.max(f0_valid)
            features['f0_median'] = np.median(f0_valid)
            features['f0_q25'] = np.percentile(f0_valid, 25)
            features['f0_q75'] = np.percentile(f0_valid, 75)
        else:
            for stat in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']:
                features[f'f0_{stat}'] = 0
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        for name, values in [('centroid', spectral_centroids), 
                              ('rolloff', spectral_rolloff),
                              ('bandwidth', spectral_bandwidth)]:
            features[f'spectral_{name}_mean'] = np.mean(values)
            features[f'spectral_{name}_std'] = np.std(values)
            features[f'spectral_{name}_min'] = np.min(values)
            features[f'spectral_{name}_max'] = np.max(values)
        
        # 3. Zero-crossing rate (voice quality indicator)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 4. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 5. MFCCs (13 coefficients, most important for speech)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        print(f"✅ Extracted {len(features)} acoustic features")
        return features
    
    def extract_all_features(self, audio_path, transcript):
        """
        Extract all features from audio and transcript
        
        Args:
            audio_path: Path to audio file
            transcript: Transcribed text
            
        Returns:
            Dict with all features
        """
        print("\n" + "=" * 70)
        print("EXTRACTING FEATURES FOR LIVE ASSESSMENT")
        print("=" * 70)
        
        # Acoustic features
        acoustic_features = self.extract_acoustic_features(audio_path)
        
        # Lexical features
        print("\n📝 Extracting lexical features...")
        lexical_features = extract_lexical(transcript)
        print(f"✅ Extracted {len(lexical_features)} lexical features")
        
        # Combine
        all_features = {**acoustic_features, **lexical_features}
        
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
