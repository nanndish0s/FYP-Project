"""
Live Assessment Pipeline
Complete end-to-end workflow for assessing candidates via live audio
"""
import sys
import os
import pickle
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.audio.record_audio import AudioRecorder
from src.audio.transcribe_audio import SpeechTranscriber
from src.features.extract_live_features import LiveFeatureExtractor

class LiveAssessmentPipeline:
    def __init__(self):
        """Initialize the live assessment pipeline"""
        print("=" * 80)
        print("INITIALIZING LIVE ASSESSMENT PIPELINE")
        print("=" * 80)
        
        # Initialize components
        self.recorder = AudioRecorder(sample_rate=16000)
        self.transcriber = SpeechTranscriber(model_size="base")
        self.feature_extractor = LiveFeatureExtractor()
        
        # Load trained models
        print("\n📦 Loading trained models...")
        self.models = {}
        for trait in ['curiosity', 'critical_thinking', 'creativity']:
            model_file = f'models/{trait}_model_44.pkl'
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    self.models[trait] = pickle.load(f)
                print(f"   ✅ Loaded {trait} model")
            else:
                # Try original 23-sample models
                model_file_old = f'models/{trait}_score_model.pkl'
                if os.path.exists(model_file_old):
                    with open(model_file_old, 'rb') as f:
                        self.models[trait] = pickle.load(f)
                    print(f"   ✅ Loaded {trait} model (23-sample version)")
        
        # Load feature names from training data
        print("\n📊 Loading feature schema...")
        ml_dataset_path = 'data/processed/ml_ready_dataset_44.csv'
        if not os.path.exists(ml_dataset_path):
            ml_dataset_path = 'data/processed/ml_ready_dataset.csv'
        
        df_train = pd.read_csv(ml_dataset_path)
        self.feature_cols = [col for col in df_train.columns 
                            if col not in ['video_id', 'transcript', 'word_count',
                                          'curiosity_score', 'critical_thinking_score', 
                                          'creativity_score', 'curiosity_reasoning',
                                          'critical_thinking_reasoning', 'creativity_reasoning']]
        print(f"   Expected features: {len(self.feature_cols)}")
        
        # Calculate feature means for imputation
        self.feature_means = df_train[self.feature_cols].mean().to_dict()
        
        print("\n✅ Pipeline ready!")
        
    def record_and_assess(self, duration=60, output_dir="data/temp"):
        """
        Record audio and assess candidate
        
        Args:
            duration: Recording duration in seconds
            output_dir: Directory to save temporary files
            
        Returns:
            Dict with assessment results
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, "live_recording.wav")
        
        print("\n" + "=" * 80)
        print("LIVE CANDIDATE ASSESSMENT")
        print("=" * 80)
        
        # Step 1: Record audio
        print(f"\n🎤 STEP 1: Recording ({duration} seconds)")
        print("="*80)
        self.recorder.record(duration=duration, output_path=audio_path)
        
        # Step 2: Transcribe
        print(f"\n📝 STEP 2: Transcription")
        print("="*80)
        transcript = self.transcriber.transcribe(audio_path)
        print(f"\nTranscript:\n{transcript[:200]}...")
        
        # Step 3: Extract features
        print(f"\n🔬 STEP 3: Feature Extraction")
        print("="*80)
        features = self.feature_extractor.extract_all_features(audio_path, transcript)
        
        # Step 4: Prepare features for prediction
        print(f"\n🧮 STEP 4: Preparing for Prediction")
        print("="*80)
        feature_vector = self._prepare_features(features)
        
        # Step 5: Predict C3 scores
        print(f"\n🎯 STEP 5: Predicting C3 Scores")
        print("="*80)
        predictions = {}
        for trait_name, model in self.models.items():
            score = model.predict(feature_vector)[0]
            # Clip to valid range
            score = np.clip(score, 1, 5)
            predictions[trait_name] = score
            print(f"   {trait_name.replace('_', ' ').title()}: {score:.2f}/5")
        
        # Compile results
        results = {
            'transcript': transcript,
            'word_count': len(transcript.split()),
            'audio_path': audio_path,
            'predictions': predictions,
            'features': features
        }
        
        print("\n" + "=" * 80)
        print("✅ ASSESSMENT COMPLETE!")
        print("=" * 80)
        
        return results
    
    def _prepare_features(self, live_features):
        """
        Prepare live features for model prediction
        
        Handles feature mismatch between live extraction (~50 features)
        and training data (539 features) by imputing missing features
        
        Args:
            live_features: Dict of extracted live features
            
        Returns:
            Feature vector ready for prediction
        """
        feature_vector = []
        
        for feature_name in self.feature_cols:
            if feature_name in live_features:
                # Use extracted feature
                feature_vector.append(live_features[feature_name])
            else:
                # Impute with training data mean
                feature_vector.append(self.feature_means.get(feature_name, 0))
        
        # Reshape for sklearn
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        print(f"   ✅ Feature vector prepared: {feature_vector.shape}")
        print(f"   Live features used: {len([f for f in self.feature_cols if f in live_features])}/{len(self.feature_cols)}")
        print(f"   Imputed features: {len([f for f in self.feature_cols if f not in live_features])}/{len(self.feature_cols)}")
        
        return feature_vector

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = LiveAssessmentPipeline()
    
    # Test microphone
    print("\n" + "=" * 80)
    print("MICROPHONE TEST")
    print("=" * 80)
    pipeline.recorder.test_microphone(duration=3)
    
    # Run live assessment
    input("\nPress Enter to start 60-second assessment...")
    results = pipeline.record_and_assess(duration=60)
    
    print(f"\n\n📊 FINAL RESULTS:")
    print("="*80)
    print(f"Transcript: {results['transcript'][:100]}...")
    print(f"Word count: {results['word_count']}")
    print(f"\nC3 Scores:")
    for trait, score in results['predictions'].items():
        print(f"  {trait.replace('_', ' ').title()}: {score:.2f}/5")
