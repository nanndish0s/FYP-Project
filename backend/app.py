"""
Flask Backend API for Voice-Based XAI Recruitment System
RESTful API for C3 predictions and SHAP explanations
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import only what we need without triggering module-level execution
from src.audio.transcribe_audio import SpeechTranscriber

# We'll import feature extraction functions directly instead of the class
import librosa
import parselmouth

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load models and data on startup
print("🚀 Loading models and data...")

# Load ML models
models = {}
models_dir = project_root / 'models'
for trait in ['curiosity', 'critical_thinking', 'creativity']:
    model_file = models_dir / f'{trait}_model_44.pkl'
    if not model_file.exists():
        model_file = models_dir / f'{trait}_score_model.pkl'
    with open(model_file, 'rb') as f:
        models[trait] = pickle.load(f)

# Load candidate data
data_dir = project_root / 'data' / 'processed'
df_candidates = pd.read_csv(data_dir / 'combined_44_with_c3_labels.csv')

# Try to load 44-sample dataset, fall back to original
ml_dataset_file = data_dir / 'ml_ready_dataset_44.csv'
if not ml_dataset_file.exists():
    ml_dataset_file = data_dir / 'ml_ready_dataset.csv'
df_ml = pd.read_csv(ml_dataset_file)

# Get feature columns
feature_cols = [col for col in df_ml.columns 
               if col not in ['video_id', 'transcript', 'word_count',
                             'curiosity_score', 'critical_thinking_score', 
                             'creativity_score', 'curiosity_reasoning',
                             'critical_thinking_reasoning', 'creativity_reasoning']]

# Calculate feature means for imputation
feature_means = df_ml[feature_cols].mean().to_dict()

print(f"✅ Loaded {len(models)} models")
print(f"✅ Loaded {len(df_candidates)} candidates")
print(f"✅ API ready!")

# ==================== ROUTES ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': len(models)})

@app.route('/api/candidates', methods=['GET'])
def get_candidates():
    """Get list of all pre-recorded candidates"""
    candidates = []
    for _, row in df_candidates.iterrows():
        candidates.append({
            'id': row['video_id'],
            'transcript': row['transcript'][:200] + '...',  # Preview
            'word_count': int(row['word_count']),
            'scores': {
                'curiosity': float(row['curiosity_score']),
                'critical_thinking': float(row['critical_thinking_score']),
                'creativity': float(row['creativity_score'])
            }
        })
    return jsonify(candidates)

@app.route('/api/candidate/<candidate_id>', methods=['GET'])
def get_candidate(candidate_id):
    """Get full candidate details"""
    candidate = df_candidates[df_candidates['video_id'] == candidate_id]
    if len(candidate) == 0:
        return jsonify({'error': 'Candidate not found'}), 404
    
    row = candidate.iloc[0]
    return jsonify({
        'id': row['video_id'],
        'transcript': row['transcript'],
        'word_count': int(row['word_count']),
        'scores': {
            'curiosity': float(row['curiosity_score']),
            'critical_thinking': float(row['critical_thinking_score']),
            'creativity': float(row['creativity_score'])
        },
        'reasoning': {
            'curiosity': row['curiosity_reasoning'],
            'critical_thinking': row['critical_thinking_reasoning'],
            'creativity': row['creativity_reasoning']
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict C3 scores from features"""
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        
        predictions = {}
        for trait_name, model in models.items():
            score = model.predict(features)[0]
            score = np.clip(score, 1, 5)
            predictions[trait_name] = float(score)
        
        # Calculate recommendation
        avg_score = np.mean(list(predictions.values()))
        if avg_score >= 4.0:
            recommendation = "STRONG_HIRE"
        elif avg_score >= 3.5:
            recommendation = "RECOMMENDED"
        elif avg_score >= 3.0:
            recommendation = "CONSIDER"
        else:
            recommendation = "NOT_RECOMMENDED"
        
        return jsonify({
            'scores': predictions,
            'average': float(avg_score),
            'recommendation': recommendation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/assess-audio', methods=['POST'])
def assess_audio():
    """Complete assessment from audio file"""
    try:
        # Save uploaded audio file
        audio_file = request.files['audio']
        temp_dir = project_root / 'data' / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        import uuid
        unique_id = uuid.uuid4().hex
        audio_path = temp_dir / f'upload_{unique_id}.wav'
        audio_file.save(str(audio_path))
        
        # Decide path to use for transcription
        processing_path = str(audio_path)
        
        # Check if file is valid WAV readable by soundfile (no conversion needed)
        try:
            import soundfile as sf
            # Try to read header to verify
            sf.info(str(audio_path))
            print("✅ Valid WAV format detected, skipping conversion.")
        except Exception:
            # Conversion attempt if efficient reading fails
            print("⚠️ Format requires conversion...")
            try:
                from pydub import AudioSegment
                compliant_wav_path = temp_dir / f'compliant_{unique_id}.wav'
                
                audio = AudioSegment.from_file(str(audio_path))
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(str(compliant_wav_path), format="wav")
                
                processing_path = str(compliant_wav_path)
            except Exception as e:
                print(f"Error converting audio: {e}")
                # We won't return 400 here, we'll try to proceed with original path as last resort
                # or let it fail downstream with standard 500
        
        
        # Transcribe
        print(f"🎤 Starting transcription process...")
        print(f"   File path: {processing_path}")
        print(f"   File exists: {os.path.exists(processing_path)}")
        if os.path.exists(processing_path):
            print(f"   File size: {os.path.getsize(processing_path)} bytes")
        
        try:
            print("   Creating transcriber...")
            transcriber = SpeechTranscriber(model_size="base")
            print("   Calling transcribe()...")
            transcript = transcriber.transcribe(processing_path)
            print(f"✅ Transcription complete: {len(transcript)} chars")
        except Exception as transcribe_error:
            print(f"❌ Transcription failed: {transcribe_error}")
            import traceback
            traceback.print_exc()
            raise
        
        # Extract simplified features (mimicking LiveFeatureExtractor)
        features = {}
        
        # Acoustic features using librosa (loading the clean WAV)
        try:
            y, sr = librosa.load(processing_path, sr=16000)
            
            # Basic acoustic features
            features['pitch_mean'] = float(np.mean(librosa.yin(y, fmin=50, fmax=500)))
            features['energy_mean'] = float(np.mean(librosa.feature.rms(y=y)))
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            
        except Exception as e:
            print(f"Warning: Acoustic feature extraction failed: {e}")
        
        # Lexical features from transcript
        words = transcript.lower().split()
        sentences = [s.strip() for s in transcript.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['vocab_diversity'] = len(set(words)) / len(words) if words else 0
        
        # Filler words
        filler_words = {'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically'}
        filler_count = sum(1 for w in words if w in filler_words)
        features['filler_word_count'] = filler_count
        features['filler_word_ratio'] = filler_count / len(words) if words else 0
        
        # Prepare feature vector with imputation
        feature_vector = []
        for feature_name in feature_cols:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(feature_means.get(feature_name, 0))
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Predict
        predictions = {}
        for trait_name, model in models.items():
            score = model.predict(feature_vector)[0]
            score = np.clip(score, 1, 5)
            predictions[trait_name] = float(score)
        
        # Calculate recommendation
        avg_score = np.mean(list(predictions.values()))
        if avg_score >= 4.0:
            recommendation = "STRONG_HIRE"
        elif avg_score >= 3.5:
            recommendation = "RECOMMENDED"
        elif avg_score >= 3.0:
            recommendation = "CONSIDER"
        else:
            recommendation = "NOT_RECOMMENDED"
        
        return jsonify({
            'transcript': transcript,
            'word_count': len(words),
            'scores': predictions,
            'average': float(avg_score),
            'recommendation': recommendation
        })
    
    except Exception as e:
        print(f"❌ Error in assess_audio: {e}")
        import traceback
        traceback.print_exc()
        # Return full traceback in development to help debugging
        tb_str = traceback.format_exc()
        return jsonify({'error': str(e), 'traceback': tb_str}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
