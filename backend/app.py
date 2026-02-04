"""
Flask Backend API for Voice-Based AI Recruitment System
RESTful API for C3 skill predictions
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
    # Prioritize RecruitView production models (2,011 samples)
    model_file = models_dir / f'{trait}_model_recruitview.pkl'
    if not model_file.exists():
        # Fallback to 44-sample prototype
        model_file = models_dir / f'{trait}_model_44.pkl'
    if not model_file.exists():
        model_file = models_dir / f'{trait}_score_model.pkl'
        
    print(f"   Loading {trait} model from {model_file.name}")
    with open(model_file, 'rb') as f:
        models[trait] = pickle.load(f)

# Load candidate data (RecruitView based)
data_dir = project_root / 'data' / 'processed'
recruitview_meta = data_dir / 'recruitview_metadata.csv'
if recruitview_meta.exists():
    df_candidates = pd.read_csv(recruitview_meta)
    # Ensure column names match what the frontend expects
    # (The frontend expects Curiosity, Critical Thinking, Creativity)
    if 'curiosity_score' in df_candidates.columns:
        df_candidates = df_candidates.rename(columns={
            'curiosity_score': 'curiosity_score',
            'critical_thinking_score': 'critical_thinking_score',
            'creativity_score': 'creativity_score'
        })
else:
    df_candidates = pd.read_csv(data_dir / 'combined_44_with_c3_labels.csv')

# Load the reference feature dataset for column alignment and imputation
ml_dataset_file = data_dir / 'recruitview_features_all.csv'
if not ml_dataset_file.exists():
    ml_dataset_file = data_dir / 'ml_ready_dataset.csv'
df_ml = pd.read_csv(ml_dataset_file)

# Get feature columns (37 features for RecruitView models)
exclude_cols = ['video_id', 'transcript', 'curiosity_score', 'critical_thinking_score', 
                'creativity_score', 'curiosity_reasoning', 'critical_thinking_reasoning', 
                'creativity_reasoning', 'file_name']
feature_cols = [col for col in df_ml.columns if col not in exclude_cols]

# Calculate feature means for imputation
feature_means = df_ml[feature_cols].mean().to_dict()

print(f"✅ Loaded {len(models)} models")
print(f"✅ Loaded {len(df_candidates)} candidates")
print(f"✅ Feature set size: {len(feature_cols)}")
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
        # Handle word count safely
        word_count = int(row['word_count']) if 'word_count' in row else len(str(row['transcript']).split())
        
        candidates.append({
            'id': row['video_id'],
            'transcript': str(row['transcript'])[:200] + '...',  # Preview
            'word_count': word_count,
            'scores': {
                'curiosity': float(row.get('curiosity_score', 0)),
                'critical_thinking': float(row.get('critical_thinking_score', 0)),
                'creativity': float(row.get('creativity_score', 0))
            }
        })
    return jsonify(candidates)

@app.route('/api/candidate/<candidate_id>', methods=['GET'])
def get_candidate(candidate_id):
    """Get full candidate details"""
    candidate = df_candidates[df_candidates['video_id'].astype(str) == str(candidate_id)]
    if len(candidate) == 0:
        return jsonify({'error': f'Candidate {candidate_id} not found'}), 404
    
    row = candidate.iloc[0]
    word_count = int(row['word_count']) if 'word_count' in row else len(str(row['transcript']).split())
    
    return jsonify({
        'id': int(row['video_id']),
        'transcript': row['transcript'],
        'word_count': word_count,
        'scores': {
            'curiosity': float(row.get('curiosity_score', 0)),
            'critical_thinking': float(row.get('critical_thinking_score', 0)),
            'creativity': float(row.get('creativity_score', 0))
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
        
        # Extract comprehensive features (Aligned with RecruitView production pipeline)
        features = {}
        
        # Acoustic features using librosa (loading at 16kHz)
        try:
            y, sr = librosa.load(processing_path, sr=16000)
            
            # 1. Pitch
            try:
                pitches = librosa.yin(y, fmin=50, fmax=500)
                features['pitch_mean'] = float(np.mean(pitches))
                features['pitch_std'] = float(np.std(pitches))
            except:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                
            # 2. Energy
            rms = librosa.feature.rms(y=y)
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            
            # 3. MFCCs (13 coeffs, mean and std)
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
            
        except Exception as e:
            print(f"❌ Acoustic extraction failed: {e}")
        
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
        features['filler_word_ratio'] = filler_count / len(words) if words else 0
        
        # Verbose Logging for Debugging
        print("\n📊 FEATURE SUMMARY FOR PREDICTION:")
        print(f"   - Pitch Mean: {features.get('pitch_mean', 0):.2f} Hz (Training Mean: ~205)")
        print(f"   - Energy Mean: {features.get('energy_mean', 0):.4f} (Training Mean: ~0.04)")
        print(f"   - Vocab Diversity: {features.get('vocab_diversity', 0):.4f} (Training Mean: ~0.74)")
        print(f"   - Word Count: {features.get('word_count', 0)}")
        print(f"   - Sentence Count: {features.get('sentence_count', 0)}")
        
        # Prepare feature vector as DataFrame to ensure perfect feature name alignment
        df_feats = pd.DataFrame([features])
        # Add missing features with means (imputation)
        for col in feature_cols:
            if col not in df_feats.columns:
                df_feats[col] = feature_means.get(col, 0)
        
        # Ensure exact column order and selection expected by the models
        df_feats = df_feats[feature_cols]
        
        # Predict
        predictions = {}
        for trait_name, model in models.items():
            # Use DataFrame to avoid "X does not have valid feature names" warnings
            score = model.predict(df_feats)[0]
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
