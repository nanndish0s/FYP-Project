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
import shap
import sqlite3
import json
import uuid
from datetime import datetime
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
CORS(app)  # Enable CORS for frontend

# ==================== DATABASE ====================
DB_PATH = project_root / 'data' / 'sessions.db'

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            recommendation      TEXT,
            overall_average     REAL,
            curiosity_avg       REAL,
            critical_thinking_avg REAL,
            creativity_avg      REAL
        );
        CREATE TABLE IF NOT EXISTS responses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            question_id   TEXT,
            question_text TEXT,
            transcript    TEXT,
            word_count    INTEGER,
            curiosity_score       REAL,
            critical_thinking_score REAL,
            creativity_score      REAL,
            average_score         REAL,
            explanations          TEXT
        );
    ''')
    conn.commit()
    conn.close()
    print(" Database initialised at", DB_PATH)

init_db()

# Human-readable labels for production feature set
FEATURE_LABELS = {
    # Pitch
    'pitch_mean': 'Average Pitch',
    'pitch_std': 'Pitch Variability',
    # Energy
    'energy_mean': 'Average Energy',
    'energy_std': 'Energy Variability',
    # Spectral / ZCR
    'spectral_centroid_mean': 'Voice Brightness',
    'zcr_mean': 'Articulation Rate',
    # Lexical
    'word_count': 'Response Length',
    'sentence_count': 'Sentence Count',
    'avg_word_length': 'Word Complexity',
    'vocab_diversity': 'Vocabulary Richness',
    'filler_word_ratio': 'Hesitation Rate',
    # MFCC means — each index maps to a specific acoustic property
    'mfcc_0_mean': 'Voice Loudness',
    'mfcc_1_mean': 'Voice Brightness',
    'mfcc_2_mean': 'Low Frequency Tone',
    'mfcc_3_mean': 'Formant Structure',
    'mfcc_4_mean': 'Mid Frequency Resonance',
    'mfcc_5_mean': 'Upper Mid Tone',
    'mfcc_6_mean': 'Tonal Texture',
    'mfcc_7_mean': 'Fine Spectral Detail',
    'mfcc_8_mean': 'High Frequency Tone',
    'mfcc_9_mean': 'Harmonic Texture',
    'mfcc_10_mean': 'Upper Harmonic Content',
    'mfcc_11_mean': 'Fine Harmonic Detail',
    'mfcc_12_mean': 'Spectral Clarity',
    # MFCC stds — how much each property varies over time (expressiveness)
    'mfcc_0_std': 'Loudness Variation',
    'mfcc_1_std': 'Brightness Variation',
    'mfcc_2_std': 'Low Tone Variation',
    'mfcc_3_std': 'Formant Variation',
    'mfcc_4_std': 'Resonance Variation',
    'mfcc_5_std': 'Tonal Expressiveness',
    'mfcc_6_std': 'Textural Variation',
    'mfcc_7_std': 'Spectral Variation',
    'mfcc_8_std': 'High Tone Variation',
    'mfcc_9_std': 'Harmonic Variation',
    'mfcc_10_std': 'Upper Harmonic Variation',
    'mfcc_11_std': 'Fine Harmonic Variation',
    'mfcc_12_std': 'Spectral Detail Variation',
}

def get_readable_name(feature_name):
    return FEATURE_LABELS.get(feature_name, feature_name.replace('_', ' ').title())

# Load models and data on startup
print(" Loading models and data...")

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

# Get feature columns — excludes quantity measures (word_count, sentence_count)
# which are retained only for the validity pre-check, not as model inputs
exclude_cols = [
    'video_id', 'transcript', 'curiosity_score', 'critical_thinking_score',
    'creativity_score', 'curiosity_reasoning', 'critical_thinking_reasoning',
    'creativity_reasoning', 'file_name',
    'word_count', 'sentence_count',
]
feature_cols = [col for col in df_ml.columns if col not in exclude_cols]

# Calculate feature means for imputation
feature_means = df_ml[feature_cols].mean().to_dict()

# Load data-driven calibration models (Isotonic Regression, trained via OOF CV)
print(" Loading calibration models...")
calibrators = {}
for trait in ['curiosity', 'critical_thinking', 'creativity']:
    cal_file = models_dir / f'{trait}_calibration.pkl'
    with open(cal_file, 'rb') as f:
        calibrators[trait] = pickle.load(f)
    print(f"   Calibrator ready: {trait}")

# Load SHAP explainers for each C3 model
print(" Loading SHAP explainers...")
explainers = {}
X_background = df_ml[feature_cols].values
for trait_name, model in models.items():
    explainers[trait_name] = shap.TreeExplainer(model, X_background)
    print(f"   SHAP explainer ready: {trait_name}")

# Load training feature statistics for distribution-aware clipping
print(" Loading feature statistics...")
with open(models_dir / 'feature_stats.pkl', 'rb') as f:
    feature_stats = pickle.load(f)
clip_low  = np.array([feature_stats['clip_low'][c]  for c in feature_cols])
clip_high = np.array([feature_stats['clip_high'][c] for c in feature_cols])
print(f"   Feature clip bounds loaded for {len(feature_cols)} features")

print(f" Loaded {len(models)} models")
print(f" Loaded {len(df_candidates)} candidates")
print(f" Feature set size: {len(feature_cols)}")
print(f" API ready!")


def clip_to_training_distribution(df_feats):
    """Clip live recording features to 3-sigma bounds of training distribution.
    Prevents out-of-distribution predictions caused by the domain gap between
    studio training audio and live browser microphone recordings."""
    arr = df_feats.values.copy().astype(float)
    arr = np.clip(arr, clip_low, clip_high)
    return pd.DataFrame(arr, columns=df_feats.columns)


def get_shap_explanation(trait_name, feature_df, top_n=5):
    """Return top N features driving the prediction for a given trait."""
    explainer = explainers[trait_name]
    shap_values = explainer.shap_values(feature_df)

    # Random Forest regressor returns (n_samples, n_features)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])[0]
    else:
        sv = np.array(shap_values)[0]

    results = []
    for feat_name, val in zip(feature_df.columns, sv):
        results.append({
            'feature': get_readable_name(feat_name),
            'impact': round(float(abs(val)), 4),
            'direction': 'positive' if val >= 0 else 'negative',
        })

    results.sort(key=lambda x: x['impact'], reverse=True)
    return results[:top_n]

# ── Research-grounded keyword lists ──────────────────────────────────────────
# Keywords derived from peer-reviewed literature (2020–2025):
# Curiosity:         Boyd & Schwartz (2021), Ceraolo et al. (2025),
#                    Sadler-Smith & Akstinaite (2022)
# Critical Thinking: Yang et al. (2023), Lohmann et al. (2024),
#                    Kharchenko et al. (2025)
# Creativity:        Ahmed & Feist (2021), López Martínez et al. (2024)
RESEARCH_KEYWORDS = {
    # Boyd & Schwartz (2021) — LIWC-22 Curiosity category
    # Ceraolo et al. (2025) — causal inquiry markers
    # Sadler-Smith & Akstinaite (2022) — insight language
    # Kashdan et al. (2020) — Five-Dimensional Curiosity Scale
    # Silvia & Christensen (2020) — curiosity-openness link
    'curiosity': [
        # Causal inquiry (Ceraolo et al., 2025)
        'why', 'how does', 'what if', 'what causes', 'how come',
        # LIWC Curiosity category (Boyd & Schwartz, 2021)
        'curious', 'wonder', 'discover', 'explore', 'fascinate',
        'interest', 'find out', 'investigate', 'question',
        # Insight language (Sadler-Smith & Akstinaite, 2022)
        'realize', 'notice', 'understand', 'effect',
        # Curiosity-openness markers (Kashdan et al., 2020)
        'independent', 'motivated', 'learned', 'passion',
        'outside', 'self-directed', 'sought', 'keen',
        # Exploratory behaviour (Silvia & Christensen, 2020)
        'stretch', 'new knowledge', 'dig deeper', 'find out why',
        'wanted to know', 'read about', 'looked into',
    ],
    # Yang et al. (2023) — argument mining markers
    # Lohmann et al. (2024) — discourse markers in argumentation
    # Kharchenko et al. (2025) — hedging as epistemic nuance
    # Koutsoumpis et al. (2022) — conscientiousness language
    'critical_thinking': [
        # Claim markers (Yang et al., 2023)
        'i think', 'i believe', 'in my opinion', 'i argue', 'i concluded',
        # Causal/reasoning (Yang et al., 2023; Lohmann et al., 2024)
        'because', 'therefore', 'thus', 'hence', 'since',
        'as a result', 'this means', 'which led to',
        # Counterargument markers (Yang et al., 2023)
        'however', 'although', 'nevertheless', 'despite',
        'on the other hand', 'in contrast', 'yet',
        # Evidence markers (Yang et al., 2023)
        'for example', 'for instance', 'according to',
        'this shows', 'this suggests', 'evidence',
        # Hedging — epistemic nuance (Kharchenko et al., 2025)
        'might', 'perhaps', 'possibly', 'it seems', 'likely',
        # Systematic reasoning
        'systematic', 'identify', 'root cause', 'isolate',
        'analyse', 'analyze', 'evaluate', 'trace', 'diagnose',
        'methodical', 'logical', 'structured', 'step by step',
        # Conscientiousness language (Koutsoumpis et al., 2022)
        'confirm', 'verify', 'validate', 'ensure', 'carefully',
    ],
    # Ahmed & Feist (2021) — Creativity & Innovation Dictionary
    # López Martínez et al. (2024) — verbal creative thinking indicators
    # Olson et al. (2021) — semantic distance and divergent thinking
    # Flynn & Allen (2025) — lexical diversity and creative fluency
    'creativity': [
        # Creativity & Innovation Dictionary (Ahmed & Feist, 2021)
        'create', 'innovate', 'invent', 'novel', 'unique',
        'imagine', 'transform', 'vision', 'creative',
        # Verbal creative thinking (López Martínez et al., 2024)
        'unconventional', 'alternative', 'original', 'unusual',
        'rather than', 'instead', 'new approach', 'different way',
        'outside the box', 'reframe', 'rethink',
        # Creative process language
        'experimented', 'tested', 'prototype', 'iteration',
        'brainstorm', 'design', 'built', 'developed',
        # Tentative/exploratory language (Ahmed & Feist, 2021)
        'maybe', 'what if', 'could we', 'what about',
        # Divergent thinking markers (Olson et al., 2021)
        'combined', 'connected', 'merged', 'adapted', 'repurposed',
        'unexpected', 'surprising', 'creative solution',
    ],
}

KEYWORD_BOOST_PER_MATCH = 0.20   # per matched keyword
KEYWORD_BOOST_MAX       = 1.20   # cap — equivalent to 6 keyword matches; differentiates good (4) from very good (6+) responses


def apply_keyword_boost(calibrated: float, transcript: str, trait_name: str) -> float:
    """Apply a small research-grounded boost for trait-relevant vocabulary."""
    if not transcript or trait_name not in RESEARCH_KEYWORDS:
        return calibrated
    text = transcript.lower()
    keywords = RESEARCH_KEYWORDS[trait_name]
    matches = sum(1 for kw in keywords if kw in text)
    boost = min(matches * KEYWORD_BOOST_PER_MATCH, KEYWORD_BOOST_MAX)
    boosted = float(np.clip(calibrated + boost, 1.0, 5.0))
    print(f"   [KEYWORD] {trait_name}: {matches} matches → +{boost:.2f} "
          f"({calibrated:.3f} → {boosted:.3f})")
    return boosted


def calibrate_score(raw_score, word_count, vocab_diversity, transcript="", question_id=None, trait_name=None):
    """
    Data-driven calibration using Isotonic Regression trained on
    out-of-fold predictions from the RecruitView dataset (2,011 samples),
    followed by a research-grounded keyword boost derived from peer-reviewed
    literature on verbal markers of curiosity, critical thinking, and creativity.
    """
    # Hard validity cap — responses under 20 words have insufficient signal
    if word_count < 20:
        print(f"   [CALIBRATION] Too short ({word_count} words) — capped at 2.4")
        return 2.4

    # Apply learned isotonic calibration
    if trait_name and trait_name in calibrators:
        calibrated = float(calibrators[trait_name].predict([raw_score])[0])
    else:
        calibrated = float(np.clip(raw_score, 1.0, 5.0))

    # Apply research-grounded keyword boost
    calibrated = apply_keyword_boost(calibrated, transcript, trait_name)

    final_score = float(np.clip(calibrated, 1.0, 5.0))
    print(f"   [CALIBRATION] Raw: {raw_score:.3f} -> Final: {final_score:.3f} "
          f"(trait: {trait_name}, words: {word_count})")
    return final_score

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
            raw_score = model.predict(features)[0]
            
            # Apply Calibration (using default values for words/div if not provided)
            # This endpoint is less common for live demo, but consistency is key
            word_count = data.get('word_count', 60)
            vocab_diversity = data.get('vocab_diversity', 0.7)
            transcript = data.get('transcript', "")
            question_id = data.get('question_id')
            
            calibrated_score = calibrate_score(raw_score, word_count, vocab_diversity, transcript=transcript, question_id=question_id)
            predictions[trait_name] = float(calibrated_score)
        
        # Calculate recommendation
        avg_score = np.mean(list(predictions.values()))
        # Thresholds aligned to QuantileTransformer uniform 1-5 scale:
        # 4.0 = top 25%, 3.45 = top ~38%, 3.0 = median
        if avg_score >= 4.0:
            recommendation = "STRONG_HIRE"
        elif avg_score >= 3.45:
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
            print(" Valid WAV format detected, skipping conversion.")
        except Exception:
            # Conversion attempt if efficient reading fails
            print(" Format requires conversion...")
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
        print(f" Starting transcription process...")
        print(f"   File path: {processing_path}")
        print(f"   File exists: {os.path.exists(processing_path)}")
        if os.path.exists(processing_path):
            print(f"   File size: {os.path.getsize(processing_path)} bytes")
        
        try:
            print("   Creating transcriber...")
            transcriber = SpeechTranscriber(model_size="base")
            print("   Calling transcribe()...")
            transcript = transcriber.transcribe(processing_path)
            print(f" Transcription complete: {len(transcript)} chars")
        except Exception as transcribe_error:
            print(f" Transcription failed: {transcribe_error}")
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
        print("\n  FEATURE SUMMARY FOR PREDICTION:")
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

        # Clip features to training distribution (3-sigma bounds)
        df_feats = clip_to_training_distribution(df_feats)

        # Predict and Calibrate
        predictions = {}
        for trait_name, model in models.items():
            # Use DataFrame to avoid "X does not have valid feature names" warnings
            raw_score = model.predict(df_feats)[0]
            
            # Apply Calibration
            calibrated_score = calibrate_score(
                raw_score,
                features['word_count'],
                features['vocab_diversity'],
                transcript=transcript,
                question_id=request.form.get('question_id'),
                trait_name=trait_name
            )
            predictions[trait_name] = float(calibrated_score)
        
        # Calculate recommendation
        avg_score = np.mean(list(predictions.values()))
        # Thresholds aligned to QuantileTransformer uniform 1-5 scale:
        # 4.0 = top 25%, 3.45 = top ~38%, 3.0 = median
        if avg_score >= 4.0:
            recommendation = "STRONG_HIRE"
        elif avg_score >= 3.45:
            recommendation = "RECOMMENDED"
        elif avg_score >= 3.0:
            recommendation = "CONSIDER"
        else:
            recommendation = "NOT_RECOMMENDED"

        # Generate SHAP explanations
        explanations = {}
        for trait_name in models.keys():
            explanations[trait_name] = get_shap_explanation(trait_name, df_feats)

        return jsonify({
            'transcript': transcript,
            'word_count': len(words),
            'scores': predictions,
            'average': float(avg_score),
            'recommendation': recommendation,
            'explanations': explanations
        })

    except Exception as e:
        print(f"❌ Error in assess_audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/assess-text', methods=['POST'])
def assess_text():
    """Complete assessment from text input"""
    try:
        data = request.json
        transcript = data.get('text', '')
        
        if not transcript:
            return jsonify({'error': 'No text provided'}), 400
            
        # Extract lexical features from transcript
        words = transcript.lower().split()
        sentences = [s.strip() for s in transcript.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        features = {}
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['vocab_diversity'] = len(set(words)) / len(words) if words else 0
        
        # Filler words
        filler_words = {'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically'}
        filler_count = sum(1 for w in words if w in filler_words)
        features['filler_word_ratio'] = filler_count / len(words) if words else 0
        
        # Text mode has no audio — acoustic features are imputed with training means
        # so the model receives a valid feature vector. Lexical features are computed
        # from the actual text response.
        for col in feature_cols:
            if col not in features:
                features[col] = feature_means.get(col, 0)
        
        # Prepare feature vector as DataFrame
        df_feats = pd.DataFrame([features])
        df_feats = df_feats[feature_cols]
        
        # Predict and Calibrate
        predictions = {}
        question_id = data.get('question_id')
        for trait_name, model in models.items():
            raw_score = model.predict(df_feats)[0]
            
            # Apply Calibration
            calibrated_score = calibrate_score(
                raw_score,
                features['word_count'],
                features['vocab_diversity'],
                transcript=transcript,
                question_id=question_id,
                trait_name=trait_name
            )
            predictions[trait_name] = float(calibrated_score)
        
        # Calculate recommendation
        avg_score = np.mean(list(predictions.values()))
        # Thresholds aligned to QuantileTransformer uniform 1-5 scale:
        # 4.0 = top 25%, 3.45 = top ~38%, 3.0 = median
        if avg_score >= 4.0:
            recommendation = "STRONG_HIRE"
        elif avg_score >= 3.45:
            recommendation = "RECOMMENDED"
        elif avg_score >= 3.0:
            recommendation = "CONSIDER"
        else:
            recommendation = "NOT_RECOMMENDED"

        # Generate SHAP explanations
        explanations = {}
        for trait_name in models.keys():
            explanations[trait_name] = get_shap_explanation(trait_name, df_feats)

        return jsonify({
            'transcript': transcript,
            'word_count': len(words),
            'scores': predictions,
            'average': float(avg_score),
            'recommendation': recommendation,
            'explanations': explanations
        })

    except Exception as e:
        print(f"❌ Error in assess_text: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== SESSION ROUTES ====================

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """Return all saved interview sessions, newest first."""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM sessions ORDER BY created_at DESC'
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/sessions', methods=['POST'])
def save_session():
    """Save a completed interview session with all question responses."""
    data = request.json
    session_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    agg = data.get('aggregateScores', {})
    conn = get_db()
    conn.execute(
        '''INSERT INTO sessions
           (id, created_at, recommendation, overall_average,
            curiosity_avg, critical_thinking_avg, creativity_avg)
           VALUES (?,?,?,?,?,?,?)''',
        (
            session_id,
            created_at,
            data.get('recommendation'),
            data.get('overallAverage'),
            agg.get('curiosity'),
            agg.get('critical_thinking'),
            agg.get('creativity'),
        )
    )

    for r in data.get('responses', []):
        scores = r.get('scores', {})
        conn.execute(
            '''INSERT INTO responses
               (session_id, question_id, question_text, transcript,
                word_count, curiosity_score, critical_thinking_score,
                creativity_score, average_score, explanations)
               VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (
                session_id,
                r.get('questionId'),
                r.get('questionText'),
                r.get('transcript'),
                r.get('wordCount'),
                scores.get('curiosity'),
                scores.get('critical_thinking'),
                scores.get('creativity'),
                r.get('average'),
                json.dumps(r.get('explanations', {})),
            )
        )

    conn.commit()
    conn.close()
    return jsonify({'id': session_id, 'created_at': created_at}), 201


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Return a single session with all its question responses."""
    conn = get_db()
    session = conn.execute(
        'SELECT * FROM sessions WHERE id = ?', (session_id,)
    ).fetchone()
    if not session:
        conn.close()
        return jsonify({'error': 'Session not found'}), 404

    responses = conn.execute(
        'SELECT * FROM responses WHERE session_id = ? ORDER BY id', (session_id,)
    ).fetchall()
    conn.close()

    result = dict(session)
    result['responses'] = []
    for r in responses:
        row = dict(r)
        row['explanations'] = json.loads(row['explanations'] or '{}')
        result['responses'].append(row)

    return jsonify(result)


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session and all its responses (CASCADE handles responses)."""
    conn = get_db()
    conn.execute('PRAGMA foreign_keys = ON')
    result = conn.execute(
        'DELETE FROM sessions WHERE id = ?', (session_id,)
    )
    conn.commit()
    conn.close()
    if result.rowcount == 0:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'deleted': session_id})


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
