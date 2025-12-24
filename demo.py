"""
Interactive Demo: Voice-Based XAI for AI Recruitment
Demonstrates the complete pipeline from transcript to C3 predictions with explanations
"""
import pandas as pd
import numpy as np
import pickle
import shap
from textwrap import wrap
import sys

print("=" * 100)
print(" " * 30 + "VOICE-BASED XAI FOR AI RECRUITMENT")
print(" " * 35 + "Interactive Demo")
print("=" * 100)

# Load all necessary data
print("\n📂 Loading system components...")
df_full = pd.read_csv('data/processed/ml_ready_dataset.csv')
df_labels = pd.read_csv('data/processed/sample_with_c3_labels.csv')

# Load models
models = {}
for trait in ['curiosity_score', 'critical_thinking_score', 'creativity_score']:
    with open(f'models/{trait}_model.pkl', 'rb') as f:
        models[trait] = pickle.load(f)

# Load SHAP explainers
with open('results/shap_explainers.pkl', 'rb') as f:
    shap_results = pickle.load(f)

feature_cols = [col for col in df_full.columns if col not in ['video_id', 'curiosity_score', 'critical_thinking_score', 'creativity_score']]

print(f"   ✓ Loaded 3 trained models")
print(f"   ✓ Loaded SHAP explainers")
print(f"   ✓ Ready with {len(df_full)} candidate samples")

# Show available samples
print("\n" + "=" * 100)
print("📋 AVAILABLE CANDIDATES")
print("=" * 100)

for idx, row in df_labels.head(10).iterrows():
    transcript_preview = row['transcript'][:80] + "..." if len(row['transcript']) > 80 else row['transcript']
    print(f"{idx + 1}. Video ID: {row['video_id']} | Words: {row['word_count']} | Preview: {transcript_preview}")

# Select a sample
print("\n" + "=" * 100)
sample_idx = 2  # Third candidate (you can change this)
print(f"🔍 ANALYZING CANDIDATE {sample_idx + 1}")
print("=" * 100)

# Get candidate data
candidate_data = df_labels.iloc[sample_idx]
candidate_features = df_full.iloc[sample_idx]

# Display transcript
print("\n📝 TRANSCRIPT:")
print("-" * 100)
transcript = candidate_data['transcript']
wrapped_transcript = '\n'.join(wrap(transcript, width=95))
print(wrapped_transcript)
print(f"\n   Word count: {candidate_data['word_count']} words")

# Display actual C3 scores (LLM-generated labels)
print("\n" + "=" * 100)
print("📊 ACTUAL C3 SCORES (LLM-Generated Labels)")
print("=" * 100)
print(f"   Curiosity:          {int(candidate_data['curiosity_score'])}/5")
print(f"   Critical Thinking:  {int(candidate_data['critical_thinking_score'])}/5")
print(f"   Creativity:         {int(candidate_data['creativity_score'])}/5")

# Extract features
X_sample = candidate_features[feature_cols].values.reshape(1, -1)

# Make predictions
print("\n" + "=" * 100)
print("🤖 MODEL PREDICTIONS")
print("=" * 100)

predictions = {}
for trait in ['curiosity_score', 'critical_thinking_score', 'creativity_score']:
    trait_name = trait.replace('_score', '').replace('_', ' ').title()
    pred = models[trait].predict(X_sample)[0]
    actual = candidate_data[trait]
    error = abs(pred - actual)
    
    predictions[trait_name] = {
        'predicted': pred,
        'actual': actual,
        'error': error
    }
    
    print(f"\n   {trait_name}:")
    print(f"      Predicted: {pred:.2f}/5")
    print(f"      Actual:    {actual:.0f}/5")
    print(f"      Error:     ±{error:.2f}")
    
    # Visual bar
    bar_length = 50
    filled = int((pred / 5) * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"      [{bar}] {pred:.2f}/5")

# SHAP Explanations
print("\n" + "=" * 100)
print("🔍 SHAP EXPLANATIONS: WHY THESE PREDICTIONS?")
print("=" * 100)

for trait in ['curiosity_score', 'critical_thinking_score', 'creativity_score']:
    trait_name = trait.replace('_score', '').replace('_', ' ').title()
    
    # Get SHAP values for this sample
    shap_vals = shap_results[trait]['shap_values'][sample_idx]
    base_value = shap_results[trait]['base_value']
    
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0] if len(base_value) > 0 else float(base_value)
    
    # Get top 10 contributing features
    feature_impacts = list(zip(feature_cols, shap_vals, X_sample[0]))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n📌 {trait_name.upper()}")
    print(f"   Base prediction (average): {float(base_value):.2f}")
    print(f"   Final prediction: {predictions[trait_name]['predicted']:.2f}")
    print(f"\n   Top 10 Contributing Features:")
    
    for rank, (feature, shap_val, feature_val) in enumerate(feature_impacts[:10], 1):
        # Readable feature name
        if 'covarep_f0' in feature:
            readable = feature.replace('covarep_f0_', 'Pitch-').replace('_', ' ').title()
        elif 'covarep_f' in feature:
            readable = feature.replace('covarep_', '').replace('_', '-').upper()
        elif 'vocab' in feature:
            readable = "Vocabulary Richness"
        elif 'filler' in feature:
            readable = "Hesitation Rate"
        else:
            readable = feature.replace('_', ' ').title()
        
        # Impact direction
        direction = "↑" if shap_val > 0 else "↓"
        impact_desc = "increases" if shap_val > 0 else "decreases"
        
        print(f"      {rank:2d}. {readable:35s} {direction} {shap_val:+.3f}  (value: {feature_val:.2f})")

# Prosodic Interpretation
print("\n" + "=" * 100)
print("🎤 PROSODIC INTERPRETATION (Voice Characteristics)")
print("=" * 100)

df_prosody = pd.read_csv('results/prosody_contributions.csv')

for trait_name in ['Curiosity', 'Critical Thinking', 'Creativity']:
    print(f"\n{trait_name}:")
    trait_prosody = df_prosody[df_prosody['trait'] == trait_name].head(5)
    
    for _, row in trait_prosody.iterrows():
        print(f"   • {row['category']:30s}: {row['avg_impact']:.4f}  ({row['description']})")

# Recruitment Recommendation
print("\n" + "=" * 100)
print("💼 RECRUITMENT RECOMMENDATION")
print("=" * 100)

avg_score = np.mean([predictions[t]['predicted'] for t in predictions])
print(f"\n   Average C3 Score: {avg_score:.2f}/5")

if avg_score >= 4.0:
    recommendation = "🌟 STRONG HIRE - Excellent soft skills across all dimensions"
elif avg_score >= 3.5:
    recommendation = "✅ RECOMMENDED - Good soft skills, suitable for role"
elif avg_score >= 3.0:
    recommendation = "⚠️  CONSIDER - Moderate soft skills, may need development"
else:
    recommendation = "❌ NOT RECOMMENDED - Soft skills below threshold"

print(f"   {recommendation}")

print(f"\n   Strengths:")
for trait_name, data in predictions.items():
    if data['predicted'] >= 4.0:
        print(f"      • High {trait_name}: {data['predicted']:.2f}/5")

if any(data['predicted'] < 3.5 for data in predictions.values()):
    print(f"\n   Areas for Development:")
    for trait_name, data in predictions.items():
        if data['predicted'] < 3.5:
            print(f"      • {trait_name}: {data['predicted']:.2f}/5")

# Summary
print("\n" + "=" * 100)
print("📈 DEMO SUMMARY")
print("=" * 100)
print("""
This demo showcased:
✓ Transcript analysis
✓ Automatic feature extraction (539 acoustic + lexical features)
✓ C3 score prediction using Random Forest models
✓ SHAP explainability showing which features influenced predictions
✓ Prosodic category analysis (voice characteristics)
✓ Recruitment recommendation based on soft skills

The system provides transparent, explainable AI for fair recruitment decisions!
""")

print("=" * 100)
print(" " * 35 + "Demo Complete!")
print("=" * 100)
