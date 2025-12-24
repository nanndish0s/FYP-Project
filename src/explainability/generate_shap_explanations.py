"""
Prosody-Aware SHAP Explainability for C3 Prediction
Maps technical acoustic features to human-readable prosodic concepts
"""
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("Phase 4: Prosody-Aware SHAP Explainability")
print("=" * 80)

# Define prosodic feature categories
PROSODY_CATEGORIES = {
    'Pitch & Intonation': {
        'description': 'Fundamental frequency patterns (F0)',
        'features': [f'covarep_f0_{stat}' for stat in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']]
    },
    'Voice Quality': {
        'description': 'Voicing characteristics and clarity',
        'features': [f'covarep_f{i}_{stat}' for i in range(1, 20) for stat in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']]
    },
    'Spectral Characteristics': {
        'description': 'Harmonic and spectral richness',
        'features': [f'covarep_f{i}_{stat}' for i in range(20, 60) for stat in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']]
    },
    'Energy & Dynamics': {
        'description': 'Speaking intensity and variation',
        'features': [f'covarep_f{i}_{stat}' for i in range(60, 74) for stat in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']]
    },
    'Vocabulary & Word Choice': {
        'description': 'Lexical sophistication',
        'features': ['vocab_diversity', 'unique_word_count', 'long_word_count', 'long_word_ratio']
    },
    'Fluency & Hesitation': {
        'description': 'Speaking smoothness',
        'features': ['filler_word_count', 'filler_word_ratio', 'stopword_ratio']
    },
    'Speech Structure': {
        'description': 'Sentence and speech organization',
        'features': ['word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length']
    },
    'Engagement Indicators': {
        'description': 'Curiosity and interaction markers',
        'features': ['question_mark_count', 'question_ratio', 'exclamation_count']
    },
    'Readability & Complexity': {
        'description': 'Speech comprehension difficulty',
        'features': ['flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability_index']
    }
}

# Human-readable feature names
FEATURE_LABELS = {
    # Pitch
    'covarep_f0_mean': 'Average Pitch',
    'covarep_f0_std': 'Pitch Variability',
    'covarep_f0_min': 'Minimum Pitch',
    'covarep_f0_max': 'Maximum Pitch',
    
    # Lexical
    'vocab_diversity': 'Vocabulary Richness',
    'filler_word_ratio': 'Hesitation Rate',
    'word_count': 'Speech Length',
    'question_mark_count': 'Number of Questions',
    'long_word_ratio': 'Complex Word Usage',
    'avg_sentence_length': 'Sentence Complexity',
    
    # Default for unmapped features
}

def get_readable_name(feature_name):
    """Convert technical feature name to human-readable label"""
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    
    # Parse COVAREP features
    if feature_name.startswith('covarep_f'):
        parts = feature_name.split('_')
        f_num = parts[0].replace('covarep_f', '')
        stat = parts[1] if len(parts) > 1 else ''
        
        # Map to prosody concepts
        if f_num == '0':
            base = 'Pitch'
        elif int(f_num) < 20:
            base = 'Voice Quality'
        elif int(f_num) < 60:
            base = f'Spectral-{f_num}'
        else:
            base = 'Energy'
        
        stat_map = {
            'mean': 'Avg', 'std': 'Variability', 'min': 'Min', 
            'max': 'Max', 'median': 'Median', 'q25': 'Lower', 'q75': 'Upper'
        }
        
        return f"{base} {stat_map.get(stat, stat)}"
    
    # Default: capitalize and replace underscores
    return feature_name.replace('_', ' ').title()

# Load data and models
print("\n1. Loading data and models...")
df = pd.read_csv('data/processed/ml_ready_dataset.csv')

# Prepare features
feature_cols = [col for col in df.columns if col not in ['video_id', 'curiosity_score', 'critical_thinking_score', 'creativity_score']]
X = df[feature_cols].values
y = df[['curiosity_score', 'critical_thinking_score', 'creativity_score']].values

print(f"   Features: {X.shape}")
print(f"   Samples: {X.shape[0]}")

# Load trained models
models = {}
for trait in ['curiosity_score', 'critical_thinking_score', 'creativity_score']:
    with open(f'models/{trait}_model.pkl', 'rb') as f:
        models[trait] = pickle.load(f)
    print(f"   Loaded: {trait}_model.pkl")

# Generate SHAP explanations
print("\n2. Generating SHAP explanations...")
shap_results = {}

for trait in ['curiosity_score', 'critical_thinking_score', 'creativity_score']:
    trait_name = trait.replace('_score', '').replace('_', ' ').title()
    print(f"\n   Computing SHAP values for {trait_name}...")
    
    model = models[trait]
    
    # Create SHAP explainer (TreeExplainer for Random Forest)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Store results
    shap_results[trait] = {
        'explainer': explainer,
        'shap_values': shap_values,
        'base_value': explainer.expected_value
    }
    
    # Get base value (handle array case)
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = base_val[0] if len(base_val) > 0 else base_val
    
    print(f"      Base value (average prediction): {float(base_val):.3f}")
    print(f"      SHAP values shape: {shap_values.shape}")

# Save SHAP results
print("\n3. Saving SHAP objects...")
with open('results/shap_explainers.pkl', 'wb') as f:
    pickle.dump(shap_results, f)
print("   Saved: results/shap_explainers.pkl")

# Analyze prosodic contributions
print("\n4. Analyzing prosodic category contributions...")
prosody_contributions = []

for trait in ['curiosity_score', 'critical_thinking_score', 'creativity_score']:
    trait_name = trait.replace('_score', '').replace('_', ' ').title()
    shap_vals = shap_results[trait]['shap_values']
    
    # Average absolute SHAP values across all samples
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    
    # Group by prosody category
    for category, info in PROSODY_CATEGORIES.items():
        category_features = info['features']
        
        # Find matching feature indices
        category_impact = 0
        feature_count = 0
        
        for feature_name in category_features:
            if feature_name in feature_cols:
                idx = feature_cols.index(feature_name)
                category_impact += mean_abs_shap[idx]
                feature_count += 1
        
        if feature_count > 0:
            avg_impact = category_impact / feature_count
            prosody_contributions.append({
                'trait': trait_name,
                'category': category,
                'description': info['description'],
                'avg_impact': avg_impact,
                'total_impact': category_impact,
                'num_features': feature_count
            })

# Save prosody analysis
df_prosody = pd.DataFrame(prosody_contributions)
df_prosody = df_prosody.sort_values(['trait', 'avg_impact'], ascending=[True, False])
df_prosody.to_csv('results/prosody_contributions.csv', index=False)
print("   Saved: results/prosody_contributions.csv")

# Display prosody analysis
print("\n5. Prosodic Category Importance:")
for trait in ['Curiosity', 'Critical Thinking', 'Creativity']:
    print(f"\n   {trait}:")
    trait_data = df_prosody[df_prosody['trait'] == trait].head(5)
    for _, row in trait_data.iterrows():
        print(f"      {row['category']}: {row['avg_impact']:.4f} ({row['description']})")

print("\n" + "=" * 80)
print("SHAP EXPLAINABILITY COMPLETE!")
print("=" * 80)
print("\n✓ SHAP values computed for all 3 traits")
print("✓ Prosodic categories analyzed")
print("✓ Results saved to results/ directory")
print("\nNext: Generate visualizations and individual explanations")
