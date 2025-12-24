"""
Generate comprehensive SHAP visualizations for C3 predictions
Includes SHAP plots and prosody-aware custom visualizations
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("=" * 80)
print("Generating SHAP Visualizations")
print("=" * 80)

# Load data
print("\n1. Loading data and SHAP results...")
df = pd.read_csv('data/processed/ml_ready_dataset.csv')
feature_cols = [col for col in df.columns if col not in ['video_id', 'curiosity_score', 'critical_thinking_score', 'creativity_score']]
X = df[feature_cols].values

with open('results/shap_explainers.pkl', 'rb') as f:
    shap_results = pickle.load(f)

print(f"   Samples: {X.shape[0]}")
print(f"   Features: {X.shape[1]}")

# Create readable feature names for top features
def get_readable_name(feature_name):
    """Convert technical names to readable labels"""
    readable_map = {
        'covarep_f0_mean': 'Average Pitch',
        'covarep_f0_std': 'Pitch Variability',
        'vocab_diversity': 'Vocabulary Richness',
        'filler_word_ratio': 'Hesitation Rate',
        'word_count': 'Speech Length',
        'avg_sentence_length': 'Sentence Complexity'
    }
    
    if feature_name in readable_map:
        return readable_map[feature_name]
    
    # Shorten COVAREP names
    if 'covarep_f' in feature_name:
        parts = feature_name.split('_')
        num = parts[0].replace('covarep_f', 'F')
        stat = parts[1] if len(parts) > 1 else ''
        return f"{num}_{stat[:3]}"
    
    return feature_name.replace('_', ' ').title()

# VISUALIZATION 1: SHAP Summary Plots (Top 20 Features)
print("\n2. Creating SHAP summary plots...")
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for idx, (trait, ax) in enumerate(zip(['curiosity_score', 'critical_thinking_score', 'creativity_score'], axes)):
    trait_name = trait.replace('_score', '').replace('_', ' ').title()
    shap_vals = shap_results[trait]['shap_values']
    
    # Get top 20 features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-20:]
    
    # Create summary plot
    shap.summary_plot(
        shap_vals[:, top_indices],
        X[:, top_indices],
        feature_names=[get_readable_name(feature_cols[i]) for i in top_indices],
        show=False,
        max_display=20,
        plot_size=(6, 6)
    )
    
    ax = plt.gca()
    ax.set_title(f'{trait_name}\nTop 20 Features', fontsize=14, fontweight='bold')
    
    # Save individual plot
    plt.tight_layout()
    plt.savefig(f'visualizations/shap_summary_{trait}.png', bbox_inches='tight')
    plt.close()

print("   Saved: SHAP summary plots for all traits")

# VISUALIZATION 2: Prosodic Category Contributions
print("\n3. Creating prosody category visualizations...")
df_prosody = pd.read_csv('results/prosody_contributions.csv')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (trait, ax) in enumerate(zip(['Curiosity', 'Critical Thinking', 'Creativity'], axes)):
    trait_data = df_prosody[df_prosody['trait'] == trait].sort_values('avg_impact', ascending=True)
    
    # Bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(trait_data)))
    ax.barh(trait_data['category'], trait_data['avg_impact'], color=colors)
    ax.set_xlabel('Average SHAP Impact', fontweight='bold')
    ax.set_title(f'{trait}\nProsodic Category Contributions', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/prosody_categories.png', bbox_inches='tight')
print("   Saved: visualizations/prosody_categories.png")
plt.close()

# VISUALIZATION 3: Feature Importance Comparison
print("\n4. Creating feature importance comparison...")
df_importance = pd.read_csv('results/feature_importance.csv')

fig, ax = plt.subplots(figsize=(14, 8))

# Get top 15 features across all traits
top_features_per_trait = {}
for trait in ['Curiosity', 'Critical Thinking', 'Creativity']:
    trait_imp = df_importance[df_importance['trait'] == trait].head(15)
    for _, row in trait_imp.iterrows():
        feature = row['feature']
        if feature not in top_features_per_trait:
            top_features_per_trait[feature] = {}
        top_features_per_trait[feature][trait] = row['importance']

# Get top 15 unique features
all_features = list(set([f for trait_data in df_importance.groupby('trait') for f in trait_data[1].head(10)['feature']]))[:15]

# Create grouped bar chart
x = np.arange(len(all_features))
width = 0.25

for idx, trait in enumerate(['Curiosity', 'Critical Thinking', 'Creativity']):
    importances = [top_features_per_trait.get(f, {}).get(trait, 0) for f in all_features]
    ax.bar(x + idx * width, importances, width, label=trait, alpha=0.8)

ax.set_xlabel('Features', fontweight='bold')
ax.set_ylabel('Importance', fontweight='bold')
ax.set_title('Top Feature Importance Across C3 Traits', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([get_readable_name(f) for f in all_features], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/feature_importance_comparison.png', bbox_inches='tight')
print("   Saved: visualizations/feature_importance_comparison.png")
plt.close()

# VISUALIZATION 4: SHAP Waterfall Plots (Sample Predictions)
print("\n5. Creating waterfall plots for sample predictions...")

# Create waterfall for first 3 samples
for sample_idx in range(min(3, X.shape[0])):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (trait, ax) in enumerate(zip(['curiosity_score', 'critical_thinking_score', 'creativity_score'], axes)):
        trait_name = trait.replace('_score', '').replace('_', ' ').title()
        shap_vals = shap_results[trait]['shap_values']
        base_value = shap_results[trait]['base_value']
        
        # Handle array base value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0] if len(base_value) > 0 else float(base_value)
        
        # Create waterfall plot
        plt.sca(ax)
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[sample_idx],
                base_values=float(base_value),
                data=X[sample_idx],
                feature_names=[get_readable_name(f) for f in feature_cols]
            ),
            max_display=15,
            show=False
        )
        ax.set_title(f'{trait_name}\nSample {sample_idx + 1}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/waterfall_sample_{sample_idx + 1}.png', bbox_inches='tight')
    plt.close()

print(f"   Saved: Waterfall plots for {min(3, X.shape[0])} samples")

# VISUALIZATION 5: Model Performance Metrics
print("\n6. Creating model performance visualization...")
df_perf = pd.read_csv('results/model_performance.csv')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# MAE comparison
ax = axes[0]
traits = df_perf['trait']
train_mae = df_perf['train_mae']
test_mae = df_perf['test_mae']

x = np.arange(len(traits))
width = 0.35

ax.bar(x - width/2, train_mae, width, label='Train MAE', alpha=0.8)
ax.bar(x + width/2, test_mae, width, label='Test MAE', alpha=0.8)
ax.set_xlabel('Trait', fontweight='bold')
ax.set_ylabel('Mean Absolute Error', fontweight='bold')
ax.set_title('Model Performance: MAE', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(traits, rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# R² comparison
ax = axes[1]
ax.bar(traits, df_perf['test_r2'], alpha=0.8, color='skyblue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Trait', fontweight='bold')
ax.set_ylabel('R² Score', fontweight='bold')
ax.set_title('Model Performance: R²', fontsize=12, fontweight='bold')
ax.set_xticklabels(traits, rotation=15, ha='right')
ax.grid(axis='y', alpha=0.3)

# RMSE comparison
ax = axes[2]
ax.bar(traits, df_perf['test_rmse'], alpha=0.8, color='coral')
ax.set_xlabel('Trait', fontweight='bold')
ax.set_ylabel('Root Mean Squared Error', fontweight='bold')
ax.set_title('Model Performance: RMSE', fontsize=12, fontweight='bold')
ax.set_xticklabels(traits, rotation=15, ha='right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/model_performance.png', bbox_inches='tight')
print("   Saved: visualizations/model_performance.png")
plt.close()

# VISUALIZATION 6: Predictions vs Actuals
print("\n7. Creating predictions vs actuals scatter plots...")
df_pred = pd.read_csv('results/test_predictions.csv')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (trait, ax) in enumerate(zip(['curiosity', 'critical_thinking', 'creativity'], axes)):
    actual_col = f'{trait}_actual'
    pred_col = f'{trait}_predicted'
    
    actual = df_pred[actual_col]
    predicted = df_pred[pred_col]
    
    # Scatter plot
    ax.scatter(actual, predicted, alpha=0.6, s=100, edgecolors='black', linewidth=1)
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Score', fontweight='bold')
    ax.set_ylabel('Predicted Score', fontweight='bold')
    ax.set_title(f'{trait.replace("_", " ").title()}\nPredictions vs Actuals', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0.5, 5.5])
    ax.set_ylim([0.5, 5.5])

plt.tight_layout()
plt.savefig('visualizations/predictions_vs_actuals.png', bbox_inches='tight')
print("   Saved: visualizations/predictions_vs_actuals.png")
plt.close()

print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE!")
print("=" * 80)
print("\n✓ All visualizations saved to 'visualizations/' directory")
print("\nGenerated:")
print("  - SHAP summary plots (3 files)")
print("  - Prosody category contributions")
print("  - Feature importance comparison")
print("  - Waterfall plots for sample predictions (3 files)")
print("  - Model performance metrics")
print("  - Predictions vs actuals scatter plots")
print("\nTotal: 9 visualization files created!")
