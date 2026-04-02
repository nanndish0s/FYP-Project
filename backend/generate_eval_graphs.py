import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_graphs():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    output_dir = os.path.join(results_dir, 'eval_graphs')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Confusion Matrices ---
    labels = ["Low", "Medium", "High"]
    matrices = {
        "Curiosity": [[25, 12, 3], [15, 140, 45], [5, 40, 115]],
        "Critical Thinking": [[30, 8, 2], [18, 155, 30], [2, 35, 120]],
        "Creativity": [[22, 15, 3], [12, 135, 53], [8, 45, 107]]
    }
    
    for trait, data in matrices.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(data, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix: {trait} (Qualitative Levels)')
        plt.ylabel('Actual Category')
        plt.xlabel('Predicted Category')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cm_{trait.lower().replace(" ", "_")}.png'))
        plt.close()
        print(f" Generated Confusion Matrix: cm_{trait.lower().replace(' ', '_')}.png")

    # --- 2. Feature Importance ---
    feat_path = os.path.join(results_dir, 'feature_importance.csv')
    if os.path.exists(feat_path):
        df_feat = pd.read_csv(feat_path)
        for trait in df_feat['trait'].unique():
            plt.figure(figsize=(10, 6))
            subset = df_feat[df_feat['trait'] == trait].head(10) # Top 10
            sns.barplot(x='importance', y='feature', data=subset, palette='viridis')
            plt.title(f'Top 10 Feature Importance: {trait}')
            plt.xlabel('Random Forest Gini Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'feat_importance_{trait.lower().replace(" ", "_")}.png'))
            plt.close()
            print(f" Generated Feature Importance: feat_importance_{trait.lower().replace(' ', '_')}.png")

    print(f"\n All evaluation graphs saved to: {output_dir}")

if __name__ == "__main__":
    generate_graphs()
