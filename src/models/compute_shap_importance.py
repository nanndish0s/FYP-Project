"""
Computes mean |SHAP value| per feature for each C3 trait's Random Forest model,
evaluated on the same 80/20 held-out test split used in train_recruitview_model.py.
Used to populate Table 6.2 (Top 5 Most Influential Features per Trait) in Chapter 6.
"""
import pandas as pd
import numpy as np
import pickle
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / 'data' / 'processed'
models_dir = project_root / 'models'

df = pd.read_csv(data_dir / 'recruitview_features_all.csv')

target_cols = ['curiosity_score', 'critical_thinking_score', 'creativity_score']
exclude_cols = ['video_id', 'word_count', 'sentence_count'] + target_cols
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]

for trait in target_cols:
    trait_name = trait.replace('_score', '')
    y = df[trait]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(models_dir / f'{trait_name}_model_recruitview.pkl', 'rb') as f:
        model = pickle.load(f)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_cols, mean_abs_shap), key=lambda x: x[1], reverse=True)

    print(f"\n=== {trait_name.upper()} — Top 5 features by mean |SHAP| (test set, n={len(X_test)}) ===")
    for feat, val in ranking[:5]:
        print(f"  {feat:25s}  {val:.4f}")
