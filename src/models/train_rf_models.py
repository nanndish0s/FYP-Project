"""
Train Random Forest models for C3 (Curiosity, Critical Thinking, Creativity) prediction
Uses acoustic + lexical features to predict soft skill scores
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Phase 3: Random Forest Model Training for C3 Prediction")
print("=" * 80)

# Load ML-ready dataset
print("\n1. Loading ML-ready dataset...")
df = pd.read_csv('data/processed/ml_ready_dataset.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Samples: {df.shape[0]}")

# Separate features and labels
print("\n2. Preparing features and labels...")
feature_cols = [col for col in df.columns if col not in ['video_id', 'curiosity_score', 'critical_thinking_score', 'creativity_score']]
target_cols = ['curiosity_score', 'critical_thinking_score', 'creativity_score']

X = df[feature_cols].values
y = df[target_cols].values
print(f"   Features (X): {X.shape}")
print(f"   Targets (y): {y.shape}")
print(f"   Feature names: {len(feature_cols)}")

# Split data (80/20 split with small dataset)
print("\n3. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")

# Train models for each C3 trait
print("\n4. Training Random Forest models...")
models = {}
results = []

for idx, trait in enumerate(target_cols):
    trait_name = trait.replace('_score', '').replace('_', ' ').title()
    print(f"\n   Training {trait_name} model...")
    
    # Create Random Forest model
    rf = RandomForestRegressor(
        n_estimators=100,           # 100 trees
        max_depth=10,               # Limit depth to prevent overfitting
        min_samples_split=5,        # Min samples to split
        min_samples_leaf=2,         # Min samples in leaf
        random_state=42,
        n_jobs=-1                   # Use all CPU cores
    )
    
    # Train model
    rf.fit(X_train, y_train[:, idx])
    
    # Predictions
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    # Evaluate
    train_mae = mean_absolute_error(y_train[:, idx], y_train_pred)
    test_mae = mean_absolute_error(y_test[:, idx], y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train[:, idx], y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test[:, idx], y_test_pred))
    train_r2 = r2_score(y_train[:, idx], y_train_pred)
    test_r2 = r2_score(y_test[:, idx], y_test_pred)
    
    print(f"      Training MAE: {train_mae:.3f}")
    print(f"      Testing MAE: {test_mae:.3f}")
    print(f"      Testing R²: {test_r2:.3f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(rf, X_train, y_train[:, idx], cv=3, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    print(f"      Cross-validation MAE: {cv_mae:.3f}")
    
    # Store model
    models[trait] = rf
    
    # Store results
    results.append({
        'trait': trait_name,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_mae': cv_mae,
        'n_features': X.shape[1],
        'n_train_samples': X_train.shape[0],
        'n_test_samples': X_test.shape[0]
    })

# Save models
print("\n5. Saving trained models...")
for trait, model in models.items():
    model_path = f'models/{trait}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   Saved: {model_path}")

# Save results
print("\n6. Saving performance results...")
df_results = pd.DataFrame(results)
df_results.to_csv('results/model_performance.csv', index=False)
print(f"   Saved: results/model_performance.csv")

# Display results summary
print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)
print(df_results.to_string(index=False))

# Feature importance analysis
print("\n7. Analyzing feature importance...")
feature_importance_data = []

for trait in target_cols:
    trait_name = trait.replace('_score', '').replace('_', ' ').title()
    model = models[trait]
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Get top 20 features
    top_indices = np.argsort(importances)[-20:][::-1]
    
    print(f"\n   Top 10 features for {trait_name}:")
    for rank, idx in enumerate(top_indices[:10], 1):
        feature_name = feature_cols[idx]
        importance = importances[idx]
        print(f"      {rank}. {feature_name}: {importance:.4f}")
        
        # Store for CSV
        feature_importance_data.append({
            'trait': trait_name,
            'rank': rank,
            'feature': feature_name,
            'importance': importance
        })

# Save feature importance
df_importance = pd.DataFrame(feature_importance_data)
df_importance.to_csv('results/feature_importance.csv', index=False)
print(f"\n   Saved: results/feature_importance.csv")

# Make predictions on test set and save
print("\n8. Saving test predictions...")
predictions_data = []

for i, idx in enumerate(range(len(X_test))):
    pred_row = {'sample_id': i + 1}
    
    for trait_idx, trait in enumerate(target_cols):
        trait_name = trait.replace('_score', '')
        actual = y_test[idx, trait_idx]
        predicted = models[trait].predict(X_test[idx].reshape(1, -1))[0]
        
        pred_row[f'{trait_name}_actual'] = actual
        pred_row[f'{trait_name}_predicted'] = predicted
        pred_row[f'{trait_name}_error'] = abs(actual - predicted)
    
    predictions_data.append(pred_row)

df_predictions = pd.DataFrame(predictions_data)
df_predictions.to_csv('results/test_predictions.csv', index=False)
print(f"   Saved: results/test_predictions.csv")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\n✓ Models trained and saved to 'models/' directory")
print(f"✓ Results saved to 'results/' directory")
print(f"\nNext: Phase 4 - SHAP Explainability")
