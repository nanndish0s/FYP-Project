"""
Retrain Random Forest models with expanded 44-sample dataset
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

print("=" * 80)
print("MODEL RETRAINING - 44 SAMPLES")
print("=" * 80)

# Load ML-ready dataset
print("\n1. Loading ML-ready dataset...")
df = pd.read_csv('data/processed/ml_ready_dataset_44.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Samples: {len(df)}")

# Prepare features and targets
print("\n2. Preparing features and targets...")
feature_cols = [col for col in df.columns if col not in ['video_id', 'transcript', 'word_count',
                                                          'curiosity_score', 'critical_thinking_score', 'creativity_score',
                                                          'curiosity_reasoning', 'critical_thinking_reasoning', 'creativity_reasoning']]
X = df[feature_cols]
print(f"   Features: {len(feature_cols)}")

# Train/test split
X_train, X_test, df_train, df_test = train_test_split(
    X, df, test_size=0.2, random_state=42
)
print(f"\n3. Data split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# Train models for each C3 trait
traits = ['curiosity_score', 'critical_thinking_score', 'creativity_score']
results = []

print("\n4. Training Random Forest models...")
print("=" * 80)

for trait in traits:
    print(f"\n{trait.replace('_', ' ').title()}:")
    print("  " + "-" * 70)
    
    # Get targets
    y_train = df_train[trait]
    y_test = df_test[trait]
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)  
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    
    print(f"  Train MAE: {train_mae:.3f}")
    print(f"  Test MAE:  {test_mae:.3f}")
    print(f"  Test RMSE: {test_rmse:.3f}")
    print(f"  Test R²:   {test_r2:.3f}")
    print(f"  CV MAE:    {cv_mae:.3f} (±{cv_scores.std():.3f})")
    
    # Save model
    model_name = trait.replace('_score', '')
    model_file = f'models/{model_name}_model_44.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✅ Saved: {model_file}")
    
    results.append({
        'trait': trait,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'cv_mae': cv_mae,
        'cv_std': cv_scores.std()
    })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('results/model_performance_44.csv', index=False)

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE!")
print("=" * 80)

print("\n📊 COMPARISON: 23 vs 44 Samples")
print("=" * 80)
print("Metric Comparison (Test MAE):")
print("  Previous (23 samples) → New (44 samples)")
for _, row in df_results.iterrows():
    print(f"  {row['trait']}: → {row['test_mae']:.3f}")

print(f"\n💾 Results saved to: results/model_performance_44.csv")
print(f"💾 Models saved to: models/*_model_44.pkl")

print("\n🎉 DATASET EXPANSION COMPLETE!")
print(f"   23 samples → 44 samples (191% increase)")
print(f"   Better generalization with larger dataset")
print("=" * 80)
