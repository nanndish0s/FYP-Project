import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

def train_recruitview_models():
    """
    Trains Random Forest models for C3 traits using the RecruitView dataset.
    """
    feature_path = 'data/processed/recruitview_features_all.csv'
    
    if not os.path.exists(feature_path):
        print(f" Feature file not found: {feature_path}")
        return

    print(f" Loading dataset from {feature_path}...")
    df = pd.read_csv(feature_path)
    print(f"   Loaded {len(df)} samples with {df.shape[1]-1} features.")

    # Define features and targets
    # Exclude non-feature columns
    target_cols = ['curiosity_score', 'critical_thinking_score', 'creativity_score']
    feature_cols = [col for col in df.columns if col not in ['video_id'] + target_cols]
    
    X = df[feature_cols]
    
    results = []
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    for trait in target_cols:
        trait_name = trait.replace('_score', '')
        print(f"\n Training model for: {trait_name.upper()}...")
        
        y = df[trait]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train Random Forest
        # Using similar parameters to the original model for consistency
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   CV Mean R²: {cv_scores.mean():.4f}")
        
        # Save model
        model_path = f'models/{trait_name}_model_recruitview.pkl'
        with open(model_path, 'rb' if os.path.exists(model_path) else 'wb') as f:
            pickle.dump(model, open(model_path, 'wb'))
            
        results.append({
            'trait': trait_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'samples': len(df)
        })

    # Save performance metrics
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/model_performance_recruitview.csv', index=False)
    print(f"\n All models trained and saved to 'models/'")
    print(f" Performance report saved to 'results/model_performance_recruitview.csv'")
    
    return df_results

if __name__ == "__main__":
    train_recruitview_models()
