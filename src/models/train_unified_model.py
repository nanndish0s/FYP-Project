import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

def train_unified_model():
    print("🚀 Starting Unified Model Training (MOSEI + RecruitView)...")
    
    # 1. Load Datasets
    recruitview_path = 'data/processed/recruitview_features_all.csv'
    mosei_path = 'data/processed/mosei_features_aligned.csv'
    
    if not os.path.exists(recruitview_path) or not os.path.exists(mosei_path):
        print("❌ Error: One or both feature files are missing.")
        return

    rv_df = pd.read_csv(recruitview_path)
    mosei_df = pd.read_csv(mosei_path)
    
    print(f"📂 Loaded {len(rv_df)} RecruitView samples.")
    print(f"📂 Loaded {len(mosei_df)} MOSEI samples.")
    
    # 2. Harmonize Columns
    features = [
        'pitch_mean', 'energy_mean', 'spec_centroid_mean', 'zcr_mean',
        'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean',
        'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean', 'mfcc_9_mean', 'mfcc_10_mean',
        'mfcc_11_mean', 'mfcc_12_mean', 'mfcc_13_mean', 'mfcc_14_mean', 'mfcc_15_mean',
        'mfcc_16_mean', 'mfcc_17_mean', 'mfcc_18_mean', 'mfcc_19_mean', 'mfcc_20_mean',
        'word_count', 'sentence_count', 'avg_word_length', 'vocab_diversity', 'filler_word_ratio'
    ]
    
    targets = ['curiosity_score', 'critical_thinking_score', 'creativity_score']
    
    # 3. Combine
    combined_df = pd.concat([rv_df[features + targets], mosei_df[features + targets]], ignore_index=True)
    print(f"📊 Total Combined Samples: {len(combined_df)}")
    
    # 4. Train & Evaluate for each trait
    performance = []
    
    for trait in targets:
        print(f"\n--- Training for {trait} ---")
        X = combined_df[features]
        y = combined_df[trait]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        preds = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"CV Mean R2: {cv_scores.mean():.4f}")
        
        performance.append({
            'Trait': trait.replace('_score', '').title(),
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_Mean_R2': cv_scores.mean(),
            'Samples': len(combined_df)
        })
        
        # Save model
        model_filename = f"models/{trait.replace('_score', '')}_model_unified.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Saved model to {model_filename}")

    # Save performance report
    perf_df = pd.DataFrame(performance)
    perf_df.to_csv('results/model_performance_unified.csv', index=False)
    print("\n✅ Unified Model Performance saved to results/model_performance_unified.csv")

if __name__ == "__main__":
    train_unified_model()
