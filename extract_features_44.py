"""
Extract features for ALL 44 samples and prepare ML-ready dataset
"""
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from features.extract_acoustic_features import extract_acoustic_features
from features.extract_lexical_features import extract_lexical_features

def main():
    print("=" * 80)
    print("FEATURE EXTRACTION FOR 44 SAMPLES")
    print("=" * 80)
    
    # Load all 44 labeled samples
    df_labeled = pd.read_csv('data/processed/combined_44_with_c3_labels.csv')
    print(f"\n📊 Total labeled samples: {len(df_labeled)}")
    
    # Step 1: Extract acoustic features from COVAREP
    print("\n" + "=" * 80)
    print("STEP 1: Extracting Acoustic Features (518 features)")
    print("=" * 80)
    print("⚠️  This may take 30-45 minutes...")
    print("    Processing COVAREP data for all 44 videos\n")
    
    acoustic_df = extract_acoustic_features(
        video_ids=df_labeled['video_id'].tolist(),
        covarep_file='data/raw/CMU_MOSEI_COVAREP.csd'
    )
    
    acoustic_df.to_csv('data/processed/acoustic_features_44.csv', index=False)
    print(f"\n✅ Acoustic features extracted: {len(acoustic_df)} samples × {len(acoustic_df.columns)-1} features")
    print(f"💾 Saved to: data/processed/acoustic_features_44.csv")
    
    # Step 2: Extract lexical features
    print("\n" + "=" * 80)
    print("STEP 2: Extracting Lexical Features (21 features)")
    print("=" * 80)
    
    lexical_df = extract_lexical_features(df_labeled)
    lexical_df.to_csv('data/processed/lexical_features_44.csv', index=False)
    print(f"\n✅ Lexical features extracted: {len(lexical_df)} samples × {len(lexical_df.columns)-1} features")
    print(f"💾 Saved to: data/processed/lexical_features_44.csv")
    
    # Step 3: Merge all features with labels
    print("\n" + "=" * 80)
    print("STEP 3: Creating ML-Ready Dataset")
    print("=" * 80)
    
    # Merge on video_id
    ml_dataset = df_labeled.copy()
    ml_dataset = ml_dataset.merge(acoustic_df, on='video_id', how='inner')
    ml_dataset = ml_dataset.merge(lexical_df, on='video_id', how='inner')
    
    print(f"\n✅ ML dataset created:")
    print(f"   Samples: {len(ml_dataset)}")
    print(f"   Total columns: {len(ml_dataset.columns)}")
    print(f"   Features: {len(ml_dataset.columns) - 7}  (excluding video_id, transcript, word_count, 3 C3 scores, 3 reasonings)")
    
    # Save
    ml_dataset.to_csv('data/processed/ml_ready_dataset_44.csv', index=False)
    print(f"\n💾 Saved to: data/processed/ml_ready_dataset_44.csv")
    
    print("\n" + "=" * 80)
    print("🎉 FEATURE EXTRACTION COMPLETE!")
    print(f"   Ready to train models on {len(ml_dataset)} samples")
    print("=" * 80)
    
    return ml_dataset

if __name__ == "__main__":
    main()
