"""
Extract and aggregate acoustic features from COVAREP for C3-labeled videos
"""
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

print("=" * 80)
print("Acoustic Feature Extraction from COVAREP")
print("=" * 80)

# Load C3-labeled transcripts
print("\n1. Loading C3-labeled transcripts...")
df_labels = pd.read_csv('data/processed/combined_44_with_c3_labels.csv')
video_ids = df_labels['video_id'].tolist()
print(f"   Loaded {len(video_ids)} labeled videos")

# Open COVAREP dataset
print("\n2. Opening COVAREP dataset...")
covarep_path = 'data/raw/CMU_MOSEI_COVAREP.csd'

# Storage for extracted features
acoustic_features = []

with h5py.File(covarep_path, 'r') as f:
    data_group = f['COVAREP']['data']
    
    print(f"\n3. Extracting features for {len(video_ids)} videos...")
    
    for video_id in tqdm(video_ids, desc="Processing videos"):
        if video_id not in data_group:
            print(f"   Warning: {video_id} not found in COVAREP!")
            acoustic_features.append(None)
            continue
        
        # Get time-series features for this video
        # Structure: data_group[video_id]['features']
        features_ts = data_group[video_id]['features'][()]  # Shape: (timesteps, 74)
        
        # Aggregate to fixed-length feature vector
        # Use various statistics: mean, std, min, max, median, 25th/75th percentile
        if len(features_ts.shape) == 2:
            # Replace NaN and Inf with 0
            features_ts = np.nan_to_num(features_ts, nan=0.0, posinf=0.0, neginf=0.0)
            
            aggregated = {
                f'covarep_f{i}_mean': np.mean(features_ts[:, i])
                for i in range(features_ts.shape[1])
            }
            aggregated.update({
                f'covarep_f{i}_std': np.std(features_ts[:, i])
                for i in range(features_ts.shape[1])
            })
            aggregated.update({
                f'covarep_f{i}_min': np.min(features_ts[:, i])
                for i in range(features_ts.shape[1])
            })
            aggregated.update({
                f'covarep_f{i}_max': np.max(features_ts[:, i])
                for i in range(features_ts.shape[1])
            })
            aggregated.update({
                f'covarep_f{i}_median': np.median(features_ts[:, i])
                for i in range(features_ts.shape[1])
            })
            aggregated.update({
                f'covarep_f{i}_q25': np.percentile(features_ts[:, i], 25)
                for i in range(features_ts.shape[1])
            })
            aggregated.update({
                f'covarep_f{i}_q75': np.percentile(features_ts[:, i], 75)
                for i in range(features_ts.shape[1])
            })
            
            # Add video_id
            aggregated['video_id'] = video_id
            
            acoustic_features.append(aggregated)
        else:
            print(f"   Warning: Unexpected shape for {video_id}: {features_ts.shape}")
            acoustic_features.append(None)

print(f"\n4. Creating feature DataFrame...")
df_acoustic = pd.DataFrame(acoustic_features)

# Move video_id to first column
cols = ['video_id'] + [col for col in df_acoustic.columns if col != 'video_id']
df_acoustic = df_acoustic[cols]

print(f"   Shape: {df_acoustic.shape}")
print(f"   Features per video: {df_acoustic.shape[1] - 1}")  # Subtract video_id column
print(f"   Videos: {df_acoustic.shape[0]}")

# Save to CSV
output_path = 'data/processed/acoustic_features_44csv'
df_acoustic.to_csv(output_path, index=False)

print(f"\n5. Saved acoustic features to: {output_path}")

# Show sample
print(f"\n6. Sample features (first 5 columns):")
print(df_acoustic.iloc[:5, :6])

print("\n" + "=" * 80)
print("Acoustic Feature Extraction Complete!")
print("=" * 80)
print(f"\nSummary:")
print(f"- Videos processed: {len([f for f in acoustic_features if f is not None])}/{len(video_ids)}")
print(f"- Features per video: {df_acoustic.shape[1] - 1}")
print(f"- Total data points: {df_acoustic.shape[0] * (df_acoustic.shape[1] - 1)}")
print(f"\n✓ Ready for model training!")
