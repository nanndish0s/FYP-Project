"""
Explore COVAREP acoustic features dataset structure
and verify our labeled videos are present
"""
import h5py
import pandas as pd
import numpy as np

# Load our labeled transcripts to get video IDs
print("=" * 80)
print("COVAREP Dataset Exploration")
print("=" * 80)

print("\n1. Loading C3-labeled transcripts...")
df_labels = pd.read_csv('data/processed/sample_with_c3_labels.csv')
our_video_ids = df_labels['video_id'].tolist()
print(f"   We have {len(our_video_ids)} labeled videos")
print(f"   First 5 video IDs: {our_video_ids[:5]}")

# Load COVAREP dataset
print("\n2. Opening COVAREP dataset...")
covarep_path = 'data/raw/CMU_MOSEI_COVAREP.csd'

found_videos = []
missing_videos = []
sample_features = None

with h5py.File(covarep_path, 'r') as f:
    print(f"   File opened successfully!")
    print(f"   Keys in file: {list(f.keys())}")
    
    # Check structure - videos are in COVAREP/data
    if 'COVAREP' in f and 'data' in f['COVAREP']:
        data_group = f['COVAREP']['data']
        print(f"\n3. Exploring 'COVAREP/data' group...")
        print(f"   Type: {type(data_group)}")
        print(f"   Keys: {list(data_group.keys())[:10]}...")  # First 10
        
        total_videos = len(data_group.keys())
        print(f"   Total videos in COVAREP: {total_videos}")
        
        # Check if our videos are present
        print(f"\n4. Checking for our 23 labeled videos...")
        
        for vid_id in our_video_ids:
            if vid_id in data_group:
                found_videos.append(vid_id)
            else:
                missing_videos.append(vid_id)
        
        print(f"   ✓ Found: {len(found_videos)}/{len(our_video_ids)}")
        if missing_videos:
            print(f"   ✗ Missing: {missing_videos}")
        
        # Examine a sample video's features
        if found_videos:
            print(f"\n5. Examining sample video: {found_videos[0]}")
            sample_features = data_group[found_videos[0]][()]
            print(f"   Shape: {sample_features.shape}")
            print(f"   Data type: {sample_features.dtype}")
            print(f"   Features (columns): {sample_features.shape[1] if len(sample_features.shape) > 1 else 'N/A'}")
            print(f"   Timesteps (rows): {sample_features.shape[0] if len(sample_features.shape) > 0 else 'N/A'}")
            print(f"   Sample values (first 3 timesteps, first 5 features):")
            if len(sample_features.shape) > 1:
                print(sample_features[:3, :5])
            else:
                print(sample_features[:10])
            
            # Check for NaN/Inf
            if len(sample_features.shape) > 1:
                nan_count = np.isnan(sample_features).sum()
                inf_count = np.isinf(sample_features).sum()
                print(f"\n   Data quality:")
                print(f"   - NaN values: {nan_count}")
                print(f"   - Inf values: {inf_count}")
            
            # Show statistics
            print(f"\n6. Feature statistics for {found_videos[0]}:")
            if len(sample_features.shape) > 1:
                print(f"   Mean across time (first 5 features): {np.mean(sample_features, axis=0)[:5]}")
                print(f"   Std across time (first 5 features): {np.std(sample_features, axis=0)[:5]}")

print("\n" + "=" * 80)
print("Exploration Complete!")
print("=" * 80)
print(f"\nSummary:")
print(f"- Labeled videos: {len(our_video_ids)}")
print(f"- Found in COVAREP: {len(found_videos)}")
print(f"- Missing: {len(missing_videos)}")
if found_videos and sample_features is not None:
    print(f"- Feature dimensions: {sample_features.shape}")
    print(f"\n✓ Ready to extract acoustic features!")
