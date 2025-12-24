"""Compare video ID formats"""
import h5py
import pandas as pd

# Get our video IDs
df_labels = pd.read_csv('data/processed/sample_with_c3_labels.csv')
our_ids = df_labels['video_id'].tolist()

print("Our video IDs (first 5):")
for vid in our_ids[:5]:
    print(f"  '{vid}' (type: {type(vid)})")

# Get COVAREP video IDs
with h5py.File('data/raw/CMU_MOSEI_COVAREP.csd', 'r') as f:
    covarep_ids = list(f['COVAREP'].keys())
    
print(f"\nCOVAREP has {len(covarep_ids)} videos")
print("\nCOVAREP video IDs (first 5):")
for vid in covarep_ids[:5]:
    print(f"  '{vid}' (type: {type(vid)})")

# Check if any match
print("\nChecking for matches...")
for our_vid in our_ids[:5]:
    if our_vid in covarep_ids:
        print(f"  ✓ {our_vid} FOUND")
    else:
        print(f"  ✗ {our_vid} NOT FOUND")
        # Try to find similar
        similar = [c for c in covarep_ids if our_vid in c or c in our_vid]
        if similar:
            print(f"    Similar: {similar[:3]}")
