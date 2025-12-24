"""
Quick script to verify and inspect the downloaded .csd file
"""
import h5py
import os

csd_file = "data/raw/debug_test.csd"

print(f"Verifying {csd_file}...")
print(f"File size: {os.path.getsize(csd_file) / (1024*1024):.2f} MB\n")

try:
    with h5py.File(csd_file, 'r') as f:
        print("✓ File is a valid HDF5 file")
        print(f"\nDataset keys: {list(f.keys())}")
        
        # Show first few entries
        print(f"\nNumber of entries: {len(f.keys())}")
        
        # Sample a few entries
        sample_keys = list(f.keys())[:3]
        for key in sample_keys:
            print(f"\nSample entry: {key}")
            print(f"  Shape: {f[key]['features'].shape if 'features' in f[key] else 'N/A'}")
            if 'features' in f[key]:
                print(f"  Data type: {f[key]['features'].dtype}")
                
except Exception as e:
    print(f"✗ Error reading file: {e}")
