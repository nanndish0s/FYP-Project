"""
Inspect the structure of downloaded CSD file in detail
"""
import h5py

csd_file = "data/raw/debug_test.csd"

with h5py.File(csd_file, 'r') as f:
    print("Root level keys:", list(f.keys()))
    
    # Navigate into glove_vectors
    gv = f['glove_vectors']
    print(f"\nglove_vectors type: {type(gv)}")
    print(f"glove_vectors keys: {list(gv.keys())[:10]}")  # Show first 10
    print(f"Total number of video IDs: {len(gv.keys())}")
    
    # Pick a sample video ID
    sample_id = list(gv.keys())[0]
    print(f"\nSample video ID: {sample_id}")
    print(f"Sample data keys: {list(gv[sample_id].keys())}")
    
    # Check the structure
    for key in gv[sample_id].keys():
        item = gv[sample_id][key]
        if isinstance(item, h5py.Dataset):
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"  {key}: type={type(item).__name__}")
