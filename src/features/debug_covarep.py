"""Quick debug of COVAREP structure"""
import h5py

print("Opening COVAREP file...")
with h5py.File('data/raw/CMU_MOSEI_COVAREP.csd', 'r') as f:
    print(f"Root keys: {list(f.keys())}")
    
    if 'data' in f:
        data_group = f['data']
        all_keys = list(data_group.keys())
        print(f"\nTotal videos in 'data': {len(all_keys)}")
        print(f"First 10 video IDs: {all_keys[:10]}")
        print(f"\nLast 10 video IDs: {all_keys[-10:]}")
        
        # Check format of video IDs  
        print(f"\nSample video ID format: '{all_keys[0]}'")
        print(f"Type: {type(all_keys[0])}")
        
        # Try to load one
        sample_data = data_group[all_keys[0]][()]
        print(f"\nSample data shape: {sample_data.shape}")
