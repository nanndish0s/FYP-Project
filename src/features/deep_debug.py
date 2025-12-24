"""Deep dive into COVAREP structure"""
import h5py

with h5py.File('data/raw/CMU_MOSEI_COVAREP.csd', 'r') as f:
    print("Level 1 - Root:")
    print(f"  Keys: {list(f.keys())}")
    
    covarep_group = f['COVAREP']
    print("\nLevel 2 - COVAREP:")
    print(f"  Type: {type(covarep_group)}")
    print(f"  Keys: {list(covarep_group.keys())}")
    
    # Check if 'data' is accessible
    print("\nAttempting to access 'data'...")
    try:
        data_item = covarep_group['data']
        print(f"  Type of 'data': {type(data_item)}")
        
        if hasattr(data_item, 'keys'):
            print(f"  Keys in data: {list(data_item.keys())[:5]}")
        else:
            # It might be a dataset
            print(f"  Shape: {data_item.shape}")
            print(f"  Dtype: {data_item.dtype}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Check metadata
    print("\nChecking 'metadata'...")
    try:
        metadata_item = covarep_group['metadata']
        print(f"  Type: {type(metadata_item)}")
        if hasattr(metadata_item, 'shape'):
            print(f"  Shape: {metadata_item.shape}")
            print(f"  Dtype: {metadata_item.dtype}")
            # Read first few entries
            meta_data = metadata_item[()]
            print(f"  First 5 entries: {meta_data[:5]}")
    except Exception as e:
        print(f"  Error: {e}")
