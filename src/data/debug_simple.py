"""
Debug script to understand the TimestampedWordVectors structure - simpler version
"""
import h5py

csd_file = "data/raw/debug_test.csd"

with h5py.File(csd_file, 'r') as f:
    print("Root keys:", list(f.keys()))
    
    gv = f['glove_vectors']
    
    # Get first video ID
    video_ids = list(gv.keys())
    first_id = video_ids[0]
    
    print(f"\nFirst video ID: '{first_id}'")
    
    video_group = gv[first_id]
    print(f"Keys in video group: {list(video_group.keys())}")
    
    # Check each key
    for key in list(video_group.keys())[:5]:  # First 5 keys only
        item = video_group[key]
        print(f"\n{key}:")
        if isinstance(item, h5py.Dataset):
            print(f"  Dataset - Shape: {item.shape}, Dtype: {item.dtype}")
            if len(item) > 0 and len(item) < 10:
                print(f"  Data: {item[:]}")
        elif isinstance(item, h5py.Group):
            print(f"  Group - Keys: {list(item.keys())[:5]}")
            # Check what's inside the group
            if len(item.keys()) > 0:
                first_subkey = list(item.keys())[0]
                subitem = item[first_subkey]
                if isinstance(subitem, h5py.Dataset):
                    print(f"    First item '{first_subkey}': Dataset - Shape: {subitem.shape}, Dtype: {subitem.dtype}")
