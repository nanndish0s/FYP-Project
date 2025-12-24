"""
Debug script to understand the TimestampedWordVectors structure
"""
import h5py

csd_file = "data/raw/debug_test.csd"

with h5py.File(csd_file, 'r') as f:
    print("Root keys:", list(f.keys()))
    
    gv = f['glove_vectors']
    print(f"\nglove_vectors type: {type(gv)}")
    
    # Get first few video IDs
    video_ids = list(gv.keys())[:5]
    print(f"\nFirst 5 video IDs: {video_ids}")
    
    # Inspect structure of first video
    first_id = video_ids[0]
    print(f"\n\nDetailed structure of '{first_id}':")
    video_group = gv[first_id]
    
    print(f"Type: {type(video_group)}")
    print(f"Keys: {list(video_group.keys())}")
    
    for key in video_group.keys():
        item = video_group[key]
        if isinstance(item, h5py.Dataset):
            print(f"\n  {key}:")
            print(f"    Type: Dataset")
            print(f"    Shape: {item.shape}")
            print(f"    Dtype: {item.dtype}")
            print(f"    Sample data (first 3):")
            print(f"      {item[:min(3, len(item))]}")
        else:
            print(f"\n  {key}: {type(item).__name__}")
