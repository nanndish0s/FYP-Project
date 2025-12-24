"""Check nested structure of video data"""
import h5py

with h5py.File('data/raw/CMU_MOSEI_COVAREP.csd', 'r') as f:
    data_group = f['COVAREP']['data']
    
    # Get first video
    video_id = 'jXQmVFcOiUI'
    print(f"Video ID: {video_id}")
    
    video_group = data_group[video_id]
    print(f"Type of video_group: {type(video_group)}")
    
    if hasattr(video_group, 'keys'):
        print(f"Keys in video_group: {list(video_group.keys())}")
        
        # Try to access 'features' or 'data' within video group
        for key in video_group.keys():
            item = video_group[key]
            print(f"\n  Key '{key}':")
            print(f"    Type: {type(item)}")
            if hasattr(item, 'shape'):
                data = item[()]
                print(f"    Shape: {data.shape}")
                print(f"    Dtype: {data.dtype}")
                print(f"    Sample: {data[:3] if len(data.shape) == 1 else data[:3, :5]}")
    else:
        # It's a dataset
        data = video_group[()]
        print(f"Shape: {data.shape}")
