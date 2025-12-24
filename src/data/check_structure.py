"""
Minimal script to check HDF5 structure
"""
import h5py
import sys

csd_file = "data/raw/debug_test.csd"

with h5py.File(csd_file, 'r') as f:
    gv = f['glove_vectors']
    
    # Get first video ID
    first_id = list(gv.keys())[0]
    video_group = gv[first_id]
    
    keys = list(video_group.keys())
    
    with open("structure_info.txt", "w") as out:
        out.write(f"First video ID: {first_id}\n")
        out.write(f"Number of keys: {len(keys)}\n")
        out.write(f"Keys: {keys[:10]}\n\n")
        
        # Check first few keys
        for key in keys[:5]:
            item = video_group[key]
            out.write(f"\n{key}: {type(item).__name__}\n")
            if isinstance(item, h5py.Dataset):
                out.write(f"  Shape: {item.shape}, Dtype: {item.dtype}\n")
                if item.shape[0] < 100:
                    out.write(f"  Sample: {item[:]}\n")
            elif isinstance(item, h5py.Group):
                subkeys = list(item.keys())
                out.write(f"  Subkeys ({len(subkeys)}): {subkeys[:5]}\n")

print("Written to structure_info.txt")
