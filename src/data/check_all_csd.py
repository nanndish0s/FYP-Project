"""
Check if there's a raw text/transcript file in CMU-MOSEI
"""
import h5py

# Check all CSD files for text data
files_to_check = [
    "data/raw/debug_test.csd",  # TimestampedWordVectors
    "data/raw/CMU_MOS EI_Labels.csd",  # Labels
]

for filepath in files_to_check:
    try:
        print(f"\n{'='*70}")
        print(f"Checking: {filepath}")
        print('='*70)
        
        with h5py.File(filepath, 'r') as f:
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"  Group: {name}")
            
            # Visit all items recursively
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"Error: {e}")
