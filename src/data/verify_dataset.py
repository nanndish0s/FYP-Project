"""
Verify the integrity and structure of downloaded CMU-MOSEI dataset files
"""
import h5py
import os

files_to_verify = [
    ("data/raw/debug_test.csd", "TimestampedWordVectors (Lexical)"),
    ("data/raw/CMU_MOSEI_COVAREP.csd", "COVAREP (Acoustic)"),
    ("data/raw/CMU_MOSEI_Labels.csd", "Labels"),
]

print("=" * 70)
print("CMU-MOSEI Dataset Verification")
print("=" * 70)

all_valid = True

for filepath, description in files_to_verify:
    print(f"\n{description}")
    print(f"File: {filepath}")
    
    # Check file exists
    if not os.path.exists(filepath):
        print(f"  ✗ File not found!")
        all_valid = False
        continue
    
    # Check file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    
    # Verify HDF5 structure
    try:
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            print(f"  ✓ Valid HDF5 file")
            print(f"  Root keys: {keys}")
            
            # For each root key, check if it's a group and count entries
            for key in keys:
                if isinstance(f[key], h5py.Group):
                    num_entries = len(f[key].keys())
                    print(f"    - {key}: {num_entries:,} entries")
                    
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
        all_valid = False

print("\n" + "=" * 70)
if all_valid:
    print("✓ All files verified successfully!")
else:
    print("✗ Some files have issues")
print("=" * 70)
