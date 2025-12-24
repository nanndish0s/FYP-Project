"""
Download all necessary CMU-MOSEI features:
1. COVAREP (Acoustic features)  
2. Labels
"""
import requests
import os
from tqdm import tqdm

def download_file(url, dest):
    """Download a file with progress bar"""
    print(f"\nDownloading: {url.split('/')[-1]}")
    print(f"Destination: {dest}")
    
    try:
        response = requests.get(url, stream=True)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest, 'wb') as f, tqdm(
                desc="Progress",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            print(f"✓ Complete: {total_size / (1024*1024):.2f} MB")
            return True
        else:
            print(f"✗ Failed: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# Base URL
base_url = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI"

# Files to download
files = [
    ("acoustic/CMU_MOSEI_COVAREP.csd", "data/raw/CMU_MOSEI_COVAREP.csd"),
    ("labels/CMU_MOSEI_Labels.csd", "data/raw/CMU_MOSEI_Labels.csd"),
]

print("=" * 60)
print("CMU-MOSEI Dataset Download")
print("=" * 60)

# Ensure directory exists
os.makedirs("data/raw", exist_ok=True)

# Download each file
for url_path, dest_path in files:
    full_url = f"{base_url}/{url_path}"
    download_file(full_url, dest_path)

print("\n" + "=" * 60)
print("Download Complete!")
print("=" * 60)
