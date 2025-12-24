import requests
import os
from tqdm import tqdm

url = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedWordVectors.csd"
dest = "data/raw/debug_test.csd"

print(f"Attempting to download {url}")
print(f"Destination: {dest}")

try:
    response = requests.get(url, stream=True)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        print(f"\n✓ Download successful! File saved to {dest}")
        print(f"  Size: {total_size / (1024*1024):.2f} MB")
    else:
        print("Download failed with status code:", response.status_code)
except Exception as e:
    print(f"An error occurred: {e}")
