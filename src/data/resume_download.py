"""
Resume CMU-MOSEI dataset download with retry logic
Downloads COVAREP and Labels files
"""
import requests
import os
from tqdm import tqdm
import time

def download_file_with_resume(url, dest, max_retries=5):
    """Download a file with resume capability and retry logic"""
    
    # Check if file exists and get its size
    resume_pos = 0
    if os.path.exists(dest):
        resume_pos = os.path.getsize(dest)
        print(f"Found existing file: {resume_pos / (1024*1024):.2f} MB")
    
    headers = {}
    if resume_pos > 0:
        headers['Range'] = f'bytes={resume_pos}-'
        print(f"Resuming download from byte {resume_pos:,}")
    
    retries = 0
    while retries < max_retries:
        try:
            print(f"\nDownloading: {url.split('/')[-1]}")
            print(f"Destination: {dest}")
            print(f"Attempt: {retries + 1}/{max_retries}")
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # Check if server supports resume
            if response.status_code == 206:
                print("✓ Server supports resume, continuing from where we left off")
                mode = 'ab'  # Append mode
            elif response.status_code == 200:
                print("✓ Server doesn't support resume, starting fresh")
                resume_pos = 0
                mode = 'wb'  # Write mode
            else:
                print(f"✗ Unexpected status code: {response.status_code}")
                return False
            
            # Get total file size
            if 'content-length' in response.headers:
                total_size = int(response.headers['content-length'])
                if response.status_code == 206:
                    total_size += resume_pos
            else:
                total_size = 0
            
            print(f"Total size: {total_size / (1024*1024):.2f} MB")
            
            # Download with progress bar
            with open(dest, mode) as f, tqdm(
                desc="Progress",
                initial=resume_pos,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)
            
            print(f"\n✓ Download complete: {total_size / (1024*1024):.2f} MB")
            return True
            
        except Exception as e:
            retries += 1
            print(f"\n✗ Error: {e}")
            if retries < max_retries:
                wait_time = min(2 ** retries, 60)  # Exponential backoff, max 60s
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                return False
    
    return False

# Base URL
base_url = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI"

# Files to download/resume
files = [
    ("acoustic/CMU_MOSEI_COVAREP.csd", "data/raw/CMU_MOSEI_COVAREP.csd"),
    ("labels/CMU_MOSEI_Labels.csd", "data/raw/CMU_MOSEI_Labels.csd"),
]

print("=" * 70)
print("CMU-MOSEI Dataset Download (Resume)")
print("=" * 70)

# Ensure directory exists
os.makedirs("data/raw", exist_ok=True)

# Download each file
success_count = 0
for url_path, dest_path in files:
    full_url = f"{base_url}/{url_path}"
    print(f"\n{'='*70}")
    if download_file_with_resume(full_url, dest_path):
        success_count += 1
    print(f"{'='*70}")

print(f"\n\n{'='*70}")
print(f"Download Summary: {success_count}/{len(files)} files completed")
print("=" * 70)
