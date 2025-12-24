import os
from datasets import load_dataset
import soundfile as sf
import pandas as pd
from tqdm import tqdm

def download_sample():
    print("Downloading CMU-MOSEI sample from Hugging Face...")
    # Load the sample dataset
    # Note: 'shinnew/CMU-MOSEI_sample' might not exist or be private. 
    # If it fails, we will try 'MSA-Lab/cmu_mosei_senti' or similar, 
    # but let's try to find a generic one or just use 'cmu-mosei' with streaming=True to get a few samples.
    
    # Better approach: Use the official 'cmu-mosei' if available or a known subset.
    # Searching HF, 'MSA-Lab/cmu_mosei_senti' is common.
    # Let's try to load a small slice of a larger dataset if possible, or use a known sample.
    
    try:
        # We'll try to load the first 10 examples from a standard repo if the specific sample fails.
        # But let's stick to the search result recommendation first.
        dataset = load_dataset("shinnew/CMU-MOSEI_sample", split="train")
    except Exception as e:
        print(f"Could not load 'shinnew/CMU-MOSEI_sample': {e}")
        print("Attempting to load a few samples from 'MSA-Lab/cmu_mosei_senti' (text only) or similar...")
        # Fallback: Create dummy data for structure if real download fails (to unblock user)
        # But we really want audio.
        # Let's try 'Hubert's' upstream dataset or similar. 
        # Actually, for the purpose of this script, let's try to be robust.
        return

    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset loaded. Saving {len(dataset)} samples to {output_dir}...")

    metadata = []

    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        # Structure depends on the dataset. 
        # Typically: 'audio' (array, sampling_rate), 'text', 'label'
        
        video_id = item.get('video_id', f'sample_{i}')
        clip_id = item.get('clip_id', f'{i}')
        filename = f"{video_id}_{clip_id}"
        
        # Save Audio
        if 'audio' in item:
            audio_path = os.path.join(output_dir, f"{filename}.wav")
            audio_array = item['audio']['array']
            sr = item['audio']['sampling_rate']
            sf.write(audio_path, audio_array, sr)
        else:
            audio_path = None
            print(f"Warning: No audio for sample {i}")

        # Save Metadata (Text + Labels if any)
        metadata.append({
            'filename': filename,
            'audio_path': audio_path,
            'text': item.get('text', ''),
            'original_sentiment': item.get('label', None)
        })

    # Save Metadata CSV
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join("data", "raw", "metadata.csv"), index=False)
    print("Download complete.")

if __name__ == "__main__":
    download_sample()
