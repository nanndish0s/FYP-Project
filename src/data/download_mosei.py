import os
import mmsdk
from mmsdk import mmdatasdk

def download_mosei(destination_dir):
    """
    Downloads CMU-MOSEI dataset features using mmsdk.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Define the dataset source (CMU-MOSEI)
    # Using the standard high-level features for now
    # We need:
    # 1. Acoustic: COVAREP (or similar if available in standard list)
    # 2. Lexical: Glove or BERT embeddings (or raw text if available as 'glove_vectors' often implies)
    # 3. Labels: All labels
    
    # Check available resources (optional, but good for debugging)
    # print(mmdatasdk.mmdataset.cmu_mosei.highlevel)

    # Recipes for standard features
    # We will try to download:
    # - CMU_MOSEI_COVAREP (Acoustic)
    # - CMU_MOSEI_TimestampedWordVectors (Lexical/Text)
    # - CMU_MOSEI_Labels (Labels)
    
    # Note: The SDK downloads .csd files.
    
    dataset = mmdatasdk.mmdataset(
        mmdatasdk.cmu_mosei.highlevel,
        destination_dir
    )

    print(f"Download complete. Files saved to {destination_dir}")

if __name__ == "__main__":
    # Path to data/raw
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'raw')
    print(f"Downloading to: {DATA_DIR}")
    download_mosei(DATA_DIR)
