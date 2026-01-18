
import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def ingest_recruitview():
    """
    Processes the locally downloaded metadata.jsonl from RecruitView.
    """
    jsonl_path = 'data/raw/recruitview/metadata.jsonl'
    output_path = 'data/processed/recruitview_metadata.csv'
    
    if not os.path.exists(jsonl_path):
        print(f"❌ Metadata file not found at {jsonl_path}")
        return None

    print(f"🚀 Processing {jsonl_path}...")
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            data.append({
                'video_id': sample.get('id', ''),
                'file_name': sample.get('file_name', ''),
                'transcript': sample.get('transcript', ''),
                'question': sample.get('question', ''),
                # Big Five
                'openness': sample.get('openness', 0),
                'conscientiousness': sample.get('conscientiousness', 0),
                'extraversion': sample.get('extraversion', 0),
                'agreeableness': sample.get('agreeableness', 0),
                'neuroticism': sample.get('neuroticism', 0),
                'overall_personality': sample.get('overall_personality', 0),
                # Performance metrics
                'interview_score': sample.get('interview_score', 0),
                'overall_performance': sample.get('overall_performance', 0)
            })
            
    df = pd.DataFrame(data)
    
    # ---------------------------------------------------------
    # SCALE MAPPING (RecruitView ~[-2, 2] -> C3 [1, 5])
    # ---------------------------------------------------------
    # Simple linear transformation: (val - min) / (max - min) * 4 + 1
    # For normalized data around 0, we'll assume a range of roughly -1.5 to 1.5 for the bulk
    def map_to_5_scale(col):
        # Using a conservative clipping to ensure 1-5 range
        min_v, max_v = -1.5, 1.5
        scaled = (col.clip(min_v, max_v) - min_v) / (max_v - min_v) * 4 + 1
        return scaled.round(2)

    df['curiosity_score'] = map_to_5_scale(df['openness'])
    df['critical_thinking_score'] = map_to_5_scale(df['conscientiousness'])
    
    # Decouple Creativity: Use a blend of High Openness and slightly lower Conscientiousness 
    # (Divergent thinking often correlates with lower order/rule-following in creative bursts)
    # Creativity = 0.7 * Openness + 0.3 * (Inverse of Conscientiousness)
    df['creativity_score'] = map_to_5_scale(0.7 * df['openness'] + 0.3 * (-df['conscientiousness']))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Processed {len(df)} samples into {output_path}")
    print("\nMapped C3 Score Previews (1-5 scale):")
    print(df[['curiosity_score', 'critical_thinking_score', 'creativity_score']].head())
    
    return df

if __name__ == "__main__":
    ingest_recruitview()
