"""
Extract lexical (text-based) features from transcripts
Simplified version using basic Python - no NLTK tokenizers
"""
import pandas as pd
import numpy as np
import re
from collections import Counter
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
from tqdm import tqdm

print("=" * 80)
print("Lexical Feature Extraction from Transcripts")
print("=" * 80)

# Load transcripts
print("\n1. Loading C3-labeled transcripts...")
df = pd.read_csv('data/processed/combined_44_with_c3_labels.csv')
print(f"   Loaded {len(df)} transcripts")

# Filler words to detect
FILLER_WORDS = [
    'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 
    'sort of', 'kind of', 'basically', 'actually', 'literally'
]

# Common stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
}

def simple_tokenize(text):
    """Simple word tokenization"""
    # Remove punctuation except apostrophes and hyphens
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    return words

def simple_sent_tokenize(text):
    """Simple sentence tokenization"""
    # Split on period, exclamation, question mark
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_lexical_features(text):
    """Extract comprehensive lexical features from text"""
    features = {}
    
    # Basic tokenization
    words = simple_tokenize(text)
    sentences = simple_sent_tokenize(text)
    
    features['word_count'] = len(words)
    features['sentence_count'] = len(sentences) if sentences else 1
    features['char_count'] = len(text)
    
    # Average lengths
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
    
    # Vocabulary diversity (unique words / total words)
    unique_words = set(words)
    features['unique_word_count'] = len(unique_words)
    features['vocab_diversity'] = len(unique_words) / len(words) if words else 0
    
    # Stopwords ratio
    stopword_count = sum(1 for w in words if w in STOPWORDS)
    features['stopword_ratio'] = stopword_count / len(words) if words else 0
    
    # Filler words count and ratio
    filler_count = 0
    text_lower = text.lower()
    for filler in FILLER_WORDS:
        filler_count += len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
    features['filler_word_count'] = filler_count
    features['filler_word_ratio'] = filler_count / len(words) if words else 0
    
    # Question marks (indicates questioning/curiosity)
    features['question_mark_count'] = text.count('?')
    features['question_ratio'] = features['question_mark_count'] / features['sentence_count']
    
    # Exclamation marks (indicates emphasis/emotion)
    features['exclamation_count'] = text.count('!')
    
    # Readability metrics
    try:
        features['flesch_reading_ease'] = flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
        features['automated_readability_index'] = automated_readability_index(text)
    except:
        features['flesch_reading_ease'] = 0
        features['flesch_kincaid_grade'] = 0
        features['automated_readability_index'] = 0
    
    # Long words (> 6 characters) - indicates vocabulary sophistication
    long_words = [w for w in words if len(w) > 6]
    features['long_word_count'] = len(long_words)
    features['long_word_ratio'] = len(long_words) / len(words) if words else 0
    
    # Numbers count (might indicate data-driven thinking)
    numbers = re.findall(r'\d+', text)
    features['number_count'] = len(numbers)
    features['number_ratio'] = len(numbers) / len(words) if words else 0
    
    # Capital letters ratio (excluding sentence starts) - might indicate emphasis
    capitals = sum(1 for c in text if c.isupper())
    features['capital_ratio'] = capitals / len(text) if text else 0
    
    return features

# Extract features for all transcripts
print("\n2. Extracting lexical features...")
lexical_features = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing transcripts"):
    features = extract_lexical_features(row['transcript'])
    features['video_id'] = row['video_id']
    lexical_features.append(features)

# Create DataFrame
print("\n3. Creating feature DataFrame...")
df_lexical = pd.DataFrame(lexical_features)

# Move video_id to first column
cols = ['video_id'] + [col for col in df_lexical.columns if col != 'video_id']
df_lexical = df_lexical[cols]

print(f"   Shape: {df_lexical.shape}")
print(f"   Features per transcript: {df_lexical.shape[1] - 1}")
print(f"   Transcripts: {df_lexical.shape[0]}")

# Save to CSV
output_path = 'data/processed/lexical_features_44.csv'
df_lexical.to_csv(output_path, index=False)

print(f"\n4. Saved lexical features to: {output_path}")

# Show sample
print(f"\n5. Sample features (first 5 columns):")
print(df_lexical.iloc[:5, :6])

# Show feature statistics
print(f"\n6. Feature Statistics:")
stats = df_lexical.iloc[:, 1:].describe()
print(stats.iloc[:, :5])  # First 5 feature stats

print("\n" + "=" * 80)
print("Lexical Feature Extraction Complete!")
print("=" * 80)
print(f"\nSummary:")
print(f"- Transcripts processed: {len(lexical_features)}")
print(f"- Features extracted: {df_lexical.shape[1] - 1}")
print(f"- Output: {output_path}")
print(f"\n✓ Ready to merge with acoustic features!")
