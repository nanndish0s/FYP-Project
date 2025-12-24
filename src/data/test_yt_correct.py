"""
Test the correct youtube-transcript-api usage with v1.2.3
"""
from youtube_transcript_api import YouTubeTranscriptApi

# Test with a known video
test_video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up

try:
    print(f"Testing with video ID: {test_video_id}")
    
    # Create API instance and fetch
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(test_video_id)
    
    print(f"Success! Type: {type(fetched_transcript)}")
    print(f"Number of snippets: {len(fetched_transcript)}")
    
    # Convert to raw data
    raw_data = fetched_transcript.to_raw_data()
    print(f"\nFirst 3 entries:")
    for i, entry in enumerate(raw_data[:3]):
        print(f"  {i+1}. {entry}")
    
    # Combine text
    full_text = ' '.join([entry['text'] for entry in raw_data])
    print(f"\nFull text (first 150 chars): {full_text[:150]}...")
    print(f"Total word count: {len(full_text.split())}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
