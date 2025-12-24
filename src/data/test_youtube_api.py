"""
Test the correct youtube-transcript-api usage
"""
from youtube_transcript_api import YouTubeTranscriptApi

# Test with a known video
test_video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up (known to have captions)

try:
    print(f"Testing with video ID: {test_video_id}")
    
    # Fetch transcript (list of transcript entries)
    transcript = YouTubeTranscriptApi.fetch(test_video_id)
    
    print(f"Success! Got transcript")
    print(f"Type: {type(transcript)}")
    print(f"Content: {transcript}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
