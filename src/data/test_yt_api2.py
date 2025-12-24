"""
Test youtube-transcript-api by checking available methods
"""
from youtube_transcript_api import YouTubeTranscriptApi
import inspect

print("Methods in YouTubeTranscriptApi:")
for name, method in inspect.getmembers(YouTubeTranscriptApi, predicate=inspect.ismethod):
    sig = inspect.signature(method)
    print(f"  {name}{sig}")

print("\n\nTrying list_transcripts:")
test_video_id = "dQw4w9WgXcQ"

try:
    # Try list method
    transcript_list = YouTubeTranscriptApi.list_transcripts(test_video_id)
    print(f"list_transcripts returned: {type(transcript_list)}")
    print(f"Content: {transcript_list}")
    
    # Try to get the first transcript
    for transcript in transcript_list:
        print(f"\nTranscript: {transcript}")
        # Try to fetch the actual text
        fetched = transcript.fetch()
        print(f"Fetched {len(fetched)} entries")
        print(f"First entry: {fetched[0]}")
        break
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
