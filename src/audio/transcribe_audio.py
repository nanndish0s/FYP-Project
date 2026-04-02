"""
Speech-to-Text Module using OpenAI Whisper
"""
import whisper
import os
import soundfile as sf
import numpy as np

class SpeechTranscriber:
    def __init__(self, model_size="base"):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
                       - tiny: fastest, least accurate
                       - base: good balance (RECOMMENDED)
                       - small: better accuracy, slower
        """
        print(f" Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        print(f" Model loaded!")
        
    def transcribe(self, audio_path, language="en"):
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (en for English)
            
        Returns:
            Transcript text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"\n Transcribing audio...")
        
        # Load audio with soundfile (bypasses ffmpeg requirement)
        audio_data, sample_rate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Convert to float32 and normalize to [-1, 1]
        audio_data = audio_data.astype(np.float32)
        if audio_data.dtype == np.int16:
            audio_data = audio_data / 32768.0
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            print(f"   Resampling from {sample_rate}Hz to 16000Hz...")
            # Simple resampling
            import scipy.signal
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = scipy.signal.resample(audio_data, num_samples)
        
        # Transcribe with numpy array instead of file path
        result = self.model.transcribe(
            audio_data,
            language=language,
            fp16=False  # Use FP32 for CPU compatibility
        )
        
        transcript = result["text"].strip()
        
        print(f" Transcription complete!")
        print(f"   Words: {len(transcript.split())}")
        print(f"   Characters: {len(transcript)}")
        
        return transcript
    
    def transcribe_with_timestamps(self, audio_path, language="en"):
        """
        Transcribe with word-level timestamps
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Dict with text and segments
        """
        print(f"\n Transcribing with timestamps...")
        
        # Load audio with soundfile
        audio_data, sample_rate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Convert to float32
        audio_data = audio_data.astype(np.float32)
        if audio_data.dtype == np.int16:
            audio_data = audio_data / 32768.0
        
        # Resample if needed
        if sample_rate != 16000:
            import scipy.signal
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = scipy.signal.resample(audio_data, num_samples)
        
        result = self.model.transcribe(
            audio_data,
            language=language,
            word_timestamps=True,
            fp16=False
        )
        
        return {
            'text': result['text'].strip(),
            'segments': result['segments']
        }

# Example usage
if __name__ == "__main__":
    transcriber = SpeechTranscriber(model_size="base")
    
    # Test transcription
    # transcript = transcriber.transcribe("data/temp/test_recording.wav")
    # print(f"\nTranscript: {transcript}")
