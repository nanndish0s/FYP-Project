"""
Audio Recording Module
Records audio from microphone for live assessment
"""
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        """
        Initialize audio recorder
        
        Args:
            sample_rate: Recording sample rate (16kHz for Whisper)
            channels: Number of audio channels (1=mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        
    def record(self, duration=60, output_path="temp_recording.wav", callback=None):
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            output_path: Where to save the recording
            callback: Optional callback function(audio_chunk, frame) for live updates
            
        Returns:
            Path to saved audio file
        """
        print(f"🎤 Recording for {duration} seconds...")
        print("   Speak now!")
        
        try:
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16'
            )
            sd.wait()  # Wait for recording to finish
            
            # Save to file
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            wav.write(output_path, self.sample_rate, audio_data)
            
            print(f"✅ Recording saved to: {output_path}")
            
            # Validate recording
            if self._validate_audio(audio_data):
                return output_path
            else:
                print("⚠️  Warning: Audio quality may be low")
                return output_path
                
        except Exception as e:
            print(f"❌ Recording error: {str(e)}")
            raise
    
    def _validate_audio(self, audio_data):
        """
        Validate audio quality
        
        Args:
            audio_data: Numpy array of audio samples
            
        Returns:
            True if audio is acceptable quality
        """
        # Check if audio is not silent
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Much more lenient threshold (was 100, now 10)
        if rms < 10:  # Completely silent
            print("   ⚠️ Audio seems very quiet. Please speak louder.")
            return False
        
        return True
    
    def list_devices(self):
        """List available audio input devices"""
        print("\n📻 Available Audio Devices:")
        print(sd.query_devices())
        
    def test_microphone(self, duration=3):
        """
        Test microphone with short recording
        
        Args:
            duration: Test duration in seconds
        """
        print(f"\n🧪 Testing microphone ({duration} seconds)...")
        print("   Say something!")
        
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16'
        )
        sd.wait()
        
        rms = np.sqrt(np.mean(audio_data**2))
        print(f"\n   Audio level (RMS): {rms:.0f}")
        
        # Much more lenient thresholds
        if rms < 10:
            print("   ❌ Microphone level too low!")
            print("   💡 Try speaking louder or adjusting microphone settings")
            print("   💡 Check Windows microphone permissions")
        elif rms > 5000:
            print("   ⚠️  Audio might be too loud (possible clipping)")
        else:
            print(f"   ✅ Microphone working well! (Level: {rms:.0f})")
        
        return rms

# Example usage
if __name__ == "__main__":
    recorder = AudioRecorder()
    
    # List devices
    recorder.list_devices()
    
    # Test microphone
    recorder.test_microphone(duration=3)
    
    # Record
    # audio_file = recorder.record(duration=30, output_path="data/temp/test_recording.wav")
