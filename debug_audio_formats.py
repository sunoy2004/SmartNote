import numpy as np
import soundfile as sf
import librosa
import os

def test_audio_formats():
    """Test different audio formats to see which ones work with librosa"""
    try:
        # Create a simple tone (440 Hz sine wave) for 2 seconds
        print("Creating test audio file...")
        duration = 2.0  # seconds
        sample_rate = 16000
        frequency = 440.0  # Hz
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Test different formats
        formats = [
            ("test_audio.wav", "WAV"),
            ("test_audio.flac", "FLAC"),
        ]
        
        for filename, format_name in formats:
            # Save file
            sf.write(filename, audio_data, sample_rate)
            print(f"\nTesting {format_name} format ({filename}):")
            print(f"File size: {os.path.getsize(filename)} bytes")
            
            # Try to load with librosa
            try:
                waveform, sr = librosa.load(filename, sr=None)
                print(f"✓ Loaded successfully: {waveform.shape}, sample rate: {sr}")
                
                # Check if resampling works
                if sr != 16000:
                    waveform_resampled = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
                    print(f"✓ Resampled successfully: {waveform_resampled.shape}")
                else:
                    print("✓ No resampling needed")
                    
            except Exception as e:
                print(f"✗ Failed to load: {e}")
            
            # Clean up
            os.remove(filename)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_formats()