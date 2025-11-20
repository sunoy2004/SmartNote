import numpy as np
import soundfile as sf
import librosa
import os
import tempfile

def test_webm_processing():
    """Test if webm audio files can be processed correctly"""
    try:
        # Create a simple tone (440 Hz sine wave) for 3 seconds
        print("Creating test audio file...")
        duration = 3.0  # seconds
        sample_rate = 16000
        frequency = 440.0  # Hz
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Test WAV format
        wav_filename = "test_audio.wav"
        sf.write(wav_filename, audio_data, sample_rate)
        print(f"Created WAV file: {wav_filename}")
        print(f"WAV file size: {os.path.getsize(wav_filename)} bytes")
        
        # Try to load WAV with librosa
        try:
            waveform, sr = librosa.load(wav_filename, sr=None)
            print(f"✓ WAV loaded successfully: {waveform.shape}, sample rate: {sr}")
        except Exception as e:
            print(f"✗ Failed to load WAV: {e}")
        
        # Test WEBM format (create a WAV file but with .webm extension)
        webm_filename = "test_audio.webm"
        sf.write(webm_filename, audio_data, sample_rate)
        print(f"\nCreated WEBM file: {webm_filename}")
        print(f"WEBM file size: {os.path.getsize(webm_filename)} bytes")
        
        # Try to load WEBM with librosa
        try:
            waveform, sr = librosa.load(webm_filename, sr=None)
            print(f"✓ WEBM loaded successfully: {waveform.shape}, sample rate: {sr}")
        except Exception as e:
            print(f"✗ Failed to load WEBM: {e}")
        
        # Clean up
        os.unlink(wav_filename)
        os.unlink(webm_filename)
        print("\n✓ Test completed successfully")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_webm_processing()