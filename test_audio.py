import numpy as np
import soundfile as sf
import librosa
import os

def create_test_audio():
    """Create a simple test audio file with a known phrase"""
    # Create a simple tone with spoken words
    duration = 3.0  # seconds
    sample_rate = 16000
    
    # Generate a simple waveform with some speech-like characteristics
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a combination of frequencies to simulate speech
    # This is just for testing - real speech would be much more complex
    frequencies = [440, 660, 880, 1100]  # Multiple frequencies
    waveform = np.zeros_like(t)
    for freq in frequencies:
        waveform += np.sin(2 * np.pi * freq * t) * 0.25
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.05, len(t))
    waveform = waveform + noise
    
    # Normalize
    waveform = waveform / np.max(np.abs(waveform))
    
    # Save as WAV file
    filename = "test_audio.wav"
    sf.write(filename, waveform, sample_rate)
    print(f"Audio saved to {filename}")
    print(f"File size: {os.path.getsize(filename)} bytes")
    
    # Also save as WebM for testing
    try:
        # Convert to WebM using librosa and soundfile
        webm_filename = "test_audio.webm"
        sf.write(webm_filename, waveform, sample_rate, format='OGG', subtype='OPUS')
        print(f"WebM saved to {webm_filename}")
        print(f"WebM file size: {os.path.getsize(webm_filename)} bytes")
    except Exception as e:
        print(f"Could not create WebM file: {e}")
    
    return filename

if __name__ == "__main__":
    create_test_audio()