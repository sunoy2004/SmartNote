import numpy as np
import soundfile as sf
import requests
import os

def create_and_test():
    """Create a test audio file with a simple tone and test the backend processing"""
    try:
        # Create a simple tone (440 Hz sine wave) for 2 seconds
        print("Creating test audio file...")
        duration = 2.0  # seconds
        sample_rate = 16000
        frequency = 440.0  # Hz
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Save as WAV file
        filename = "test_tone.wav"
        sf.write(filename, audio_data, sample_rate)
        print(f"Audio saved to {filename}")
        
        # Check file size
        file_size = os.path.getsize(filename)
        print(f"File size: {file_size} bytes")
        
        # Test the backend
        url = "http://localhost:8000/voice-to-notes"
        with open(filename, "rb") as f:
            files = {"file": (filename, f, "audio/wav")}
            print("Sending to backend...")
            response = requests.post(url, files=files)
            
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Transcript: {result['transcript']}")
            print(f"Summary: {result['summary']}")
        else:
            print(f"Error: {response.text}")
            
        # Clean up
        os.remove(filename)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_and_test()