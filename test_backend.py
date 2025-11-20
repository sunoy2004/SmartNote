"""
Test script to verify backend functionality
"""
import requests
import sys
import os

def test_backend():
    """Test the backend voice-to-notes endpoint"""
    try:
        # Test health endpoint first
        print("Testing health endpoint...")
        health_response = requests.get("http://localhost:8000/health")
        print(f"Health status: {health_response.status_code}")
        print(f"Health response: {health_response.json()}")
        
        # Create a simple test WAV file with silence
        import numpy as np
        import soundfile as sf
        
        # Generate 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        waveform = np.zeros(samples)
        
        # Save as WAV file
        test_file = "test_silence.wav"
        sf.write(test_file, waveform, sample_rate)
        print(f"Created test file: {test_file}")
        
        # Test voice-to-notes endpoint
        print("Testing voice-to-notes endpoint...")
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'audio/wav')}
            response = requests.post("http://localhost:8000/voice-to-notes", files=files)
        
        print(f"Voice-to-notes status: {response.status_code}")
        if response.status_code == 200:
            print(f"Voice-to-notes response: {response.json()}")
        else:
            print(f"Error response: {response.text}")
            
        # Clean up test file
        os.remove(test_file)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backend()