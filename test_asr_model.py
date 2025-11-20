import numpy as np
import soundfile as sf
import librosa
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

def test_asr_model():
    """Test the ASR model with a known audio file"""
    try:
        print("Testing ASR model...")
        
        # Create a simple test tone (not speech, but should produce some output)
        print("Creating test audio...")
        duration = 2.0  # seconds
        sample_rate = 16000
        frequency = 440.0  # Hz
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Save as WAV file
        test_file = "test_tone.wav"
        sf.write(test_file, audio_data, sample_rate)
        print(f"Created test file: {test_file}")
        
        # Test the ASR handler
        print("Loading ASR handler...")
        from backend.asr_handler import ASRHandler
        
        # Initialize ASR handler
        asr_handler = ASRHandler()
        print("ASR handler loaded successfully")
        
        # Test transcription
        print("Testing transcription...")
        transcript = asr_handler.transcribe(test_file)
        print(f"Transcript: '{transcript}'")
        
        # Clean up
        os.remove(test_file)
        
        print("\nModel checkpoint information:")
        model_path = os.path.join("checkpoints", "asr", "best_model.pth")
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"ASR model exists: {model_path} ({size} bytes)")
        else:
            print(f"ASR model not found: {model_path}")
            
    except Exception as e:
        print(f"Error testing ASR model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_asr_model()