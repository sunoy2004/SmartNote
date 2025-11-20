"""
Simple audio test to isolate the issue
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

def test_simple_audio():
    """Test simple audio processing"""
    try:
        print("Step 1: Creating test audio file...")
        import numpy as np
        import soundfile as sf
        
        # Generate 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        waveform = np.zeros(samples)
        
        # Save as WAV file
        test_file = "simple_test.wav"
        sf.write(test_file, waveform, sample_rate)
        print(f"✓ Created test file: {test_file}")
        
        print("Step 2: Testing librosa load...")
        import librosa
        y, sr = librosa.load(test_file, sr=None)
        print(f"✓ Loaded audio: shape={y.shape}, sr={sr}")
        
        print("Step 3: Testing audio processor...")
        from training.utils_audio import AudioProcessor
        processor = AudioProcessor()
        mel_spec = processor.process_audio(test_file)
        print(f"✓ Mel spectrogram shape: {mel_spec.shape}")
        
        print("Step 4: Testing ASR model forward pass...")
        from models.asr.asr_model import create_model
        import torch
        model = create_model()
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            output = model(mel_spec)
        print(f"✓ Model output shape: {output.shape}")
        
        print("Step 5: Testing greedy decode...")
        from models.asr.decode import greedy_decode
        transcripts = greedy_decode(output)
        print(f"✓ Decoded transcript: '{transcripts[0]}'")
        
        # Clean up
        os.remove(test_file)
        
        print("All steps completed successfully!")
        
    except Exception as e:
        print(f"Test failed at step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_audio()