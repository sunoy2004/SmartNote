import numpy as np
import soundfile as sf
import librosa
import torch
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

def debug_asr_output():
    """Debug the ASR model outputs directly"""
    try:
        print("Debugging ASR model outputs...")
        
        # Create a simple test tone
        print("Creating test audio...")
        duration = 2.0  # seconds
        sample_rate = 16000
        frequency = 440.0  # Hz
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Save as WAV file
        test_file = "debug_tone.wav"
        sf.write(test_file, audio_data, sample_rate)
        print(f"Created test file: {test_file}")
        
        # Load and process audio like the ASR handler does
        print("Processing audio...")
        from training.utils_audio import AudioProcessor
        audio_processor = AudioProcessor()
        mel_spec = audio_processor.process_audio(test_file)
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        
        # Load model
        print("Loading ASR model...")
        from models.asr.asr_model import create_model
        model = create_model()
        
        # Load checkpoint
        checkpoint_path = os.path.join("checkpoints", "asr", "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully")
        else:
            print(f"Model checkpoint not found: {checkpoint_path}")
            return
            
        model.eval()
        
        # Run inference
        print("Running inference...")
        with torch.no_grad():
            log_probs = model(mel_spec)
            print(f"Log probs shape: {log_probs.shape}")
            
            # Check the range of log probabilities
            print(f"Log probs min: {log_probs.min()}, max: {log_probs.max()}")
            
            # Check if all values are very negative (low confidence)
            if log_probs.max() < -10:
                print("WARNING: All log probabilities are very negative, indicating low confidence")
            
            # Run greedy decode
            from models.asr.decode import greedy_decode
            transcripts = greedy_decode(log_probs)
            print(f"Transcripts: {transcripts}")
            
        # Clean up
        os.remove(test_file)
        
    except Exception as e:
        print(f"Error debugging ASR model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_asr_output()