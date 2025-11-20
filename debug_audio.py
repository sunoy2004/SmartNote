"""
Debug script to test audio processing
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

def test_audio_processing():
    """Test audio processing directly"""
    try:
        # Create a test WAV file
        import numpy as np
        import soundfile as sf
        
        # Generate 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        waveform = np.zeros(samples)
        
        # Save as WAV file
        test_file = "debug_test.wav"
        sf.write(test_file, waveform, sample_rate)
        print(f"Created test file: {test_file}")
        
        # Test audio processor directly
        from training.utils_audio import AudioProcessor
        processor = AudioProcessor()
        
        print("Testing audio processor...")
        mel_spec = processor.process_audio(test_file)
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        print(f"Mel spectrogram dtype: {mel_spec.dtype}")
        
        # Test ASR handler directly
        from backend.asr_handler import ASRHandler
        asr_handler = ASRHandler(supabase_connector=None)
        
        print("Testing ASR transcription...")
        transcript = asr_handler.transcribe(test_file)
        print(f"Transcript: '{transcript}'")
        
        # Test summarizer handler
        from backend.summarizer_handler import SummarizerHandler
        summarizer_handler = SummarizerHandler(supabase_connector=None)
        
        print("Testing summarization...")
        summary = summarizer_handler.summarize(transcript)
        print(f"Summary: '{summary}'")
        
        # Clean up
        os.remove(test_file)
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_processing()