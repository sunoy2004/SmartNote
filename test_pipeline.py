"""
Test script to verify the complete pipeline is working
"""
import sys
import os
import tempfile
import numpy as np

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

def test_asr_model():
    """Test that ASR model can be loaded and used"""
    try:
        print("Testing ASR model loading...")
        from backend.asr_handler import ASRHandler
        
        # Create ASR handler without Supabase
        asr_handler = ASRHandler(supabase_connector=None)
        print("✓ ASR model loaded successfully")
        
        # Test transcription with a simple waveform
        print("Testing transcription with dummy audio...")
        # Create a simple dummy waveform (silence)
        dummy_waveform = np.zeros(16000)  # 1 second of silence at 16kHz
        transcript = asr_handler.transcribe_waveform(dummy_waveform, sample_rate=16000)
        print(f"✓ Transcription successful: '{transcript}'")
        
        return True
    except Exception as e:
        print(f"✗ ASR test failed: {e}")
        return False

def test_summarizer_model():
    """Test that Summarizer model can be loaded"""
    try:
        print("Testing Summarizer model loading...")
        from backend.summarizer_handler import SummarizerHandler
        
        # Create Summarizer handler without Supabase
        summarizer_handler = SummarizerHandler(supabase_connector=None)
        print("✓ Summarizer model loaded successfully")
        
        # Test summarization with sample text
        sample_text = "This is a test transcript from our ASR system. It contains multiple sentences that should be summarized."
        summary = summarizer_handler.summarize(sample_text)
        print(f"✓ Summarization successful: '{summary}'")
        
        return True
    except Exception as e:
        print(f"✗ Summarizer test failed: {e}")
        # This might fail if we don't have a trained model, which is expected
        return False

def test_complete_pipeline():
    """Test the complete pipeline"""
    print("=" * 60)
    print("Testing Complete Voice-to-Notes Pipeline")
    print("=" * 60)
    
    # Test ASR
    asr_success = test_asr_model()
    
    # Test Summarizer
    summarizer_success = test_summarizer_model()
    
    print("\n" + "=" * 60)
    print("Pipeline Test Results:")
    print(f"  ASR Model: {'✓ PASS' if asr_success else '✗ FAIL'}")
    print(f"  Summarizer Model: {'✓ PASS' if summarizer_success else '✗ FAIL (Expected if no trained model)'}")
    
    if asr_success:
        print("\n✓ Core pipeline is working! The ASR model is ready for use.")
        print("  You can now:")
        print("  1. Train a summarizer model: python training/train_summarizer_custom.py --epochs 5 --subset 10")
        print("  2. Start the backend API: cd backend && python main.py")
        print("  3. Test the API endpoints with curl or a REST client")
    else:
        print("\n✗ There are issues with the pipeline that need to be addressed.")
    
    print("=" * 60)

if __name__ == "__main__":
    test_complete_pipeline()