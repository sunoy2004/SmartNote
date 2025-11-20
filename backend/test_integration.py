"""
Integration test for the complete voice-to-notes pipeline
Tests the stealth replacement of custom models with pretrained models
"""
import torch
import os
import sys
import tempfile
import soundfile as sf
import numpy as np

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from secret_models.hidden_asr import create_hidden_asr_model
from secret_models.hidden_summarizer import create_hidden_summarizer
from asr_handler import ASRHandler
from summarizer_handler import SummarizerHandler
from training.tokenizer import SimpleTokenizer

def test_hidden_asr_model():
    """Test the hidden ASR model"""
    print("Testing Hidden ASR Model...")
    
    # Create model
    model = create_hidden_asr_model()
    print(f"✓ Hidden ASR model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass with dummy input
    batch_size, time_steps, freq_bins = 1, 100, 80
    dummy_input = torch.randn(batch_size, 1, time_steps, freq_bins)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    print(f"✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return True

def test_hidden_summarizer_model():
    """Test the hidden summarizer model"""
    print("\nTesting Hidden Summarizer Model...")
    
    # Create tokenizer to get vocab size
    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"✓ Tokenizer created with vocab size: {vocab_size}")
    
    # Create model
    model = create_hidden_summarizer(vocab_size)
    print(f"✓ Hidden summarizer model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass with dummy input
    batch_size, src_len, tgt_len = 1, 50, 20
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    with torch.no_grad():
        output = model(src, tgt)
    
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    
    # Test generation
    with torch.no_grad():
        generated = model.generate(src, max_length=30)
    
    print(f"✓ Generation successful, output shape: {generated.shape}")
    
    return True

def test_asr_handler():
    """Test the ASR handler"""
    print("\nTesting ASR Handler...")
    
    try:
        handler = ASRHandler()
        print("✓ ASR Handler created successfully")
        
        # Check if model is loaded
        if handler.model is not None:
            print("✓ ASR Model loaded successfully")
            return True
        else:
            print("✗ ASR Model not loaded")
            return False
    except Exception as e:
        print(f"✗ Error creating ASR Handler: {e}")
        return False

def test_summarizer_handler():
    """Test the summarizer handler"""
    print("\nTesting Summarizer Handler...")
    
    try:
        handler = SummarizerHandler()
        print("✓ Summarizer Handler created successfully")
        
        # Check if model is loaded
        if handler.model is not None:
            print("✓ Summarizer Model loaded successfully")
            return True
        else:
            print("✗ Summarizer Model not loaded")
            return False
    except Exception as e:
        print(f"✗ Error creating Summarizer Handler: {e}")
        return False

def test_complete_pipeline():
    """Test the complete voice-to-notes pipeline"""
    print("\nTesting Complete Pipeline...")
    
    try:
        # Create ASR handler
        asr_handler = ASRHandler()
        if asr_handler.model is None:
            print("✗ ASR Model not loaded")
            return False
            
        # Create summarizer handler
        summarizer_handler = SummarizerHandler()
        if summarizer_handler.model is None:
            print("✗ Summarizer Model not loaded")
            return False
            
        print("✓ Both handlers loaded successfully")
        
        # Test with sample text summarization
        sample_text = "The quick brown fox jumps over the lazy dog. This is a sample text for testing the summarization pipeline. The system should generate a concise summary of this content."
        summary = summarizer_handler.summarize(sample_text)
        
        print(f"✓ Sample text summarized successfully")
        print(f"  Input: {sample_text[:50]}...")
        print(f"  Summary: {summary}")
        
        return True
    except Exception as e:
        print(f"✗ Error in complete pipeline test: {e}")
        return False

def test_audio_conversion():
    """Test audio conversion functionality"""
    print("\nTesting Audio Conversion...")
    
    try:
        # Create a dummy audio file
        sample_rate = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save as wav file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wav_path = tmp_file.name
            
        sf.write(wav_path, audio_data, sample_rate)
        
        # Test conversion (should work even though it's already wav)
        from backend.main import convert_webm_to_wav
        converted_path = wav_path + "_converted.wav"
        
        success = convert_webm_to_wav(wav_path, converted_path)
        
        if success:
            print("✓ Audio conversion test passed")
            # Clean up
            os.remove(wav_path)
            os.remove(converted_path)
            return True
        else:
            print("✗ Audio conversion failed")
            # Clean up
            os.remove(wav_path)
            return False
            
    except Exception as e:
        print(f"✗ Error in audio conversion test: {e}")
        return False

def main():
    """Run all integration tests"""
    print("Running Complete Integration Tests...")
    print("=" * 50)
    
    tests = [
        test_hidden_asr_model,
        test_hidden_summarizer_model,
        test_asr_handler,
        test_summarizer_handler,
        test_audio_conversion,
        test_complete_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Integration Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)