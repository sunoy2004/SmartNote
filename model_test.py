"""
Simple model test to verify that our models can be instantiated and run
"""
import torch
import sys
import os

# Add current directory to path
sys.path.append(".")

def test_asr_model():
    """Test ASR model instantiation"""
    try:
        from models.asr.asr_model import create_model
        
        # Create model
        model = create_model()
        print("✓ ASR model created successfully")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 1, 100, 80)  # [batch, channel, time, freq]
        with torch.no_grad():
            output = model(dummy_input)
            print(f"  Forward pass output shape: {output.shape}")
            
        print("✓ ASR model forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ ASR model test failed: {e}")
        return False

def test_summarizer_model():
    """Test Summarizer model instantiation"""
    try:
        from models.summarizer.transformer import create_summarizer
        
        # Create model with small vocab size for testing
        model = create_summarizer(vocab_size=100)
        print("✓ Summarizer model created successfully")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with dummy input
        dummy_src = torch.randint(0, 100, (1, 50))  # [batch, seq_len]
        dummy_tgt = torch.randint(0, 100, (1, 50))  # [batch, seq_len]
        
        with torch.no_grad():
            output = model(dummy_src, dummy_tgt)
            print(f"  Forward pass output shape: {output.shape}")
            
        print("✓ Summarizer model forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Summarizer model test failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*50)
    print("Model Test Script")
    print("="*50)
    
    success = True
    success &= test_asr_model()
    print()
    success &= test_summarizer_model()
    
    print("\n" + "="*50)
    if success:
        print("✓ All model tests passed!")
        print("Models are ready for training.")
    else:
        print("✗ Some model tests failed.")
    print("="*50)

if __name__ == "__main__":
    main()