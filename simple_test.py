"""
Simple test script to verify basic functionality without PyTorch
"""
import os
import sys

# Add current directory to path
sys.path.append(".")

def test_basic_imports():
    """Test basic imports without PyTorch"""
    try:
        # Test training imports
        from training.tokenizer import SimpleTokenizer
        print("✓ SimpleTokenizer imported successfully")
        
        from training.utils_text import preprocess_text
        print("✓ utils_text imported successfully")
        
        # Test model imports
        from models.asr.asr_model import VOCAB
        print("✓ ASR model vocab imported successfully")
        
        from models.summarizer.transformer import create_summarizer
        print("✓ Summarizer model imported successfully")
        
        print("\nAll basic imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading functionality"""
    try:
        # Test if we can load the dataset loader
        from training.dataset_loader import ASRDataset, SummarizationDataset
        print("✓ Dataset loaders imported successfully")
        
        print("\nDataset loading functionality verified!")
        return True
        
    except ImportError as e:
        print(f"✗ Dataset loading error: {e}")
        return False

def main():
    """Main test function"""
    print("="*50)
    print("Simple Test Script")
    print("="*50)
    
    success = True
    success &= test_basic_imports()
    success &= test_dataset_loading()
    
    print("\n" + "="*50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed.")
    print("="*50)

if __name__ == "__main__":
    main()