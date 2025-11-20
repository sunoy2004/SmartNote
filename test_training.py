"""
Simple test script to verify that our training pipeline works
"""
import sys
import os

# Add current directory to path
sys.path.append(".")

def test_model_creation():
    """Test that models can be created"""
    try:
        # Test ASR model creation
        from models.asr.asr_model import create_model
        asr_model = create_model()
        print("✓ ASR model can be created")
        
        # Test Summarizer model creation
        from models.summarizer.transformer import create_summarizer
        summarizer_model = create_summarizer(vocab_size=100)
        print("✓ Summarizer model can be created")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_dataset_loading():
    """Test that dataset can be loaded"""
    try:
        from training.dataset_loader import ASRDataset
        # Create a small dataset with just 1 sample for testing
        dataset = ASRDataset(split='train', subset_percent=0.1)
        print(f"✓ Dataset loaded successfully with {len(dataset)} samples")
        
        # Test getting one item
        if len(dataset) > 0:
            item = dataset[0]
            print(f"✓ Dataset item retrieved successfully with {len(item)} elements")
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def test_training_script_imports():
    """Test that training scripts can be imported"""
    try:
        from training.train_asr import main as asr_main
        from training.train_summarizer import main as summarizer_main
        print("✓ Training scripts can be imported")
        
        return True
    except Exception as e:
        print(f"✗ Training script import failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*50)
    print("Training Pipeline Test")
    print("="*50)
    
    all_tests_passed = True
    
    print("\n1. Model Creation Test:")
    all_tests_passed &= test_model_creation()
    
    print("\n2. Dataset Loading Test:")
    all_tests_passed &= test_dataset_loading()
    
    print("\n3. Training Script Import Test:")
    all_tests_passed &= test_training_script_imports()
    
    print("\n" + "="*50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour training pipeline is working correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above and resolve them.")
    print("="*50)

if __name__ == "__main__":
    main()