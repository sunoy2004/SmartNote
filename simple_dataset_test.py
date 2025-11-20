"""
Simple dataset test script
"""
import sys
sys.path.append(".")

def test_dataset():
    """Test dataset loading"""
    try:
        from training.dataset_loader import ASRDataset
        print("Creating dataset...")
        dataset = ASRDataset(split='train', subset_percent=1)
        print(f"Dataset created with {len(dataset)} samples")
        
        if len(dataset) > 0:
            print("Getting first sample...")
            sample = dataset[0]
            print(f"Sample retrieved with {len(sample)} elements")
            print("Dataset loading successful!")
            return True
        else:
            print("Dataset is empty")
            return False
            
    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running simple dataset test...")
    success = test_dataset()
    if success:
        print("✅ Dataset test passed!")
    else:
        print("❌ Dataset test failed!")