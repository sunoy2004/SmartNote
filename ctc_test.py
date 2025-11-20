"""
Simple CTC test to verify the fix
"""
import sys
sys.path.append(".")
import torch

def test_ctc_compatibility():
    """Test that the data is compatible with CTC loss"""
    try:
        from training.dataset_loader import get_asr_dataloaders
        print("Creating dataloaders...")
        train_loader, val_loader, test_loader = get_asr_dataloaders(batch_size=1, subset_percent=1)
        print("Dataloaders created successfully")
        
        # Get one batch to test
        print("Getting first batch...")
        for batch in train_loader:
            inputs, targets, input_lengths, target_lengths = batch
            print(f"Batch shapes: inputs={inputs.shape}, targets={targets.shape}")
            print(f"Lengths: input_lengths={input_lengths}, target_lengths={target_lengths}")
            
            # Verify that input_lengths are valid
            max_input_length = inputs.size(2)  # time dimension
            print(f"Max input length: {max_input_length}")
            print(f"Actual input lengths: {input_lengths}")
            
            # Check if all input lengths are within valid range
            if torch.all(input_lengths <= max_input_length):
                print("✅ Input lengths are valid for CTC loss")
                return True
            else:
                print("❌ Input lengths exceed maximum allowed by CTC loss")
                return False
        
        print("No batches found")
        return False
        
    except Exception as e:
        print(f"CTC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running CTC compatibility test...")
    success = test_ctc_compatibility()
    if success:
        print("✅ CTC compatibility test passed!")
    else:
        print("❌ CTC compatibility test failed!")