"""
Debug CTC issue to understand the length mismatch
"""
import sys
sys.path.append(".")
import torch

def debug_ctc_issue():
    """Debug the CTC length issue"""
    try:
        from training.dataset_loader import get_asr_dataloaders
        print("Creating dataloaders...")
        train_loader, val_loader, test_loader = get_asr_dataloaders(batch_size=1, subset_percent=1)
        print("Dataloaders created successfully")
        
        # Get one batch to debug
        print("Getting first batch...")
        for batch in train_loader:
            inputs, targets, input_lengths, target_lengths = batch
            print(f"Inputs shape: {inputs.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Input lengths: {input_lengths}")
            print(f"Target lengths: {target_lengths}")
            
            # Check the actual dimensions
            batch_size = inputs.size(0)
            channels = inputs.size(1)
            time_steps = inputs.size(2)  # This should match input_lengths
            freq_bins = inputs.size(3)
            
            print(f"Actual dimensions - Batch: {batch_size}, Channels: {channels}, Time: {time_steps}, Freq: {freq_bins}")
            
            # Check if input_lengths match the actual time dimension
            for i in range(batch_size):
                actual_time = inputs[i].size(1)  # time dimension for this sample
                reported_length = input_lengths[i].item()
                print(f"Sample {i}: Actual time={actual_time}, Reported length={reported_length}")
                
                if actual_time != reported_length:
                    print(f"❌ MISMATCH: Sample {i} has time dimension {actual_time} but reported length {reported_length}")
                else:
                    print(f"✅ MATCH: Sample {i} has consistent dimensions")
            
            # Test CTC loss directly
            print("\nTesting CTC loss...")
            from models.asr.asr_model import create_model
            model = create_model()
            model.eval()
            
            with torch.no_grad():
                log_probs = model(inputs)
                print(f"Model output shape: {log_probs.shape}")
                
                # Transpose for CTC: [time, batch, num_classes]
                log_probs_t = log_probs.transpose(0, 1)
                print(f"Transposed for CTC: {log_probs_t.shape}")
                
                # Try CTC loss
                criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
                try:
                    loss = criterion(log_probs_t, targets, input_lengths, target_lengths)
                    print(f"✅ CTC loss computation successful: {loss.item()}")
                    return True
                except Exception as e:
                    print(f"❌ CTC loss computation failed: {e}")
                    return False
        
        print("No batches found")
        return False
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running CTC debug...")
    success = debug_ctc_issue()
    if success:
        print("✅ CTC debug completed successfully!")
    else:
        print("❌ CTC debug failed!")